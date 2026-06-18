# vllm-ascend CPU 卸载方案总览

本文档系统整理 vllm-ascend 代码库中所有与 CPU 卸载（CPU Offloading）相关的方案，包括各自的原理、配置项与使用方法。

## 目录

- [一、方案总览与选型矩阵](#一方案总览与选型矩阵)
- [二、KV Cache CPU 卸载（三套实现）](#二kv-cache-cpu-卸载三套实现)
  - [2.1 NPUOffloadingSpec（基于 OffloadingConnector）](#21-npuoffloadingspec基于-offloadingconnector)
  - [2.2 AscendSimpleCPUOffloadConnector（上游 SimpleCPUOffloadConnector 的 NPU 适配）](#22-ascendsimplecpuoffloadconnector上游-simplecpuoffloadconnector-的-npu-适配)
  - [2.3 CPUOffloadingConnector（自带 Prefix Caching 的独立实现）](#23-cpuoffloadingconnector自带-prefix-caching-的独立实现)
  - [2.4 三套 KV Cache 卸载实现对比](#24-三套-kv-cache-卸载实现对比)
- [三、模型权重 CPU 卸载（NPUPrefetchOffloader）](#三模型权重-cpu-卸载npuprefetchoffloader)
- [四、Sleep Mode（基于 CaMemAllocator 的权重/KV 卸载）](#四sleep-mode基于-camemallocator-的权重kv-卸载)
- [五、CaMem 分配器（CANN 内存可插拔分配器）](#五camem-分配器cann-内存可插拔分配器)
- [六、CPU 绑定（CPU Binding，主机侧优化）](#六cpu-绑定cpu-binding主机侧优化)
- [七、SSD 卸载（Mooncake 后端的 SSD Offload）](#七ssd-卸载mooncake-后端的-ssd-offload)
- [八、权重预取与离线权重加载](#八权重预取与离线权重加载)
- [九、相关环境变量与配置项汇总](#九相关环境变量与配置项汇总)
- [十、底层 C++ 实现：swap_blocks_batch](#十底层-c-实现swap_blocks_batch)

---

## 一、方案总览与选型矩阵

vllm-ascend 中存在 **多套不同层次、不同目的的 CPU 卸载方案**，可归纳为下表：

| 方案 | 卸载对象 | 卸载目标 | 触发方式 | 典型场景 |
|------|----------|----------|----------|----------|
| **NPUOffloadingSpec** | KV Cache | CPU pinned mem | LRU 自动淘汰 | 单实例 NPU 显存不足，长上下文 |
| **AscendSimpleCPUOffloadConnector** | KV Cache | CPU pinned mem | 上游 SimpleCPUOffloadScheduler | 与上游 API 一致的简单 KV 卸载 |
| **CPUOffloadingConnector** | KV Cache | CPU 共享内存 | prefix cache 命中/淘汰 | 跨 DP 共享 CPU KV、大容量 swap、需 CPU prefix caching |
| **NPUPrefetchOffloader** | 模型权重 | CPU pinned mem | 分组轮转预取 | 显存不足运行大模型，权重 > 显存 |
| **Sleep Mode (Level 1)** | 模型权重 | CPU pinned mem | `llm.sleep(level=1)` | RL 训练阶段释放显存 |
| **Sleep Mode (Level 2)** | 权重 + KV | 丢弃（不保留） | `llm.sleep(level=2)` | 切换模型 |
| **CaMemAllocator** | NPU 显存（通用） | CPU pinned mem | sleep/wake_up API | Sleep Mode 底层基础设施 |
| **CPU Binding** | （非卸载，CPU 亲和性） | - | 自动（默认启用） | ARM 多路服务器主机侧优化 |
| **Mooncake SSD Offload** | KV Cache | SSD 磁盘 | mooncake.json 配置 | KV Pool 跨节点共享、超大容量 |
| **Weight Prefetch** | （非卸载，预取优化） | - | additional_config | decode 阶段权重预取加速 |
| **Offline Weight Load** | 模型权重 | CPU→NPU 重载 | sleep + manual load | RL 训练后权重更新 |

---

## 二、KV Cache CPU 卸载（三套实现）

vllm-ascend 中存在 **三套独立的 KV Cache CPU 卸载实现**，分别面向不同场景。

### 2.1 NPUOffloadingSpec（基于 OffloadingConnector）

**文件路径：**
- [vllm_ascend/kv_offload/npu.py](file:///workspace/vllm_ascend/kv_offload/npu.py) - `NPUOffloadingSpec` 类
- [vllm_ascend/kv_offload/cpu_npu.py](file:///workspace/vllm_ascend/kv_offload/cpu_npu.py) - `CpuNpuOffloadingHandler` 类
- [docs/source/user_guide/feature_guide/kv_cache_cpu_offload.md](file:///workspace/docs/source/user_guide/feature_guide/kv_cache_cpu_offload.md) - 用户文档

**关键类与函数签名：**

```python
# npu.py
class NPUOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None = None)
    def get_manager(self) -> OffloadingManager  # 返回 CPUOffloadingManager（scheduler 侧）
    def get_handlers(self, kv_caches, attn_backends) -> Iterator[...]  # 返回 CpuNpuOffloadingHandler（worker 侧）

# cpu_npu.py
@dataclass
class Transfer:
    job_id: int
    stream: torch.npu.Stream
    start_event: torch.npu.Event
    end_event: torch.npu.Event
    num_bytes: int

class CpuNpuOffloadingHandler(OffloadingHandler):
    def __init__(self, gpu_block_size, cpu_block_size, num_cpu_blocks, gpu_caches, attn_backends)
    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool
    def get_finished(self) -> list[TransferResult]
    def wait(self, job_ids: set[int]) -> None
```

**工作原理：**
1. 基于 vLLM v1 的 `OffloadingConnector` 框架，提供 Ascend NPU 专用实现。
2. `CpuNpuOffloadingHandler` 使用两条独立的 NPU Stream：`d2h_stream`（NPU→CPU）和 `h2d_stream`（CPU→NPU），实现异步传输。
3. CPU 端预分配 `num_cpu_blocks` 个 block 的 pinned memory 张量（每个 layer 一对 key/value）。
4. 通过 `torch.ops._C_ascend.swap_blocks_batch` 调用底层 `aclrtMemcpyBatchAsync` 进行批量 DMA 拷贝。
5. 使用 numpy 向量化广播构建扁平指针数组，避免 Python 循环开销。
6. 使用 Event 池（`_event_pool`）避免重复分配 Event 对象。
7. CPU block 大小需为 NPU block 大小的整数倍（`block_size_factor = cpu_block_size // gpu_block_size`）。

**执行流程：**
1. 正常推理时，KV cache 块在 NPU 上计算并存储。
2. NPU 显存满时，不活跃的 KV cache 块通过专用 D2H NPU stream 异步拷贝到 CPU 内存。
3. 当请求与之前计算的数据共享前缀，且 NPU 上 prefix cache 未命中但 CPU 内存中存在时，通过专用 H2D NPU stream 异步回载。
4. CPU block 池使用 LRU 淘汰策略管理有限 CPU 内存。

**配置参数（通过 `kv_connector_extra_config`）：**

| 参数 | 说明 |
|------|------|
| `kv_connector` | 必须为 `"OffloadingConnector"` |
| `kv_role` | 设为 `"kv_both"` 同时启用存储和加载 |
| `num_cpu_blocks` | CPU 内存中分配的 block 数量 |
| `block_size` | CPU 侧 block 大小（典型值 128），应为 NPU block 大小的整数倍 |
| `spec_name` | 必须为 `"NPUOffloadingSpec"` |
| `spec_module_path` | 必须为 `"vllm_ascend.kv_offload.npu"` |

**使用示例（Python API）：**

```python
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "num_cpu_blocks": 1000,
        "block_size": 128,
        "spec_name": "NPUOffloadingSpec",
        "spec_module_path": "vllm_ascend.kv_offload.npu",
    },
)

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    gpu_memory_utilization=0.5,
    kv_transfer_config=kv_transfer_config,
)

sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
outputs = llm.generate(["Hello, my name is"], sampling_params)
```

**使用示例（在线服务）：**

```bash
vllm serve Qwen/Qwen3-0.6B \
    --gpu-memory-utilization 0.5 \
    --kv-transfer-config '{
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "num_cpu_blocks": 1000,
            "block_size": 128,
            "spec_name": "NPUOffloadingSpec",
            "spec_module_path": "vllm_ascend.kv_offload.npu"
        }
    }'
```

**适用场景：** 单实例内 NPU 显存不足时，将不活跃的 KV cache 块卸载到 CPU 内存，支持 LRU 淘汰与 prefix cache 命中后异步回载。

---

### 2.2 AscendSimpleCPUOffloadConnector（上游 SimpleCPUOffloadConnector 的 NPU 适配）

**文件路径：**
- [vllm_ascend/simple_kv_offload/__init__.py](file:///workspace/vllm_ascend/simple_kv_offload/__init__.py) - 模块说明
- [vllm_ascend/simple_kv_offload/copy_backend.py](file:///workspace/vllm_ascend/simple_kv_offload/copy_backend.py) - `NPUDmaCopyBackend` 类
- [vllm_ascend/simple_kv_offload/npu_mem_ops.py](file:///workspace/vllm_ascend/simple_kv_offload/npu_mem_ops.py) - `BatchMemcpyParams`、`build_params`、`copy_blocks`
- [vllm_ascend/simple_kv_offload/worker.py](file:///workspace/vllm_ascend/simple_kv_offload/worker.py) - `SimpleCPUOffloadNPUWorker` 类
- [vllm_ascend/distributed/kv_transfer/kv_pool/simple_cpu_offload/simple_cpu_offload_connector.py](file:///workspace/vllm_ascend/distributed/kv_transfer/kv_pool/simple_cpu_offload/simple_cpu_offload_connector.py) - `AscendSimpleCPUOffloadConnector` 类

**注册位置：** [vllm_ascend/distributed/kv_transfer/__init__.py](file:///workspace/vllm_ascend/distributed/kv_transfer/__init__.py) 第 67-81 行覆盖了上游 `SimpleCPUOffloadConnector` 注册。

**关键类与函数签名：**

```python
# npu_mem_ops.py
DIRECTION_H2D = 0
DIRECTION_D2H = 1

class BatchMemcpyParams(NamedTuple):
    src_bases: np.ndarray  # 每个子张量的 data_ptr
    dst_bases: np.ndarray
    bpb: np.ndarray  # bytes per block
    num_sub_tensors: int
    direction: int

def build_params(src_caches, dst_caches, direction) -> BatchMemcpyParams
def copy_blocks(src_block_ids, dst_block_ids, params: BatchMemcpyParams) -> None

# copy_backend.py
class NPUDmaCopyBackend:
    def init(self, npu_caches, cpu_caches, device, load_stream, store_stream) -> None
    def launch_copy(self, src_blocks, dst_blocks, is_store, event_idx, events_list) -> None
    def shutdown(self) -> None
    def _copy_loop(self) -> None  # 后台工作线程主循环

# worker.py
class SimpleCPUOffloadNPUWorker(SimpleCPUOffloadWorker):
    def __init__(self, vllm_config, kv_cache_config, cpu_capacity_bytes)
    def register_kv_caches(self, kv_caches) -> None

# simple_cpu_offload_connector.py
class AscendSimpleCPUOffloadConnector(SimpleCPUOffloadConnector):
    def __init__(self, vllm_config, role, kv_cache_config=None) -> None
```

**工作原理：**
1. 这是 vLLM 上游 `SimpleCPUOffloadConnector` 的 NPU 适配版本，scheduler 侧逻辑完全复用上游（平台无关），仅替换 worker 侧的 CUDA 拷贝后端为 NPU 版本。
2. `NPUDmaCopyBackend` 在独立的后台守护线程中执行拷贝任务（FIFO 队列），每个任务在专用 NPU stream 上发出拷贝并记录 Event，主线程可轮询 Event 而无需同步设备。
3. 预构建两套 `BatchMemcpyParams`：`_store_params`（D2H，NPU→CPU）和 `_load_params`（H2D，CPU→NPU）。
4. `SimpleCPUOffloadNPUWorker.register_kv_caches` 处理 NPU 特有布局：
   - K/V 在 Ascend 上是 **分离的独立分配**（不像 CUDA 那样堆叠在一个外层维度下）。
   - runner 会为每个 KV 张量过度分配 `+alignment`（2 MiB）用于对齐，因此 view 大小必须来自张量自身的 shape/stride，而非 `storage.nbytes()`。
   - 通过 `untyped_storage().data_ptr()` 去重，避免同一 NPU 分配被多次注册。
   - CPU 镜像通过 `torch.zeros(pin_memory=True)` 分配（因为 `cudaHostRegister` 是 CUDA 专有的）。
   - `torch.npu.Stream` 不支持 `priority=` 参数，因此使用普通传输 stream。

**配置方式：** 使用上游的 `SimpleCPUOffloadConnector` 名称即可（已被 vllm-ascend 自动覆盖为 NPU 版本）。

**适用场景：** 与上游 SimpleCPUOffloadConnector 行为一致，适合需要简单 CPU KV 卸载且希望与上游保持一致 API 的用户。

---

### 2.3 CPUOffloadingConnector（自带 Prefix Caching 的独立实现）

**文件路径：**
- [vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py](file:///workspace/vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py) - `CPUOffloadingConnector`、`CPUOffloadingConnectorScheduler`、`CPUOffloadingConnectorWorker`
- [vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_kv_cache_manager.py](file:///workspace/vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_kv_cache_manager.py) - `CPUKVCacheManager`、`CPUCacheStats`
- [vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/metadata.py](file:///workspace/vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/metadata.py) - `MetadataServer`、`MetadataServerProc`、`MLAConfig`

**关键类与函数签名：**

```python
# cpu_offload_connector.py
@dataclass
class ReqMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]
    num_scheduled_tokens: int
    num_computed_tokens: int
    num_gpu_computed_tokens: int
    num_cpu_computed_tokens: int

class CPUOffloadingConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config=None)

class CPUOffloadingConnectorScheduler:
    def __init__(self, vllm_config)
    def get_num_new_matched_tokens(self, ori_request, num_computed_tokens) -> tuple[int, bool]
    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata
    def request_finished(self, ori_request)

class CPUOffloadingConnectorWorker:
    def __init__(self, vllm_config)
    def register_kv_caches(self, kv_caches)
    def start_load_kv(self) -> None
    def wait_for_layer_load(self) -> None
    def load_kv_layer(self, layer: int)
    def get_finished(self) -> set[str]
    def _save_listener(self)  # 后台保存线程

# cpu_kv_cache_manager.py
class CPUKVCacheManager:
    def __init__(self, kv_cache_spec, num_cpu_blocks, caching_hash_algo="builtin", ...)
    def get_matched_num_and_touch(self, request) -> tuple[int, bool]  # CPU prefix cache 命中
    def allocate_slots(self, req_to_num_tokens, unallocated_req_ids) -> dict[str, list[int]]
    def cache_and_free_slots(self, request_id: str)

# metadata.py
class MetadataServer:
    METADATA_SERVER_ADDRESS = f"ipc://{envs.VLLM_RPC_BASE_PATH}/metadata.ipc"
    DEFAULT_CPU_SWAP_SPACE_GB = 800
    class ZMQRPCClient: ...
    def init_cpu_kv_caches(self, pp_rank, tp_rank, kv_cache_specs, mla_config) -> ...
    def post_init(self)  # 初始化 CPUKVCacheManager
    def serve_step(self)  # ZMQ RPC 服务循环

class MetadataServerProc:
    @staticmethod
    def run_metadata_server(vllm_config)  # 独立进程入口
```

**工作原理（与前两套显著不同）：**

这是一套 **完整的、独立的 CPU KV Cache 卸载实现**，自带 prefix caching 能力。架构分为三层：

1. **Scheduler 侧**（`CPUOffloadingConnectorScheduler`）：通过 ZMQ RPC 与 Metadata Server 通信，查询 CPU prefix cache 命中数、分配 CPU block 槽位。
2. **Worker 侧**（`CPUOffloadingConnectorWorker`）：使用 `torch.npu.Stream`（`load_stream`/`save_stream`）进行 NPU↔CPU 的异步 `copy_`，逐 layer 加载/保存 KV cache。后台 `_save_listener` 线程处理请求完成后的 KV 保存。
3. **Metadata Server**（`MetadataServer`/`MetadataServerProc`）：独立进程，仅在 `data_parallel_rank==0 && tp_rank==0 && pp_rank==0` 启动。使用 ZMQ ROUTER/DEALER socket 提供 RPC 服务，管理 `CPUKVCacheManager`（含 BlockPool、prefix cache 哈希、LRU 等）。

**关键特性：**
- **CPU 内存使用共享内存（`SharedMemory`）** 跨进程共享 KV cache 张量，通过 `torch.frombuffer(shm.buf, ...)` 在 worker 进程中重建张量视图。
- **CPU 容量配置：** `cpu_swap_space_gb`（默认 800 GB），通过 `kv_connector_extra_config` 传入。
- **swap_in_threshold：** 控制何时触发 CPU→NPU 的 swap-in。
- **MLA 支持：** 通过 `MLAConfig(nope_dim, rope_dim)` 处理 DeepSeek MLA 的 rope/nope 分离存储；MLA 时 KV cache 在不同 TP 间共享（`tp_rank=0`）。
- **TP 协调：** worker 通过 `tp_group.send_object/recv_object` 协调多 TP rank 的完成状态，只有所有 TP rank 都完成发送后才释放 CPU 槽位。

**配置参数：**

| 参数 | 说明 |
|------|------|
| `kv_connector` | `"CPUOffloadingConnector"` |
| `kv_role` | `"kv_both"` 等 |
| `kv_connector_extra_config.cpu_swap_space_gb` | CPU swap 空间大小（GB），默认 800 |
| `kv_connector_extra_config.swap_in_threshold` | 触发 swap-in 的阈值 |
| 前提条件 | `enable_prefix_caching=True`（否则 connector 为 no-op） |

**适用场景：** 需要 CPU 侧 prefix caching 能力、跨 DP 共享 CPU KV cache、大容量 CPU swap 空间（数百 GB）的场景。

---

### 2.4 三套 KV Cache 卸载实现对比

| 特性 | NPUOffloadingSpec | AscendSimpleCPUOffloadConnector | CPUOffloadingConnector |
|------|------|------|------|
| 基础框架 | vLLM OffloadingConnector | vLLM SimpleCPUOffloadConnector（上游适配） | 独立实现 KVConnectorBase_V1 |
| CPU Prefix Caching | 通过 CPUOffloadingManager（LRU） | 复用上游 SimpleCPUOffloadScheduler | 自带 CPUKVCacheManager（完整 prefix cache） |
| 跨进程共享 | 否 | 否 | 是（SharedMemory + ZMQ RPC） |
| 传输底层 | `swap_blocks_batch`（aclrtMemcpyBatchAsync） | `swap_blocks_batch`（aclrtMemcpyBatchAsync） | `torch.Tensor.copy_`（non_blocking） |
| CPU 容量配置 | `num_cpu_blocks` | `cpu_capacity_bytes` | `cpu_swap_space_gb`（默认 800GB） |
| MLA 支持 | 通过 spec | 通过 spec | 显式 MLAConfig |
| 独立进程 | 否 | 否 | 是（MetadataServerProc） |

---

## 三、模型权重 CPU 卸载（NPUPrefetchOffloader）

**文件路径：**
- [vllm_ascend/model_executor/offloader/prefetch.py](file:///workspace/vllm_ascend/model_executor/offloader/prefetch.py) - `NPUPrefetchOffloader`、`_NPUModuleOffloader`
- [vllm_ascend/worker/model_runner_v1.py](file:///workspace/vllm_ascend/worker/model_runner_v1.py) 第 277-290 行（集成位置）

**关键类与函数签名：**

```python
class NPUPrefetchOffloader(BaseOffloader):
    def __init__(self, group_size, num_in_group, prefetch_step, offload_params=None, mode="cpu")
    def wrap_modules(self, modules_generator) -> list[nn.Module]
    def _hook_module_forward(self, index, module)
    def _wait_for_layer(self, layer_idx)
    def _start_prefetch(self, layer_idx)
    def sync_prev_onload(self)
    def join_after_forward(self)
    def post_init(self)

class _NPUModuleOffloader:
    def __init__(self, mode, module, copy_stream, whitelist_param_names, layer_idx)
    def post_init(self)
    def sync_cpu_storage(self)
    def get_param_infos(self) -> list[ParamInfo]
    def assign_buffer_slot(self, pool, slot_idx)
    def start_onload_to_static()
```

**工作原理：**
1. 这是 vLLM 上游 `PrefetchOffloader` 的 NPU 移植版，将所有 `torch.cuda.*` API 替换为 `torch.npu.*`。
2. **核心思想：** 将模型权重分组（`group_size`），其中 `num_in_group` 个 module 的权重卸载到 CPU，仅保留 `prefetch_step` 个 module 的权重在 NPU 显存中循环复用（Static Buffer Pool）。
3. **预取机制：** 在当前 layer forward 时，异步预取后续 `prefetch_step` 个 layer 的权重到 NPU（通过 `copy_stream`）；同时通过 `torch.ops.vllm.wait_prefetch` / `torch.ops.vllm.start_prefetch` 自定义 op 实现流同步。
4. **ACL Graph 兼容：** 通过 `torch.npu.is_current_stream_capturing()` 区分 eager/graph 模式，在 graph capture 时使用 Event 同步，eager 模式使用 stream 同步。
5. **pinned memory：** CPU 存储使用 pinned memory（`should_pin_memory()`）以获得最佳 H2D 带宽。
6. **StaticBufferPool：** 预分配 `prefetch_step` 个 slot 的静态 buffer 池，避免运行时分配。

**配置方式（通过 vLLM 的 `offload_config.prefetch`）：**

```python
# 在 NPUModelRunner.__init__ 中自动替换
offload_cfg = vllm_config.offload_config
if (offload_cfg is not None
        and getattr(offload_cfg, "prefetch", None) is not None
        and getattr(offload_cfg.prefetch, "offload_group_size", 0) > 0):
    set_offloader(NPUPrefetchOffloader(
        group_size=offload_cfg.prefetch.offload_group_size,
        num_in_group=offload_cfg.prefetch.offload_num_in_group,
        prefetch_step=offload_cfg.prefetch.offload_prefetch_step,
        offload_params=offload_cfg.prefetch.offload_params,
    ))
```

**ACL Graph 集成：** [vllm_ascend/compilation/acl_graph.py](file:///workspace/vllm_ascend/compilation/acl_graph.py) 在 graph capture 前后调用 `get_offloader().sync_prev_onload()` 和 `join_after_forward()` 确保拷贝流同步。

**限制：** 当 `offload_config.uva.cpu_offload_gb == 0` 时才允许重新初始化 input batch，即 **UVA CPU offload 与 input batch 重初始化互斥**。

**适用场景：** NPU 显存不足以容纳完整模型权重时，通过分组轮转 + 异步预取运行大模型。

---

## 四、Sleep Mode（基于 CaMemAllocator 的权重/KV 卸载）

**文件路径：**
- [vllm_ascend/worker/worker.py](file:///workspace/vllm_ascend/worker/worker.py) 第 200-254 行 - `NPUWorker.sleep()` / `wake_up()` 方法
- [vllm_ascend/device_allocator/camem.py](file:///workspace/vllm_ascend/device_allocator/camem.py) - `CaMemAllocator` 类（核心实现）
- [examples/offline_inference_sleep_mode_npu.py](file:///workspace/examples/offline_inference_sleep_mode_npu.py) - 离线推理示例
- [examples/offline_weight_load.py](file:///workspace/examples/offline_weight_load.py) - 带 sleep mode 的权重加载示例
- [docs/source/user_guide/feature_guide/sleep_mode.md](file:///workspace/docs/source/user_guide/feature_guide/sleep_mode.md) - 用户文档

**关键 API：**

```python
# worker.py
class NPUWorker:
    def sleep(self, level: int = 1) -> None
    def wake_up(self, tags: list[str] | None = None) -> None
```

**两个 Sleep Level：**

- **Level 1 Sleep：**
  - 动作：将模型权重卸载到 CPU pinned memory，丢弃 KV cache。
  - 内存：模型权重移至 CPU 内存；KV cache 被遗忘。
  - 适用场景：后续要复用同一模型。
  - 实现：`allocator.sleep(offload_tags=("weights",))` —— weights tag 的张量卸载到 CPU，kv_cache tag 的张量直接丢弃。

- **Level 2 Sleep：**
  - 动作：丢弃模型权重和 KV cache（不保留 CPU 备份）。
  - 内存：权重和 KV cache 内容都被遗忘。
  - 适用场景：切换到不同模型或更新当前模型。
  - 实现：`allocator.sleep(offload_tags=tuple())` —— 所有张量都直接丢弃。Level 2 还会先保存 model buffers 到 CPU，wake_up 时恢复。

**Wake Up：**
- `allocator.wake_up(tags)` —— 对之前卸载的张量，重新 `create_and_map` 并通过 `ACL_MEMCPY_HOST_TO_DEVICE` 从 CPU 拷回 NPU；对丢弃的张量，仅重新分配空内存。
- wake_up 后会对 MOE 模型的 `w2_weight`/`w13_weight` 执行 `transpose(1,2)` 恢复。

**Tag 机制：**
vllm-ascend 使用两个 tag 标记 NPU 显存分配：
- `"weights"`：模型权重（在 `NPUWorker.load_model` 中，第 544-548 行）
- `"kv_cache"`：KV cache（在 `NPUWorker.initialize_from_config` 中，第 765-767 行）

**使用示例（离线推理）：**

```python
import os
import torch
from vllm import LLM, SamplingParams
from vllm.utils.mem_constants import GiB_bytes

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ASCEND_ENABLE_NZ"] = "0"

prompt = "How are you?"
free, total = torch.npu.mem_get_info()
used_bytes_baseline = total - free

llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", enable_sleep_mode=True)
sampling_params = SamplingParams(temperature=0, max_completion_tokens=10)
output = llm.generate(prompt, sampling_params)

# Level 1: 卸载权重到 CPU，丢弃 KV cache
llm.sleep(level=1)

free_npu_bytes_after_sleep, total = torch.npu.mem_get_info()
used_bytes = total - free_npu_bytes_after_sleep - used_bytes_baseline
# 0.5B 模型，1GiB 权重，sleep 后显存占用应小于 1GiB
assert used_bytes < 1 * GiB_bytes

# 从 CPU 恢复权重
llm.wake_up()
output2 = llm.generate(prompt, sampling_params)
assert output[0].outputs[0].text == output2[0].outputs[0].text
```

**使用示例（在线服务）：**

```bash
export VLLM_SERVER_DEV_MODE="1"        # 必须开启 dev mode 才暴露 sleep/wake_up 端点
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_MODELSCOPE="True"
export VLLM_ASCEND_ENABLE_NZ="0"

vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-sleep-mode

# sleep level 1
curl -X POST http://127.0.0.1:8000/sleep \
    -H "Content-Type: application/json" \
    -d '{"level": "1"}'

# 查询是否在 sleep 状态
curl -X GET http://127.0.0.1:8000/is_sleeping

# sleep level 2
curl -X POST http://127.0.0.1:8000/sleep \
    -H "Content-Type: application/json" \
    -d '{"level": "2"}'

# wake up
curl -X POST http://127.0.0.1:8000/wake_up

# 带 tag 唤醒（tags 必须在 ["weights", "kv_cache"] 中）
curl -X POST "http://127.0.0.1:8000/wake_up?tags=weights"
```

**环境变量要求：**
- `VLLM_ASCEND_ENABLE_NZ=0`（sleep mode 与 NZ 模式不兼容）
- 需要从源码编译并设置 `COMPILE_CUSTOM_KERNELS=1`（< v0.12.0rc1）
- `PYTORCH_NPU_ALLOC_CONF` 中不能包含 `expandable_segments:True`（与内存池不兼容）
- `VLLM_SERVER_DEV_MODE=1`（在线服务暴露 sleep/wake_up 端点）

**适用场景：** RL 训练（PPO/GRPO/DPO）中，generation 与 training 阶段使用不同并行策略时，释放 NPU 显存给训练阶段使用。

---

## 五、CaMem 分配器（CANN 内存可插拔分配器）

**文件路径：**
- [vllm_ascend/device_allocator/camem.py](file:///workspace/vllm_ascend/device_allocator/camem.py) - 核心实现
- [csrc/camem_allocator.cpp](file:///workspace/csrc/camem_allocator.cpp) - C++ 底层实现
- [tests/ut/device_allocator/test_camem.py](file:///workspace/tests/ut/device_allocator/test_camem.py) - 单元测试
- [vllm_ascend/patch/platform/patch_camem_allocator.py](file:///workspace/vllm_ascend/patch/platform/patch_camem_allocator.py) - 补丁

**关键 API：**

```python
# 底层 C 扩展（从 vllm_ascend_C 导入）
init_module(python_malloc_fn, python_free_func)
python_create_and_map(py_device, py_alignedSize, py_d_mem, py_p_memHandle)
python_unmap_and_release(py_device, py_alignedSize, py_d_mem, py_p_memHandle)

# Python 层
HandleType = tuple[int, int, int, int]  # py_device, py_alignedSize, py_d_mem, py_p_memHandle

@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: torch.Tensor | None = None

class CaMemAllocator:  # 单例
    instance = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CaMemAllocator"
    def python_malloc_callback(self, allocation_handle: HandleType) -> None
    def python_free_callback(self, ptr: int) -> HandleType
    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None
    def wake_up(self, tags: list[str] | None = None) -> None
    @contextmanager
    def use_memory_pool(self, tag: str | None = None)
    def get_current_usage(self) -> int
```

**工作原理（深入）：**

1. CaMemAllocator 是一个 **单例的 CANN 内存可插拔分配器**，基于 AscendCL（ACL）底层 API 实现 NPU 显存的"卸载/丢弃"语义。
2. 通过 `torch.npu.memory.NPUPluggableAllocator` 注册自定义的 malloc/free 回调。
3. **malloc 回调**（`python_malloc_callback`）：记录分配的 handle 和当前 tag 到 `pointer_to_data` 字典。
4. **free 回调**（`python_free_callback`）：从字典中取出 handle 返回给 C 层释放，并清理 `cpu_backup_tensor`。
5. **sleep 时的卸载逻辑：**
   ```python
   for ptr, data in self.pointer_to_data.items():
       if data.tag in offload_tags:
           cpu_backup_tensor = torch.empty(size_in_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
           memcpy(cpu_ptr, dest_max, ptr, size_in_bytes, ACL_MEMCPY_DEVICE_TO_HOST)
           data.cpu_backup_tensor = cpu_backup_tensor
       unmap_and_release(handle)  # 所有 handle 都释放 NPU 显存
   ```
6. **wake_up 时的恢复逻辑：**
   ```python
   for ptr, data in self.pointer_to_data.items():
       if tags is None or data.tag in tags:
           create_and_map(handle)  # 重新分配 NPU 显存
           if data.cpu_backup_tensor is not None:
               memcpy(ptr, dest_max, cpu_ptr, size_in_bytes, ACL_MEMCPY_HOST_TO_DEVICE)
               data.cpu_backup_tensor = None
   ```
7. **单例必要性：** C 扩展使用全局变量存储回调函数指针，多实例会互相覆盖导致 free 回调失效。

**Tag 机制：** 通过 `use_memory_pool(tag=...)` 上下文管理器，所有在该上下文中分配的 NPU 张量都会被打上对应 tag。

**底层依赖：** 需要编译 `vllm_ascend_C` 扩展（`init_module`、`python_create_and_map`、`python_unmap_and_release`），并依赖 AscendCL 的 `memcpy`。若导入失败，sleep mode 会被禁用并告警。

**补丁说明：** [vllm_ascend/patch/platform/patch_camem_allocator.py](file:///workspace/vllm_ascend/patch/platform/patch_camem_allocator.py) 补丁 vLLM 的 `model_config_module.is_cumem_allocator_available` 使其返回 True，因为 NPUPlatform 声明支持 sleep mode 且 vllm-ascend 使用 CaMemAllocator。避免在 ModelConfig 验证阶段（早于自定义 op 初始化）导入扩展。

**适用场景：** Sleep Mode 的底层基础设施，不直接被用户调用。

---

## 六、CPU 绑定（CPU Binding，主机侧优化）

**文件路径：**
- [vllm_ascend/cpu_binding.py](file:///workspace/vllm_ascend/cpu_binding.py) - 核心实现
- [tests/ut/device_allocator/test_cpu_binding.py](file:///workspace/tests/ut/device_allocator/test_cpu_binding.py) - 完整单元测试
- [docs/source/user_guide/feature_guide/cpu_binding.md](file:///workspace/docs/source/user_guide/feature_guide/cpu_binding.md) - 用户文档
- [docs/source/developer_guide/Design_Documents/cpu_binding.md](file:///workspace/docs/source/developer_guide/Design_Documents/cpu_binding.md) - 设计文档

> **注意：** CPU Binding 严格来说不是"卸载"，而是主机侧 CPU 亲和性优化。作为 vllm-ascend 中以 "cpu_" 开头的核心特性，本文档一并说明。

**关键常量与函数签名：**

```python
MASK_BIT = 32
MIN_CPUS_PER_NPU = 5  # 2(IRQ) + 1(main) + 1(acl) + 1(release)
TOPO_AFFINITY_MODE = "topo_affinity"
GLOBAL_SLICE_MODE = "global_slice"

DEVICE_BINDING_MODE: dict[AscendDeviceType, str] = {
    AscendDeviceType.A2: TOPO_AFFINITY_MODE,
    AscendDeviceType.A3: GLOBAL_SLICE_MODE,
    AscendDeviceType._310P: TOPO_AFFINITY_MODE,
}

def is_arm_cpu() -> bool
def execute_command(cmd: list[str]) -> tuple[str, int]
def bind_cpus(rank_id: int) -> None  # 入口函数

class DeviceInfo:
    def __init__(self)
    @staticmethod
    def expand_cpu_list(allowed_list_str: str) -> list[int]
    def get_all_logic_npus(self) -> list[int]
    @staticmethod
    def get_npu_map_info() -> dict[str, dict[str, str]]
    def get_running_npus(self) -> list[int]
    def parse_allowed_cpus(self) -> list[int]
    def parse_topo_affinity(self) -> dict[int, list[int]]

class CpuAlloc:
    def __init__(self, rank_id: int)
    @staticmethod
    def cpu_to_mask(cpu: int) -> str
    @staticmethod
    def get_threads_map(thread_message: str) -> dict[str, dict[str, list[str]]]
    @staticmethod
    def bind(pid: str, cpus: list[int], bind_sub_thread: bool) -> None
    def average_distribute(self, groups) -> dict[int, list[int]]
    def extend_numa(self, cpu_list: list[int]) -> list[int]
    def build_cpu_node_map(self) -> None
    def build_global_slice_cpu_pool(self) -> None
    @staticmethod
    def _binding_mode() -> str
    def build_cpu_pools(self) -> None
    def allocate(self) -> None
    def print_plan(self) -> None
    def bind_memory(self, pid: str, npu: int) -> None
    def bind_threads(self) -> None
    def bind_npu_irq(self) -> None
    def run_all(self) -> None
```

**工作原理：**

1. **CPU Binding 是 Ascend 原生的主机侧优化**，用于 ARM 多路服务器。**从 v0.18.0rc1 起默认启用**（`enable_cpu_binding=True`）。
2. **不改变模型执行逻辑或数值结果**，仅控制 worker 进程、关键运行时线程、内存页、NPU IRQ 的 CPU 放置。
3. **数据来源：**
   - 允许的 CPU 列表：`/proc/self/status` 的 `Cpus_allowed_list`
   - 逻辑 NPU 映射：`npu-smi info -m`
   - 运行中的 NPU：`npu-smi info` 进程表，按 `ASCEND_RT_VISIBLE_DEVICES` 过滤
   - 拓扑亲和性：`npu-smi info -t topo`
   - CPU NUMA 映射：`lscpu -e=CPU,NODE`
4. **两种策略：**
   - **`global_slice`（A3 默认）：** A3 的 HCCS 互联使每个 NPU 到每个 NUMA 节点距离几乎相同，因此按全局逻辑 NPU ID 切分 `allowed_cpus`。关键性质：两个独立 worker 进程即使共享同一 cpuset，因都按全局 NPU ID 空间切分，得到 **不重叠的 CPU 池**。要求 `base >= 5`。
   - **`topo_affinity`（A2/310P 默认）：** 从 NPU 拓扑亲和性出发，与 `allowed_cpus` 取交集，单 NUMA 时扩展到下一个 NUMA 节点。会包含非运行但亲和性重叠的 NPU 作为候选，避免两个独立单卡 worker 选到相同 CPU 范围。
5. **角色分配（每个 NPU 池至少 5 CPU）：**
   - `pool[0]`, `pool[1]`：SQ/CQ IRQ 绑定
   - `pool[2:-2]`：主 worker 进程及子线程
   - `pool[-2]`：ACL 线程
   - `pool[-1]`：release 线程
6. **条件性主机调优：**
   - **内存迁移：** 使用 `migratepages` 将 worker 进程已有页面迁移到所选 NUMA 节点（缺失则跳过，仅影响性能）。
   - **IRQ 绑定：** 当 `/proc/irq` 可写时，将 NPU IRQ 处理绑定到对应 NPU 保留的 CPU；会先停止 `irqbalance` 服务。
7. **入口：** `bind_cpus(rank_id)` —— 仅在 ARM CPU 上执行，非 ARM 直接跳过。

**配置方式：**

```bash
# 默认启用，无需配置
vllm serve Qwen/Qwen2.5-7B-Instruct

# 禁用
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --additional-config '{"enable_cpu_binding": false}'
```

```python
# 离线禁用
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", additional_config={"enable_cpu_binding": False})
```

**配置项位置：** [vllm_ascend/ascend_config.py](file:///workspace/vllm_ascend/ascend_config.py) 第 135 行 `self.enable_cpu_binding = additional_config.get("enable_cpu_binding", True)`。

**依赖工具：** `util-linux`（taskset）、`numactl`（migratepages）、`procps`/`procps-ng`（ps）。官方镜像 v0.18.0rc1 起已包含。

**适用场景：** ARM 多路服务器主机侧优化，降低跨 NUMA 流量、减少线程抢占、提升多 worker 隔离性。

---

## 七、SSD 卸载（Mooncake 后端的 SSD Offload）

> **注意：** 这是 KV cache 到 SSD 的卸载（非 CPU），但作为存储层级卸载体系的一部分在此一并说明。

**文件路径：**
- [vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py](file:///workspace/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py) - `MooncakeStoreConfig`、`MooncakeBackend`
- [docs/source/user_guide/feature_guide/kv_pool.md](file:///workspace/docs/source/user_guide/feature_guide/kv_pool.md) - 部署文档

**关键配置：**

```python
@dataclass
class MooncakeStoreConfig:
    metadata_server: str
    global_segment_size: int | str
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    preferred_segment: bool
    prefer_alloc_in_same_node: bool
    enable_ssd_offload: bool = False
    ssd_offload_path: str = ""
```

**工作原理：**
1. 通过 `mooncake.json` 配置 `enable_ssd_offload: true` 和 `ssd_offload_path`（绝对路径）启用。
2. 每个 TP rank 使用独立的 SSD 子目录（`rank_0/`, `rank_1/`, ...）避免 bucket 文件冲突。
3. 需要 mooncake >= v0.3.11，且 `MooncakeDistributedStore.setup()` 需支持 `enable_ssd_offload`/`ssd_offload_path` 参数（运行时检测）。
4. **磁盘用量控制环境变量：**
   - `MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE`：驱逐阈值（字节），0 表示用 90% 物理磁盘容量
   - `MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY`：`none`/`fifo`/`lru`
   - `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES`：全局最大磁盘用量（默认 2TB）
5. **Master 启动：** `mooncake_master --rpc_port=50051 --enable_offload=true`

**mooncake.json 配置示例：**

```json
{
    "local_hostname": "xx.xx.xx.xx",
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "use_ascend_direct": true,
    "device_name": "",
    "master_server_address": "xx.xx.xx.xx:50088",
    "global_segment_size": "1GB",
    "enable_ssd_offload": true,
    "ssd_offload_path": "/nvme/mooncake_offload"
}
```

**适用场景：** KV Pool 跨节点共享、超大容量 KV cache 存储。

---

## 八、权重预取与离线权重加载

### 8.1 Weight Prefetch（权重预取优化）

> **注意：** 权重预取不是"卸载"，而是 decode 阶段的权重预取加速优化，作为权重相关特性一并说明。

**文件路径：**
- [vllm_ascend/ops/weight_prefetch.py](file:///workspace/vllm_ascend/ops/weight_prefetch.py) - `WeightPrefetchMethod`、`ModuleWeightPrefetchConfig`、`maybe_npu_prefetch`
- [vllm_ascend/ascend_config.py](file:///workspace/vllm_ascend/ascend_config.py) 第 585-602 行 - `WeightPrefetchConfig`

**关键常量与函数签名：**

```python
SUPPORTED_MODULES = ["attn", "mlp", "moe"]
MOE_PREFETCH_TOKEN_THRESHOLD = 96
MAX_PREFETCH_WEIGHT_SIZE = 18 * 1024 * 1024

@dataclass
class ModuleWeightPrefetchConfig:
    module_name: str
    enable: bool = False
    is_active_this_forward: bool = False
    prefetch_ratio: dict = field(default_factory=dict)
    linear_prefix_map: dict = field(default_factory=dict)

class WeightPrefetchMethod:
    def __init__(self, weight_prefetch_config: WeightPrefetchConfig)
    def maybe_prefetch_attn_weight_preprocess(self, layer_cls_name, weight, start_flag) -> None
    def maybe_prefetch_attn_weight_postprocess(self, layer_cls_name, stop_flag) -> None
    def maybe_prefetch_moe_weight_preprocess(self, hidden_states, prefix)
    def maybe_prefetch_moe_weight_postprocess(self, stop_flag)
    def maybe_prefetch_mlp_weight_preprocess(self, prefetch_layer_name, x_dependency, curr_layer_prefix=None)
    def maybe_prefetch_mlp_weight_postprocess(self, stop_flag)
    def maybe_prefetch_mla_or_sla_weight_in_current_stream(self, inputs, dependency, max_size=0, linear_layer=None) -> None

def maybe_npu_prefetch(inputs, dependency, max_size=0, offset=0, *, enabled=True) -> None

class WeightPrefetchConfig:
    prefetch_ratio: dict = {
        "attn": {"qkv": 1.0, "o": 1.0},
        "moe": {"gate_up": 0.8},
        "mlp": {"gate_up": 1.0, "down": 1.0},
    }
    def __init__(self, weight_prefetch_config: dict):
        self.enabled = weight_prefetch_config.get("enabled", False)
        self.prefetch_ratio = weight_prefetch_config.get("prefetch_ratio", self.prefetch_ratio)
```

**工作原理：**
1. 通过 `torch.ops.vllm.prefetch_preprocess` 和 `torch.ops.vllm.prefetch_postprocess` 自定义 op 触发权重预取。
2. 支持三种模块：`attn`（qkv/o_proj）、`mlp`（gate_up/down_proj）、`moe`（gate_up）。
3. MOE 预取有 token 阈值（`MOE_PREFETCH_TOKEN_THRESHOLD=96`），仅当 token 数 >= 96 时激活。
4. MLP 预取仅在 `num_tokens < 500` 时激活（decode 场景）。
5. 单次预取权重大小上限 `MAX_PREFETCH_WEIGHT_SIZE = 18MB`。
6. `maybe_npu_prefetch` 直接调用 `torch_npu.npu_prefetch` 进行底层预取。

**配置方式（通过 `additional_config.weight_prefetch_config`）：**

```python
{
    "weight_prefetch_config": {
        "enabled": True,
        "prefetch_ratio": {
            "attn": {"qkv": 1.0, "o": 1.0},
            "moe": {"gate_up": 0.8},
            "mlp": {"gate_up": 1.0, "down": 1.0}
        }
    }
}
```

### 8.2 离线权重加载（External Launcher + Sleep Mode）

**文件路径：**
- [examples/offline_weight_load.py](file:///workspace/examples/offline_weight_load.py) - 外部启动器推理示例
- [examples/save_sharded_state_310.py](file:///workspace/examples/save_sharded_state_310.py) - 310P 分片状态保存

**`offline_weight_load.py` 工作原理：**
1. 使用 `distributed_executor_backend="external_launcher"` 模式，由外部脚本通过 `multiprocessing.Process` 启动各 rank。
2. 支持 `--enable-sleep-mode`：先正常推理，然后 `llm.sleep(level=1)` 卸载权重，验证释放的显存 >= 模型权重大小，再 `llm.wake_up()` 唤醒，**手动通过 `load_and_merge_safetensors` + `runmodel.load_weights(sd.items())` 重新加载权重**，最后再次推理验证一致性。
3. 适用于 RL 场景中训练阶段更新权重后重新加载到推理引擎。
4. 要求 sleep mode 下 `temperature=0` 且必须提供 `--model-weight-gib`。

**命令行示例：**

```bash
# 单节点 Dense 模型
python examples/offline_weight_load.py \
    --model="Qwen/Qwen2.5-0.5B-Instruct" --tp-size=1 --proc-per-node=2

# MOE 模型
python examples/offline_weight_load.py \
    --model="Qwen/Qwen3-30B-A3B" --tp-size=2 --proc-per-node=2 --enable-expert-parallel

# 多节点
python examples/offline_weight_load.py \
    --model="Qwen/Qwen3-30B-A3B" --tp-size=2 --node-size=2 --node-rank=0 \
    --proc-per-node=2 --enable-expert-parallel \
    --master-addr=10.99.48.128 --master-port=13345
```

### 8.3 ShardedStateLoader310（310P 专用）

**文件路径：**
- [vllm_ascend/_310p/sharded_state_loader_310p.py](file:///workspace/vllm_ascend/_310p/sharded_state_loader_310p.py) - `ShardedStateLoader310(ShardedStateLoader)`
- [tests/ut/_310p/test_sharded_state_loader_310p.py](file:///workspace/tests/ut/_310p/test_sharded_state_loader_310p.py) - 测试

继承自上游 `ShardedStateLoader`，为 310P 设备适配分片状态加载。将每个 worker 的模型 state dict 直接保存为 checkpoint，支持大 TP 模型的快速加载路径（每个 worker 只读自己的分片），加载时使用 `load_format="sharded_state"`。

---

## 九、相关环境变量与配置项汇总

### 9.1 环境变量（[vllm_ascend/envs.py](file:///workspace/vllm_ascend/envs.py)）

与 CPU 卸载直接相关的环境变量：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `VLLM_ASCEND_ENABLE_BATCH_MEMCPY` | `None`（自动检测） | 控制 KV cache offloading 的 `aclrtMemcpyBatchAsync` 编译路径。`"1"` 强制启用，`"0"` 强制禁用，`None` 从 CANN headers 自动检测 |
| `VLLM_ASCEND_ENABLE_NZ` | `1` | NZ 模式（sleep mode 要求设为 `0`） |

Sleep mode 相关环境变量：

| 环境变量 | 说明 |
|----------|------|
| `VLLM_SERVER_DEV_MODE=1` | 在线服务暴露 sleep/wake_up 端点 |
| `VLLM_USE_MODELSCOPE=True` | 使用 ModelScope 下载模型 |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | 多进程启动方式 |
| `COMPILE_CUSTOM_KERNELS=1` | 编译自定义 kernel（< v0.12.0rc1 需要） |
| `PYTORCH_NPU_ALLOC_CONF` | 不能含 `expandable_segments:True` |

Mooncake SSD offload 相关环境变量：

| 环境变量 | 说明 |
|----------|------|
| `MOONCAKE_CONFIG_PATH` | mooncake.json 路径 |
| `MOONCAKE_MASTER` | master 地址（覆盖 json 配置） |
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | 全局 segment 大小 |
| `MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE` | SSD bucket 最大总量 |
| `MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY` | SSD 驱逐策略（`none`/`fifo`/`lru`） |
| `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` | SSD 全局上限（默认 2TB） |

CPU binding 相关环境变量：

| 环境变量 | 说明 |
|----------|------|
| `ASCEND_RT_VISIBLE_DEVICES` | 过滤运行中的 NPU |

### 9.2 AscendConfig 中的卸载相关配置（[vllm_ascend/ascend_config.py](file:///workspace/vllm_ascend/ascend_config.py)）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable_cpu_binding` | bool | `True` | 启用 ARM 服务器 CPU 绑定 |
| `weight_prefetch_config` | dict | `{}` | 权重预取配置（含 `enabled`、`prefetch_ratio`） |
| `weight_nz_mode` | int | `1` | NZ 模式（sleep mode 要求设为 0） |

`WeightPrefetchConfig` 默认 `prefetch_ratio`：

```python
{
    "attn": {"qkv": 1.0, "o": 1.0},
    "moe": {"gate_up": 0.8},
    "mlp": {"gate_up": 1.0, "down": 1.0},
}
```

---

## 十、底层 C++ 实现：swap_blocks_batch

**文件路径：** [csrc/torch_binding.cpp](file:///workspace/csrc/torch_binding.cpp) 第 123-161 行及第 2238-2239 行注册

```cpp
void swap_blocks_batch(const torch::Tensor& src_ptrs,
                       const torch::Tensor& dst_ptrs,
                       const torch::Tensor& sizes,
                       int64_t direction) {
    // 校验 CPU int64 张量
    // direction: 0=H2D, 1=D2H, 2=D2D
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    // path 1: aclrtMemcpyBatchAsync (CANN 8.5+)
    #if defined(CANN_MEMCPY_BATCH_ASYNC)
    if (memcpy_kind != ACL_MEMCPY_DEVICE_TO_DEVICE) {
        // 使用 aclrtMemcpyBatchAsync 批量异步拷贝
    }
    #endif
    // path 2: fallback 逐个 aclrtMemcpyAsync
}

// 注册（第 2238-2239 行）
ops.def("swap_blocks_batch(Tensor x, Tensor y, Tensor z, int direction) -> ()");
ops.impl("swap_blocks_batch", torch::kCPU, &vllm_ascend::swap_blocks_batch);
```

这是 KV cache CPU 卸载（NPUOffloadingSpec 和 AscendSimpleCPUOffloadConnector）的底层批量 DMA 传输原语，优先使用 CANN 8.5+ 的 `aclrtMemcpyBatchAsync`，否则回退到逐个 `aclrtMemcpyAsync`。受 `VLLM_ASCEND_ENABLE_BATCH_MEMCPY` 环境变量控制编译路径。

---

## 附录：相关文档清单

所有相关文档均位于 [docs/source/](file:///workspace/docs/source/)：

| 文档路径 | 内容 |
|----------|------|
| [user_guide/feature_guide/kv_cache_cpu_offload.md](file:///workspace/docs/source/user_guide/feature_guide/kv_cache_cpu_offload.md) | NPUOffloadingSpec 使用指南 |
| [user_guide/feature_guide/sleep_mode.md](file:///workspace/docs/source/user_guide/feature_guide/sleep_mode.md) | Sleep Mode 使用指南 |
| [user_guide/feature_guide/cpu_binding.md](file:///workspace/docs/source/user_guide/feature_guide/cpu_binding.md) | CPU Binding 用户指南 |
| [user_guide/feature_guide/kv_pool.md](file:///workspace/docs/source/user_guide/feature_guide/kv_pool.md) | AscendStore/KV Pool 部署指南（含 SSD offload） |
| [user_guide/feature_guide/weight_prefetch.md](file:///workspace/docs/source/user_guide/feature_guide/weight_prefetch.md) | 权重预取使用指南 |
| [developer_guide/Design_Documents/cpu_binding.md](file:///workspace/docs/source/developer_guide/Design_Documents/cpu_binding.md) | CPU Binding 设计文档（详细原理） |
