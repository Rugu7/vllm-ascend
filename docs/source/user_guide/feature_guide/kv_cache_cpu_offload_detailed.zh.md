# vllm-ascend KV Cache CPU 卸载方案详解

本文档详细整理 vllm-ascend 中三套 KV Cache CPU 卸载方案的实现原理、关键流程与时序图。

## 目录

- [一、三套方案概览](#一三套方案概览)
- [二、方案一：NPUOffloadingSpec（OffloadingConnector 框架）](#二方案一npuoffloadingspecoffloadingconnector-框架)
  - [2.1 架构与组件](#21-架构与组件)
  - [2.2 初始化流程](#22-初始化流程)
  - [2.3 D2H 卸载时序图（NPU→CPU）](#23-d2h-卸载时序图npucpu)
  - [2.4 H2D 回载时序图（CPU→NPU）](#24-h2d-回载时序图cpunpu)
  - [2.5 关键实现细节](#25-关键实现细节)
  - [2.6 配置与使用](#26-配置与使用)
- [三、方案二：AscendSimpleCPUOffloadConnector（上游适配）](#三方案二ascendsimplecpuoffloadconnector上游适配)
  - [3.1 架构与组件](#31-架构与组件)
  - [3.2 初始化与注册流程](#32-初始化与注册流程)
  - [3.3 拷贝任务时序图](#33-拷贝任务时序图)
  - [3.4 关键实现细节](#34-关键实现细节)
  - [3.5 配置与使用](#35-配置与使用)
- [四、方案三：CPUOffloadingConnector（独立实现 + Prefix Cache）](#四方案三cpuoffloadingconnector独立实现--prefix-cache)
  - [4.1 架构与组件](#41-架构与组件)
  - [4.2 三层架构逻辑流程图](#42-三层架构逻辑流程图)
  - [4.3 启动与初始化时序图](#43-启动与初始化时序图)
  - [4.4 请求处理时序图（Prefix Cache 命中场景）](#44-请求处理时序图prefix-cache-命中场景)
  - [4.5 KV Save 时序图（请求完成后异步保存）](#45-kv-save-时序图请求完成后异步保存)
  - [4.6 关键实现细节](#46-关键实现细节)
  - [4.7 配置与使用](#47-配置与使用)
- [五、三套方案对比](#五三套方案对比)
- [六、选型建议](#六选型建议)

---

## 一、三套方案概览

vllm-ascend 提供三套独立的 KV Cache CPU 卸载实现，分别面向不同场景：

| 方案 | Connector 名称 | 核心文件 | 适用场景 |
|------|---------------|----------|----------|
| 方案一 | `OffloadingConnector` + `NPUOffloadingSpec` | `vllm_ascend/kv_offload/` | 单实例 NPU 显存不足，LRU 淘汰 |
| 方案二 | `SimpleCPUOffloadConnector`（自动覆盖为 NPU 版） | `vllm_ascend/simple_kv_offload/` | 与上游 API 一致的简单 KV 卸载 |
| 方案三 | `CPUOffloadingConnector` | `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/` | 跨 DP 共享、CPU prefix caching、大容量 swap |

---

## 二、方案一：NPUOffloadingSpec（OffloadingConnector 框架）

### 2.1 架构与组件

```mermaid
graph TB
    subgraph "Scheduler 进程"
        Sched[vLLM Scheduler]
        Mgr[CPUOffloadingManager<br/>LRU BlockPool]
        Sched --> Mgr
    end

    subgraph "Worker 进程"
        Worker[vLLM Worker]
        Handler[CpuNpuOffloadingHandler<br/>双 Stream + Event 池]
        NPU_KV[NPU KV Cache<br/>num_gpu_blocks × block_size]
        CPU_KV[CPU KV Cache<br/>num_cpu_blocks × cpu_block_size]
        Worker --> Handler
        Handler -.D2H.-> CPU_KV
        Handler -.H2D.-> NPU_KV
    end

    Mgr <-.TransferSpec/TransferResult.-> Handler

    style Mgr fill:#e1f5ff
    style Handler fill:#fff4e1
    style NPU_KV fill:#ffe1e1
    style CPU_KV fill:#e1ffe1
```

**核心组件：**
- `NPUOffloadingSpec`：spec 工厂，分别在 scheduler 侧创建 `CPUOffloadingManager`，worker 侧创建 `CpuNpuOffloadingHandler`。
- `CPUOffloadingManager`（来自 vLLM v1）：scheduler 侧的 CPU block 池管理器，使用 LRU 策略。
- `CpuNpuOffloadingHandler`：worker 侧的 NPU↔CPU 异步传输处理器。

### 2.2 初始化流程

```mermaid
flowchart TD
    A[LLM 启动<br/>kv_transfer_config 配置] --> B[NPUOffloadingSpec.__init__]
    B --> C{读取 extra_config}
    C --> D[num_cpu_blocks 检查<br/>必须指定]
    C --> E[block_size_factor<br/>= cpu_block_size // gpu_block_size]

    E --> F[get_manager<br/>scheduler 侧]
    F --> G[创建 CPUOffloadingManager<br/>LRU BlockPool]

    E --> H[get_handlers<br/>worker 侧]
    H --> I[创建 CpuNpuOffloadingHandler]
    I --> J[创建 d2h_stream<br/>创建 h2d_stream]
    I --> K[分配 CPU pinned memory<br/>每层一对 key/value 张量]
    I --> L[预计算 base_ptrs<br/>和 block_size_in_bytes]
    I --> M[初始化 Event 池]

    style D fill:#ffe1e1
    style G fill:#e1f5ff
    style K fill:#e1ffe1
```

### 2.3 D2H 卸载时序图（NPU→CPU）

当 NPU 显存满且需要新 block 时，scheduler 决定淘汰某些 block 到 CPU：

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant M as CPUOffloadingManager
    participant H as CpuNpuOffloadingHandler
    participant DS as Default NPU Stream
    participant D2H as D2H NPU Stream
    participant CPU as CPU Memory

    S->>M: 决定淘汰 NPU block A<br/>分配 CPU block B
    M->>H: transfer_async(job_id, GPULoadStoreSpec→CPULoadStoreSpec)
    Note over H: src_spec.block_ids=[A]<br/>dst_spec.block_ids=[B]

    H->>H: expand_block_ids<br/>CPU block B 展开为子 block
    H->>H: numpy 向量化构建<br/>扁平 src/dst/sizes 数组
    H->>H: 从 Event 池获取 start_event, end_event

    H->>D2H: stream.wait_stream(current_stream)<br/>等待模型计算完成
    H->>D2H: stream.wait_event(last_transfer.end_event)<br/>保证传输顺序

    H->>D2H: record start_event
    H->>D2H: torch.ops._C_ascend.swap_blocks_batch<br/>(aclrtMemcpyBatchAsync D2H)
    H->>D2H: record end_event

    H-->>M: 返回 True（已提交）

    Note over DS,D2H: 模型计算与 D2H 传输<br/>在不同 stream 上并行

    loop 轮询
        S->>H: get_finished()
        H->>D2H: end_event.query() 非阻塞查询
        alt 传输完成
            H->>H: 回收 start_event, end_event 到池
            H-->>S: TransferResult(job_id, success, size, time)
        end
    end
```

### 2.4 H2D 回载时序图（CPU→NPU）

当 prefix cache 在 NPU 未命中但 CPU 命中时：

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant M as CPUOffloadingManager
    participant H as CpuNpuOffloadingHandler
    participant H2D as H2D NPU Stream
    participant DS as Default NPU Stream
    participant CPU as CPU Memory

    S->>M: prefix cache 命中 CPU block B<br/>分配 NPU block A
    M->>H: transfer_async(job_id, CPULoadStoreSpec→GPULoadStoreSpec)

    H->>H: expand_block_ids<br/>CPU block B 展开为子 block
    H->>H: numpy 向量化构建指针数组

    Note over H: H2D 不需要 wait_stream<br/>（CPU 数据已就绪）
    H->>H2D: stream.wait_event(last_transfer.end_event)<br/>保证 H2D 传输顺序

    H->>H2D: record start_event
    H->>H2D: swap_blocks_batch (aclrtMemcpyBatchAsync H2D)
    H->>H2D: record end_event

    H-->>M: 返回 True

    S->>H: wait(job_ids) 阻塞等待
    H->>H2D: end_event.synchronize()
    H-->>S: 完成

    Note over H2D,DS: 后续模型 forward 使用 block A 前<br/>default_stream.wait_stream(h2d_stream)<br/>或显式同步
```

### 2.5 关键实现细节

**1. CPU block 与 NPU block 的映射关系：**

```
NPU block_size = 16 (示例)
CPU block_size = 128 (典型值)
block_size_factor = 128 / 16 = 8

1 个 CPU block = 8 个 NPU block
expand_block_ids([0, 1, 3], factor=8) = [0,1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15, 24,25,26,27,28,29,30,31]
```

**2. numpy 向量化构建指针数组（避免 Python 循环）：**

```python
# 形状: (num_sub_tensors, num_pairs) -> ravel to 1D
bsz_col = self._block_size_in_bytes_arr[:, None]          # (T, 1)
all_src = (src_base_ptrs[:, None] + src_block_ids[None, :] * bsz_col).ravel()
all_dst = (dst_base_ptrs[:, None] + dst_block_ids[None, :] * bsz_col).ravel()
all_sizes = np.broadcast_to(bsz_col, (num_sub_tensors, num_pairs)).ravel().copy()
```

**3. Event 池复用：** 避免每次传输都创建新的 `torch.npu.Event`，减少分配开销。

**4. 传输顺序保证：**
- D2H：`d2h_stream.wait_stream(current_stream)` 确保读到的是计算完成的数据。
- 同方向连续传输：`stream.wait_event(last_transfer.end_event)` 保证 FIFO 顺序。

### 2.6 配置与使用

```python
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "num_cpu_blocks": 1000,        # CPU 内存中分配的 block 数量
        "block_size": 128,             # CPU 侧 block 大小
        "spec_name": "NPUOffloadingSpec",
        "spec_module_path": "vllm_ascend.kv_offload.npu",
    },
)

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    gpu_memory_utilization=0.5,
    kv_transfer_config=kv_transfer_config,
)
```

**配置参数说明：**

| 参数 | 说明 |
|------|------|
| `kv_connector` | 必须为 `"OffloadingConnector"` |
| `kv_role` | `"kv_both"` 同时启用存储和加载 |
| `num_cpu_blocks` | CPU 内存中分配的 block 数量，每个 block 消耗 `block_size × num_layers × (key_size + value_size)` 字节 |
| `block_size` | CPU 侧 block 大小，应为 NPU block 大小的整数倍，典型值 128 |
| `spec_name` | 必须为 `"NPUOffloadingSpec"` |
| `spec_module_path` | 必须为 `"vllm_ascend.kv_offload.npu"` |

---

## 三、方案二：AscendSimpleCPUOffloadConnector（上游适配）

### 3.1 架构与组件

```mermaid
graph TB
    subgraph "Scheduler 进程"
        Sched[vLLM Scheduler]
        SchedMgr[SimpleCPUOffloadScheduler<br/>上游复用，平台无关]
        Sched --> SchedMgr
    end

    subgraph "Worker 进程"
        Worker[vLLM Worker]
        WH[SimpleCPUOffloadNPUWorker<br/>继承上游 SimpleCPUOffloadWorker]
        Backend[NPUDmaCopyBackend<br/>后台守护线程]
        NPU_KV[NPU KV Cache<br/>K/V 分离分配]
        CPU_KV[CPU KV Cache<br/>pinned mirror]
        Worker --> WH
        WH --> Backend
        Backend -.D2H store.-> CPU_KV
        Backend -.H2D load.-> NPU_KV
    end

    SchedMgr <-.metadata.-> WH

    style SchedMgr fill:#e1f5ff
    style WH fill:#fff4e1
    style Backend fill:#ffe1e1
    style CPU_KV fill:#e1ffe1
```

**核心组件：**
- `AscendSimpleCPUOffloadConnector`：继承上游 `SimpleCPUOffloadConnector`，scheduler 侧逻辑完全复用。
- `SimpleCPUOffloadNPUWorker`：继承上游 `SimpleCPUOffloadWorker`，替换 CUDA 拷贝后端为 NPU 版本。
- `NPUDmaCopyBackend`：后台守护线程执行 FIFO 拷贝任务。
- `BatchMemcpyParams`：预构建的批量拷贝参数（src_bases, dst_bases, bpb）。

### 3.2 初始化与注册流程

```mermaid
flowchart TD
    A["vllm-ascend 启动"] --> B["register_connector<br/>覆盖上游 SimpleCPUOffloadConnector"]
    B --> C["用户配置<br/>kv_connector=SimpleCPUOffloadConnector"]

    C --> D["AscendSimpleCPUOffloadConnector.__init__"]
    D --> E["super().__init__<br/>创建临时 CUDA worker"]
    E --> F{role == WORKER?}
    F -- 否 --> G["connector 为 no-op"]
    F -- 是 --> H["替换为 SimpleCPUOffloadNPUWorker"]

    H --> I["register_kv_caches"]
    I --> J["遍历 kv_caches<br/>_flatten_kv_value 展开 K/V"]
    J --> K["按 untyped_storage.data_ptr 去重<br/>避免同一 NPU 分配重复注册"]
    K --> L["_build_block_views<br/>从 tensor shape/stride 构建 view<br/>而非 storage.nbytes"]
    L --> M["计算 total_bytes_per_block"]
    M --> N["num_cpu_blocks =<br/>cpu_capacity_bytes // total_bytes_per_block"]
    N --> O["分配 CPU pinned mirror<br/>torch.zeros pin_memory=True"]
    O --> P["创建 load_stream, store_stream<br/>普通 NPU Stream 无 priority"]
    P --> Q["NPUDmaCopyBackend.init<br/>build_params 构建 store/load 参数<br/>启动后台拷贝线程"]

    style B fill:#ffe1e1
    style K fill:#fff4e1
    style L fill:#fff4e1
    style Q fill:#e1f5ff
```

### 3.3 拷贝任务时序图

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant W as SimpleCPUOffloadNPUWorker
    participant B as NPUDmaCopyBackend
    participant Q as FIFO Queue
    participant T as Copy Thread
    participant SS as Store Stream
    participant LS as Load Stream
    participant E as Events List

    Note over S: Scheduler 决定 store/load 一批 block

    rect rgb(255, 240, 240)
        Note over S,W: Store 阶段 (D2H)
        S->>W: save_kv_blocks(src_blocks, dst_blocks, event_idx)
        W->>B: launch_copy(src, dst, is_store=True, event_idx, events_list)
        B->>Q: put(store_task)
        Q->>T: get(store_task)
        T->>SS: with torch.npu.stream(store_stream)
        T->>SS: copy_blocks → swap_blocks_batch D2H
        T->>SS: event.record(store_stream)
        T->>E: append((event_idx, event))
    end

    rect rgb(240, 255, 240)
        Note over S,W: Load 阶段 (H2D)
        S->>W: load_kv_blocks(src_blocks, dst_blocks, event_idx)
        W->>B: launch_copy(src, dst, is_store=False, event_idx, events_list)
        B->>Q: put(load_task)
        Q->>T: get(load_task)
        T->>LS: with torch.npu.stream(load_stream)
        T->>LS: copy_blocks → swap_blocks_batch H2D
        T->>LS: event.record(load_stream)
        T->>E: append((event_idx, event))
    end

    Note over S,W: 主线程轮询 events_list
    loop 轮询
        S->>W: get_finished()
        W->>E: 遍历 events_list
        alt event.query() 为 True
            W-->>S: 返回完成的 job
        end
    end
```

### 3.4 关键实现细节

**1. K/V 分离分配的处理：**

Ascend 上 K 和 V 是 **分离的独立分配**（不像 CUDA 堆叠在一个外层维度下），因此必须遍历每个子张量：

```python
def _flatten_kv_value(value):
    if isinstance(value, torch.Tensor):
        return [value]
    return [t for t in value if isinstance(torch.Tensor)]  # (k_cache, v_cache) → [k, v]
```

**2. storage 去重：**

```python
unique_caches = {}
seen_ptrs = set()
for layer_name, value in kv_caches.items():
    for sub_idx, tensor in enumerate(_flatten_kv_value(value)):
        ptr = tensor.untyped_storage().data_ptr()
        if ptr in seen_ptrs:
            continue  # 同一 NPU 分配可能被多层引用
        seen_ptrs.add(ptr)
        unique_caches.update(self._build_block_views(...))
```

**3. view 大小来自 tensor 自身（非 storage.nbytes）：**

```python
# runner 会为每个 KV 张量过度分配 +alignment (2 MiB) 用于对齐
# 因此 view 大小必须来自 tensor.shape/stride，而非 storage.nbytes()
page_size_bytes = tensor.stride(0) * el  # 而非 storage.nbytes()
data_bytes = num_blocks * page_size_bytes
raw = torch.empty(0, dtype=torch.int8, device=tensor.device).set_(
    storage, storage_offset_bytes, (data_bytes,)
)
return {key: raw.view(num_blocks, page_size_bytes)}
```

**4. CPU pinned memory 分配方式差异：**

```python
# CUDA 上游：cudaHostRegister（CUDA 专有）
# NPU 适配：torch.zeros(pin_memory=True)
self.cpu_kv_caches = {
    name: torch.zeros(..., device="cpu", pin_memory=pin_memory)
    for name, t in unique_caches.items()
}
```

**5. Stream priority 差异：**

```python
# CUDA 上游：load_stream = torch.cuda.Stream(priority=-1)  # 最低优先级让步计算
# NPU 适配：torch.npu.Stream()  # 不支持 priority 参数
# 影响：失去"always yield"提示，但传输仍在独立 stream 上与 forward 并行
```

### 3.5 配置与使用

使用上游的 `SimpleCPUOffloadConnector` 名称即可（vllm-ascend 启动时自动覆盖为 NPU 版本）：

```bash
vllm serve Qwen/Qwen3-0.6B \
    --kv-transfer-config '{
        "kv_connector": "SimpleCPUOffloadConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "cpu_capacity_bytes": 5368709120
        }
    }'
```

---

## 四、方案三：CPUOffloadingConnector（独立实现 + Prefix Cache）

### 4.1 架构与组件

```mermaid
graph TB
    subgraph "Scheduler 进程"
        Sched[vLLM Scheduler]
        CS[CPUOffloadingConnectorScheduler<br/>ZMQ RPC Client]
        Sched --> CS
    end

    subgraph "Worker 进程"
        Worker[vLLM Worker]
        CW[CPUOffloadingConnectorWorker<br/>load_stream + save_stream<br/>+ save_thread]
        NPU_KV[NPU KV Cache]
        CPU_KV[CPU KV Cache<br/>SharedMemory 跨进程共享]
        Worker --> CW
        CW <-.load/save.-> NPU_KV
        CW <-.load/save.-> CPU_KV
    end

    subgraph "Metadata Server 进程<br/>仅 dp_rank=0, tp_rank=0, pp_rank=0"
        MS[MetadataServer<br/>ZMQ ROUTER socket]
        CKM[CPUKVCacheManager<br/>BlockPool + Prefix Cache + LRU]
        MS --> CKM
    end

    CS <-.ZMQ RPC.-> MS
    CW <-.ZMQ RPC.-> MS

    style CS fill:#e1f5ff
    style CW fill:#fff4e1
    style MS fill:#ffe1e1
    style CKM fill:#ffe1e1
    style CPU_KV fill:#e1ffe1
```

**三层架构：**
1. **Scheduler 侧**（`CPUOffloadingConnectorScheduler`）：通过 ZMQ RPC 与 Metadata Server 通信。
2. **Worker 侧**（`CPUOffloadingConnectorWorker`）：使用 `torch.npu.Stream` 异步加载/保存 KV cache。
3. **Metadata Server**（`MetadataServer`/`MetadataServerProc`）：独立进程，管理 `CPUKVCacheManager`。

### 4.2 三层架构逻辑流程图

```mermaid
flowchart TB
    subgraph 启动阶段
        A1[Worker 启动] --> A2{dp_rank==0<br/>AND tp_rank==0<br/>AND pp_rank==0?}
        A2 -- 是 --> A3[启动 Metadata Server 线程<br/>MetadataServerProc.run_metadata_server]
        A2 -- 否 --> A4[跳过]
        A3 --> A5[MetadataServer.__init__<br/>创建 ZMQ ROUTER socket<br/>读取 cpu_swap_space_gb 默认 800GB]
        A5 --> A6[等待 ready RPC 成功]

        A4 --> A6
        A6 --> A7[Worker.register_kv_caches]
        A7 --> A8[ZMQ RPC: init_cpu_kv_caches<br/>传递 pp_rank, tp_rank, kv_cache_specs, mla_config]
        A8 --> A9[MetadataServer 创建 SharedMemory<br/>按 pp_rank/tp_rank 分配]
        A9 --> A10[返回 SharedMemory dict<br/>Worker 通过 torch.frombuffer 重建张量视图]
        A10 --> A11[MetadataServer.post_init<br/>创建 CPUKVCacheManager<br/>注册 RPC 函数]
    end

    subgraph 推理阶段
        B1[新请求到达] --> B2[Scheduler: get_num_new_matched_tokens]
        B2 --> B3[ZMQ RPC: get_matched_num_and_touch<br/>查询 CPU prefix cache 命中]
        B3 --> B4{命中数 >= swap_in_threshold?}
        B4 -- 是 --> B5[返回命中数, load_async=True]
        B4 -- 否 --> B6[返回 0, load_async=False]
        B5 --> B7[Scheduler: build_connector_meta<br/>ZMQ RPC: allocate_slots 分配 CPU block]
        B7 --> B8[Worker: bind_connector_metadata<br/>构建 load_block_mapping]
        B8 --> B9[Worker: start_load_kv<br/>逐 layer 异步 H2D copy]
        B9 --> B10[模型 forward<br/>每层前 wait_for_layer_load 同步]
    end

    subgraph 完成阶段
        C1[请求完成] --> C2[Scheduler: request_finished<br/>ZMQ RPC: record_request_cache_and_free_slots]
        C2 --> C3[Worker: save_input_queue.put<br/>触发后台 save_thread]
        C3 --> C4[save_thread: 逐 layer D2H copy<br/>到 CPU SharedMemory]
        C4 --> C5[save_thread: save_output_queue.put]
        C5 --> C6[get_finished: TP 协调<br/>所有 TP rank 完成后]
        C6 --> C7[ZMQ RPC: cache_and_free_slots<br/>CPUKVCacheManager.cache_blocks<br/>写入 prefix cache]
    end

    style A3 fill:#ffe1e1
    style A9 fill:#ffe1e1
    style B3 fill:#e1f5ff
    style C4 fill:#fff4e1
    style C7 fill:#e1ffe1
```

### 4.3 启动与初始化时序图

```mermaid
sequenceDiagram
    participant W as Worker (rank 0)
    participant MS as Metadata Server
    participant SHM as SharedMemory
    participant CKM as CPUKVCacheManager

    Note over W,MS: 仅 dp_rank=0, tp_rank=0, pp_rank=0 的 Worker 启动 MS

    W->>W: init_metadata_server<br/>启动 daemon 线程
    W->>MS: MetadataServerProc.run_metadata_server

    MS->>MS: 创建 ZMQ ROUTER socket<br/>绑定 ipc://.../metadata.ipc
    MS->>MS: 读取 cpu_swap_space_gb<br/>默认 800GB

    W->>MS: ZMQ RPC: ready
    MS-->>W: True

    loop 重试直到 ready
        W->>MS: ZMQ RPC: ready
        MS-->>W: True
    end

    W->>MS: ZMQ RPC: init_cpu_kv_caches<br/>(pp_rank, tp_rank, kv_cache_specs, mla_config)

    alt MLA 模型
        MS->>MS: tp_rank = 0<br/>（MLA 跨 TP 共享 KV）
        MS->>SHM: available_memory //= pp_size // num_layers
    else 非 MLA
        MS->>SHM: available_memory //= world_size // num_layers
    end

    MS->>SHM: 为每层创建 SharedMemory<br/>name=cpu_kv_cache_{pp}_{tp}_{layer}
    MS-->>W: 返回 (shm_dict, layer_size, dtype, mla_config)

    W->>W: torch.frombuffer(shm.buf)<br/>重建张量视图
    alt MLA
        W->>W: tensor.split([nope_dim, rope_dim], dim=-1)
    end

    MS->>MS: post_init
    MS->>CKM: CPUKVCacheManager(layer, num_cpu_blocks)
    CKM->>CKM: BlockPool(num_cpu_blocks, enable_prefix_caching=True)
    MS->>MS: 注册 RPC 函数:<br/>get_matched_num_and_touch<br/>allocate_slots<br/>record_request_cache_and_free_slots<br/>cache_and_free_slots
```

### 4.4 请求处理时序图（Prefix Cache 命中场景）

```mermaid
sequenceDiagram
    participant Req as 新请求
    participant S as Scheduler
    participant CS as ConnectorScheduler
    participant MS as Metadata Server
    participant CKM as CPUKVCacheManager
    participant W as Worker
    participant LS as Load Stream
    participant NPU as NPU KV Cache

    Req->>S: 请求到达，带 block_hashes

    S->>CS: get_num_new_matched_tokens(request, num_computed_tokens)
    CS->>MS: ZMQ RPC: get_matched_num_and_touch(request)
    MS->>CKM: get_matched_num_and_touch(request)

    CKM->>CKM: find_longest_cache_hit<br/>(block_hashes, max_length)
    CKM->>CKM: block_pool.touch(computed_blocks)<br/>更新 LRU
    CKM-->>MS: (num_cpu_computed_tokens, False)
    MS-->>CS: (num_cpu_computed_tokens, False)

    CS->>CS: num_cpu_computed - num_computed >= swap_in_threshold?
    alt 满足阈值
        CS-->>S: (num_cpu_computed_tokens, load_async=True)
    else 不满足
        CS-->>S: (0, load_async=False)
    end

    S->>S: update_state_after_alloc<br/>allocated_req_ids.add(req_id)

    S->>CS: build_connector_meta(scheduler_output)
    CS->>MS: ZMQ RPC: allocate_slots(num_tokens, unallocated_req_ids)
    MS->>CKM: allocate_slots
    CKM->>CKM: get_num_blocks_to_allocate
    alt 空间不足
        CKM->>CKM: _release_ahead_touch<br/>req_failed_to_allocate=True
    else 空间足够
        CKM->>CKM: save_new_computed_blocks<br/>allocate_new_blocks
    end
    CKM-->>MS: {req_id: [cpu_block_ids]}
    MS-->>CS: {req_id: [cpu_block_ids]}

    CS->>CS: 构建 ReqMeta<br/>(gpu_block_ids, cpu_block_ids, ...)
    CS-->>S: CPUOffloadingConnectorMetadata

    S->>W: bind_connector_metadata(metadata)
    W->>W: 构建 load_block_mapping<br/>[(cpu_block_id, gpu_block_id), ...]

    S->>W: start_load_kv (forward 开始)
    W->>LS: load_kv_layer(0)<br/>with torch.npu.stream(load_stream)
    LS->>NPU: gpu_layer[gpu_id].copy_(cpu_layer[cpu_id], non_blocking=True)

    loop 每层 forward
        S->>W: wait_for_layer_load(layer_name)
        W->>LS: load_stream.synchronize()<br/>（等待当前层加载完成）
        W->>LS: load_kv_layer(next_layer)<br/>（启动下一层加载）
        W->>NPU: 模型 forward 当前层
    end
```

### 4.5 KV Save 时序图（请求完成后异步保存）

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant CS as ConnectorScheduler
    participant MS as Metadata Server
    participant W as Worker
    participant SQ as save_input_queue
    participant ST as save_thread
    participant SS as Save Stream
    participant TP as TP Group
    participant CKM as CPUKVCacheManager

    Note over S: 请求完成
    S->>CS: request_finished(request)
    CS->>MS: ZMQ RPC: record_request_cache_and_free_slots(request)
    MS->>CKM: record_request_cache_and_free_slots<br/>req_to_free[req_id] = request

    S->>W: bind_connector_metadata<br/>(含 finished_req_ids)
    W->>SQ: put((req_id, req_meta))

    Note over ST: 后台 save_thread 主循环
    SQ->>ST: get((req_id, req_meta))
    ST->>ST: 构建 save_block_mapping<br/>[(gpu_block_id, cpu_block_id), ...]

    ST->>SS: with torch.npu.stream(save_stream)

    alt MLA 模型
        Note over ST: start=tp_rank, step=tp_world_size<br/>不同 TP rank 保存不同 block
    else 非 MLA
        Note over ST: start=0, step=1<br/>所有 block 由当前 rank 保存
    end

    loop 每个 block
        SS->>SS: cpu_layer[cpu_id].copy_(gpu_layer[gpu_id], non_blocking=True)
    end

    ST->>SS: save_stream.synchronize()
    ST->>W: save_output_queue.put(req_id)

    Note over W: get_finished 轮询 save_output_queue
    W->>W: done_sending = {req_id}

    alt tp_world_size > 1
        Note over W: TP 协调
        alt tp_rank == 0
            W->>TP: recv_object from rank 1, 2, ...
            W->>W: done_sending_count[req_id] += 1 (每个 rank)
            alt done_sending_count == tp_world_size
                W->>ST: 启动 _sending_finished 线程
                ST->>MS: ZMQ RPC: cache_and_free_slots(req_id)
                MS->>CKM: cache_and_free_slots
                CKM->>CKM: single_type_manager.cache_blocks<br/>写入 prefix cache 哈希
                CKM->>CKM: _free_slots(req_id)
            end
        else tp_rank > 0
            W->>TP: send_object(done_sending, dst=0)
        end
    else tp_world_size == 1
        W->>MS: ZMQ RPC: cache_and_free_slots(req_id)
        MS->>CKM: cache_and_free_slots
    end
```

### 4.6 关键实现细节

**1. SharedMemory 跨进程共享：**

```python
# MetadataServer 创建
shared_memory_dict[layer_name] = SharedMemory(
    name=f"cpu_kv_cache_{pp_rank}_{tp_rank}_{layer_name}",
    create=True, size=nbytes
)

# Worker 通过 ZMQ RPC 收到 shm 名称后重建张量
for key, shm in memory_dict.items():
    tensor = torch.frombuffer(shm.buf, dtype=layer_dtype).reshape(layer_size)
    if mla_config is not None:
        tensor = tensor.split([mla_config.nope_dim, mla_config.rope_dim], dim=-1)
    result[key] = tensor
```

**2. MLA 模型的特殊处理：**

```python
# MLA: KV cache 跨 TP 共享（tp_rank=0 创建即可）
if use_mla:
    tp_rank = 0  # 所有 TP rank 共享同一份
    available_memory //= pipeline_parallel_size  # 按 PP 切分
    layer_size = (num_blocks, block_size, num_kv_heads, head_size)
else:
    available_memory //= world_size  # 按 world_size 切分
    layer_size = (2, num_blocks, block_size, num_kv_heads, head_size)  # K/V 堆叠

# MLA 保存时按 TP rank 分担
if self.use_mla:
    start, step = self.tp_rank, self.tp_world_size  # 不同 rank 保存不同 block
else:
    start, step = 0, 1
```

**3. swap_in_threshold 阈值控制：**

```python
# Scheduler 侧：只有 CPU 命中数超过阈值才触发 swap-in
if num_cpu_computed_tokens - num_computed_tokens >= self.swap_in_threshold:
    return num_cpu_computed_tokens - num_computed_tokens, load_async
else:
    return 0, load_async  # 命中太少不值得 swap-in
```

**4. CPUKVCacheManager 的 prefix cache 机制：**

```python
class CPUKVCacheManager:
    def __init__(self, kv_cache_spec, num_cpu_blocks, ...):
        self.block_pool = BlockPool(num_cpu_blocks, True, block_size, ...)  # enable_prefix_caching=True
        self.single_type_manager = get_manager_for_kv_cache_spec(...)
        self.req_to_block_hashes = defaultdict(list)  # 缓存 block 哈希避免重复计算

    def get_matched_num_and_touch(self, request):
        # 复用 vLLM 的 find_longest_cache_hit
        computed_blocks = self.single_type_manager.find_longest_cache_hit(
            block_hashes=block_hashes, max_length=max_cache_hit_length, ...
        )
        self.block_pool.touch(computed_blocks)  # 更新 LRU
        return num_computed_tokens, False

    def cache_and_free_slots(self, request_id):
        # 请求完成后，将 block 写入 prefix cache 哈希表
        self.single_type_manager.cache_blocks(request, num_tokens)
        self._free_slots(request_id)
```

**5. TP 协调机制：**

```python
# tp_rank > 0：发送完成信号给 rank 0
if self.tp_rank == 0:
    for i in range(1, self.tp_world_size):
        other_ranks_finished_ids.extend(self.tp_group.recv_object(src=i))
    # 所有 rank 都完成后才释放 CPU 槽位
    for req_id in list(self.done_sending_count.keys()):
        if self.done_sending_count[req_id] == self.tp_world_size:
            all_done_sending.add(req_id)
    # 异步 RPC 避免 blocking
    sending_finished_thread = threading.Thread(target=self._sending_finished, args=(all_done_sending,))
else:
    self.tp_group.send_object(done_sending, dst=0)
```

### 4.7 配置与使用

```python
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="CPUOffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "cpu_swap_space_gb": 800,         # CPU swap 空间大小（GB），默认 800
        "swap_in_threshold": 0,           # 触发 swap-in 的阈值
    },
)

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    enable_prefix_caching=True,           # 前提条件：必须启用 prefix caching
    kv_transfer_config=kv_transfer_config,
)
```

**配置参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `kv_connector` | - | `"CPUOffloadingConnector"` |
| `kv_role` | - | `"kv_both"` 等 |
| `cpu_swap_space_gb` | `800` | CPU swap 空间大小（GB） |
| `swap_in_threshold` | `0` | CPU 命中数超过此阈值才触发 swap-in |
| `enable_prefix_caching` | - | **必须为 True**，否则 connector 为 no-op |

---

## 五、三套方案对比

### 5.1 架构对比

```mermaid
graph LR
    subgraph "方案一 NPUOffloadingSpec"
        A1[Scheduler: CPUOffloadingManager<br/>LRU BlockPool]
        A2[Worker: CpuNpuOffloadingHandler<br/>双 Stream + Event 池]
        A1 <-.TransferSpec.-> A2
    end

    subgraph "方案二 AscendSimpleCPUOffload"
        B1[Scheduler: SimpleCPUOffloadScheduler<br/>上游复用]
        B2[Worker: SimpleCPUOffloadNPUWorker<br/>+ NPUDmaCopyBackend 守护线程]
        B1 <-.metadata.-> B2
    end

    subgraph "方案三 CPUOffloadingConnector"
        C1[Scheduler: ConnectorScheduler<br/>ZMQ RPC Client]
        C2[Worker: ConnectorWorker<br/>load/save stream + save_thread]
        C3[MetadataServer 独立进程<br/>CPUKVCacheManager + Prefix Cache]
        C1 <-.ZMQ RPC.-> C3
        C2 <-.ZMQ RPC.-> C3
        C2 <-.SharedMemory.-> C3
    end

    style A1 fill:#e1f5ff
    style A2 fill:#fff4e1
    style B1 fill:#e1f5ff
    style B2 fill:#fff4e1
    style C1 fill:#e1f5ff
    style C2 fill:#fff4e1
    style C3 fill:#ffe1e1
```

### 5.2 特性对比表

| 特性 | 方案一 NPUOffloadingSpec | 方案二 AscendSimpleCPUOffload | 方案三 CPUOffloadingConnector |
|------|------|------|------|
| **基础框架** | vLLM OffloadingConnector | vLLM SimpleCPUOffloadConnector（上游适配） | 独立实现 KVConnectorBase_V1 |
| **CPU Prefix Caching** | 通过 CPUOffloadingManager（LRU） | 复用上游 SimpleCPUOffloadScheduler | 自带 CPUKVCacheManager（完整 prefix cache） |
| **跨进程共享** | 否 | 否 | 是（SharedMemory + ZMQ RPC） |
| **独立进程** | 否 | 否 | 是（MetadataServerProc） |
| **传输底层** | `swap_blocks_batch`（aclrtMemcpyBatchAsync） | `swap_blocks_batch`（aclrtMemcpyBatchAsync） | `torch.Tensor.copy_`（non_blocking） |
| **CPU 容量配置** | `num_cpu_blocks` | `cpu_capacity_bytes` | `cpu_swap_space_gb`（默认 800GB） |
| **MLA 支持** | 通过 spec | 通过 spec | 显式 MLAConfig |
| **TP 协调** | 不涉及 | 不涉及 | tp_group.send_object/recv_object |
| **后台线程** | 无（主线程轮询 Event） | 有（_copy_loop 守护线程） | 有（_save_listener 线程） |
| **Block 映射** | CPU block = N × NPU block | 1:1（block_size 相同） | 1:1（block_size 相同） |
| **Event 机制** | Event 池复用 | events_list 轮询 | stream.synchronize() |
| **scheduler 侧逻辑** | vLLM 上游 CPUOffloadingManager | vLLM 上游 SimpleCPUOffloadScheduler | 自研 ConnectorScheduler |
| **worker 侧逻辑** | 自研 CpuNpuOffloadingHandler | 继承上游 + 替换 backend | 自研 ConnectorWorker |

### 5.3 传输机制对比

```mermaid
graph TB
    subgraph "方案一 传输机制"
        direction LR
        T1A[transfer_async 提交] --> T1B[主线程在专用 stream<br/>record start_event]
        T1B --> T1C[swap_blocks_batch<br/>aclrtMemcpyBatchAsync]
        T1C --> T1D[record end_event]
        T1D --> T1E[get_finished 轮询<br/>end_event.query 非阻塞]
    end

    subgraph "方案二 传输机制"
        direction LR
        T2A[launch_copy 提交] --> T2B[put 到 FIFO Queue]
        T2B --> T2C[守护线程 get]
        T2C --> T2D[在专用 stream<br/>copy_blocks → swap_blocks_batch]
        T2D --> T2E[event.record]
        T2E --> T2F[append 到 events_list<br/>主线程轮询]
    end

    subgraph "方案三 传输机制"
        direction LR
        T3A[load_kv_layer/save<br/>逐 layer 提交] --> T3B[在 load_stream/save_stream<br/>tensor.copy_ non_blocking]
        T3B --> T3C[stream.synchronize<br/>阻塞等待]
    end

    style T1C fill:#ffe1e1
    style T2D fill:#ffe1e1
    style T3B fill:#e1f5ff
```

---

## 六、选型建议

```mermaid
flowchart TD
    Start[需要 KV Cache CPU 卸载] --> Q1{需要跨 DP 共享<br/>CPU KV cache?}
    Q1 -- 是 --> Q2{需要 CPU 侧<br/>完整 prefix caching?}
    Q2 -- 是 --> S3[方案三: CPUOffloadingConnector<br/>支持 SharedMemory 跨进程共享<br/>自带 CPUKVCacheManager<br/>默认 800GB swap 空间]
    Q2 -- 否 --> S3

    Q1 -- 否 --> Q3{希望与上游 vLLM<br/>API 完全一致?}
    Q3 -- 是 --> S2[方案二: AscendSimpleCPUOffloadConnector<br/>复用上游 scheduler 逻辑<br/>仅替换 worker 拷贝后端]
    Q3 -- 否 --> Q4{需要 CPU block<br/>聚合多个 NPU block?}
    Q4 -- 是 --> S1[方案一: NPUOffloadingSpec<br/>block_size_factor 聚合<br/>Event 池复用<br/>numpy 向量化构建指针]
    Q4 -- 否 --> Q5{单实例长上下文<br/>需要 LRU 淘汰?}
    Q5 -- 是 --> S1
    Q5 -- 否 --> S2

    style S1 fill:#e1f5ff
    style S2 fill:#fff4e1
    style S3 fill:#ffe1e1
```

### 选型要点

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| 单实例 NPU 显存不足，长上下文 | 方案一 | LRU 淘汰 + block 聚合减少传输次数 |
| 与上游 vLLM API 一致的简单卸载 | 方案二 | scheduler 逻辑上游复用，维护成本低 |
| 跨 DP 共享 CPU KV cache | 方案三 | SharedMemory 跨进程共享 |
| 需要 CPU 侧 prefix caching | 方案三 | 自带 CPUKVCacheManager + BlockPool |
| 大容量 CPU swap（数百 GB） | 方案三 | 默认 800GB，可配置 |
| MLA 模型 + CPU 卸载 | 方案三 | 显式 MLAConfig 处理 rope/nope 分离 |
| RL 训练场景 | 方案一/二 | 单实例内 LRU 淘汰即可 |

### 性能考量

- **传输效率：** 方案一和方案二使用 `aclrtMemcpyBatchAsync` 批量 DMA，方案三使用 `tensor.copy_` 逐 layer 传输。
- **CPU 内存利用率：** 方案三的 prefix caching 能显著减少重计算，但需要额外的哈希计算和 BlockPool 管理。
- **跨进程开销：** 方案三的 ZMQ RPC 和 SharedMemory 同步有额外开销，但支持跨 DP 共享。
- **TP 扩展性：** 方案三显式处理 TP 协调，适合大规模 TP 部署；方案一/二不涉及 TP 协调。
