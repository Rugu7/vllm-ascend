# Prefix Cache 系列文档

> 基于 vLLM main 分支（commit `aa4990a9a2024b3f93f1f26f828931f7301daa15`, 2026-06-22）与 vllm-ascend main 分支源码分析。

本系列文档系统分析 vLLM 与 vllm-ascend 的 prefix cache 实现，涵盖核心原理、命中计算、外部缓存机制及 Ascend 定制实现。

## 文档目录

1. [Prefix Cache 核心原理机制](./01_prefix-cache-core-mechanism.zh.md)
   - 三层架构总览
   - 链式 Block Hash 内容寻址
   - KVCacheBlock 与 LRU 自由队列
   - BlockPool 哈希索引
   - KVCacheManager 调度器门面
   - KVCacheCoordinator 多 group 协调
   - 端到端调度流程时序图

2. [Prefix Cache 命中计算方法](./02_prefix-cache-hit-calculation.zh.md)
   - `get_computed_blocks` 入口
   - `FullAttentionManager` 命中扫描
   - `SlidingWindowManager` 反向扫描
   - `MambaManager` 状态重建
   - Hybrid 模型不动点算法
   - `allocate_slots` 三阶段分配
   - 部分命中场景分析
   - 抢占与恢复
   - Cascade Attention 公共前缀

3. [External Prefix Cache 外部前缀缓存机制](./03_external-prefix-cache.zh.md)
   - KVConnector 框架架构
   - `get_num_new_matched_tokens` 命中计算
   - 异步加载状态机
   - OffloadingConnector 参考实现
   - vllm-ascend 的 Mooncake P2P / AscendStore KV Pool
   - Layerwise KV 传输
   - NPU 原生内存传输基础设施
   - Recompute Scheduler PD 分离回退

4. [vllm-ascend Prefix Cache 定制实现](./04_vllm-ascend-prefix-cache.zh.md)
   - 压缩 Prefix Cache（DeepSeek V4）
   - `CompressAttentionManager` 逻辑块哈希
   - `AscendHybridKVCacheCoordinator`
   - `AscendMLAAttentionSpec` 扩展
   - `AscendMambaManager` 反向扫描
   - PCP 对 Prefix Cache 的影响
   - KVComp Hamming 稀疏注意力
   - 与上游 vLLM 的差异总结

## 关键 Mermaid 图表索引

| 图表类型 | 位置 | 说明 |
|---|---|---|
| 架构图 | 01 - 第1节 | 三层 Prefix Cache 架构 |
| 流程图 | 01 - 第3.2节 | LRU 双向链表 |
| 流程图 | 01 - 第6.2节 | Hybrid 协调器迭代不动点 |
| 时序图 | 01 - 第8节 | 端到端调度流程 |
| 流程图 | 02 - 第2.1节 | FullAttention 命中扫描 |
| 流程图 | 02 - 第3节 | SlidingWindow 反向扫描 |
| 流程图 | 02 - 第4节 | Mamba 状态重建 |
| 流程图 | 02 - 第5节 | Hybrid 多 group 命中协调 |
| 流程图 | 02 - 第6.2节 | allocate_slots 三阶段分配 |
| 流程图 | 02 - 第7.1节 | 调度器命中计算全流程 |
| 时序图 | 02 - 第9节 | 抢占与恢复 |
| 架构图 | 03 - 第2.1节 | KVConnector 角色划分 |
| 状态图 | 03 - 第3.3节 | 异步加载状态机 |
| 时序图 | 03 - 第3.4节 | 异步加载块预留 |
| 时序图 | 03 - 第6.2节 | PD 分离命中流程 |
| 架构图 | 03 - 第7.1节 | AscendStore 架构 |
| 时序图 | 03 - 第7.6节 | 完整命中流程 |
| Gantt 图 | 03 - 第8.1节 | Layerwise 传输与计算重叠 |
| 时序图 | 03 - 第8.3节 | Layerwise 传输时序 |
| 时序图 | 03 - 第10.3节 | PD 回退流程 |
| 流程图 | 04 - 第2.5节 | 压缩命中扫描 |
| 时序图 | 04 - 第6节 | 端到端压缩 Prefix Cache 流程 |
| 流程图 | 04 - 第5.2节 | Mamba 命中语义 |
| 流程图 | 04 - 第8.1节 | DualChunkSwap PCP 模式 |

## 源码版本

- **vLLM**: main 分支，commit `aa4990a9a2024b3f93f1f26f828931f7301daa15`（2026-06-22）
- **vllm-ascend**: main 分支（2026-06-22）

## 核心结论

1. **内容寻址 + 链式哈希** 是 prefix cache 的基础，O(1) 单块查表，前缀偏差自动失效
2. **LRU + 引用计数** 实现高效块回收，有哈希的块归队尾保留复用价值
3. **多 group 不动点算法** 保证 hybrid 模型各 attention 类型命中长度一致
4. **两阶段分配** 避免一个 group 误淘汰另一个 group 的命中块
5. **外部 prefix cache** 通过 `KVConnector` 框架统一接入，支持异步加载与 PD 分离
6. **vllm-ascend 的压缩 prefix cache** 通过逻辑块哈希支持 DeepSeek V4 的 MLA compression
7. **Layerwise KV 传输** 通过 `AttentionComputeStartGate` 实现传输与计算重叠
8. **RecomputeScheduler** 为 PD 分离提供优雅的回退机制
