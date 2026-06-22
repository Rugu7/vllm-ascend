# DeepSeek-V4-Flash 分析文档

> 基于 vllm-ascend main 分支源码分析。

## 文档目录

1. [DSv4-Flash KV 计算与 Prefix Cache 分析](./01_dsv4-flash-kv-and-prefix-cache.zh.md)
   - 异构层架构（C4/C128/SWA）
   - 6/7 元组 KV cache 布局
   - Compressor 与 Indexer 机制
   - Lightning Indexer 自定义算子
   - IndexCache 跨层 top-k 复用
   - 逻辑块哈希的压缩 prefix cache
   - AscendHybridKVCacheCoordinator 的 DSv4 处理
   - 两个 Full-Attention Group 截断 quirk
   - EAGLE/MTP 交互

## 核心要点

DSv4-Flash 是 DeepSeek V4 系列的轻量变体，采用异构层架构：

- **C4 层** (compress_ratio=4)：Indexer + Compressor + SWA，逻辑块 512 token
- **C128 层** (compress_ratio=128)：Compressor + SWA，逻辑块 16384 token
- **SWA 层** (compress_ratio=0)：仅滑动窗口注意力

KV cache 为 6 元组（A3）/ 7 元组（A5），包含 compress_kv、swa_kv、state、indexer_state、indexer_k、indexer_scale（+ A5 的 indexer_full）。

Prefix cache 采用逻辑块哈希机制：`compress_ratio` 个物理块哈希为 1 个逻辑块，保证压缩原子性，无部分命中。

## 相关文档

- [Prefix Cache 系列文档](../prefix-cache/README.md)
