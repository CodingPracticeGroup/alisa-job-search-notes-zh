# 第六章　GPU（GPUs）

> 译自 Alisa's Book of LLMs "GPUs" 一节；底稿与约定见 [`05-scaling-laws.md`](05-scaling-laws.md) 页眉说明。

- **高带宽内存**（high bandwidth memory, HBM）是 GPU 的主存
  - 从 GPU 的视角看，它是"慢内存"
  - A100 上是 40GB 或 80GB
- **静态随机存取存储器**（static RAM, SRAM）是片上的小容量快存
  - A100 上总共约 20MB
