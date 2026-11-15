# How to Scale Your Model

> 状态：**已完成——直接采用本组已有中译（全译，0–12 章齐全），本仓库不重复翻译。**

- 原文：<https://jax-ml.github.io/scaling-book/>（开源书，源码仓库 `jax-ml/scaling-book`）
- 原书作者：Jacob Austin, Sholto Douglas, Roy Frostig, Anselm Levskaya, Charlie Chen, Sharad Vikram, Federico Lebron, Peter Choy, Vinay Ramasesh, Albert Webson, Reiner Pope（Google DeepMind；署名译文本的 byline，2026-11-15 核实），原书标注发表于 2025-02-04
- **中文翻译（本组仓库）**：[《How to Scale Your Model》简体中文翻译站](https://codingpracticegroup.github.io/scaling-book/)
  - 章节直达：[引言](https://codingpracticegroup.github.io/scaling-book/) · [1 Roofline](https://codingpracticegroup.github.io/scaling-book/roofline/) · [2 TPU](https://codingpracticegroup.github.io/scaling-book/tpus/) · [3 分片矩阵乘法](https://codingpracticegroup.github.io/scaling-book/sharding/) · [4 Transformer 数学](https://codingpracticegroup.github.io/scaling-book/transformers/) · [5 训练并行](https://codingpracticegroup.github.io/scaling-book/training/) · [6 训练 LLaMA 3](https://codingpracticegroup.github.io/scaling-book/applied-training/) · [7 推理](https://codingpracticegroup.github.io/scaling-book/inference/) · [8 部署 LLaMA 3](https://codingpracticegroup.github.io/scaling-book/applied-inference/) · [9 性能剖析](https://codingpracticegroup.github.io/scaling-book/profiling/) · [10 JAX](https://codingpracticegroup.github.io/scaling-book/jax-stuff/) · [11 结论与延伸阅读](https://codingpracticegroup.github.io/scaling-book/conclusion/) · [12 GPU](https://codingpracticegroup.github.io/scaling-book/gpus/)
- 收录于：Alisa 博客 *Appendix: learning resources*
- 引用可用性验证：2026-11-15（首页与 `applied-inference` 章均 HTTP 200、正文为中文）

## 范围（已核实，据译本目录）

从系统视角讲大模型在 TPU/GPU 上的扩展：roofline 分析、TPU 硬件与集群互联、分片矩阵乘法、Transformer 参数量与 FLOPs 推导、训练/推理的并行策略（数据/张量/流水/专家并行、ZeRO、重计算等）、LLaMA 3 训练与部署实操、性能剖析与 JAX 编程、GPU 专章，各章附练习题。

## 面试关联

对应 Alisa 正文所述"对模型扩展与系统设计做判断"的能力（技术讨论轮的资源估算类问题）。此关联为本仓库判断。
