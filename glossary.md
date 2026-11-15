# glossary.md —— 术语对照表

> 全仓库译文统一遵循本表；新增术语随翻译补充。规则：通用术语用通行中文译法并在首次出现处括注英文；专名与缩写保留英文。

## 保留英文（不翻译）

| 词 | 说明 |
| --- | --- |
| Transformer, attention, decoder-only | 模型/机制专名 |
| PPO, GRPO, DPO, RLHF, SFT, MLE, MDP, CLT, CDF/PDF/PMF | 缩写 |
| RoPE, RMSNorm, SwiGLU, LoRA, GPT, BERT, LLaMA, ViT | 架构/方法专名 |
| PyTorch, numpy, JAX, LeetCode, NeetCode, Notion, ChatGPT, Claude | 工具与产品名 |
| offer, offer letter, Research Scientist | 职场惯用语，保留原文 |

## 通用译法

| 英文 | 中文（首次出现写法） | 备注 |
| --- | --- | --- |
| MTS（Member of Technical Staff） | 技术组成员（首现括注英文全称） | 职场头衔；行文中缩写 MTS 可保留 |
| recruiter | recruiter（招聘专员） | 公司侧对接候选人的角色 |
| headcount | headcount（招聘名额） | |
| ghosted | 被鸽（ghosted） | 公司不给任何回音 |
| withdrawn | 主动退出（withdrawn） | 候选人主动结束流程 |
| exploding offer | "限时作废"式 offer（exploding offer） | 首译保留引号解释 |
| ML coding (interview) | ML 编码面试 | 博客"面试的种类"一节 |
| general coding | 通用编程面试 | |
| technical discussion | 技术讨论面试 | |
| research discussion | 研究讨论面试 | |
| behavioral interview | 行为面试 | |
| job talk | 工作报告（job talk） | |
| tokenizer | 分词器（tokenizer） | |
| backpropagation | 反向传播（backpropagation / backprop） | |
| forward pass / backward pass | 前向传播 / 反向传播 | |
| decoding | 解码 | LLM 生成语境 |
| off-by-one (error) | off-by-one（差一）错误 | |
| network(ing) | 社交联络 / networking | 求职语境 |
| first-step analysis | 首步分析（first-step analysis） | 随机过程求期望技巧，Math Notes |
| memoryless | 无记忆性 | 几何/指数分布性质 |
| expectation | 期望（expectation） | 概率论语境 |
| variance | 方差 | |
| linearity of expectation | 期望的线性性 | |
| indicator (variable) | 示性（指示）随机变量 | $\mathbf 1[\cdot]$ |
| Markov chain | 马尔可夫链 | |
| coupon collector | 赠券收集问题（coupon collector） | 经典概率谜题 |
| layer normalization | 层归一化（Layer Norm / LN） | |
| residual connection | 残差连接 | |
| Add & Norm | Add & Norm（残差+层归一化块的图中写法，保留原文） | |
| multi-head attention | 多头注意力（multi-head attention） | |
| positional embedding | 位置嵌入（positional embedding） | |
| positional encoding | 位置编码 | 与 learned embedding 区分 |
| cross-attention | 交叉注意力 | |
| self-attention | 自注意力 | |
| embedding (matrix) | 嵌入 / 嵌入矩阵 | |
| context(ual) representation | 上下文（相关）表示 | 讲义 Note 10 的核心区分 |
| non-contextual | 非上下文相关 | |
| speculative decoding | 投机解码（speculative decoding） | |
| KV cache | KV 缓存（保留 KV） | |
| batching & packing | 批处理与拼接（batching & packing） | |
| Flash Attention | Flash Attention（保留原文） | |
| tensor/data/pipeline/expert parallelism | 张量/数据/流水/专家并行 | scaling book 译站同用法 |
| mixed precision | 混合精度 | |
| activation | 激活值；激活函数语境下称"激活函数（activation function）" | 按上下文区分 |
| gradient clipping | 梯度裁剪 | |
| logit | logit（对数几率），一般保留英文 | |
| softmax | softmax（保留原文） | |
| policy gradient | 策略梯度（policy gradient） | |
| REINFORCE | REINFORCE（算法名，保留原文） | |
| reward-to-go | 后续回报（reward-to-go） | 从时刻 $t$ 起的未来回报 $G_t$ |
| return | 回报 | 累计（折扣）奖励 $R(\tau)$ |
| advantage (function) | 优势（函数） | $A = G_t - \hat V(s_t)$ |
| value function | 价值函数 | $\hat V(s_t)$ |
| baseline | 基线 | 降方差技巧 |
| importance sampling (weighting) | 重要性采样（加权） | |
| importance sampling ratio | 重要性采样比 | $r_t(\theta)=\pi_\theta/\pi_{\theta_\text{old}}$ |
| clip / clipping | 裁剪（clip） | |
| on-policy / off-policy | 同策略 / 异策略（on-/off-policy） | |
| stop gradient | 停止梯度（stop gradient, sg） | 不回传梯度的算子 |
| forking tokens | 分叉词元（forking tokens） | |
| GRPO / CISPO / PPO / TRPO / SFT | 保留原文 | |
| RLOO / REINFORCE++ / Dr. GRPO | 保留原文（算法名） | 均为 GRPO/REINFORCE 变体 |
| DPO (direct preference optimization) | DPO（直接偏好优化） | 首现括注后保留缩写 |
| Bradley-Terry model | Bradley-Terry 模型（保留人名） | |
| surrogate objective | 替代目标（surrogate objective） | |
| bias correction (Adam) | 偏差修正 | 滑动平均初期 |
| straight-through estimator (STE) | 直通估计器（straight-through estimator, STE） | |
| Gumbel-Max / Gumbel-Softmax | Gumbel-Max / Gumbel-Softmax（保留原文） | |
| online softmax (trick) | 在线 softmax（技巧） | |
| log-sum-exp trick | logsumexp 技巧 | 数值稳定 |
| RLHF | RLHF（保留原文） | |
| SFT (supervised fine-tuning) | SFT（监督微调） | 首现括注后保留缩写 |
| distribution shift | 分布偏移（distribution shift） | |
| rollout | 采样展开（rollout） | 按当前策略生成一条完整轨迹 |
| trajectory | 轨迹 | $\tau$ |
| MDP (Markov Decision Process) | 马尔可夫决策过程（MDP） | |

## 用法示例

- 首现：`自注意力（self-attention）`；后续直接用中文或缩写。
- 图中文字（Add & Norm、Feed-Forward 等）保留英文，正文用中文对照。
