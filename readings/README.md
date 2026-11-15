# readings/ —— 附录文章收集与翻译

本目录对应 Alisa 博客文末 *Appendix: learning resources* 中的文章类资料。原始清单见仓库根 [README.md](../README.md)。

## 处理原则

1. **一律全文翻译**。
2. **先落快照再动笔**：动工第一步是把原文抓取/另存到 `../snapshots/readings/`，文件顶部记录来源 URL 与抓取日期；译文只以快照为准。
3. 元信息表中"未核实"的字段（作者、篇幅）在抓取时补全，不预填猜测。

## 文件索引

| 本地文件 | 原文 | 来源 / 作者 | 处理类型 | 状态 | 建议优先级 |
| --- | --- | --- | --- | --- | --- |
| [self-attention-transformers-cs224n.md](self-attention-transformers-cs224n.md) | Self-Attention & Transformers | Stanford CS224n 讲义，John Hewitt | 全文翻译 | **已完成**（2026-11-15；底稿为识图转写，公式 (1)–(36) 与图 1–6 图注齐备） | 高 |
| [illustrated-gpt2.md](illustrated-gpt2.md) | The Illustrated GPT-2 | jalammar.github.io（Jay Alammar） | 引用本组已有中译 | **已完成**（外链） | 高 |
| [backpropagation-cs231n.md](backpropagation-cs231n.md) | Backpropagation | CS231n 课程笔记（页内未署名） | 全文翻译 | **已完成**（2026-11-15） | 高 |
| [policy-gradient-for-lms.md](policy-gradient-for-lms.md) | Introduction to Policy Gradient for LMs | ivison.id.au，Hamish Ivison | 全文翻译 | **已完成**（2026-11-15） | 中 |
| [grpo-intro.md](grpo-intro.md) | Lightweight Guide to understanding GRPO and RL principles | gitlostmurali.com（"Musings of Murali"） | 全文翻译 | **已完成**（2026-11-15） | 中 |
| [how-to-scale-your-model.md](how-to-scale-your-model.md) | How to Scale Your Model | jax-ml（开源书，GitHub 同名仓库） | 引用本组已有中译（全译） | **已完成**（外链） | 高 |

建议优先级是本仓库的判断（前三篇与"实现/解释 Transformer、反向传播"这些高频面试环节直接相关），不是 Alisa 的排序——原文附录为无序列表，未标注顺序。

## 不单独建文件的资源

以下三项非文章，不进入本目录，在根 README 的资源清单中维护：

- LeetCode 75 / NeetCode Blind 75（题单）
- Stanford CS336: Language Modeling from Scratch（课程，作业 HW1 另行关注）
