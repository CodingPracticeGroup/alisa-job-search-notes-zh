# Alisa 求职笔记 · 中文译本全集

收集并翻译 Alisa 在求职博客及其笔记中公开的学习资料，做成有组织的中文版本，供 AI / LLM 方向求职准备使用。

## 在线阅读

全站已发布为 GitHub Pages 静态站点（三栏布局：左侧导航、正文、右侧本页目录，公式与脚注原生渲染）：

**<https://codingpracticegroup.github.io/alisa-job-search-notes-zh/>**

若上方链接尚未生效，说明 Pages 来源还没切换：在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**，之后每次推送到 main 会自动构建部署（工作流：`.github/workflows/deploy-pages.yml`）。

本地预览（需要 Node.js ≥ 18）：

```bash
npm install
npm run dev       # 开发预览，改动热更新
npm run build     # 生产构建，产物在 .vitepress/dist/
```

站点以仓库内的 `.md` 译文为唯一内容源（`index.md` 是首页）；`README.md` 与 `snapshots/` 只作为仓库档案，不参与站点构建。

## 背景

Alisa 是华盛顿大学（UW）NLP 方向博士生，在临近毕业完成工业界求职（Research Scientist / MTS 岗位）后，于博客发表了 [Notes on the Industry Job Search](https://alisawuffles.github.io/blog/job-search/)（页面标注发表日期 2026-06-20）。文章基于她 11 家公司、57 场面试的一手经历，系统介绍了：

- **求职时间线**：从开始面试到拿到结果的全过程节奏（参考了 Nathan Lambert 的时间线图形式）；
- **7 类面试分类**：ML 编码（ML coding）、通用编程（general coding）、技术讨论、研究讨论、行为面试、数学面试、工作报告（Job Talk），并逐一说明考察重点与准备方向；
- **准备方法**：以记笔记、画图、做题、从零实现为核心的复习方式；
- **薪酬谈判**与求职心态。

她在文中公开了自己手写的两份 Notion 笔记，被读者称为"LLM 面试圣经"，并给出了她实际使用过的课程与阅读清单。这些资料精准对应 AI Research / MTS 岗位面试的实际考察范围（实现、推导、解释、debug、系统判断），本仓库即以此为主题做收集与中译。

## 资料清单

以下链接均已对照博客原文核实（"正文"指博客正文中的引用，"附录"指博客文末 *Appendix: learning resources*）。

### 一手来源（本仓库的翻译对象）

| 资料 | 说明 | 原始链接 |
| --- | --- | --- |
| 博客文章 | Notes on the Industry Job Search，全文 | <https://alisawuffles.github.io/blog/job-search/>（[源码仓库](https://github.com/alisawuffles/alisawuffles.github.io/blob/main/blog/job-search/index.html)） |
| Alisa's Book of LLMs | 她求职期间持续维护的 LLM 知识笔记，覆盖神经网络基础到后训练的广泛内容 | <https://alisawuffles.notion.site/alisa-s-book-of-llms> |
| Alisa's Math Notes | 为数学面试整理的笔记，涉及概率、线性代数、微积分等 | <https://alisawuffles.notion.site/math-notes> |

### 附录推荐资源（按需翻译或做导读）

| 类别 | 资源 | 链接 |
| --- | --- | --- |
| 编码题单 | LeetCode 75 | <https://leetcode.com/studyplan/leetcode-75/> |
| 编码题单 | NeetCode Blind 75 | <https://neetcode.io/practice/practice/blind75> |
| 系统课程 | Stanford CS336: Language Modeling from Scratch | <https://cs336.stanford.edu/spring2025/> |
| 概念阅读 | Self-Attention & Transformers（CS224n 讲义） | [PDF](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf) |
| 概念阅读 | The Illustrated GPT-2（Jay Alammar） | <https://jalammar.github.io/illustrated-gpt2/> |
| 概念阅读 | Backpropagation（CS231n） | <https://cs231n.github.io/optimization-2/> |
| RL 相关 | Introduction to Policy Gradient for LMs | <https://ivison.id.au/2026/02/09/policy-gradient.html> |
| RL 相关 | Lightweight Guide to understanding GRPO and RL principles | <https://gitlostmurali.com/blog/grpo-intro> |
| 系统 / 扩展 | How to Scale Your Model（scaling book） | <https://jax-ml.github.io/scaling-book/> |

> 已有本组中译、不重复翻译的条目（均已在 `readings/` 登记外链）：
>
> - **The Illustrated GPT-2** —— [Jay Alammar 博客中文翻译站](https://codingpracticegroup.github.io/jay-alammar-blog-zh/posts/illustrated-gpt2.html)（该译站另收录 Illustrated Transformer、Illustrated BERT 等多篇，见[站点目录](https://codingpracticegroup.github.io/jay-alammar-blog-zh/)）
> - **How to Scale Your Model** —— [简体中文翻译站](https://codingpracticegroup.github.io/scaling-book/)（全书 0–12 章全译）

### 正文中额外提到的资源

- **CS336 Homework 1**（从零实现并调试 Transformer）：Alisa 特别强调这是面试中回报率最高的练习之一，建议练到"肌肉记忆"的程度。仓库：<https://github.com/stanford-cs336/assignment1-basics>

## 目录结构

```
alisa-job-search-notes-zh/
├── README.md              # 本文件：资料索引与翻译进度（不参与站点构建）
├── index.md               # 静态站点首页
├── .vitepress/            # 站点配置（导航、侧栏、目录、公式、搜索）
├── .github/workflows/     # GitHub Pages 自动部署工作流
├── blog/                  # 博客《Notes on the Industry Job Search》全文中译
├── llm-book/              # Alisa's Book of LLMs 中译，按章节拆分
├── math-notes/            # Alisa's Math Notes 中译
├── glossary.md            # 术语对照表（全仓库译文统一遵循）
├── readings/              # 附录文章的译文或导读，每篇一个文件（索引与进度见其 README）
├── snapshots/             # 各资料的原文快照，供翻译对照（约定见其 README）
└── assets/                # 插图、时间线图等静态资源
```

当前进度：博客、readings 四篇（另两篇登记外链）、Book of LLMs 全部 11 章、Math Notes 均已中译完成（2026-11-15）；各目录细节状态以子目录 README 为准。

另：两份 Notion 笔记均无法直接 HTTP 抓取，已改由手动导出获取——Book of LLMs 已导出并转换为英文底稿（公式全量保留 LaTeX，见 [`llm-book/`](llm-book/README.md)）；Math Notes 以 FireShot 整页捕获（无文字层），已走识图管线完成底稿与中译（见 [`math-notes/`](math-notes/README.md)）。

## 翻译约定

先快照后翻译：每个资料先在 `snapshots/` 落原文快照，译文只以快照为准；只从原始英文翻译，不以网络现版本为准。读者定位为中国高考后、大学一年级的学生，译文追求信达雅，术语首现解释。术语统一维护 `glossary.md` 对照表：通用术语用通行中文译法并首次出现处括注英文，专名与缩写保留英文。无文字层的 PDF 走识图管线：渲染页面 → 切条带 → 逐条识图转写英文底稿 → 由底稿翻译。每份译文头部注明原文出处、所用底稿与日期；原稿疑误照录并以〔转写注〕标明。
