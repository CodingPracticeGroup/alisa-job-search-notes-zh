# snapshots/readings/ 抓取记录

统一信息：抓取日期 2026-11-15；方式 `curl -sL`（无 JS 渲染需求，四站均服务端返回完整正文）；全部 HTTP 200。HTML 原件保留，`.md` 为 pandoc（`-f html -t gfm`）转换的可读副本，供翻译对照；图片多为站点绝对 URL，未镜像到本地。

## cs224n-self-attention-transformers-2023-draft.pdf

- 来源：https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf
- 标题（元数据）：[draft] Note 10: Self-Attention & Transformers；课程 CS 224n, Winter 2023
- 作者（首页署名）：John Hewitt（课程负责人 Christopher Manning, John Hewitt）
- 体量：18 页，纯文本约 7,166 词（提取件：`cs224n-self-attention-transformers-2023-draft.txt`）
- 页图：`cs224n-pages/`（`pdftoppm -png -r 170`，18 张 PNG，2026-11-15）；识图转写件：`cs224n-extract-p*.md`（按页码段分件，由主会话逐页识图转写——endpoint 仅单会话，不用子代理）。翻译底稿以识图转写件合并结果为准，`.txt` 仅作对照

## cs231n-optimization-2.html / .md

- 来源：https://cs231n.github.io/optimization-2/
- 作者：页面未署名（CS231n 课程公开笔记）
- 体量：约 4,513 词（Markdown 副本统计）

## ivison-policy-gradient-for-lms.html / .md

- 来源：https://ivison.id.au/2026/02/09/policy-gradient.html
- 作者：Hamish Ivison（HTML meta name="author" 与页面标题署名）
- 体量：约 1,998 词（Markdown 副本统计）

## gitlostmurali-grpo-intro.html / .md

- 来源：https://gitlostmurali.com/blog/grpo-intro
- 作者：站点署名 "Musings of Murali"，页内未见全名，待确认
- 体量：约 2,182 词（Markdown 副本统计）

## 处理类型（2026-11-15 项目决定）

四篇均做全文翻译。
