# snapshots/math-notes/ 捕获记录

- 资料：Alisa's Math Notes，<https://alisawuffles.notion.site/math-notes>
- 获取：2026-11-15 项目成员用 FireShot 浏览器扩展整页捕获，原始件 `FireShot Capture 003 - math - [alisawuffles.notion.site].pdf`（3.5 MB）存本目录，为权威快照
- 特性（实测）：单页长图 PDF（1169×17086 px 有效内容，高约 10 屏），**无文字层**（pdftotext 提取 0 词）——文字与公式均为位图，必须走识图管线
- 处理：`pdftoppm -r 150` 渲染 → `convert -trim` 去空白 → 按 1800px 高、150px 重叠切成 11 条带（`seg-00.png` … `seg-10.png`），主会话逐条识图转写
- 转写产物：**`alisa-math-notes-en.md`**（2026-11-15 完成）——约 4.6k 词、131 条块级公式、覆盖原页全部 12 个顶层小节（Distributions → Math things）；分段文件 part1–6 保留备查
- 原稿疑误已照录并以〔转写注〕标出：geometric series 求和式、AM-GM 不等号方向；另有两处快照固有限制——① FireShot 捕获在 "Handy equations → exponential series" 行截止，该节若有后续内容不在快照内；② 若原页存在收起状态的 toggle，其内容不会出现（本次未发现收起块迹象）
- 注意：FireShot 为页面渲染捕获，折叠块（toggle）若处于收起状态则内容不会出现在快照中；转写时如发现疑似被截断的小节需回原页核对。
