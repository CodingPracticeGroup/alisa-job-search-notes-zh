# snapshots/llm-book/ 导出记录

- 资料：Alisa's Book of LLMs，<https://alisawuffles.notion.site/alisa-s-book-of-llms>
- 获取：2026-11-15 项目成员在公开页手动导出（HTML 格式），原始导出包 `e22d4a79-9e3a-412b-a9ae-a48a95f1931f_ExportBlock-cb4500cc-57af-4456-9fb3-fa3674967cd5.zip`（12.5 MB）存本目录，为权威快照
- 导出包内容：Notion 静态 HTML（1.9 MB）+ 12 张 PNG 图片（原目录名 `Alisa’s book of LLMs`，已改名 `assets/`）
- 转换：`convert_notion.py`（可复跑）→ 英文底稿 **`alisa-book-of-llms-en.md`**
  - 块级公式 151 条、行内公式 707 条，全部从 HTML 的 `data-notion-equation` / `notion-inline-equation` 属性恢复 LaTeX，**无需识图**；
  - 正文引用图片 12 张（已转为 Markdown 图片语法；`rnn-2.png` 此前因文件名含括号未被引用检测命中，实际在 "Other architectures" 一节使用）；
  - 规模：11 个顶层部分、56 个标题、约 1.35 万词；原页自带目录，转换后保留为纯文本列表。
- 注意：底稿以本次导出为快照，不随 Notion 在线版更新；原文公式无编号，译文如需编号另行添加；原文标题层级（h1=部分、h2/h3=小节）已保留。
- 章节体量统计（译前规划用，2026-11-15）：

| 部分 | 词数 | 公式数 | 图片 |
| --- | ---: | ---: | ---: |
| Neural net basics | 2,654 | 165 | 3 |
| Mathy things | 1,573 | 108 | 0 |
| The modern transformer LM | 2,450 | 243 | 2 |
| Inference | 1,276 | 31 | 1 |
| Scaling laws | 165 | 10 | 0 |
| GPUs | 42 | 0 | 0 |
| Other architectures | 506 | 47 | 1 |
| Post-training | 2,064 | 192 | 0 |
| Precision | 494 | 0 | 0 |
| Parallelism | 1,816 | 60 | 3 |
| Multimodality | 227 | 2 | 1 |
