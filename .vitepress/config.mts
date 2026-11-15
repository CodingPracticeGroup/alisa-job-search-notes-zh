import { defineConfig } from 'vitepress'
import footnote from 'markdown-it-footnote'
import cjkFriendly from 'markdown-it-cjk-friendly'

const REPO = 'https://github.com/CodingPracticeGroup/alisa-job-search-notes-zh'

const chapters = [
  { text: '01 神经网络基础', link: '/llm-book/01-neural-net-basics' },
  { text: '02 数学工具箱', link: '/llm-book/02-mathy-things' },
  { text: '03 现代 Transformer 语言模型', link: '/llm-book/03-modern-transformer-lm' },
  { text: '04 推理', link: '/llm-book/04-inference' },
  { text: '05 缩放定律', link: '/llm-book/05-scaling-laws' },
  { text: '06 GPU', link: '/llm-book/06-gpus' },
  { text: '07 其他架构', link: '/llm-book/07-other-architectures' },
  { text: '08 后训练', link: '/llm-book/08-post-training' },
  { text: '09 精度', link: '/llm-book/09-precision' },
  { text: '10 并行', link: '/llm-book/10-parallelism' },
  { text: '11 多模态', link: '/llm-book/11-multimodality' },
]

const readings = [
  { text: '反向传播（CS231n）', link: '/readings/backpropagation-cs231n' },
  { text: '自注意力与 Transformer（CS224n）', link: '/readings/self-attention-transformers-cs224n' },
  { text: '语言模型的策略梯度', link: '/readings/policy-gradient-for-lms' },
  { text: 'GRPO 与强化学习原理入门', link: '/readings/grpo-intro' },
  { text: '图解 GPT-2（本组已有中译·外链）', link: 'https://codingpracticegroup.github.io/jay-alammar-blog-zh/posts/illustrated-gpt2.html' },
  { text: 'How to Scale Your Model（本组已有全译·外链）', link: 'https://codingpracticegroup.github.io/scaling-book/' },
]

export default defineConfig({
  lang: 'zh-CN',
  title: 'Alisa 求职笔记 · 中文译本全集',
  description: 'Alisa 求职博客、Book of LLMs（全 11 章）、数学笔记与延伸读物的中文翻译',
  appearance: true,
  lastUpdated: false,
  cleanUrls: true,
  ignoreDeadLinks: [
    /^https?:\/\/localhost/,
    /翻译-方法论/,
  ],
  srcExclude: [
    'README.md',
    '**/README.md',
    'snapshots/**',
  ],
  markdown: {
    math: true,
    theme: {
      light: 'github-light',
      dark: 'github-dark',
    },
    config(md) {
      md.use(footnote) // 支持 [^n] 译注脚注
      md.use(cjkFriendly) // 修复中文语境 **加粗** 不解析
    },
  },
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '求职博客', link: '/blog/job-search' },
      { text: 'Book of LLMs', link: '/llm-book/01-neural-net-basics' },
      { text: '数学笔记', link: '/math-notes/math-notes' },
      { text: '延伸读物', link: '/readings/backpropagation-cs231n' },
      { text: '术语表', link: '/glossary' },
    ],
    sidebar: {
      '/blog/': [
        {
          text: 'Notes on the Industry Job Search',
          items: [{ text: '工业界求职笔记（全文）', link: '/blog/job-search' }],
        },
      ],
      '/llm-book/': [
        { text: 'Alisa\u2019s Book of LLMs', collapsed: false, items: chapters },
      ],
      '/math-notes/': [
        { text: '数学笔记', items: [{ text: '概率与微积分速记', link: '/math-notes/math-notes' }] },
      ],
      '/readings/': [
        { text: '延伸读物', collapsed: false, items: readings },
      ],
      '/glossary': [
        { text: '术语对照表', items: [{ text: '术语对照表', link: '/glossary' }] },
      ],
      '/': [
        {
          text: '开始',
          items: [
            { text: '阅读指南', link: '/' },
            { text: '术语对照表', link: '/glossary' },
          ],
        },
      ],
    },
    outline: { level: [2, 3], label: '本页目录', wide: true },
    sidebarMenuLabel: '目录',
    returnToTopLabel: '返回顶部',
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索', buttonEraseText: '清除' },
          suggestions: { noResultsText: '没有找到结果', resetButtonTitle: '清除' },
          modal: { displayDetails: '详情', noResultsText: '没有结果', footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' } },
        },
      },
    },
    socialLinks: [{ icon: 'github', link: REPO }],
    footer: {
      message: '以仓库内快照为唯一底稿的中文翻译 · 读者定位：理工科新生',
    },
    editLink: undefined,
  },
})
