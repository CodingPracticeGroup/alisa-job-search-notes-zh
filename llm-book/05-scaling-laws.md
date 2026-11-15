# 第五章　缩放定律（Scaling laws）

> 译自 Alisa's Book of LLMs（<https://alisawuffles.notion.site/alisa-s-book-of-llms>）"Scaling laws" 一节；底稿 `../snapshots/llm-book/alisa-book-of-llms-en.md`（2026-11-15 快照）。术语见根 `glossary.md`。

- **最大更新参数化**（maximal update parameterization, $\mu P$）
  - 核心痛点：小规模实验调出的超参数，放大规模后就不适用了
  - 标准参数化下 → 网络变宽时，各层的**更新量级**彼此失衡（宽度一变就乱）
  - $\mu P$ 按层调整初始化尺度与各层学习率，使更新相对于权重的**幅度**在任何宽度下保持不变
  - 它处理的首先是**宽度**方向上的缩放
- **把学习率拟合成算力 $C$ 的函数**

$$\text{LR}(C)=\beta C^{-\alpha}\implies\log\text{LR}(C)=\log\beta-\alpha\log C$$

- **把损失拟合成算力 $C$ 的函数**
  - 必须引入不可约损失项 $\mathcal L_\infty$（数据本身的熵），否则 $C\to\infty$ 时损失会错误地趋向 0

$$\mathcal L(C)=\mathcal L_\infty+\beta C^{-\alpha}\implies \log(\mathcal L(C)-\mathcal L_\infty)=\log\beta-\alpha\log C$$

- 方程怎么拟合？用最小二乘法
  - 最小二乘的目标是最小化**残差平方和**
    - 分普通（线性）最小二乘与非线性最小二乘
  - 线性最小二乘有闭式解
    - 对 $y=X\beta$，解为 $\beta=(X^\top X)^{-1}X^\top y$
  - 非线性最小二乘靠迭代精化求解

$$S=\sum_i(y_i-f(x_i))^2$$
