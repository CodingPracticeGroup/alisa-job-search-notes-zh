# Alisa 的数学笔记（Math Notes）——概率与微积分速记

> 译自：Alisa's Math Notes（<https://alisawuffles.notion.site/math-notes>）
> 底稿：`../snapshots/math-notes/alisa-math-notes-en.md`（FireShot 整页快照识图转写，2026-11-15）
> 译注日期：2026-11-15。〔译注〕为译者添加；〔转写注〕为转写阶段对原稿疑误处的标注（照录原则），一并保留。术语见根 `glossary.md`。
> 说明：这是 Alisa 为面试备战整理的**速记式**笔记，原文即条目体（bullet notes），无行文铺陈；译文保持条目体例。每条推导都完整可跟着算。

本页目录：

- 分布（Distributions）
  - 离散分布：Bernoulli / Binomial / Poisson / Geometric
  - 连续分布：Uniform / Exponential / Gaussian
- 期望与方差（Expectations & variance）
- 函数一览（Zoo of functions）
- 不等式（Inequalities）
- 条件概率（Conditional probabilities）
- 极限定理（Limit theorems）
- 一些题型（Some problem types）：首步分析 / 用期望数数 / $n$ 个随机变量的最大最小
- 组合计数（Combinatorics）
- 马尔可夫链（Markov chains）
- 谜题（Puzzles）：生日问题 / 赌徒破产 / 随机游走 / 赠券收集 / 信息论谜题 / 蓄水池采样 / 秘书问题 / 蒙特霍尔 / 最长连续正面 / 球投盒 / 路径计数
- 统计（Statistics）：最大似然估计 / 偏差与方差
- 数学工具（Math things）：泰勒级数 / 求导法则 / 常用公式

> "一句话认概念"速查表（原文置于开头，是全笔记的索引）：
>
> | 描述 | 对应概念 |
> | --- | --- |
> | *无记忆*、*等待* | 几何分布、指数分布 |
> | 反复试验、有成有败 | Bernoulli、二项分布 |
> | 大样本、求平均 | 中心极限定理（CLT）、近似高斯 |
> | 只知道均值/方差就要给界 | Markov / Chebyshev 不等式 |
> | 从数据估计参数 | MLE、MAP、贝叶斯更新 |
> | 根据新证据更新信念 | 贝叶斯公式 |
> | 随机变量的函数 | Jensen 不等式 |
> | 数"平均有多少个满足条件" | 期望线性性 + 示性变量 |
> | 期望命中时间 / 返回概率 | 首步分析、解递推 |
> | 马尔可夫性（未来只依赖现在） | 首步分析 |

# 分布（Distributions）

|  | 离散 | 连续 |
| --- | --- | --- |
| 数事件个数 | **二项分布**：$n$ 次试验里成功几次？ | **泊松分布**：固定窗口里来多少事件？ |
| 数等待时长 | **几何分布**：到第一次成功要试几次？ | **指数分布**：到第一次事件要等多久？ |

## 离散分布

### Bernoulli（伯努利）

- **伯努利**随机变量 $X\sim\mathrm{Bernoulli}(p)$：一次试验，成功概率 $p$

$$P(X=1)=p,\ P(X=0)=1-p$$

  - 紧凑的 PMF 写法（PMF：概率质量函数，即"每个取值各有多大概率"的函数）

$$P(X=x)=p^{x}(1-p)^{1-x}$$

- $\mathbb E[X]=p,\ \mathrm{Var}[X]=p(1-p)$（$\mathbb E$：期望，按概率加权的平均；$\mathrm{Var}$：方差，偏离均值的平方的平均）
  - 证明
    - $\mathbb E[X]=1\cdot p+0\cdot(1-p)=p$
    - $\mathbb E[X^2]=1^2\cdot p+0^2\cdot(1-p)=p$
    - $\mathrm{Var}[X]=\mathbb E[X^2]-\mathbb E[X]^2=p-p^2=p(1-p)$
- **类别分布（categorical）**把 Bernoulli 从 2 个结果推广到 $k$ 个结果

$$P(X=i)=p_i$$

  - 每个结果 $i$ 相当于一个 Bernoulli$(p_i)$，故 $\mathbb E[X_i]=p_i$、$\mathrm{Var}(X_i)=p_i(1-p_i)$
  - 常用技巧
    - **示性变量平方不变**：对任何取值于 $\{0,1\}$ 的 $X$，有 $X^2=X$，于是 $\mathbb E[X^2]=\mathbb E[X]$

### 二项分布（Binomial）

- **二项**随机变量 $X\sim\mathrm{Binomial}(n,p)$：$n$ 次独立 Bernoulli$(p)$ 试验的成功次数
  - $X=\sum_{i=1}^{n}X_i$，其中 $X_i\sim\mathrm{Bernoulli}(p)$

$$P(X=k)=\binom{n}{k}p^{k}(1-p)^{n-k}$$

  - 道理：含 $k$ 次成功的每一种具体序列概率都是 $p^{k}(1-p)^{n-k}$，而这样的序列共 $\binom{n}{k}$ 种
- **二项式定理**

$$(a+b)^{n}=\sum_{k=0}^{n}{\binom{n}{k}}a^{k}b^{n-k}$$

- 可以验证所有概率加起来为 1：$\sum_{k}P(X=k)=1$

$$\sum_{k=0}^{n}{\binom{n}{k}}p^{k}(1-p)^{n-k}=(p+(1-p))^{n}=1^{n}=1$$

- $\mathbb E[X]=np$，$\mathrm{Var}[X]=np(1-p)$
  - 证明
    - $\mathbb E[X]=\mathbb E[\sum_{i}X_{i}]=\sum_{i}\mathbb E[X_{i}]=np$
    - $\mathrm{Var}[X]=\mathrm{Var}[\sum_{i}X_{i}]=\sum_{i}\mathrm{Var}[X_{i}]=np(1-p)$（第二个等号成立是因为诸 $X_i$ 相互独立）
  - 常用技巧
    - 拆成一串 Bernoulli，用期望的线性性
    - 把 $X^{2}$ 拆成交叉项

$$X^{2}=\left(\sum_{i}X_{i}\right)^{2}=\sum_{i}X_{i}^{2}+\sum_{i\neq j}X_{i}X_{j}$$

  - 这样求期望就容易了（利用示性变量的性质 $X_{i}^{2}=X_{i}$）

$$\mathbb E[X^{2}]=\sum_{i}\mathbb E[X_{i}]+\sum_{i\neq j}\mathbb E[X_{i}X_{j}]=n\mathbb E[X_{i}]+n(n-1)\mathbb E[X_{i}X_{j}]$$

### 泊松分布（Poisson）

- **泊松**随机变量 $X\sim$ Poisson$(\lambda)$：当事件以平均速率 $\lambda$ 相互独立地发生时，固定区间内发生的事件数

$$P(X=k)=\frac{\lambda^{k}e^{-\lambda}}{k!}$$

- 长度为 $t$ 的区间内的事件数服从 $N(t)\sim$ Poisson$(\lambda t)$
- 用二项分布极限推导：把固定区间切成 $n$ 个极小的子区间，小到每个子区间最多发生一个事件
  - 此时总数服从 Binomial$(n,p)$，其中 $np=\lambda$

$$P(X{-}k)=\binom{n}{k}p^{k}(1-p)^{n-k}$$

$$=\frac{n!}{k!(n-k)!}\left(\frac{\lambda}{n}\right)^{k}\left(1-\frac{\lambda}{n}\right)^{n-k}$$

$$=\frac{\lambda^{k}}{k!}\underbrace{\frac{n!}{(n-k)!\,n^{k}}}_{\to 1}\underbrace{\left(1-\frac{\lambda}{n}\right)^{n}}_{\to e^{-\lambda}}\underbrace{\left(1-\frac{\lambda}{n}\right)^{-k}}_{\to 1}$$

- $\mathbb E[X]=\mathrm{Var}[X]=\lambda$
  - $\mathbb E[X]=\lambda$ 其实就是"速率 $\lambda$"的定义本身
- 独立泊松变量之和仍是泊松，速率相加
  - 可用矩母函数（MGF）唯一性证明

$$X+Y\sim\mathrm{Poisson}(\lambda_{1}+\lambda_{2})$$

### 几何分布（Geometric）

- **几何**随机变量 $X\sim$ Geometric$(p)$：到第一次成功为止需要的 Bernoulli 试验次数（包含成功那次），每次成功概率 $p$

$$P(X=k)=(1-p)^{k-1}\cdot p$$

  - 尾概率（"$>k$"的概率）是 $(1-p)$ 的纯幂

$$P(X>k)=(1-p)^{k}$$

- **无记忆性**：$\mathrm P(X>m+n\mid X>m)=\mathrm P(X>n)$
  - 意思是：已经失败了 $m$ 次这一事实完全不影响未来
  - **几何分布是正整数上唯一的无记忆分布** ⟺ 任何无记忆的离散"等待时间"分布必是几何分布
- **几何级数**形如 $\sum_{r=0}^{\infty}r^{r}$〔转写注：原图如此；按几何级数惯例应为 $\sum_r x^r$，疑为原文笔误，照录〕
  - $|r|<1$ 时收敛到 $\frac{1}{1-r}$，否则发散
  - 一个好记的验证：取 $r=\frac12$，$\sum_{i=0}^{\infty}1/2^{i}=2$
  - 自检：概率总和为 1

$$\sum_{k=1}^{\infty}(1-p)^{k-1}p=p\sum_{i=0}^{\infty}(1-p)^{i}$$

$$=p\cdot\frac{1}{1-(1-p)}\quad\text{几何级数求和}$$

$$=p\cdot\frac{1}{p}=1$$

- 期望等待时间 $\mathbb E[X]=1/p$
  - 直接算：$X=k$ 表示 $k-1$ 次失败

$$\mathbb E[X]=\sum_{k=1}^{\infty}k\cdot P(X=k)$$

  - 解出来有点绕，但确实得到 $\mathbb E[X]=1/p$
  - 用**首步分析**（first-step analysis，见后文题型一节）解：第一次抛掷，以概率 $p$ 成功（$X=1$），以概率 $1-p$ 失败——失败就回到原点（$X=1+X'$，$X'$ 与原问题同分布）

$$\mathbb E[X]=p\cdot1+(1-p)(1+\mathbb E[X])$$

  - 解这个方程即得 $\mathbb E[X]=\frac{1}{p}$
  - 直觉：每次试验都有 $p$ 的机会结束游戏，所以平均要 $\frac{1}{p}$ 次
- $\mathrm{Var}[X]=\frac{1-p}{p^{2}}$
  - $p\to0$ 时，均值和方差一起爆炸
  - 成功越罕见，等待时间的波动越大
- 常用技巧
  - 首步分析
  - 换下标 $k=k-1$，把 $\sum_{k=1}^{\infty}$ 变成 $\sum_{n=0}^{\infty}$

## 连续分布

### 均匀分布（Uniform）

- **均匀**随机变量 $X\sim\mathrm{Uniform}(a,b)$：$X$ 等可能地落在区间 $[a,b]$ 的任何位置
- $\mathbb E[X]=\frac{a+b}{2}$

$$\mathbb E[X]=\int_{a}^{b}x\cdot\frac{1}{b-a}\,dx$$

$$=\frac{1}{b-a}\cdot\frac{b^{2}-a^{2}}{2}\quad\text{微积分基本定理}$$

$$=\frac{(b-a)(b+a)}{2(b-a)}=\frac{a+b}{2}$$

- $\mathrm{Var}(X)=\frac{(b-a)^{2}}{12}$
- **PDF**（概率密度函数：概率 = 密度曲线下的面积）

$$f_{X}(x)=\frac{1}{b-a}\quad(a\le x\le b)$$

- **CDF**（累积分布函数：$F(x)=P(X\le x)$）

$$F_{X}(x)=\begin{cases}0 & x<a\\ \frac{x-a}{b-a} & a\le x\le b\\ 1 & x>b\end{cases}$$

- 两个均匀分布之差是三角分布
- $n$ 个均匀随机点把区间分成的 $n+1$ 段，分布彼此相同

### 指数分布（Exponential）

- **指数**随机变量 $X\sim\mathrm{Exp}(\lambda)$：事件以每小时（每单位时间）平均 $\lambda$ 次的速率发生时，到下一个事件的**等待时间**
  - 等价说法：泊松过程中第一个事件前的等待时间

$$f_{X}(x)=\lambda e^{-\lambda x},\quad x\ge 0$$

- 尾公式（又叫生存函数）

$$P(X>x)=e^{-\lambda x}$$

- **无记忆性**：$P(X>s+t\mid X>s)=P(X>t)$
  - 已经等了 $s$ 个时间单位这一事实无关紧要
  - 指数分布是 $[0,\infty)$ 上唯一的连续无记忆分布
  - 它是几何分布的连续版：连续时间里的等待
- $\mathbb E[X]=\frac{1}{\lambda}$，$\mathrm{Var}(X)=\frac{1}{\lambda^{2}}$
  - 平均每单位时间发生 $\lambda$ 个事件，则事件之间平均要等 $1/\lambda$ 个单位时间
- **泊松与指数是一家**
  - 两个等价命题：
    - 单位时间内的事件数服从 Poisson$(\lambda)$
    - 相邻事件的间隔服从 Exp$(\lambda)$
  - "下一小时会有几个事件？" → 泊松
  - "下一个事件还要等多久？" → 指数
- 指数变量取最小值仍是指数：若 $X_{1}\sim\mathrm{Exp}(\lambda_{1})$、$X_{2}\sim\mathrm{Exp}(\lambda_{2})$ 独立，则

$$\min(X_{1},X_{2})\sim\mathrm{Exp}(\lambda_{1}+\lambda_{2})$$

  - 直觉：邮件以速率 $\lambda_1$ 到达、短信以速率 $\lambda_2$ 到达，那么"先来邮件或先来短信"的等待时间是指数分布，速率 $\lambda_1+\lambda_2$
  - 证明

$$P(\min(X_{1},X_{2})>t)=P(X_{1}>t\ \text{且}\ X_{2}>t)$$

$$=P(X_{1}>t)P(X_{2}>t)\quad\text{独立性}$$

$$=e^{-\lambda_{1}t}e^{-\lambda_{2}t}=e^{-(\lambda_{1}+\lambda_{2})t}$$

  - 这正是 $\mathrm{Exp}(\lambda_{1}+\lambda_{2})$ 的尾函数

### 高斯分布（Gaussian，正态分布）

- **高斯**随机变量 $X\sim\mathcal{N}(\mu,\sigma^{2})$：均值 $\mu$、方差 $\sigma^2$
- 一般正态与标准正态的关系
  - 减去均值平移到 0，除以标准差缩放到单位方差

$$Z=\frac{X-\mu}{\sigma}\sim\mathcal{N}(0,1)$$

  - 例题：身高服从 $\mu=70$、$\sigma=3$ 的正态分布，身高超过 76 英寸的人占多少？

$$X\sim\mathcal{N}(70,9)\implies Z=\frac{X-70}{3}\sim\mathcal{N}(0,1)$$

$$P(X>76)=P(Z>2)$$

- **分位数** $z_{\alpha}$ 是满足下式的值：

$$P(Z\leq z_{\alpha})=\alpha\iff P(Z>z_{\alpha})=1-\alpha$$

  - 也就是说，$z_\alpha$ 是数轴上"左边恰有概率质量 $\alpha$"的那个点
  - 对称性：$z_{1-\alpha}=-z_{\alpha}$
  - 双侧区间：$P(|Z|\leq z_{1-\alpha/2})=1-\alpha\iff P(|Z|>z_{1-\alpha/2})=\alpha$
    - $(1-\alpha)$ 置信区间就用 $z_{\alpha/2}$ 和 $z_{1-\alpha/2}$ 做端点
  - 写成界的形式：$P(|Z|\leq c)\geq1-\alpha\iff c\geq z_{1-\alpha/2}$
  - 由此，若 $X\sim\mathcal{N}(\mu,\sigma^{2})$，其 $\alpha$ 分位数为 $x_{\alpha}=\mu+\sigma\cdot z_{\alpha}$

$$P(X\leq x_{\alpha})=P\left(Z\leq\frac{x_{\alpha}-\mu}{\sigma}\right)=\alpha\implies z_{\alpha}=\frac{x_{\alpha}-\mu}{\sigma}\implies x_{\alpha}=\mu+\sigma\cdot z_{\alpha}$$

- 独立高斯之和仍是高斯：若 $X_{1}\sim\mathcal{N}(\mu_{1},\sigma_{1}^{2})$、$X_{2}\sim\mathcal{N}(\mu_{2},\sigma_{2}^{2})$，则

$$X_{1}+X_{2}\sim\mathcal{N}(\mu_{1}+\mu_{2},\sigma_{1}^{2}+\sigma_{2}^{2})$$

# 期望与方差（Expectations & variance）

- **期望**

$$\mathbb E[X]=\sum_{x}xP(X=x)\quad\text{或}\quad\int_{-\infty}^{\infty}xf(x)\,dx$$

  - 函数的期望

$$\mathbb E[g(x)]=\sum_{x}g(x)P(X=x)\quad\text{或}\quad\int_{-\infty}^{\infty}g(x)f(x)\,dx$$

  - **尾公式**（非负随机变量专用）

$$\mathbb E[X]=\int_{0}^{\infty}P(X>x)\,dx=\int_{0}^{\infty}1-F(x)\,dx$$

- **方差**

$$\mathrm{Var}(X)=\mathbb E[(X-\mathbb E[X])^{2}]=\mathbb E[X^{2}]-(\mathbb E[X])^{2}$$

- **期望的线性性**（不管独不独立都成立）

$$\mathbb E[aX+bY+c]=a\mathbb E[X]+b\mathbb E[Y]+c$$

- **独立**的 $X$、$Y$ 有

$$\mathbb E[XY]=\mathbb E[X]\mathbb E[Y]$$

- 若 $X$、$Y$ 独立，则

$$\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)$$

# 函数一览（Zoo of functions）

- **概率质量函数（PMF）**：离散随机变量每个取值的概率

$$p(k)=P(X=k)$$

  - 性质
    - $p(k)\geq0$
    - $\sum_{k}p(k)=1$
- **概率密度函数（PDF）**：连续随机变量的"密度"，概率等于曲线下面积

$$f(x)\ \text{满足}\ P(a\le x\le b)=\int_{a}^{b}f(x)\,dx$$

  - 对 CDF 求导得 PDF：$f(x)=F'(x)$
- **累积分布函数（CDF）**：离散、连续通用

$$F(x)=P(X\leq x)$$

  - 离散：$F(x)=\sum_{k\leq x}p(k)$
  - 连续：$F(x)=\int_{-\infty}^{x}f(x)\,dx$
  - 注意 $P(a\le x\le b)=F(b)-F(a)$

# 不等式（Inequalities）

- **Markov 不等式**：对任何非负随机变量 $X$ 和 $a>0$，

$$P(X\geq a)\leq{\frac{\mathbb E[X]}{a}}$$

  - 直觉：$\mathbb E[X]$ 不可能比"大于 $a$ 的那部分取值"的贡献还小
  - 证明

$$\mathbb E[X]\geq\sum_{x\geq a}xP(X=x)\geq a\sum_{x\geq a}P(X=x)=aP(X\geq a)$$

- **Chebyshev 不等式**：对任何均值 $\mu$、方差 $\sigma^2$ 的随机变量 $X$，

$$P(|X-\mu|\geq k\sigma)\leq{\frac{1}{k^{2}}}$$

  - 至少有 $1-1/k^{2}$ 的概率质量落在均值 $k$ 个标准差以内
    - 例如：75% 的概率质量落在 2 个标准差以内
  - 证明：对 $(X-\mu)^{2}$ 用 Markov

$$P(|X-\mu|\geq k\sigma)=P((X-\mu)^{2}\geq k^{2}\sigma^{2})\leq{\frac{\mathbb E[(X-\mu)^{2}]}{k^{2}\sigma^{2}}}={\frac{\sigma^{2}}{k^{2}\sigma^{2}}}={\frac{1}{k^{2}}}$$

  - 换元 $\epsilon=k\sigma$ 得另一种写法

$$P(|X-\mu|\geq\epsilon)\leq{\frac{\sigma^{2}}{\epsilon^{2}}}$$

- **Jensen 不等式**：描述期望与凸/凹变换的相互作用

$$\varphi\ \text{凸}\implies\varphi[\mathbb E[X]]\leq\mathbb E[\varphi(X)]$$

$$\varphi\ \text{凹}\implies\varphi[\mathbb E[X]]\geq\mathbb E[\varphi(X)]$$

〔译注：凸函数（convex）像碗，$\varphi''\ge0$；凹函数（concave）像拱，$\varphi''\le0$。对凸函数，"先平均再取函数" ≤ "先取函数再平均"。〕

  - 记不住就用 $\varphi(x)=x^2$（凸）配上 $X\in\{-1,1\}$：
    - $\varphi(\mathbb E[X])=\varphi(0)=0$
    - $\mathbb E[\varphi(X)]=1$
  - 凸函数满足 $\varphi''\geq0$ 处处成立（斜率一直在涨）
    - $x^2$、$x^4$、…（任何偶数次幂）
    - $|x|$
    - $a^x$（$a>1$）
    - $1/x$（$x>0$）
    - $\max(x_1,x_2,\dots)$
  - 凹函数满足 $\varphi''\leq0$ 处处成立（斜率一直在降）
    - $\sqrt{x}$
    - $\log x$
    - $x^p$（$0<p<1$）
    - $\min(x_1,x_2,\dots)$
  - 线性函数既凸又凹
  - 值得背下来：$\log\mathbb E[X]\geq\mathbb E[\log X]$
- **AM-GM 不等式**（均值-几何均值不等式）：算术平均恒 $\geq$ 几何平均

$$\frac{a_1+\dots+a_n}{n}\geq\sqrt[n]{a_1\dotsb a_n}$$

〔转写注：原图为 $\leq$，按 AM-GM 惯例应为 $\geq$，疑为原稿笔误；译文按数学事实订正为 $\geq$（后文 "log mean" 一节亦以正确方向重述同一事实）。〕

  - 证明：两边取对数，用 Jensen 不等式（利用 $\log$ 是凹函数），再取指数
- **Cauchy-Schwarz 不等式**

$$\mathbb E[XY]^{2}\leq\mathbb E[X^{2}]\,\mathbb E[Y^{2}]$$

  - 另一种写法

$$\left(\sum_{i=1}^{n}x_iy_i\right)^{2}\leq\left(\sum_{i=1}^{n}x_i^{2}\right)\left(\sum_{i=1}^{n}y_i^{2}\right)$$

- **并集界（union bound）**

$$P(A_1\cup A_2\cup\dots\cup A_n)\leq P(A_1)+P(A_2)+\dots+P(A_n)$$

---

# 条件概率（Conditional probabilities）

- 设我们有两个随机变量 $X$ 与 $Y$
  - **联合分布**：$P(X=x,Y=y)$
  - **边际分布**：$P(X=x)=\sum_y P(X=x,Y=y)$
  - **条件分布**：$P(X=x\mid Y=y)$
- **贝叶斯公式**

$$P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}$$

  - $P[A]$ 是**先验**（看到证据 $B$ 之前我们相信的东西）
  - $P[A\mid B]$ 是**后验**（看到证据 $B$ 之后我们相信的东西）
  - $P[B]$ 用全概率公式算
- **全期望公式**（也叫**塔式法则 tower property**）

$$\mathbb E[X]=\sum_y\mathbb E[X\mid Y=y]\cdot P(Y=y)\quad\text{或}\quad\int_{-\infty}^{\infty}\mathbb E[X\mid Y=y]f(y)\,dy$$

  - 也可写成 $\mathbb E[X]=\mathbb E_Y[\mathbb E[X\mid Y]]$
    - 注意这种写法里 $\mathbb E[X\mid Y]$ 本身就是一个以 $Y$ 表示的随机变量
- **全概率公式**

$$P(A)=\sum_y P(A\mid Y=y)\cdot P(Y=y)$$

# 极限定理（Limit theorems）

- **中心极限定理（CLT）**：$n$ 个独立同分布（iid）随机变量 $X_i$（均值 $\mu$、方差 $\sigma^2$）的**平均值**近似高斯——不管单个 $X_i$ 长什么样

$$\bar X_n=\frac{1}{n}\sum_{i=1}^{n}X_i\approx\mathcal{N}\left(\mu,\frac{\sigma^2}{n}\right)$$

  - 方差怎么来的：

$$\mathrm{Var}(\bar X_n)=\mathrm{Var}\left(\frac{1}{n}\sum_{i=1}^{n}X_i\right)=\frac{1}{n^2}\mathrm{Var}\left(\sum_{i=1}^{n}X_i\right)=\frac{1}{n^2}\sum_{i=1}^{n}\mathrm{Var}(X_i)=\frac{n\cdot\sigma^2}{n^2}=\frac{\sigma^2}{n}$$

  - **标准误（standard error）**就是 $\bar X_n$ 的标准差，即 $\sigma/\sqrt{n}$
    - 含义：误差按 $\sqrt n$ 缩小，想把误差降 10 倍得采 100 倍的样本
  - $(1-\alpha)$ 置信区间：

$$\bar X_n\pm z_{1-\alpha/2}\cdot\frac{\sigma}{\sqrt{n}}$$

- **大数定律**：$n$ 个 iid 随机变量 $X_i$（均值 $\mu$）的平均值收敛到真值

$$P(|\overline{X}_n-\mu|>\epsilon)\to0\quad\text{对任意 }\epsilon>0$$

  - 由 Chebyshev 证明（代入 $\epsilon=k\sigma$）：

$$P(|\overline{X}_{n}-\mu|>\epsilon)\leq\frac{\mathrm{Var}(\overline{X}_{n})}{\epsilon^{2}}=\frac{\sigma^{2}}{n\epsilon^{2}}\to0$$

# 一些题型（Some problem types）

## 首步分析（First-step analysis）

- 问：要得到连续两次正面，平均要抛多少次硬币？
  - 设
    - $E_0$ = 刚抛出反面（或刚开始）时，到结束还需要的期望次数
    - $E_1$ = 刚抛出正面时，到结束还需要的期望次数

$$E_0=\underbrace{\frac{1}{2}(1+E_0)}_{\text{反面，重新开始}}+\underbrace{\frac{1}{2}(1+E_1)}_{\text{正面}}$$

$$E_1=\underbrace{\frac{1}{2}(1+E_0)}_{\text{反面，前功尽弃}}+\underbrace{\frac{1}{2}\cdot0}_{\text{连续两次正面，结束}}$$

  - 解这个线性方程组得 $E_0=6$

> ✅ 首步分析技巧：盯住过程的第一步，把想要求的期望用它自己表示出来，然后解线性方程组。

## 用期望数数（Counting in expectation）

- 问："平均有多少个东西满足某个性质的？"
  - 定义 $X_i=\mathbf{1}[\text{第 }i\text{ 个东西有该性质}]$
  - 令 $N=\sum_i X_i$（示性变量之和就是计数）
  - 逐个算 $\mathbb E[X_i]=P(X_i)$（多用对称性）
  - 用期望的线性性：$\mathbb E[N]=\sum_i P(X_i)$
- 例题：随机排列中**不动点**（站在自己位置上的人）的期望个数
  - 由对称性，位置 $i$ 拿到任何物品的概率相同，故 $P(X_i=1)=1/n$，于是 $\mathbb E[N]=n\cdot\frac{1}{n}=1$

## $n$ 个随机变量的最大与最小

- 若 $X_1,\ldots,X_n$ 独立同分布，公共 CDF 为 $F_X$
  - 最大值 $M=\max_i X_i$：

$$F_M(x)=P(M\leq x)=P(\text{所有 }X_i\leq x)=[F_X(x)]^{n}$$

  - 最小值 $m=\min_i X_i$：

$$P(m>x)=P(\text{每个 }X_i>m)=[1-F_X(x)]^{n}$$

$$F_m(x)=1-P(m>x)$$

- 求 min/max 的期望有两条路
  - 三步套路：CDF → 求导得 PDF → 积分
  - 非负随机变量可直接用尾公式 $\mathbb E[X]=\int_0^{\infty}P(X>x)\,dx$
- 拿 $X_i\sim$Uniform$[0,1]$ 实操一遍
  - 最大 $M=\max_i X_i$
    - CDF：$F_M(x)=\prod_i P(X_i\leq x)=x^n$
    - PDF：$f_M(x)=\frac{d}{dx}x^n=nx^{n-1}$
    - 期望：$\mathbb E[M]=\int_0^1 x\cdot nx^{n-1}\,dx=n\int_0^1 x^{n}\,dx=\frac{n}{n+1}$
    - 直觉：$n$ 个均匀点把线段分成 $n+1$ 段，由对称性各段分布相同，所以每段平均长 $1/(n+1)$，最大点到右端的距离期望就是它
    - 用尾公式同样可得：$\mathbb E[M]=\int_0^1(1-x^{n})\,dx=\left.x-\frac{x^{n+1}}{n+1}\right|_0^1=\frac{n}{n+1}$
  - 最小 $m=\min_i X_i$
    - $P(m>x)=(1-x)^n$
    - CDF：$F_m(x)=1-P(m\geq x)=1-(1-x)^n$
    - PDF：$f_m(x)=\frac{d}{dx}F_m(x)=n(1-x)^{n-1}$
    - 期望：$\mathbb E[m]=\int_0^1 x\cdot n(1-x)^{n-1}\,dx=\frac{1}{n+1}$
- $X_i\sim$Exponential$(\lambda)$
  - min
    - $P(m>x)=[P(X_i>x)]^{n}=e^{-n\lambda x}$
    - $\mathbb E[m]=\int_0^{\infty}e^{-n\lambda x}\,dx=\frac{1}{n\lambda}$
  - max 复杂得多，但要知道它随 $n$ **按对数增长**
- $X_i\sim\mathcal N(0,1)$：$\mathbb E[\max_i X_i]\approx\sqrt{2\ln n}$（这个推导相当难）

# 组合计数（Combinatorics）

- **组合（combination）**：从 $n$ 个里选 $k$ 个，不计顺序

$$\binom{n}{k}=\frac{n!}{k!(n-k)!}=\frac{P(n,k)}{k!}$$

  - 与排列的关系：每组 $k$ 个物品可以排出 $k!$ 种顺序，组合把这些顺序并成同一种
- **排列（permutation）**：从 $n$ 个里排 $k$ 个，顺序要紧

$$P(n,k)=\frac{n!}{(n-k)!}$$

- $n$ 个物品中有 $n_1$ 个同型、$n_2$ 个同型……时的排法数（多重排列）

$$\binom{n}{n_1,\dots,n_m}=\frac{n!}{n_1!\dotsb n_m!}$$

  - 先全排列，再除以同型物品内部的相互交换
- **错排（derangement）**：没有任何不动点的排列
- **可重复组合（combination with replacement）**：从 $n$ 种里可重复地选 $k$ 个

$$\binom{n+k-1}{k}$$

  - 这是**隔板法（stars and bars）**：一种选法对应一串"$k$ 个星 + $n-1$ 道隔板"的图案，也就是从 $n+k-1$ 个位置里挑 $k$ 个放星
- 问：把非负整数 $n$ 写成 $k$ 个非负整数之和，有多少种写法？
  - 把 $n$ 个星与 $k-1$ 道隔板排成一排即可
- 四格总览：

| 顺序要紧？ | 可重复？ | 计数 |
| --- | --- | --- |
| 是 | 否 | $\frac{n!}{(n-k)!}$（排列） |
| 否 | 否 | $\binom{n}{k}$（组合） |
| 是 | 是 | $n^k$ |
| 否 | 是 | $\binom{n+k-1}{k}$（隔板法） |

- **容斥原理（inclusion-exclusion）**

$$P(A\cup B)=P(A)+P(B)-P(A\cap B)$$

$$P(A\cup B\cup C)=P(A)+P(B)+P(C)-P(A\cap B)-P(B\cap C)-P(A\cap C)+P(A\cap B\cap C)$$

〔转写注：原图三事件式末项符号为 $-P(A\cap B\cap C)$，按容斥原理应为 $+$；三事件公式在下一行"以计数形式"写出时符号正确，可证原图为笔误。计数形式照原文方向保留如下。〕

  - 写成计数形式

$$|A\cup B\cup C|=|A|+|B|+|C|-|A\cap B|-|A\cap C|-|B\cap C|+|A\cap B\cap C|$$

# 马尔可夫链（Markov chains）

- **马尔可夫链**是一列随机变量 $X_0,X_1,X_2,\dots$，取值于状态空间 $S$，满足**马尔可夫性**：未来只依赖现在
- 有限状态链可用**转移矩阵** $P$ 描述，其中 $P_{ij}=P(X_{t+1}=j\mid X_t=i)$
  - 每行加起来是 1，本身是一个"下一状态"的概率分布
- 若 $\pi_t$ 是时刻 $t$ 的状态分布，则 $\pi_{t+1}=\pi_t P$
- **平稳分布**满足 $\pi=\pi P$
  - 温和条件下，链有唯一平稳分布，且不管从哪出发都有 $\pi_t\to\pi$
- 返回某状态的期望时间恰是其平稳概率的**倒数**

$$\mathbb E[\text{返回状态 }i]=\frac{1}{\pi_i}$$

- 求期望吸收时间的一般套路
  - 用首步分析写递推
    - 记 $h_j$ = 从状态 $j$ 出发到被吸收的期望时间
    - 目标吸收态 $A$：$h_A=0$

$$h_i=1+\sum_j P_{ij}h_j$$

  - 记 $q_j$ = 从状态 $j$ 出发最终被"吸进目标态"的概率
    - 目标吸收态 $q_A=1$，其他吸收态 $q_B=0$

$$q_i=\sum_j P_{ij}q_j$$

  - 解递推
    - 小链：直接解线性方程组
    - 有结构的链：先猜闭式再验证
      - 猜的办法：先算小例子找答案，再试着推广
      - 从"函数形状"上想什么形式合理
    - 代入边界条件定出常数
- 常用技巧
  - 对称性可以合并等价状态
  - 圆上的随机游走：把 $h_i$ 重新参数化为"从某个**距离**出发的期望命中时间"，比从具体状态出发简单
  - 用**平稳分布**抄近路算期望命中时间
    - 在"每个节点看起来都一样"的图上（环图、完全图等），相邻两点间的期望命中时间恒为 $n-1$（$n$ 为节点数）
      - 因为 $\mathbb E[\text{返回自身}]=n$（平稳分布处处 $1/n$）
      - 而 $\mathbb E[\text{返回自身}]=1+h$：每个相邻点贡献 $\frac{1}{d}h$，$d$ 个邻居加起来恰好是 $h$

# 谜题（Puzzles）

## 生日问题

- 问：要多少人 $n$，才能使"有两个人同一天生日"的概率超过 50%？
- $P(\text{撞生日})=1-P(\text{生日全不同})$
- 设一年 $d$ 天、共 $n$ 人

$$P(\text{生日全不同})=\frac{d}{d}\cdot\frac{d-1}{d}\cdot\frac{d-2}{d}\cdots\frac{d-n+1}{d}=\frac{d!}{(d-n)!\cdot d^{n}}$$

## 赌徒破产（Gambler's ruin）

- 你带着 $\$a$ 开局
  - 每局以概率 $p$ 赢 $\$1$，以概率 $1-p$ 输 $\$1$
  - 到 $\$N$ 收手（赢！），到 $\$0$ 破产
  - 问：到达 $\$N$ 的概率？
- 设 $P_i$ = 从 $i$ 出发到达 $N$ 的概率
  - 边界条件：$P_0=0$，$P_N=1$
- 对任意 $0<i<N$：

$$q_i=p\cdot q_{i+1}+(1-p)\cdot q_{i-1}$$

  - 这是二阶线性递推
- 公平游戏（$p=\frac12$）时，从 $a$ 出发到达 $N$ 的概率就是 $a/N$
  - 做法一：用两种方式表示停时上的期望
    - 设 $T$ 为停止时刻，$W_T$ 为 $T$ 时的财富
      - 无漂移，故 $\mathbb E[W_T]=a$
      - 而 $W_T$ 只能取 $0$ 或 $N$，于是

$$\mathbb E[W_T]=q_a\cdot N+(1-q_a)\cdot0=q_a N$$

  - 两式相等得 $q_a=a/N$
  - 做法二：解递推

$$q_i=\frac{1}{2}q_i+\frac{1}{2}q_{i-1}\implies q_{i+1}-q_i=q_i-q_{i-1}$$

  - 逐差全相等 → $q$ 是 $i$ 的线性函数
    - 代入 $q_0=0$、$q_N=1$ 得 $q_i=i/N$
  - 这就是"两端有吸收壁"的随机游走
- 一般地，**鞅（martingale）**是满足 $\mathbb E[M_i]=\mathbb E[M_n]$ 的一列随机变量〔译注：鞅的标准定义是 $\mathbb E[M_{n}\mid M_0,\dots,M_{n-1}]=M_{n-1}$，即"对未来的最佳预测就是现在"；笔记取其期望不变这一弱化表述〕
- 期望吸收时间 $h_a=a(N-a)$
  - 直接推导不容易，但用它满足首步分析递推关系可以很快**验证**

## 随机游走（Random walk）

- 简单对称**随机游走**：$S_n=X_1+\dots+X_n$，每个 $X_i=\pm1$ 等概率
  - 显然 $\mathbb E[S_n]=\mathbb E[X_i]=0$
  - 每个 $X_i$ 的方差 $\mathbb E[X_i^2]-(\mathbb E[X_i])^2=1-0=1$

$$\mathrm{Var}(S_n)=\sum_{i=1}^{n}\mathrm{Var}(X_i)=n$$

  - 到原点的典型距离（用标准差度量）按 $\sqrt n$ 增长
- 有偏情形（$X_i=1$ 的概率为 $p$，否则 $-1$）：

$$\mathbb E[S_n]=n\big(p\cdot1+(1-p)\cdot(-1)\big)=n(2p-1)$$

## 赠券收集（Coupon collector）

- $n$ 种赠券，每买一盒麦片随机等概率得一张。集齐全部 $n$ 种平均要买多少盒？
  - 已经集了 $k$ 种时，收集到"新的一种"的时间 $X_k\sim$Geom$\big(\frac{n-k}{n}\big)$
  - 第 $k$ 阶段的期望长度

$$\mathbb E[X_k]=\frac{n}{n-k}$$

  - 总期望

$$\mathbb E[X]=\sum_{k=0}^{n-1}\mathbb E[X_k]=\sum_{k=0}^{n-1}\frac{n}{n-k}=n\sum_{j=1}^{n}\frac{1}{j}=n\cdot H_n$$

  - 其中 $H_n=1+1/2+1/3+\dots+1/n$ 是第 $n$ 个**调和数**
  - 最后一个阶段平均要 $n$ 盒！

## 信息论谜题

- 思路：数"信息量"
- 若要从 $k$ 种可能中分辨出答案，而每次检验有 $m$ 种结果，则至少要 $\lceil\log_m k\rceil$ 次检验
  - 这是下界
  - 能不能达到这个下界，取决于题目的具体约束
- 9 枚硬币中 1 枚较重，用天平称 2 次找出它：
  - 分成 3 组各 3 枚
    - 左边沉 → 假币在左边
    - 右边沉 → 假币在右边
    - 平衡 → 假币在没称的那组
  - 再称剩下 3 枚中的 2 枚

## 蓄水池采样（Reservoir sampling）

- 数据流 $x_1,x_2,x_3,\dots$ 逐个到来，结束时从所有见过的物品中等概率**随机返回 1 个**（事先不知道流长 $n$）
- 算法
  - 先保留第一个：$s=x_1$
  - 对 $n=2,3,4,\dots$
    - 以概率 $1/n$ 用 $x_n$ 替换 $s$
- 用归纳法证明正确
  - 命题：处理完 $n$ 个物品后，每个 $x_1,\dots,x_n$ 留在 $s$ 里的概率都是 $1/n$
    - 基础（$n=1$）：平凡成立，$s=x_1$，概率 $1=1/1$
    - 归纳步（严格归纳假设：上一步结束时各物品等概率被保留）：
      - $s=x_n$（发生了替换）：概率 $1/n$
      - $s=x_k$（$k<n$，没被替换）：需要 (1) 上一步结束时 $s=x_k$——由归纳假设概率 $1/(n-1)$；且 (2) 第 $n$ 步拒绝了 $x_n$——概率 $(n-1)/n$

$$\frac{1}{n-1}\cdot\frac{n-1}{n}=\frac{1}{n}$$

  - 于是 $n$ 步之后每个 $x_k$ 都在蓄水池里，概率 $1/n$

## 秘书问题（Secretary problem）

- 问题：$n$ 个候选人随机顺序依次面试，每人面试完**立刻**决定录用或拒绝。目标是最大化"录用到最优者"的概率。采用策略：先拒绝前 $k$ 个，之后遇到比前 $k$ 个都好的就录用（一直没遇到就录用最后一个）。最优的 $k$ 是多少？
- 设 $P(k)$ = 阈值为 $k$ 的策略录到最优者的概率，$b$ = 最优者的位置
  - 策略成功当且仅当：
    1. $b>k$：最优者出现在前 $k$ 之后
    2. 位置 $1,\dots,b-1$ 中最出色的那个人恰好落在 $1,\dots,k$（否则在到达 $b$ 之前就会先录用别人）

$$P(k)=\sum_{b=k+1}^{n}P(\text{最优者在 }b)\,P(\text{前 }b-1\text{ 中最优者出现在}\le k)$$

$$=\sum_{b=k+1}^{n}\frac{1}{n}\cdot\frac{k}{b-1}=\frac{k}{n}\sum_{m=k}^{n-1}\frac{1}{m}$$

$$=\frac{k}{n}\left(H_{n-1}-H_{k-1}\right)\approx\frac{k}{n}\big(\log(n-1)-\log(k-1)\big)=\frac{k}{n}\log\frac{n}{k}$$

  - 令 $r=\frac{k}{n}$，则 $P\approx r\log\frac{1}{r}=-r\log r$（右注：大 $n$ 时 $H_m\approx\log m$）
  - 求导：$-\frac{d}{dr}r\log r=-\log r-1$
    - 令其为 0：$\log r=-1\implies r=\frac{1}{e}\implies k=\frac{n}{e}$

## 蒙特霍尔问题（Monty Hall）

- 设 $A=\{1,2,\dots,n\}$ 表示车的位置，$B$ 为 Monty 打开的门集合
- 我们选门 $c$（不失一般性）
- 要求的是每个未开门 $j$ 的后验 $P(A=j\mid B)$
- 贝叶斯公式：$P(A=j\mid B)=\frac{P(B\mid A=j)P(A=j)}{P(B)}$
  - 比较不同 $j$ 的 $P(A=j\mid B)$ 只需比较 $P(B\mid A=j)$：先验 $P(A=j)=1/n$ 处处相同，分母也公共
- 情形 1：Monty 打开 $n-2$ 扇门，只留我们选的门和另一扇
  - 记 $r$ = 剩下的那扇门
  - 车在 $r$ 时，Monty 有 $n-1$ 种"留哪扇关着"的选法，故 $P(B\mid A=r)=1/(n-1)$
  - 车在我们选的 $c$ 时，Monty 没有自由度，$P(B\mid A=c)=1$〔转写注：原图此处两行均写作 $P(B\mid A=r)$，据上下文，后一行主目应为 $c$〕
  - 车在被打开的门 $k\in B$ 后时，$P(B\mid A=k)=0$
  - 于是后验之比为 $P(A=r\mid B)\propto 1/(n-1)$、$P(A=c\mid B)\propto 1$
  - 归一化得 $P(A=r\mid B)=1/n$、$P(A=c\mid B)=(n-1)/n$——换门是对的
- 情形 2：Monty 只开 1 扇门
  - 记 $R$ 为其余门中的某一扇
  - 车在 $t$ 时 Monty 有 $n-1$ 种开门选法：$P(B\mid A=t)=1/(n-1)$
  - 车在 $r$ 时只剩 $n-2$ 种选法（不能开我们选的 $c$，也不能开有车的那扇）：$P(B\mid A=r)=1/(n-2)$
  - 若 $R=\{k\}$ 是已开的门：$P(B\mid A=k)=0$
  - 于是 $P(A=t\mid B)\propto 1/(n-1)$、$P(A=r\mid B)\propto 1/(n-2)$
  - 剩下的门每一扇都比原来略微更可能藏车
- 一般结论：被打开的那些门的概率之和（$k/n$，$k$ 为开门数）会摊到其余未开未选的门上（共 $n-k-1$ 扇）

## 最长连续正面（Longest run of heads）

- 抛一枚公平硬币 $n$ 次，$L_n$ = 最长连续正面的长度
- $\mathbb E[L_n]$ 没有干净的闭式，但有个广为人知的近似：$\mathbb E[L_n]\approx\log_2(n)$

## 球投盒（Throwing balls into bins）

- 把 $n$ 个球等概率投进 $m$ 个盒子：大约 37% 的盒子是空的，37% 恰好一个球，其余有两个以上

## 路径计数（Path counting）

- 从 $(0,0)$ 走到 $(m,n)$、每步只能右移或上移，有多少条路径？
  - 每条路径恰由 $m$ 步右、$n$ 步上组成
  - 一条路径由"哪 $m$ 步是向右"唯一确定
  - 答案是 $\binom{m+n}{m}$
- 其中经过 $(a,b)$ 的有多少条？
  - 到 $(a,b)$ 的路径数 × 从 $(a,b)$ 到 $(m,n)$ 的路径数

$$\binom{a+b}{a}\cdot\binom{(m-a)+(n-b)}{m-a}$$

# 统计（Statistics）

## 最大似然估计（Maximum likelihood estimation, MLE）

- 数据 $x_1,\dots,x_n$ 抽自某个带未知参数 $\theta$ 的分布
- **最大似然估计**：选让已观测数据"概率最大"的 $\hat\theta$

$$L(\theta)=\prod_{i=1}^{n}p(x_i;\theta),\qquad\hat\theta_\text{MLE}=\arg\max_\theta L(\theta)$$

- 对数似然技巧：取对数，把连乘变求和

$$\ell(\theta)=\log L(\theta)=\sum_{i=1}^{n}\log p(x_i;\theta)$$

- 对 $\theta$ 求导、置零、解出 $\hat\theta$
  - 再用二阶导数或端点值检查它是极大而非极小
- **例题**：$x_1,\dots,x_n$ iid 服从 Bernoulli$(p)$
  - PMF：$P(X=x;p)=p^{x}(1-p)^{1-x}$
  - 对数似然

$$\ell(p)=\sum_i x\log p+(1-x)\log(1-p)$$

  - 记 $k$ 为 1 的个数：$\ell(p)=k\log p+(n-k)\log(1-p)$
  - 求导

$$\ell'(p)=\frac{k}{p}-\frac{n-k}{1-p}=0\implies p=\frac{k}{n}$$

  - MLE 就是样本频率
- **例题**：$x_1,\dots,x_n$ iid 服从 Exp$(\lambda)$
  - PDF：$f(x;\lambda)=\lambda e^{-\lambda x}$
  - 对数似然

$$\ell(\lambda)=\sum_i\log\big[\lambda e^{-\lambda x_i}\big]=n\log\lambda-\lambda\sum_i x_i$$

  - 求导

$$\ell'(\lambda)=\frac{n}{\lambda}-\sum_i x_i=0\implies\hat\lambda_\text{MLE}=\frac{n}{\sum_i x_i}$$

  - MLE 是样本均值的倒数
      - 合理，因为 Exp$(\lambda)$ 的均值就是 $1/\lambda$
- **例题**：$x_1,\dots,x_n$ iid 服从 Uniform$(0,\theta)$
  - PDF：$f(x;\theta)=1/\theta$（$0\le x\le\theta$）
  - 似然

$$L(\theta)=\begin{cases}\frac{1}{\theta^{n}}&\max_i x_i\le\theta\\ 0&\text{否则}\end{cases}$$

  - 要在 $\theta\geq\max_i x_i$ 里挑最小者，所以 $\hat\theta=\max_i x_i$
- 模型训练最小化交叉熵损失 $=-\frac{1}{N}\sum_i\log p(x_i)$，等价于最大化对数似然 $=\sum_i\log p(x_i)$
  - 换句话说：**训练就是让已经发生的数据看起来尽量可能**

## 偏差与方差（Bias & variance）

- 真参数 $\theta$，估计量 $\hat\theta$
  - 估计量是数据的函数；数据随机，所以 $\hat\theta$ 本身是随机变量
- **偏差（bias）**：平均而言 $\hat\theta$ 离 $\theta$ 多远？

$$\mathrm{bias}(\hat{\theta})=\mathbb{E}[\hat{\theta}]-\theta$$

- **方差（variance）**：$\hat{\theta}$ 随数据晃动的幅度

$$\mathrm{var}(\hat{\theta})=\mathbb{E}\big[(\hat{\theta}-\mathbb{E}[\hat{\theta}])^{2}\big]$$

- **均方误差（MSE）**

$$\mathrm{MSE}(\hat{\theta})=\mathbb{E}\big[(\hat{\theta}-\theta)^{2}\big]$$

  - 可分解为 偏差² + 方差

$$\mathrm{MSE}(\hat{\theta})=\mathrm{Bias}(\hat{\theta})^{2}+\mathrm{Var}(\hat{\theta})$$

# 数学工具（Math things）

## 泰勒级数（Taylor series）

- **泰勒级数**只用函数在**一个点**的信息，用多项式去逼近函数

$$f(x)\sim\sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(x-a)^{n}$$

- 在 $a=0$ 处展开的叫**麦克劳林级数（Maclaurin series）**
- $e^x$ 在 $x=0$ 处的泰勒级数是

$$e^{x}=1+x+\frac{x^{2}}{2!}+\frac{x^{3}}{3!}+\dots=\sum_{n=0}^{\infty}\frac{x^{n}}{n!}$$

  - 收敛半径无穷大，处处收敛到 $e^x$
- 想在 $x=0$ 附近近似 $f(x)$：
  - 0 阶近似：$f(x)\approx f(0)$
    - 一条过 $(0,f(0))$ 的水平线
  - 1 阶近似：$f(x)\approx f(0)+f'(0)x$
    - 切线：在 $x=0$ 处与 $f$ 的值和斜率都对齐
  - 2 阶近似：$f(x)\approx f(0)+f'(0)x+\frac{f''(0)}{2!}x^{2}$
    - 抛物线：连值、斜率、弯曲程度都对齐
  - 每多一项就多对上一阶导数信息，多项式就在 $x=0$ 附近"贴"得更紧

## 求导法则（Derivative rules）

- 背下来：

$$\frac{d}{dx}x^{n}=nx^{n-1},\qquad\frac{d}{dx}e^{x}=e^{x},\qquad\frac{d}{dx}\log x=\frac{1}{x}$$

$$\frac{d}{dx}\sin x=\cos x,\qquad\frac{d}{dx}\cos x=-\sin x$$

- 复合运算法则

$$\frac{d}{dx}f(g(x))=f'(g(x))\cdot g'(x)\qquad\text{链式法则（chain rule）}$$

$$\frac{d}{dx}f(x)g(x)=f'(x)g(x)+f(x)g'(x)\qquad\text{乘积法则（product rule）}$$

$$\frac{d}{dx}\frac{f(x)}{g(x)}=\frac{f'(x)g(x)-f(x)g'(x)}{g(x)^{2}}\qquad\text{商法则（quotient rule）}$$

## 常用公式（Handy equations）

- 第 $n$ 个调和数 $H_n=\sum_{k=1}^{n}\frac{1}{k}\approx\log n$（$n$ 很大时）
- 几何均值 $\sqrt{xy}\leq\frac{x+y}{2}$（$x,y>0$）
- 关于 $e$ 的几个恒等式

$$e=\lim_{n\to\infty}\left(1+\frac{1}{n}\right)^{n}$$

  - 把 $n$ 换成 $-n$：

$$\frac{1}{e}=\lim_{n\to\infty}\left(1-\frac{1}{n}\right)^{n}$$

  - 把 $n$ 换成 $n/x$：

$$e^{x}=\lim_{n\to\infty}\left(1+\frac{x}{n}\right)^{n}$$

- 指数级数

$$e=\sum_{k=0}^{\infty}\frac{1}{k!},\qquad e^{x}=\sum_{n=0}^{\infty}\frac{x^{n}}{n!},\qquad e^{-1}=1-\frac{1}{1!}+\frac{1}{2!}-\frac{1}{3!}+\dots$$

〔译注：快照在"指数级数"一行截止，该节若有后续内容未包含在快照内；日后如补全将另行追加。〕
