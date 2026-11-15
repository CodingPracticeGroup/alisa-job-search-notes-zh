# 第二章　数学工具箱（Mathy things）

> 译自 Alisa's Book of LLMs "Mathy things" 一节；底稿与约定见 [`05-scaling-laws.md`](05-scaling-laws.md) 页眉说明。

## 信息论（Information theory）

- **交叉熵**（cross entropy）

$$\mathrm{CE}(p,q)=-\mathbb E_p[\log q]=-\sum_{x\in\mathcal X}p(x)\log q(x)$$

- **KL 散度**（KL divergence）

$$\mathrm{KL}(p\mid\mid q)=\sum_{x\in\mathcal X}p(x)(\log p(x)-\log q(x))$$

- **熵**（entropy）

$$H(p)=-\sum_{x\in\mathcal X}p(x)\log p(x)$$

  - 已知 logits $x_i$ 时的另一种常用形式

$$H(p)=\log\sum_ie^{x_i}-\underbrace{\frac{\sum_ie^{x_i}x_i}{\sum_i e^{x_i}}}_{\mathbb E[x]}$$

- $p$ 与 $q$ 之间的交叉熵 = 两者之间的 KL 散度 + $p$ 自身不可再约的熵

$$\mathrm{CE}(p,q)=\mathrm{KL}(p\mid\mid q)+H(p)$$

  - 证明

$$\begin{align*}\mathrm{KL}(p\mid\mid q)=\sum_{x\in\mathcal X}p(x)(\log p(x)-\log q(x)) \\ &=\sum_{x\in\mathcal X}p(x)\log p(x)-\sum_{x\in\mathcal X}p(x)\log q(x) \\ &=-H(p)+\mathrm{CE}(p,q) \end{align*}$$

- **交叉熵损失**（cross-entropy loss）
  - 当目标分布是 one-hot 时，交叉熵损失就是下一个词元的负对数似然

$$\mathcal L(x)=-\sum_{t=1}^T\log p(x_t\mid x_{<t})$$

  - 它也等价于 KL 散度（差一个与模型无关的常数）
  - 实现损失
    - 用 `F.cross_entropy()` 时，logits 和标签在内部已经错位对齐
      - `loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=pad_idx)`

  ```python
  shift_logits = logits[:, :-1, :]
  shift_labels = input_ids[:, 1:]
  logprobs = F.log_softmax(shift_logits, dim=-1)
  token_logprobs = logprobs.gather(index=shift_labels.unsqueeze(-1), dim=-1).squeeze(-1)
  # 构造损失掩码
  masked_logprobs = -token_logprobs * mask.float()
  return masked_logprobs.sum() / mask.sum()
  ```

## 数值稳定性与小技巧

- 一般而言要盯住：
  - $x$ 很大时的 $\exp(x)$ → 溢出到 $+\infty$
  - $x$ 接近 0 时的 $\log(x)$ → 下溢到 $-\infty$（$\log(0)=-\infty$）
  - $x$ 接近 1 时的 $\log(x)$ → 精度问题（$\log(1)=0$ 附近曲线几乎平掉）
- 计算 `softmax(x)`
  - 直接算不稳定：$x_i$ 很大时 $\exp(x_i)$ 会溢出
  - 利用"softmax 对整体减去一个常数保持不变"这一性质：

$$\begin{align*}\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} \\ &= \frac{e^{x_i-c}\cdot e^c}{\sum_j e^{x_j-c}\cdot e^c} \\ &= \frac{e^{x_i-c}\cdot e^c}{e^c\cdot \sum_j e^{x_j-c}} \\ &= \frac{e^{x_i-c}}{\sum_j e^{x_j-c}} \\ &=\text{softmax}(x-c)_i \end{align*}$$

  - 给所有 $x_i$ 减去 $x_\text{max}$ 就能保证指数都不大（最大的指数项就是 $\exp(0)=1$），所以数值稳定的实现是

$$\text{softmax}(x)_i=\frac{e^{x_i-x_\text{max}}}{\sum_j e^{x_j-x_\text{max}}}$$

- 计算 `log(softmax(x))`
  - 先 softmax 再取 log 不好：接近 0 的输入（低概率类别）取对数不稳定
  - 改用 `x - logsumexp(x)`，既不用真的把 tiny 概率算出来，而 `logsumexp(x)` 本身是稳定的
  - 两者等价：

$$\begin{align*}\log(\text{softmax}(x))_i = \log \frac{e^{x_i}}{\sum_j e^{x_j}} \\ &= \log e^{x_i}-\log\sum_j e^{x_j} \\ &= x_i-\log\sum_j e^{x_j} \end{align*}$$

- 计算 `log(sum(exp(x)))`（logsumexp）
  - 为什么不稳定？
    - $x_i$ 稍大，$\exp(x_i)$ 就溢出为无穷大
      - 例如 float32 下 $x_i\approx 83$ 就炸了
    - 所有 $x_i$ 都很负时，和会下溢成 0，$\log(0)=-\infty$
    - 和接近 1 时有精度问题，因为 $\log(1)$ 附近不稳定
      - 我们在蒸馏项目里踩的就是这个坑
  - 直觉：想办法把 $x_i$ 变小——减一个常数就行！最后再把常数加回来
  - `logsumexp(x)` 的实现：

$$\begin{align*}\log\sum_i e^{x_i}=\log\sum_i\left(e^{x_i-x_\text{max}} \cdot e^{x_\text{max}}\right) \\ &= \log \left(e^{x_\text{max}}\sum_i e^{x_i-x_\text{max}}\right) \\ &=\log e^{x_\text{max}}+\log\sum_ie^{x_i-x_\text{max}} \\ &= x_\text{max}+\log\sum_i e^{x_i-x_\text{max}} \end{align*}$$

  - 最大的项是 $e^0=1$（当 $x_i=x_\text{max}$），不会溢出
    - 括号里的和至少为 1，永远不会算 $\log(0)$
- 朴素地算 softmax 要扫两遍：先求 $x_\text{max}$（为了稳定），再算分母 $\sum_j e^{x_j-x_\text{max}}$
  - 稳定 softmax：

$$\text{softmax}(x)_i=\frac{e^{x_i-x_\text{max}}}{\sum_j e^{x_j-x_\text{max}}}$$

  - **在线 softmax 技巧**（online softmax）：把求 $x_\text{max}$ 和求分母合并成**一趟**完成
    - 思路：分母可以一边维护"到目前为止的最大值"一边算，每当最大值刷新就把已算的部分整体重标定
    - 维护
      - 运行最大值 $m_k=\max(x_1,\dots,x_k)$
      - 运行（平移后的）分母 $d_k=\sum_j e^{x_j-m_k}$
    - 更新规则：遇到 $x_{k+1}$ 时

$$\begin{gather*}m_{k+1}\leftarrow\max(m_k,x_{k+1}) \\ d_{k+1}\leftarrow d_k\cdot e^{m_k-m_{k+1}}+e^{x_{k+1}-m_{k+1}} \end{gather*}$$

  - 验证 $d_{k+1}$ 更新的正确性：

$$\begin{align*}d_{k+1}=\sum_{j=1}^{k+1}e^{x_j-m_{k+1}} \\ &= e^{x_{k+1}-m_{k+1}}+\sum_{j=1}^k e^{x_j-m_{k+1}}\quad\text{拆出最后一项} \\ &= e^{x_{k+1}-m_{k+1}}+\sum_{j=1}^k e^{x_j-m_k+m_k-m_{k+1}}\quad\text{恒等变形} \\ &= e^{x_{k+1}-m_{k+1}}+\sum_{j=1}^k e^{x_j-m_k}e^{m_k-m_{k+1}} \\ &= e^{x_{k+1}-m_{k+1}}+e^{m_k-m_{k+1}}\underbrace{\sum_{j=1}^k e^{x_j-m_k}}_{d_k}\quad\text{提出常数} \\ &= \underbrace{d_k\cdot e^{m_k-m_{k+1}}}_\text{旧项重标定}+\underbrace{e^{x_{k+1}-m_{k+1}}}_\text{新项}\quad \end{align*}$$

  - 注意最大值没变时 $m_k=m_{k+1}$，重标定因子 $e^{m_k-m_{k+1}}=1$，旧项不动
    - 若目标是返回 softmax 本身，还要再扫一遍 logits 计算 $e^{x_i-m_S}/d_S$
- 同样的技巧还能在"logits 与数值逐条到来"时维护一个加权和：

$$o=\sum_ip_iv_i=\sum_i\frac{e^{x_i}}{\sum_j e^{x_j}}v_i=\frac{\sum_i e^{x_i}v_i}{\sum_i e^{x_i}}$$

  - 对 FlashAttention 来说，$o$ 就是每个查询 $q$ 的注意力输出，其中 $x_i=q\cdot k_i$、$v_i$ 是值向量
  - 数值稳定的做法：分子分母同乘 $e^{-x_\text{max}}$（避免 $e^{x_i}$ 溢出）

$$o=\frac{\sum_i e^{x_i-x_\text{max}}v_i}{\sum_i e^{x_i-x_\text{max}}}$$

  - 除了 $m_k$、$d_k$，再维护运行分子 $o_k\in\mathbb R^H$（$H$ 是每个 $v_i$ 的维度，即 FlashAttention 的头维度）

$$o_k=\sum_{i=1}^ke^{x_i-m_k}v_i$$

  - 更新方式和 $d_k$ 完全同理（推导一模一样）

$$o_k=o_k\cdot e^{m_k-m_{k+1}}+ \underbrace{e^{x_{k+1}-m_{k+1}}v_{k+1}}_\text{新项}$$

  - 处理完全部 $N$ 个 logits 后，得到 $(m_N,d_N,o_N)$，真正的注意力输出就是 $o_N/d_N$

> 当含 $e^{x}$ 的表达式需要数值稳定时，标准工具就是：乘以 $e^{-m}$（$m$ 取一个大数，通常是 $m=x_\text{max}$），看看什么留下来。

## 基础统计（Basic statistics）

- $p$ 值：在零假设为真的前提下，看到（至少这么极端的）数据的概率
  - 关键：它是"零假设为真时数据出现"的概率，**不是**"数据出现时零假设为真"的概率
- 这些组之间有差别吗？
  - **Kolmogorov-Smirnov 检验**：给定同一连续变量的两组观测，它们来自同一个底层分布吗？
    - 度量两个样本 CDF 之间的最大垂直距离
  - **卡方（Chi-squared）检验**：给定同一类别变量的两组观测，它们来自同一个底层分布吗？
  - **t 检验**：两组连续观测的均值不同吗？
    - 假设：数据服从正态分布
    - 单样本版：一组数据的均值是否等于某个值
    - 双样本版：两组数据的均值是否相同
    - 配对版：对同一批对象的两次测量是否有系统性差异
      - 本质上就是伪装成配对形式的单样本 t 检验
      - 拿到逐样本的差值（+1、0 或 −1），检验差值是否显著偏离 0
  - **方差分析 ANOVA**（F 检验）：t 检验向多于两组的推广——$k$ 组里有任何一组均值不同吗？
  - **McNemar 检验**：在同一个数据集上比较两个分类器
    - 两个模型在同一测试集上评测、每条样本非对即错——这个最常见场景下它大概是最合适的检验
- 这些变量之间有关系吗？
  - **Pearson 相关检验**：检验两个连续变量之间是否存在**线性**关系
    - 返回相关系数 $r$，以及"$r$ 是否显著不为 0"的 p 值
    - 完全看不到非线性关系（完美的抛物线关系 → $r\approx0$）
    - 想象在散点图上拟一条直线
  - **Spearman 相关检验**：同样的思路，度量的是**单调**关联
    - 先把两个变量都换成秩（最小值记 1，依此类推），再对秩算 Pearson 相关
    - 唯一要紧的是*顺序*，不是具体数值
  - Pearson 与 Spearman 的取舍
    - Pearson 对离群点更敏感；Spearman 对离群点稳健
    - Pearson 会低估"非线性但单调"的关系
    - Pearson 的解释更直观
  - 互信息（mutual information）：能捕捉任意依赖关系，包括非线性的
    - 但它严格来说不是一种统计检验

## 梯度穿过采样（Gradient flow through sampling）

- **Gumbel-Max 技巧**
  - 给定 logits $z_1,\dots,z_k$，要按对应的类别分布采样，可以：
    1. 从 $\mathrm{Gumbel}(0,1)$ 分布独立抽取噪声 $g_1,\dots,g_k$
    2. 对 $(z_1+g_1,\dots,z_k+g_k)$ 取 argmax
- **Gumbel-Softmax**：把 argmax 换成 softmax：$\mathrm{softmax}((z_1+g_1,\dots,z_k+g_k)/\tau)$
  - softmax 处处可微！
  - 高温 → 梯度平滑；低温 → 接近离散样本
  - 它的目的**不是**"让东西可微"（softmax 本来就可微），而是让它**随机**
  - 对比三种做法：
    - 普通 softmax：$y = \mathrm{softmax}(\alpha)$——确定的（永远算出同一份软混合）
    - 真正的类别采样：$y=\mathrm{one\_hot}(\mathrm{sample}(\mathrm{softmax}(\alpha)))$——离散、不可导
    - Gumbel-Softmax：$y=\mathrm{softmax}((\alpha+G)/\tau)$——随机的（Gumbel 噪声）、近离散的（低温）、可导的
    - 鱼与熊掌兼得：既有探索又有梯度！
- **直通估计器**（straight-through estimator, STE）：假装 $f$ 是恒等函数
  - 前向：$y=f(x)$（照常应用那个不可微函数）
  - 反向：$\frac{\partial\mathcal L}{\partial x}=\frac{\partial\mathcal L}{\partial y}$——上游梯度原样往下传
  - 若 $f(x)$ 是递减函数，这样做会反号（信号方向反了）
  - 但 STE 通常用在单调递增的操作上（四舍五入、量化、递增的阶梯函数）
    - 方向是对的，即便大小不准

## 理论计算机科学（Theoretical CS）

- **正则语言**（regular language）：能用**有限状态机**（也叫有限自动机）识别的语言
  - 确定性有限自动机（DFA）
- **上下文无关语言**（context-free language）：能用下推自动机识别的语言——相当于有限状态机外加一个栈
  - 栈提供了无限容量，但只能访问栈顶
  - 编程语言的语法正是由上下文无关语言构造的（配对括号、嵌套函数调用、闭合的 HTML 标签等）
- 任何有限语言都是平凡的正则语言——状态够多就能把所有合法字符串一一枚举出来
- 任何 DFA 都可以用 ReLU RNN 编码
  - 给定一个 DFA：
    - 状态集 $Q=\{q_1,\dots,q_k\}$
    - 字母表 $\Sigma=\{\sigma_1,\dots,\sigma_m\}$
    - 转移函数 $\delta:Q\times\Sigma\to Q$
    - 起始状态 $q_1$
    - 接受状态集 $F\subseteq Q$
  - 构造一个隐维度为 $k$ 的 RNN，隐状态是"当前处于哪个状态"的 one-hot 编码
  - 需要构造 $W_h\in\mathbb R^{k\times k}$、$W_x\in\mathbb R^{k\times m}$、$b\in\mathbb R^k$
  - 对每条转移 $\delta(q_i,\sigma_k)=q_j$，令 $(W_h)_{ji}=1$、$(W_x)_{jk}=1$、$b_j=-1$
