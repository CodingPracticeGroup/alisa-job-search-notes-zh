# 面向语言模型的策略梯度入门

> 译自：<https://ivison.id.au/2026/02/09/policy-gradient.html>（页面标注日期 2026-02-09）
> 作者：Hamish Ivison（原文为其研究组会"RL101"讲座的笔记整理）；底稿：`../snapshots/readings/ivison-policy-gradient-for-lms.md`（2026-11-15 快照）
> 译注日期：2026-11-15。术语见根 `glossary.md`。
> 读者提示：全文只需要两个数学工具——期望（$\mathbb E$，即"按概率加权的平均"）与多元函数的梯度（$\nabla$，即"每个变量上的敏感度"）；概率图、求和号 $\sum$、乘积号 $\prod$ 按定义可逐行读懂。文中"奖励/回报/折扣"等概念首次出现时均给出解释。

这些笔记改编自我在几个研究组的组会上做的一次"RL101"讲座，改动很小。目标是把策略梯度（policy gradient）讲得相当完整，且专门站在语言模型（LM）的语境下。我会一路推到 CISPO（说实话，它差不多是当下最好的算法之一），但不覆盖到 PPO 的全部。笔记底子是当年我在华盛顿大学（UW）上强化学习课时写的。

## 术语

我们使用标准的马尔可夫决策过程（Markov Decision Process, MDP）模型[^1]

$$\mathcal{M} = (\mathcal{S},\mathcal{A},P,r,\gamma,\rho_0).$$

| **术语** | **MDP 记号** | **LM 语境下的对应物** | **LM 记号** |
|----|----|----|----|
| 状态（State） | $s\in\mathcal{S}$ | 提示词 + 已生成前缀 | $s_t \equiv x_{0:t}$ |
| 动作（Action） | $a\in\mathcal{A}$ | 下一个词元 | $a_t \equiv x_{t+1}$ |
| 转移（Transition） | $P(s'\mid s,a)$ | 追加词元（通常是确定性的） | $s_{t+1}=(s_t,a_t)$ |
| 奖励（Reward） | $r(s,a)$ | 终止得分 / 奖励模型 | $r_T$（常给在 EOS 处） |
| 折扣（Discount） | $\gamma\in[0,1)$ | （可选的）词元位置折扣 | $\gamma$ |
| 初始状态分布 | $\rho_0$ | 提示词分布 | $p(\text{prompt})$ |
| 策略（Policy） | $\pi_\theta(a\mid s)$ | 语言模型的下一词分布 | $p_\theta(x_{t+1}\mid x_{0:t})$ |
| 轨迹（Trajectory） | $\tau=(s_0,a_0,\dots,s_T)$ | 补全 / 采样（rollout） | $\tau \equiv x_{0:T}$ |
| 回报（Return） | $R(\tau)=\sum_{t=0}^{T-1}\gamma^t r(s_t,a_t)$ | 对一条补全的标量打分 | $R(x_{0:T})$ |


**策略 / 语言模型分布。** 在一般 RL 记号中，策略是一个条件分布 $\pi_\theta(a\mid s)$。对于自回归语言模型，我们直接认定

$$\pi_\theta(a_t\mid s_t) \equiv p_\theta(x_{t+1}\mid x_{0:t}).$$

有了这个对应，一整条补全（轨迹）的概率就可以分解为

$$\pi_\theta(\tau) \equiv p_\theta(x_{0:T}) = p(x_0)\prod_{t=0}^{T-1} p_\theta(x_{t+1}\mid x_{0:t}),$$

其中 $p(x_0)$ 表示提示词的分布（或者被当作固定条件处理）。

注意，我们把折扣称为"可选"，是因为绝大多数 LM RL 工作直接取 $\gamma = 1$，且只在轨迹结尾给一次奖励。不过有时我们会改掉这一点（比如能拿到密集奖励（dense reward）的场景）。

## 推导策略梯度

在策略梯度方法里，我们假设策略 $\pi_\theta$ 是可微、带参数的，并用（随机）梯度**上升**来更新参数、最大化目标 $J(\theta)$：

$$\theta \leftarrow \theta + \alpha\widehat{\nabla_\theta J(\theta)}.$$

一个重要的初始恒等式是"对数求导技巧"（log-derivative trick）：

$$\begin{aligned} \nabla_\theta \log \pi_\theta(\tau) &= \frac{\nabla_\theta \pi_\theta(\tau)}{\pi_\theta(\tau)} \quad\text{（当 } \pi_\theta(\tau)>0 \text{）}\\ \nabla_\theta \pi_\theta(\tau) &= \pi_\theta(\tau)\nabla_\theta \log \pi_\theta(\tau) \end{aligned}$$

我们想最大化期望奖励，即令

$$\begin{aligned} J(\theta) &= \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)] \\ &= \int \pi_\theta(\tau)\,R(\tau)\,d\tau \end{aligned}$$

现在来求导：

$$\begin{aligned} \nabla_\theta J(\theta) &= \nabla_\theta \int \pi_\theta(\tau)\,R(\tau)\,d\tau \\ &= \int \nabla_\theta\pi_\theta(\tau)\,R(\tau)\,d\tau \\ &= \int \pi_\theta(\tau)\,\nabla_\theta\log\pi_\theta(\tau)\,R(\tau)\,d\tau \end{aligned}$$

然后，把轨迹的对数概率展开。最终形式：

$$\begin{aligned} \log\pi_\theta(\tau) &= \log\rho_0(s_0) + \sum_{t=0}^{T-1}\log\pi_\theta(a_t\mid s_t) + \sum_{t=0}^{T-1}\log P(s_{t+1}\mid s_t,a_t) \\ \nabla_\theta\log\pi_\theta(\tau) &= \nabla_\theta\log\rho_0(s_0) + \nabla_\theta\sum_{t=0}^{T-1}\log\pi_\theta(a_t\mid s_t) + \nabla_\theta\sum_{t=0}^{T-1}\log P(s_{t+1}\mid s_t,a_t) \\ &= \sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t) \end{aligned}$$

有趣的是，转移动力学（环境规则）**完全消失了**！当然，它们仍然在隐式地影响轨迹的分布。所以直接把它代回去即可：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,R(\tau)\right]$$

接下来定义"后续回报"（reward-to-go，即从时刻 $t$ 起的未来回报）：

$$G_t \triangleq \sum_{k=t}^{T-1} \gamma^{k-t}\,r(s_k,a_k)$$

关键观察：在我们的导数里，真正有用的只有后续回报。把完整回报写成"过去"部分加"未来"部分：

$$\begin{aligned} R(\tau) &= \sum_{k=0}^{T-1} \gamma^k r(s_k,a_k)\\ &= \underbrace{\sum_{k=0}^{t-1} \gamma^k r(s_k,a_k)}_{\triangleq\,R_{<t}} + \underbrace{\sum_{k=t}^{T-1} \gamma^k r(s_k,a_k)}_{\triangleq\,G_t} \end{aligned}$$

那么，对任意固定时刻 $t$，

$$\begin{aligned} \mathbb{E}_{\tau\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,R(\tau)\big] &= \mathbb{E}_{\tau\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,(G_t + R_{<t})\big]\\ &= \mathbb{E}_{\tau\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\big] + \underbrace{\mathbb{E}_{\tau\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,R_{<t}\big]}_{=0} \end{aligned}$$

最后一项为零，是因为在以 $s_t$ 为条件时，$R_{<t}$ 构成一个合法的基线（baseline）。这一点先放一放（记个"大头针"），我们稍后回来解释。

于是得到 REINFORCE 损失：

$$\mathcal{L}_{\text{REINFORCE}}(\theta) \triangleq -\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1} \log\pi_\theta(a_t\mid s_t)\,G_t\right]$$

以及最朴素的算法：

1. 初始化策略参数 $\theta$。
2. 对迭代 $k=0,1,2,\dots$ 重复：
   1. 用当前策略 $\pi_\theta$ 采样一批轨迹 $\{\tau^{(i)}\}_{i=1}^N$。
   2. 对每条轨迹 $\tau^{(i)}=(s_0^{(i)},a_0^{(i)},\dots,s_T^{(i)})$，计算各时刻的奖励 $r(s_t^{(i)},a_t^{(i)})$ 和后续回报 $G_t^{(i)}$。
   3. 构造（使用后续回报的）蒙特卡洛梯度估计：
      $$\widehat{\nabla_\theta J(\theta)} = \frac{1}{N}\sum_{i=1}^N\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta\big(a_t^{(i)}\mid s_t^{(i)}\big)\,G_t^{(i)}$$
   4. 梯度上升更新参数：$\theta \leftarrow \theta + \alpha\widehat{\nabla_\theta J(\theta)}$

### REINFORCE 的问题

REINFORCE 有三个核心问题：

1. **高方差**：奖励值可能波动极大，甚至量级都很奇怪。设想一个恒为负的奖励：基本上，我们需要把数值归一化一下。
2. **步长未知**：我们其实不知道该用多大的学习率，而且与普通机器学习不同，这个"合适的学习率"还可能随训练过程变化。
3. **数据复用**：我们无法复用之前几轮的数据，因为损失和梯度都是针对**当前**策略定义的。

我讲其中两个。步长问题对 CISPO 这类方法影响不大，但它确实重要，也是 TRPO、PPO 等方法的发展动机之一。先处理方差——这是最要紧的。

### 基线方法

一个标准的降方差技巧，是减去一个只依赖当前状态的基线：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,(G_t - b(s_t))\right]$$

需要证明两件事：（1）这样做是无偏的，不改变期望结果；（2）这确实能降低方差。

#### 无偏性

先把损失拆开：

$$\begin{aligned} \nabla_\theta J(\theta) &= \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,(G_t - b(s_t))\right] \\ &= \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\right] - \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,b(s_t)\right] \end{aligned}$$

接下来只看基线项。考察单个时间步 $t$ 的情形，关键是把期望拆开：

$$\begin{aligned} \mathbb{E}_{\tau\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,b(s_t)\big] &= \mathbb{E}_{s_t,a_t\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,b(s_t)\big]\\ &= \mathbb{E}_{s_t}\left[\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,b(s_t)\;\big|\;s_t\big]\right]\\ &= \mathbb{E}_{s_t}\left[b(s_t)\,\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\big]\right]\\ &= \mathbb{E}_{s_t}\left[b(s_t)\,\sum_a \pi_\theta(a\mid s_t)\,\nabla_\theta\log\pi_\theta(a\mid s_t)\right]\\ &= \mathbb{E}_{s_t}\left[b(s_t)\,\sum_a \nabla_\theta\pi_\theta(a\mid s_t)\right]\\ &= \mathbb{E}_{s_t}\left[b(s_t)\,\nabla_\theta\sum_a \pi_\theta(a\mid s_t)\right] = \mathbb{E}_{s_t}\left[b(s_t)\,\nabla_\theta 1\right] = 0 \end{aligned}$$

（倒数第二步用的是对数求导技巧的逆用：$\pi_\theta(a\mid s_t)\nabla_\theta\log\pi_\theta(a\mid s_t)=\nabla_\theta\pi_\theta(a\mid s_t)$；最后一步用的是"任何状态的下一步概率总和恒等于 1"，而常数的梯度是 0。）

因此

$$\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,b(s_t)\right] = 0,$$

把 $G_t$ 换成 $G_t-b(s_t)$ 不改变期望（估计量保持无偏）。

注意，这就是为什么基线只依赖"到当前状态为止"是合法的：如果把**当前动作**也算进基线，上面的推导中就无法把 $b$ 提到期望外面。所以，基线可以依赖除当前动作以外的一切。

#### 降方差

既然知道可以这样做，接下来要问：什么样的基线能真正降方差？

最优基线的推导更繁琐，我也没找到清晰的说法，所以这里采用一个更高层面的近似[^2]。这些近似在一般情况下并不严格成立，但能帮我们对"基线为何能降方差"建立直觉。考察单条轨迹的方差：

$$\begin{aligned} \mathrm{Var}\Big(\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,(G_t-b(s_t))\Big) &\approx^{(i)} \sum_{t=0}^{T-1}\mathbb{E}_{\tau}\Big[\big(\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,(G_t-b(s_t))\big)^2\Big]\\ &\approx^{(ii)} \sum_{t=0}^{T-1}\mathbb{E}_{\tau}\Big[\big(\nabla_\theta\log\pi_\theta(a_t\mid s_t)\big)^2\Big]\,\mathbb{E}_{\tau}\Big[\big(G_t-b(s_t)\big)^2\Big] \end{aligned}$$

这里 (i) 略去了不同时间步之间的交叉协方差项，(ii) 进一步把"得分项"（log 概率梯度）与"中心化后的回报项"近似看作相互独立。我们无法控制 log 概率那部分的方差，所以能做的就是只最小化 $G_t - b(s_t)$！这说明：好的基线就是我们基于 $s_t$ 对 $G_t$ 的最好猜测——**而这正是价值函数（value function）！**

于是得到**优势函数（advantage function）**：

$$A_{\pi_\theta}(s_t, a_t) = G_t - \hat{V}_{\pi_\theta}(s_t)$$

注意这里 $G_t$ 是"在状态 $s_t$ 下采取 $a_t$ 之后实际拿到的回报"，它正是 Q 值的（无偏蒙特卡洛）估计。直觉上，我们把**按绝对奖励加权**换成了**按"该动作比策略平均动作好多少"加权**。这自然地降低方差，因为我们相当于把"策略本身已经有多好"考虑了进来。

### 复用旧数据

再讲 REINFORCE 的另一个大问题：数据复用。

在现有设定下旧数据没法直接复用，因为我们的一切推导都建立在"轨迹来自当前模型"（$\tau \sim \pi_\theta$）的前提上。一旦走完一步梯度更新，旧轨迹在技术上就来自另一个分布了（$\tau \sim \pi_{\theta_{t-1}}$）。

可以用**重要性加权（importance weighting）**修复这个问题。核心思想是：从别的分布采样也可以，只要我们按"当前分布下的概率"给样本重新加权。为简化，先定义

$$\begin{aligned} \mathbb{E}_{\tau\sim\pi_{\theta}}\big[f(\tau)\big] &= \int \pi_{\theta}(\tau)\,f(\tau)\,d\tau \\ &= \int \pi_{\theta_{\mathrm{old}}}(\tau)\,\frac{\pi_{\theta}(\tau)}{\pi_{\theta_{\mathrm{old}}}(\tau)}\,f(\tau)\,d\tau \\ &= \mathbb{E}_{\tau\sim\pi_{\theta_{\mathrm{old}}}}\Big[w(\tau)\,f(\tau)\Big] \end{aligned}$$

其中

$$w(\tau) \triangleq \frac{\pi_{\theta}(\tau)}{\pi_{\theta_{\mathrm{old}}}(\tau)} = \prod_{t=0}^{T-1}\frac{\pi_{\theta}(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}$$

把这个技巧直接用到原来的策略梯度表达式上：

$$\begin{aligned} \nabla_\theta J(\theta) &= \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\right]\\ &= \mathbb{E}_{\tau\sim\pi_{\theta_{\mathrm{old}}}}\left[w(\tau)\,\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\right] \end{aligned}$$

也可以在**词元层面**定义这个比值：

$$r_{t}(\theta) = \frac{\pi_{\theta}(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}$$

## CISPO

事实上，到这里我们基本已经凑齐了当下最先进（SOTA）的 RL 算法 CISPO[^3] 的全部零件！

CISPO 的损失是：

$$J_{\mathrm{CISPO}}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D},\;\{o^i\}_{i=1}^G\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q)}\left[ \frac{1}{\sum_{i=1}^G |o^i|} \sum_{i=1}^G\sum_{t=1}^{|o^i|} \mathrm{sg}\big(r_{i,t}(\theta)\big)\,\hat A_{i,t}\,\log \pi_\theta\big(o^i_t\mid q,o^i_{<t}\big) \right]$$

注意这里我们对组（下标 $i$）和时刻（一次采样的词元数）取平均。$\mathrm{sg}$ 表示"停止梯度"（stop gradient），即不让梯度回传到计算重要性比值 $r$ 所用的那些 log 概率上。$G$ 是组大小（同一个提示词对应的采样条数），用来计算优势 $\hat A$：

$$\hat A_{i,t} = \frac{R_i - \mathrm{mean}(\{R_j\}_{j=1}^G)}{\mathrm{std}(\{R_j\}_{j=1}^G)}$$

注意，直觉上这与前面做的完全是同一件事，只不过用**组内统计量**替代了"用一个学出来的函数估计价值"。

## 其他优势估计

其他一些有意思的优势估计：

1. **VinePPO**[^4]：用词元级的蒙特卡洛 rollout 来估计价值。贵得惊人，但可以说是最好的做法。
2. **GIGPO**[^5]：直接利用词元相似度把相似状态归并到一起。对"智能体式（agentic）、基于状态"的任务（例如操作浏览器）效果非常好——这类任务能拿到真实状态，且状态本身很简单。
3. **REINFORCE++**[^6]：干脆把整个批次的平均奖励当价值估计！当你的任务都差不多时，这很合理。
4. **RLOO**[^7]：基本就是 GRPO，但去掉标准差，并且用"留一法"（leave-one-out）替代"用全部样本取平均"。

> **最后一点说明**
>
> 你可能会问：GRPO、REINFORCE++ 这类方法里，基线居然直接用了轨迹自己的奖励——这不是把当前动作也包含进去了吗？实际上，这恰好是一个合法的"特例"！
>
> 首先，用**其他**轨迹完全合法，因为它们与当前动作完全独立（本来就是不同的样本）。奇怪的是"自己这条轨迹的奖励"。考虑用批内平均后续回报当基线：
>
> $$ b(s_t) \equiv \frac{1}{B}\sum_{n=1}^{B} G^n_t \quad\text{，其中 } B \text{ 是批次大小} $$
>
> 设 $x$ 表示当前样本在批次中的编号，则：
>
> $$ \begin{aligned} \nabla_\theta J(\theta) &= \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,(G^x_t - b(s_t))\right]\\ &= \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\Big(G^x_t - \frac{1}{B}\sum_{n=1}^{B} G^n_t\Big)\right] \\ &= \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\Big(G^x_t - \frac{1}{B}G^x_t - \frac{1}{B}\sum_{n=1,\ne x}^{B}G^n_t\Big)\right] \\ &= \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\Big(\frac{B-1}{B}G^x_t - \frac{1}{B}\sum_{n=1,\ne x}^{B}G^n_t\Big)\right] \end{aligned} $$
>
> 我们知道 $\frac{1}{B}\sum_{n=1,\ne x}^{B}G^n_t$ 作为基线是合法的，所以只需分析 $\frac{B-1}{B}G^x_t$。直观上很清楚：它只是把梯度整体缩小了 $\frac{B-1}{B}$ 倍，不会给最终结果带来偏差。严格来说，像 RLOO 那样用"留一"估计可以避免这层缩放，更好一些，但用这里的形式也无妨！
>
> 关键在于，这只在**基线里用了奖励本身**时成立。其他包含 $a_t$ 的函数用作基线都是不合法的。

就到这里，谢谢各位听讲！

---


## 参考文献

另可参考：Weng, L.（2018）. [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)；Sutton & Barto（1998）. *Reinforcement Learning: An Introduction*（第 13 章）。

[^1]: 译注：MDP 是描述"智能体反复感知状态、采取行动、获得奖励"这类问题的标准数学框架：$\mathcal{S}$ 是状态集合，$\mathcal{A}$ 是动作集合，$P$ 是转移规则，$r$ 是奖励规则，$\gamma$ 是折扣因子（越接近 1 越重视远期奖励），$\rho_0$ 是起点分布。
[^2]: Seita, D. (2017). [Going Deeper Into Reinforcement Learning: Fundamentals of Policy Gradients](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/).
[^3]: MiniMax et al. (2025). [MiniMax-M1: Scaling Test-Time Compute](https://arxiv.org/abs/2501.08313).
[^4]: Kazemnejad et al. (2025). [VinePPO: Refining Credit Assignment](https://arxiv.org/abs/2410.01679).
[^5]: Feng et al. (2025). [Grouping in Group Policy Optimization for LLM](https://arxiv.org/abs/2504.02763).
[^6]: Hu et al. (2025). [REINFORCE++: Stabilizing Critic-Free Policy Optimization](https://arxiv.org/abs/2501.03262).
[^7]: Ahmadian et al. (2024). [Back to Basics: Revisiting REINFORCE Style Optimization](https://arxiv.org/abs/2402.14740).
