# 第八章　后训练（Post-training）

> 译自 Alisa's Book of LLMs "Post-training" 一节；底稿与约定见 [`05-scaling-laws.md`](05-scaling-laws.md) 页眉说明。术语对照见根 `glossary.md`。

## 策略梯度（Policy gradients）

- 记号
  - 动作 $a_t\in\mathcal V$：时间步 $t$ 的下一个词元
  - 状态 $s_t$：时间步 $t$ 的文本前缀 $(s_0,a_0,\dots,a_{t-1})$
  - $a_t\sim\pi_\theta(\cdot\mid s_t)$：语言模型策略
  - $s_0\sim p_0$：提示 $s_0$ 从提示起始分布中采样
  - $\tau$：轨迹（有限步），也叫 rollout（生成回线）、episode（回合）
  - $R(\tau)$：轨迹 $\tau$ 的奖励
- 目标

$$\text{最大化 }J(\theta)=\mathbb E_{\tau\sim\pi_\theta} [R(\tau)]$$

- 可以用梯度上升来优化：

$$\theta_{k+1}=\theta_k+\alpha\nabla_\theta J(\theta_k)$$

- **朴素策略梯度**（vanilla policy gradient，又名 **REINFORCE**）：目标的梯度可以写成

$$\nabla_\theta J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t) R(\tau)\right]$$

  - 这本质上和 SFT（监督微调）是同一类更新，只是数据 $\tau$ 改由策略自己采样，且梯度按 $R(\tau)$ 加权
    - 若 $R(\tau)$ 为正，就往"增大 $\tau$ 中每个词元 $a_t$ 的 $\log\pi_\theta(a_t\mid s_t)$"的方向走
      - 否则反向走
    - $|R(\tau)|$ 越大，步子迈得越大
  - 梯度推导：

$$\begin{align*}\nabla_\theta J(\theta)=\nabla_\theta\mathbb E_{\tau\sim\pi_\theta} [R(\tau)] \\ &=\nabla_\theta\sum_\tau P(\tau\mid\theta)\,R(\tau) \\ &=\sum_\tau\nabla_\theta P(\tau\mid\theta)\,R(\tau) \\ &=\sum_\tau P(\tau\mid\theta)\nabla_\theta\log P(\tau\mid\theta)\,R(\tau)\quad\text{对数导数技巧：}\nabla P=P\,\nabla \log P \\ &=\mathbb E_{\tau\sim\pi_\theta}\nabla_\theta\log P(\tau\mid\theta)\,R(\tau) \\ &=\mathbb E_{\tau\sim\pi_\theta}\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\, R(\tau) \end{align*}$$

- 实践中，从策略 $\pi_\theta$ 采样 $N$ 条轨迹 $\tau^{(i)}$（起始状态 $s_0^{(i)}\sim p_0$）来估计 $\nabla_\theta J(\theta)$：

$$\hat g=\frac 1N\sum_{i=1}^N\sum_{t=1}^T\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})\,R(\tau^{(i)})$$

- 所谓"策略梯度损失"只是一个标量 `pg_loss`，让 `pg_loss.backward()` 算出的梯度恰好等于近似策略梯度 $\hat g$：

$$L(\theta)=\frac 1N\sum_{i=1}^N\sum_{t=1}^T\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})\,R(\tau^{(i)})$$

  - 它不是通常意义上的"损失"：$L(\theta)$ 不告诉你策略有多好，只是产生正确梯度的工具
  - 目标函数也不是固定的——它由当前策略采出的样本构成

**带基线的策略梯度（baselined policy gradient）**

- 朴素策略梯度的问题：估计的方差非常大
  - 设想一道很简单的题，所有回答都得正奖励
    - 没有基线时所有回答都被"加强"，包括批次里较差的那些！
      - 从训练全程平均看没什么，但每一步更新噪声都很大
    - 如果拿*平均*奖励当基线，低于平均的回答就不会被加强
- **带基线的策略梯度**：在梯度估计中把 $R(\tau)$ 换成 $R(\tau)-b(s_t)$，$b$ 是基线函数

$$\nabla_\theta J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_{t=0}^T\nabla_\theta\log\pi_\theta(a_t\mid s_t) (R(\tau)-b(s_t))\right]$$

  - 只要 $b(s_t)$ 只依赖状态 $s_t$（不依赖动作 $a_t$），就不会给 $\nabla J(\theta)$ 的估计引入偏差
  - 我们希望 $b(s_t)$ 与 $R(\tau)$ 相关，这样 $R(\tau)-b(s_t)$ 就小 → 梯度噪声更小
  - 这不改变被优化的目标，只改变目标梯度的估计方式
- 为什么带基线的策略梯度仍是无偏估计？
  - 只需证明基线项 $B$ 的期望为 0：

$$B=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_{t=0}^T\nabla_\theta\log\pi_\theta(a_t\mid s_t)b(s_t)\right]$$

  - 记 $X_t=\nabla_\theta\log\pi_\theta(a_t\mid s_t)b(s_t)$，则

$$B=\mathbb E_{\tau\sim\pi_\theta}\sum_{t=0}^T X_t$$

  - 先把期望移进求和号：

$$B=\sum_{t=0}^T\mathbb E_{\tau\sim\pi_\theta} X_t$$

  - 之前：每条轨迹先对时间步求和，再对轨迹取平均
    - 现在：每个时间步先对轨迹取平均，再对时间步求和
  - 对整条轨迹平均 $X_t$，等价于只对 $(s_t,a_t)$ 平均——因为 $X_t$ 只依赖 $(s_t,a_t)$，轨迹其余部分与它无关
    - 再把 $(s_t,a_t)$ 上的联合期望分解为"先对 $s_t$、再对给定 $s_t$ 下的 $a_t$"：

$$B=\sum_{t=0}^T\mathbb E_{s_t,a_t} X_t=\sum_{t=0}^T\mathbb E_{s_t}\left[\mathbb E_{a_t\mid s_t} X_t\right]$$

  - 代入 $X_t$ 的定义：

$$\begin{align*}\mathbb E_{a_t\mid s_t} X_t=\mathbb E_{a_t\mid s_t}\nabla_\theta\log\pi_\theta (a_t\mid s_t)\,b(s_t) \\ &= b(s_t)\,\mathbb E_{a_t\mid s_t}\nabla_\theta\log\pi_\theta (a_t\mid s_t)\quad\text{因为 }b\text{ 不依赖 }a_t！ \\ &= b(s_t)\,\sum_{a_t}\pi_\theta(a_t\mid s_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t) \\ &= b(s_t)\,\sum_{a_t}\nabla_\theta\pi_\theta(a_t\mid s_t)\quad\text{再次使用对数导数技巧} \\ &=b(s_t)\nabla_\theta\sum_{a_t}\pi_\theta(a_t\mid s_t) \\ &= b(s_t)\nabla_\theta 1\quad\text{全部动作的概率之和为 1} \\ &= b(s_t)\cdot 0 \\ &= 0 \end{align*}$$

  - 注意：能把 $b(s_t)$ 提出对 $a_t\mid s_t$ 的期望，正因为 $b$ 只依赖 $s_t$；这才用得上"概率求和恒为 1"〔原文此处误写 $\sum_a\pi(a\mid s_t)=0$，应为 $=1$〕

  - 代回即得：

$$B=\sum_{t=0}^T\mathbb E_{s_t}[0]=0$$

- PPO 常用的基线取法是 $V_\psi(s_t)$——对"看到部分序列 $s_t$ 时的*期望*奖励"的估计

$$\nabla_\theta J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_{t=0}^T\nabla_\theta\log\pi_\theta(a_t\mid s_t) (R(\tau)-V_\psi(s_t))\right]$$

  - 无偏，因为 $V_\psi$ 只依赖状态
  - 每个词元收到的信号：最终奖励是高于还是低于"走到这里时看起来大概率会拿到的奖励"？
    - 把"看起来要砸"的回答（$V_\psi(s_t)$ 小）做成"好回答"（$R(\tau)$ 高）的词元，分到更多功劳
- 与优势（advantage）的联系：
  - $R(\tau)$ 是 **Q 函数** $Q^\pi(s_t,a_t)$（在该状态选该动作、之后按 $\pi$ 走的期望奖励）的一次抽样
  - $V_\psi(s_t)$ 是**价值函数** $V^\pi(s_t)$（只给状态时的期望奖励，$V^\pi(s)=\sum_{a\sim\pi(\cdot\mid s)}Q^\pi(s,a)$）的估计
  - 所以 $R(\tau)-V_\psi(s_t)$ 是 $Q^\pi(s,a)-V^\pi(s)=A^\pi(s,a)$ 的估计——**优势函数**：动作 $a$ 比状态 $s$ 下的平均水准好多少
    - 特别地，这是优势的**蒙特卡洛估计**：用单条轨迹的 $R(\tau)$ 来估 $Q^\pi$
- 基线函数大体都在估 $\mathcal V^\pi(s_t)$，即"从当前状态出发的期望回报"：
  - 学习出来的价值函数【上面那种】
  - RLOO：每个提示采 $G$ 个回答，某条的基线取*其余*回答的平均奖励（与 GRPO 非常像，差别在于保留自身下标 $i$、且不做标准差归一化）
  - 批量估计（REINFORCE++）：直接用批次内平均奖励

**异策略策略梯度（off-policy policy gradient）**

- 同策略（on-policy）的问题：每做一步梯度都得从当前策略重新生成，可策略其实没怎么变
- 异策略学习：从别的策略采样轨迹，优化的是另一个策略
  - 通常用旧版策略 $\pi_{\theta_\text{old}}$ 的轨迹来优化当前策略 $\pi_\theta$
- 使用**替代目标**（surrogate objective）$\mathcal J^\text{surrogate}$：

$$\mathcal J^\text{surrogate}(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\sum_{t=1}^T\underbrace{\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}}_{r_t}\, R(\tau)\right]$$

  - 比值 $r_t=\pi_\theta/\pi_{\theta_\text{old}}$ 是重要性采样式（importance sampling）的重新加权项
    - 重要性采样背景：想要 $\mathbb E_{x\sim p} f(x)$ 却只有 $x\sim q$ 的样本，可改写为

$$\mathbb E_{x\sim p}\,f(x)=\mathbb E_{x\sim q}\,\left[\frac{p(x)}{q(x)}f(x)\right]$$

  - 直觉：若 $x$ 在 $p$ 下比在 $q$ 下更可能出现，这条样本就该多算一点分；反之少算
  - 注意：这优化的已不是原目标 $\mathcal J(\theta)$！
  - *真正的*异策略估计是把 $\mathcal J(\theta)$ 用重要性采样原样改写，但比值连乘导致方差爆炸：

$$\mathcal J(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\frac{P(\tau\mid\theta)}{P(\tau\mid\theta_\text{old})}\,R(\tau)\right]=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\prod_{t=1}^T\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}\,R(\tau)\right]$$

  - $\mathcal J^\text{surrogate}$  essentially 是把"连乘"换成了"逐时间步求和"

- $\mathcal J^\text{surrogate}$ 导出如下**异策略策略梯度**：

$$\nabla_\theta \mathcal J^\text{surrogate}(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\sum_t\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\, R(\tau)\right]$$

  - 理由：$r_t$ 中只有分子依赖 $\theta$，再套用对数导数技巧即可
  - 实践中按下式估计（$N$ = 每批轨迹条数）：

$$\hat g_\text{off-policy}=\frac 1N\sum_{i=1}^N\sum_{t=1}^T\frac{\pi_\theta(a_t^{(i)}\mid s_t^{(i)})}{\pi_{\theta_\text{old}}(a_t^{(i)}\mid s_t^{(i)})}\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})R(\tau^{(i)})$$

## PPO

- 提出了对重要性权重 $r_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}$ 的**裁剪**（clip）机制
  - 常规替代目标是

$$\mathcal J^\text{surrogate}(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\sum_tr_tA_t\right]$$

  - PPO 用价值函数作基线，按惯例把策略梯度里的 $R(\tau)$ 换成逐时间步优势 $A_t=R(\tau)-V_\psi(s_t)$
  - **裁剪替代目标**把 $r_t$ 限制在 $[1-\epsilon,1+\epsilon]$ 内：

$$\mathcal J^\text{CLIP}(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\sum_t\min(r_tA_t,\text{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\right]$$

- 在一批轨迹上做多次梯度步时，裁剪保证稳定
  - 用无偏性换取更稳的更新
  - 只要没走离 $\pi_{\theta_\text{old}}$ 太远，裁剪目标就是很好的近似
- 裁剪起作用的四种情形：
  - $r_t>1+\epsilon$ 且 $A_t>0$：取 $(1+\epsilon)A_t$
    - $\nabla\mathcal J(\theta)$ 不再依赖 $\theta$ → 梯度为 0
    - 相对旧策略已经大幅提高了 $a_t$ 的概率——好动作别继续硬推了
  - $r_t<1-\epsilon$ 且 $A_t<0$：取 $(1-\epsilon)A_t$
    - 梯度同样为 0
    - 坏动作也别继续硬压了
  - $r_t>1+\epsilon$ 且 $A_t<0$：取 $r_tA_t$
    - $\nabla\mathcal J(\theta)$ 依赖 $\theta$【见异策略策略梯度】
    - 也就是说此时允许 $r_t>1+\epsilon$——合理，因为 $A_t<0$ 是坏词元，应该继续往下压
    - 裁剪是不对称的：只在"策略正顺着优势方向走"时刹车，但不阻止你纠正错误
  - $r_t<1-\epsilon$ 且 $A_t>0$：取 $r_tA_t$
    - $\nabla\mathcal J(\theta)$ 依赖 $\theta$【见异策略策略梯度】
- PPO 的用法：从当前策略收集一批轨迹，然后用裁剪替代目标做多步梯度更新

## RLHF

- 引入 KL 惩罚，防止 $\pi_\theta$ 离参考模型 $\pi_\text{ref}$ 太远：

$$\mathcal J^\text{RLHF}(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[R(\tau)-\beta D_{\text{KL}}(\pi_\theta\mid\mid\pi_\text{ref})\right]$$

- 实践中 KL 惩罚逐词元计算，直接并入逐词元奖励
- 奖励模型怎么训练：
  - Bradley-Terry 模型：$y_w$ 胜过 $y_l$ 的概率是

$$P(y_w\succ y_l)=\frac{\exp(R(x,y_w))}{\exp(R(x,y_w))+\exp(R(x,y_l))}=\sigma(R(x,y_w)-R(x,y_l))$$

  - 给语言模型加分类头训练，最大化

$$\mathcal L(\varphi)=-\log P(y_w\succ y_l)$$

  - 这就是二元交叉熵损失 $-(y\log p+(1-y)\log(1-p))$，其中真实标签恒取 $y=1$，$p$ 按 Bradley-Terry 计算

## GRPO

- GRPO 把奖励 $R(\tau)$ 换成"相对于一组采样轨迹的优势"
  - 注意：它不能干净地归为某种基线选择——它还在减去基线后除以了归一化因子
  - 对问题 $q$ 的一组生成 $\{o^{(i)}\}_{i=1}^G$，每条由词元 $o^{(i)}_1,\dots,o^{(i)}_T$ 组成：
    - 用 $R(q,o^{(i)})$ 算出每条生成的奖励 $\mathbf r=\{r^{(i)}\}_{i=1}^G$
    - 组内相对优势估计 $A^{(i)}$ 为

$$A^{(i)}=\frac{r^{(i)}-\text{mean}(\mathbf r)}{\text{std}(\mathbf r)}$$

  - 同一条回答里的所有词元共享同一个 $A^{(i)}$
  - 目标函数：

$$\mathcal J^\text{GRPO-CLIP}(\theta)=\frac 1G\sum_{i=1}^G\frac 1T\sum_{t=1}^T\min\left(r_tA^{(i)},\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A^{(i)}\right)$$

  - 其中 $r_t$ 就是逐词元概率比（这里的记号惯例是用 $o_t$ 代替 $a_t$）：

$$r_t=\frac{\pi_\theta(o_t\mid s_t)}{\pi_{\theta_\text{old}}(o_t\mid s_t)}$$

  - 通过"组内比较"算优势，省掉了 PPO 需要的评论家（价值网络），比 PPO 简单
- GRPO 融合了三个想法：
  - 用 $\pi_{\theta_\text{old}}$ 的异策略策略梯度：$\mathcal J^\text{surrogate}(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\sum_tr_tR(\tau)\right]$
  - 裁剪机制【PPO】：$\mathcal J^\text{CLIP}(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\sum_t\min(r_t R(\tau),\text{clip}(r_t,1-\epsilon,1+\epsilon)R(\tau))\right]$
  - 组归一化优势【DeepSeek R1】：$\mathcal J^\text{GRPO-CLIP}(\theta)=\frac 1G\sum_{i=1}^G\frac 1T\sum_{t=1}^T\min\left(r_tA^{(i)},\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A^{(i)}\right)$
- 算法
  - 策略模型 $\pi_\theta\leftarrow\pi_{\theta_\text{init}}$
  - 对每个大步（`n_grpo_steps`）：
    - 从 $\mathcal D$ 采一批 $\mathcal D_b$
    - 更新旧策略 $\pi_{\theta_\text{old}}\leftarrow\pi_\theta$
    - 对 $\mathcal D_b$ 中每个问题 $q$，从 $\pi_{\theta_\text{old}}(\cdot\mid q)$ 采 $G$ 个输出 $\{o^{(i)}\}_{i=1}^G$
    - 用 $R(q,o^{(i)})$ 算出每个输出的奖励 $\{r^{(i)}\}_{i=1}^G$
    - 用组内相对优势估计计算 $A^{(i)}$
    - 对每个训练步（`n_train_steps_per_rollout_batch`）：
      - 最大化 GRPO 目标来更新 $\pi_\theta$
  - 通过重放（replay）机制持续训练更新 $r_\varphi$〔转写注：原图此行上下文在快照中仅一句，$r_\varphi$ 应指奖励/判别模型参数，照录〕
- GRPO 实质上用的是**经验基线**而不是学出来的基线
  - 不再学习"预测期望奖励"，直接采样来测量它
  - 多花算力（每个提示生成 $G$ 条），少占显存（不必存价值网络）
- Dr. GRPO 修复 GRPO 的两个问题：
  - GRPO 中 $\text{std}(\mathbf r)$ 很小时（题太易或太难），奖励会被放大 → 这组变得"更重要"
    - 这个偏差高估了太易/太难的问题
  - GRPO 的奖励按生成长度归一（乘 $\frac{1}{\lvert o_i\rvert}$）
    - 在正确回答里：越短梯度越大 → 短答案被额外强化
    - 在错误回答里：越长惩罚越小 → 做不对时"写长点"反而划算

## DPO

- 带 KL 正则的目标 $\mathcal J^\text{RLHF}$ 有闭式最优解：

$$\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_\text{ref}(y\mid x)\exp\left(\frac1\beta R(y)\right)$$

  - 反解出奖励：

$$R(y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_\text{ref}(y\mid x)}+\beta\log Z(x)$$

- DPO 损失就是"观察到的偏好"的负对数似然：

$$\mathcal L^\text{DPO}(\theta)=-\mathbb E_{(x,y_w,y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_\text{ref}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_\text{ref}(y_l\mid x)}\right)\right]$$

  ——把上面的 $R$ 直接代入 Bradley-Terry 的似然，奖励模型就被"参数化为策略本身"，无需显式训练奖励模型，也无需在线生成。
