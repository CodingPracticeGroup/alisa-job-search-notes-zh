# [transcription] Alisa's Math Notes — 英文底稿（识图转写）

> 来源：`FireShot Capture 003 - math - [alisawuffles.notion.site].pdf`（原页 <https://alisawuffles.notion.site/math-notes>，2026-11-15 由项目成员整页捕获；无文字层，位图渲染）。
> 转写方式：主会话 pdftoppm 渲染 + 分条带逐屏识图（2026-11-15），分段文件 `alisa-math-notes-en-part1..6.md`。
> 说明：〔转写注〕为转录者对原稿疑误处的标注，照录不改。本底稿用于中文全文翻译，翻译仅以本文件为据。

---

# math

- Distributions
  - Discrete
    - Bernoulli
    - Binomial
    - Poisson
    - Geometric
  - Continuous
    - Uniform
    - Exponential
    - Gaussian
- Expectations & variance
- Zoo of functions
- Inequalities
- Conditional probabilities
- Limit theorems
- Some problem types
  - First-step analysis
  - Counting in expectation
  - Matrix of first RVs
- Combinatorics
- Markov chains
- Puzzles
  - Birthday problem
  - Gambler's ruin
  - Random walk
  - Coupon collector
  - Information-theoretic puzzles
  - Reservoir sampling
  - Secretary problem
  - Monty Hall
  - Longest run of heads
  - Throwing balls into bins
  - Path counting
- Statistics
  - Maximum likelihood estimation
  - Bias & variance
- Math things
  - Taylor series
  - Derivative rules
  - Handy equations

| description | concept |
| --- | --- |
| *memoryless*, *waiting* | geometric, exponential |
| trials with success & failure | Bernoulli, binomial |
| large sample, average | CLT, approximately Gaussian |
| bound given only mean / variance | Markov / Chebyshev inequalities |
| estimating parameter from data | MLE, MAP, Bayesian update |
| update belief from evidence | Bayes rule |
| function of a RV | Jensen's inequality |
| expected count of things | linearity + indicators |
| expected hitting time / return probability | first-step analysis, solve recurrences |
| Markov property (future depends only on present) | first-step analysis |

# Distributions

|  | discrete | continuous |
| --- | --- | --- |
| counting | **binomial**: how many successes in n trials? | **Poisson**: how many events in fixed window? |
| waiting | **geometric**: how many trials until first success? | **exponential**: how long until first event? |

## Discrete

### Bernoulli

- **bernoulli** RV $X\sim\mathrm{Bernoulli}(p)$ models a single trial with $p$ probability of success

$$P(X=1)=p, P(X=0)=1-p$$

  - compact PMF form

$$P(X=x)=p^{x}(1-p)^{1-x}$$

- $\mathbb E[X]=p,\ \mathrm{Var}[X]=p(1-p)$
  - proof
    - $\mathbb E[X]=1\cdot p+0\cdot(1-p)=p$
    - $\mathbb E[X^2]=1^2\cdot p+0^2\cdot(1-p)=p$
    - $\mathrm{Var}[X]=\mathbb E[X^2]-\mathbb E[X]^2=p-p^2=p(1-p)$
- **categorical** RV generalizes Bernoulli from 2 to $k$ outcomes

$$P(X=i)=p_i$$

  - each item $i$ is Bernoulli$(p_i)$, so $\mathbb E[X_i]=p_i$ and $\mathrm{Var}(X_i)=p_i(1-p_i)$
  - handy techniques
    - **indicator squaring**: for any binary variable $X\in\{0,1\}$, $X^2=X$ so $\mathbb E[X^2]=\mathbb E[X]$

### Binomial

- **binomial** RV $X\sim\mathrm{Binomial}(n,p)$ is the sum of $n$ independent Bernoulli$(p)$ trials
  - $X=\sum_{i=1}^{n}X_i$ where $X_i\sim\mathrm{Bernoulli}(p)$

$$P(X=k)=\binom{n}{k}p^{k}(1-p)^{n-k}$$

  - every sequence with $k$ successes has the same probability $p^{k}(1-p)^{n-k}$, and there are $\binom{n}{k}$ such sequences
- **Binomial theorem**

$$(a+b)^{n}=\sum_{k=0}^{n}{\binom{n}{k}}a^{k}b^{n-k}$$

- we can verify that $\sum_{k}P(X=k)=1$

$$\sum_{k=0}^{n}{\binom{n}{k}}p^{k}(1-p)^{n-k}=(p+(1-p))^{n}=1^{n}=1$$

- $\mathbb E[X]=np$, $\mathrm{Var}[X]=np(1-p)$
  - proof
    - $\mathbb E[X]=\mathbb E[\sum_{i}X_{i}]=\sum_{i}\mathbb E[X_{i}]=np$
    - $\mathrm{Var}[X]=\mathrm{Var}[\sum_{i}X_{i}]=\sum_{i}\mathrm{Var}[X_{i}]=np(1-p)$ (second equality follows because the $X_{i}$ are independent)
  - handy techniques
    - decompose into Bernoulli and use linearity of expectation
    - decompose $X^{2}$ into cross-terms

$$X^{2}=\left(\sum_{i}X_{i}\right)^{2}=\sum_{i}X_{i}^{2}+\sum_{i\neq j}X_{i}X_{j}$$

    - this makes computing expectation easy (using the fact that $X_{i}^{2}=X_{i}$ for indicator variables)

$$\mathbb E[X^{2}]=\sum_{i}\mathbb E[X_{i}]+\sum_{i\neq j}\mathbb E[X_{i}X_{j}]=n\mathbb E[X_{i}]+n(n-1)\mathbb E[X_{i}X_{j}]$$

### Poisson

- **Poisson** RV $X\sim$ Poisson$(\lambda)$ models the number of events in a fixed interval when they occur independently at an average rate of $\lambda$ over that interval

$$P(X=k)=\frac{\lambda^{k}e^{-\lambda}}{k!}$$

- the number of events in interval of length $t$ distributed as $N(t)\sim$ Poisson$(\lambda t)$
- derivation using binomial limit: divide the fixed interval into $n$ tiny time intervals, assuming it is small enough that at most one event occurs in each interval
  - then the total count is Binomial$(n,p)$ with $np=\lambda$

$$P(X-k)=\binom{n}{k}p^{k}(1-p)^{n-k}$$

$$=\frac{n!}{k!(n-k)!}\left(\frac{\lambda}{n}\right)^{k}\left(1-\frac{\lambda}{n}\right)^{n-k}$$

$$=\frac{\lambda^{k}}{k!}\underbrace{\frac{n!}{(n-k)!n^{k}}}_{\to1}\underbrace{\left(1-\frac{\lambda}{n}\right)^{n}}_{\to e^{-\lambda}}\underbrace{\left(1-\frac{\lambda}{n}\right)^{-k}}_{\to1}$$

- $\mathbb E[X]=\mathrm{Var}[X]=\lambda$
  - $\mathbb E[X]=\lambda$ is basically the definition of rate $\lambda$
- sums of independent Poisson are Poisson, with rates that add
  - can be derived using MGF uniqueness

$$X+Y\sim\mathrm{Poisson}(\lambda_{1}+\lambda_{2})$$

### Geometric

- **geometric** RV $X\sim$ Geometric$(p)$ models the number of Bernoulli trials $X$ until first success (inclusive), where each trial has $p$ probability of success

$$P(X=k)=(1-p)^{k-1}\cdot p$$

  - the tail probability is a pure power of $(1-p)$

$$P(X>k)=(1-p)^{k}$$

- **memoryless**: $\mathrm{P}(X>m+n\mid X>m)=\mathrm{P}(X>n)$
  - the fact that you've failed $m$ times already is irrelevant
  - the **geometric distribution** is the only memoryless distribution on the positive integers $\iff$ any discrete waiting-time distribution that is memoryless must be geometric
- the **geometric series** looks like $\sum_{r=0}^{\infty}r^{r}$〔转写注：原图如此；按 geometric series 惯例应为 $\sum_{r}x^{r}$，疑为原文笔误，照录〕
  - converges for $|r|<1$ to $\frac{1}{1-r}$, diverges otherwise
  - can try to remember this by checking that for $r=2$, $\sum_{i=0}^{\infty}1/2^{i}=2$
  - sanity check: PMF sums to 1

$$\sum_{k=1}^{\infty}(1-p)^{k-1}p=p\sum_{i=0}^{\infty}(1-p)^{i}$$

$$=p\cdot\frac{1}{1-(1-p)}\quad\text{sum of geometric series}$$

$$=p\cdot\frac{1}{p}$$

$$=1$$

- expected wait time $\mathbb E[X]=1/p$
  - direct solution: $X=k$ means $k-1$ failures

$$\mathbb E[X]=\sum_{k=1}^{\infty}k\cdot P(X=k)$$

  - a little tricky to solve but gives $\mathbb E[X]=1/p$
  - solution based on first-step analysis: on the first flip, we succeed with probability $p$ ($X=1$) and fail with probability $1-p$, in which case we're back to where we started ($X=1+X'$)……〔该条在条带底缘，完整内容见下一段〕

（接上：geometric 期望的 first-step analysis 解法）

$$\mathbb E[X]=p\cdot1+(1-p)(1+\mathbb E[X])$$

  - algebra then produces $\mathbb E[X]=\frac{1}{p}$
  - intuition: every trial has $p$ chance of ending the process, so on average we need $1/p$ attempts
- $\mathrm{Var}[X]=\frac{1-p}{p^{2}}$
  - as $p\to0$, both mean and variance blow up
  - rare successes have high variability in waiting time
- handy techniques
  - first-step analysis
  - re-index using $k=k-1$, turning $\sum_{k=1}^{\infty}$ to $\sum_{n=0}^{\infty}$

## Continuous

### Uniform

- **uniform** RV $X\sim\mathrm{Uniform}(a,b)$ on $[a,b]$ means $X$ is equally likely to be anywhere on the interval $[a,b]$
- $\mathbb E[X]=\frac{a+b}{2}$

$$\mathbb E[X]=\int_{a}^{b}x\cdot\frac{1}{b-a}dx$$

$$=\frac{1}{b-a}\frac{b^{2}-a^{2}}{2}\quad\text{fundamental theorem of calc}$$

$$=\frac{(b-a)(b+a)}{2(b-a)}$$

$$=\frac{a+b}{2}$$

- $\mathrm{Var}(X)=\frac{(b-a)^{2}}{12}$
- **PDF**

$$f_{X}(x)=\frac{1}{b-a}\quad\text{for }a\le x\le b$$

- **CDF**

$$F_{X}(x)=\begin{cases}0 & x<a\\ \frac{x-a}{b-a} & a\le x\le b\\ 1 & x>b\end{cases}$$

- difference between two uniform distributions is triangular
- the $n+1$ gaps between $n$ uniform points are distributed the same way

### Exponential

- **exponential** RV $X\sim\mathrm{Exp}(\lambda)$ with rate $\lambda$ models waiting time until the next event, where events happen at an average rate of $\lambda$ per unit time
  - equivalently, waiting time until first event in a Poisson process with rate $\lambda$

$$f_{X}(x)=\lambda e^{-\lambda x},\quad x\ge0$$

- tail formula (aka survival function)

$$P(X>x)=e^{-\lambda x}$$

- **memoryless**: $P(X>s+t\mid X>s)=P(X>t)$
  - the fact that you've waited for $s$ units of time already is irrelevant
  - the exponential is the only continuous distribution on $[0,\infty)$ that is memoryless
  - continuous analog to the geometric: waiting time in continuous time
- $\mathbb E[X]=\frac{1}{\lambda}$, $\mathrm{Var}(X)=\frac{1}{\lambda^{2}}$
  - if events happen at rate $\lambda$/unit time on average, then on average you need to wait $1/\lambda$ units of time between events
- relation between **Poisson and exponential**
  - two equivalent statements
    - number of events in unit interval is Poisson$(\lambda)$
    - inter-event time is Exp$(\lambda)$
  - how many events in the next hour? = Poisson
  - how long until the next event? = exponential
- minimum of exponentials: if $X_{1}\sim\mathrm{Exp}(\lambda_{1})$ and $X_{2}\sim\mathrm{Exp}(\lambda_{2})$ are independent, then

$$\min(X_{1},X_{2})\sim\operatorname{Exp}(\lambda_{1}+\lambda_{2})$$

  - intuition: if emails arrive at rate $\lambda_{1}$ and texts at rate $\lambda_{2}$, then the waiting time for "either an email or a text" is exponential with rate $\lambda_{1}+\lambda_{2}$
  - proof

$$P(\min(X_{1},X_{2})>t)=P(X_{1}>t\text{ and }X_{2}>t)$$

$$=P(X_{1}>t)P(X_{2}>t)\quad\text{independence}$$

$$=e^{-\lambda_{1}t}e^{-\lambda_{2}t}$$

$$=e^{-(\lambda_{1}+\lambda_{2})t}$$

  - which is the tail function of $\mathrm{Exp}(\lambda_{1}+\lambda_{2})$

### Gaussian

- **Gaussian** RV $X\sim\mathcal{N}(\mu,\sigma^{2})$ with mean $\mu$ and variance $\sigma^{2}$
- relationship between general and standard normal RV
  - subtract the mean to shift to 0, divide by standard dev to rescale to unit variance

$$Z=\frac{X-\mu}{\sigma}\sim\mathcal{N}(0,1)$$

  - example: if heights are normally distributed with $\mu=70$ and $\sigma=3$, what fraction of people are taller than 76 inches?

$$X\sim\mathcal{N}(70,9)\implies Z=\frac{X-70}{3}\sim\mathcal{N}(0,1)$$

$$P(X>76)=P(Z>2)$$

- the **percentile** $z_{\alpha}$ is the value such that

$$P(Z\leq z_{\alpha})=\alpha\iff P(Z>z_{\alpha})=1-\alpha$$

  - i.e., $z_{\alpha}$ is the point on the axis with probability mass $\alpha$ to the left
  - symmetry: $z_{1-\alpha}=-z_{\alpha}$
  - 2-sided intervals: $P(|Z|\leq z_{1-\alpha/2})=1-\alpha\iff P(|Z|>z_{1-\alpha/2})=\alpha$
    - a $(1-\alpha)$ confidence interval uses $z_{\alpha/2}$ and $z_{1-\alpha/2}$ as endpoints
  - stated as bounds: $P(|Z|\leq c)\geq1-\alpha\iff c\geq z_{1-\alpha/2}$
  - this also means that for $X\sim\mathcal{N}(\mu,\sigma^{2})$, the $\alpha$-percentile is $x_{\alpha}=\mu+\sigma\cdot z_{\alpha}$

$$P(X\leq x_{\alpha})=P\left(Z\leq\frac{x_{\alpha}-\mu}{\sigma}\right)=\alpha\implies z_{\alpha}=\frac{x_{\alpha}-\mu}{\sigma}\implies x_{\alpha}=\mu+\sigma\cdot z_{\alpha}$$

- sum of independent Gaussians is Gaussian: if $X_{1}\sim\mathcal{N}(\mu_{1},\sigma_{1}^{2})$ and $X_{2}\sim\mathcal{N}(\mu_{2},\sigma_{2}^{2})$, then

$$X_{1}+X_{2}\sim\mathcal{N}(\mu_{1}+\mu_{2},\sigma_{1}^{2}+\sigma_{2}^{2})$$

# Expectations & variance

- **expectation**

$$\mathbb E[X]=\sum_{x}xP(X=x)\quad\text{or}\quad\int_{-\infty}^{\infty}xf(x)dx$$

  - expectation of a function over $X$

$$\mathbb E[g(x)]=\sum_{x}g(x)P(X=x)\quad\text{or}\quad\int_{-\infty}^{\infty}g(x)f(x)dx$$

  - tail formula for non-negative RVs

$$\mathbb E[X]=\int_{0}^{\infty}P(X>x)dx=\int_{0}^{\infty}1-F(x)dx$$

- **variance**

$$\mathrm{Var}(X)=\mathbb E[(X-\mathbb E[X])^{2}]$$

$$=\mathbb E[X^{2}]-(\mathbb E[X])^{2}$$

- **linearity of expectation** (regardless of independence)

$$\mathbb E[aX+bY+c]=a\mathbb E[X]+b\mathbb E[Y]+c$$

- for **independent** $X$ and $Y$

$$\mathbb E[XY]=\mathbb E[X]\mathbb E[Y]$$

- if $X$ and $Y$ are independent, then

$$\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)$$

# Zoo of functions

- **probability mass function** (PMF) of discrete RVs

$$p(k)=P(X=k)$$

  - properties
    - $p(k)\geq0$
    - $\sum_{k}p(k)=1$
- **probability density function** (PDF) of continuous RVs

$$f(x)\text{ such that }P(a\le x\le b)=\int_{a}^{b}f(x)dx$$

  - differentiate CDF to get PDF: $f(x)=F'(x)$
- **cumulative distribution function** (CDF) for both discrete and continuous RVs

$$F(x)=P(X\leq x)$$

  - for discrete: $F(x)=\sum_{k\leq x}p(k)$
  - for continuous: $F(x)=\int_{-\infty}^{x}f(x)dx$
  - note $P(a\le x\le b)=F(b)-F(a)$

# Inequalities

- **Markov's inequality**: for any non-negative RV $X$ and $a>0$,

$$P(X\geq a)\leq{\frac{\mathbb E[X]}{a}}$$

  - intuitively, $\mathbb E[X]$ cannot be smaller than the contribution of values greater than $a$
  - proof

$$\mathbb E[X]\geq\sum_{x\geq a}xP(X=x)\geq a\sum_{x\geq a}P(X=x)=aP(X\geq a)$$

- **Chebyshev's inequality**: for any RV $X$ with mean $\mu$ and variance $\sigma^{2}$

$$P(|X-\mu|\geq k\sigma)\leq{\frac{1}{k^{2}}}$$

  - at least $1-1/k^{2}$ of the distribution lies within $k$ standard deviations of the mean
    - e.g., 75% of probability mass sits within 2 standard deviations
  - proof: apply Markov to $(X-\mu)^{2}$

（接上：Chebyshev 不等式的证明）

$$P(|X-\mu|\geq k\sigma)=P((X-\mu)^2\geq k^2\sigma^2)\leq{\frac{\mathbb E[(X-\mu)^2]}{k^2\sigma^2}}={\frac{\sigma^2}{k^2\sigma^2}}={\frac{1}{k^2}}$$

  - another form, by substituting $\epsilon=k\sigma$

$$P(|X-\mu|\geq\epsilon)\leq{\frac{\sigma^2}{\epsilon^2}}$$

- **Jensen's inequality** describes how expectations interact with convex/concave transformations

$$\varphi\ \text{convex}\implies\varphi[\mathbb E[X]]\leq\mathbb E[\varphi(X)]$$

$$\varphi\ \text{concave}\implies\varphi[\mathbb E[X]]\geq\mathbb E[\varphi(X)]$$

  - to remember, use $\varphi(x)=x^2$ (convex) with $X\in\{-1,1\}$
    - $\varphi(\mathbb E[X])=\varphi(0)=0$
    - $\mathbb E[\varphi(X)]=1$
  - convex functions have $\varphi''\geq0$ everywhere (slope is increasing)
    - $x^2$, $x^4$, … ($x^n$ for any even $n$)
    - $|x|$
    - $a^x$ (for $a>1$)
    - $1/x$ (for $x>0$)
    - $\max(x_1,x_2,\dots)$
  - concave functions have $\varphi''\leq0$ everywhere (slope is decreasing)
    - $\sqrt{x}$
    - $\log x$
    - $x^p$ for $0<p<1$
    - $\min(x_1,x_2,\dots)$
  - linear functions are both convex and concave
  - worth memorizing: $\log\mathbb E[X]\geq\mathbb E[\log X]$
- **AM-GM inequality** (arithmetic mean, geometric mean): arithmetic mean always $\geq$ geometric mean

$${\frac{a_1+\dots+a_n}{n}}\leq\sqrt[n]{a_1\dotsb a_n}$$

〔转写注：原图不等号如此（AM $\leq$ GM），按 AM-GM 惯例应为 $\geq$，疑为原稿笔误，照录。〕

  - proof: take log of both sides, apply Jensen's inequality (using the fact that log is concave), then exponentiate both sides
- **Cauchy-Schwarz inequality**

$$\mathbb E[XY]^2\leq\mathbb E[X^2]\mathbb E[Y^2]$$

  - another form

$$\left(\sum_{i=1}^{n}x_iy_i\right)^2\leq\left(\sum_{i=1}^{n}x_i^2\right)\left(\sum_{i=1}^{n}y_i^2\right)$$

- **union bound**

$$P(A_1\cup A_2\cup\dots\cup A_n)\leq P(A_1)+P(A_2)+\dots+P(A_n)$$

# Conditional probabilities

- suppose we have two random variables, $X$ and $Y$
  - **joint distribution**: $P(X=x,Y=y)$
  - **marginal distribution**: $P(X=x)=\sum_y P(X=x,Y=y)$
  - **conditional distribution**: $P(X=x\mid Y=y)$
- **Bayes rule**

$$P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}$$

  - $P[A]$ is the **prior** (what we believed before seeing $B$)
  - $P[A\mid B]$ is the **posterior** (what we believed after seeing $B$)
  - calculate $P[B]$ with law of total probability
- **law of total expectation** (also known as the **tower property**)

$$\mathbb E[X]=\sum_y\mathbb E[X\mid Y=y]\cdot P(Y=y)\quad\text{or}\quad\int_{-\infty}^{\infty}\mathbb E[X\mid Y=y]f(y)dy$$

  - can also be stated as $\mathbb E[X]=\mathbb E_Y[\mathbb E[X\mid Y]]$
    - note that in this statement, $\mathbb E[X\mid Y]$ is itself *RV* expressed in terms of $Y$
- **law of total probability**

$$P(A)=\sum_y P(A\mid Y=y)\cdot P(Y=y)$$

# Limit theorems

- **central limit theorem**: mean of $n$ iid RVs $X_i$ with mean $\mu$ and variance $\sigma^2$ is Gaussian, no matter what the $X_i$ look like

$$\bar X_n={\frac{1}{n}}\sum_{i=1}^{n}X_i\approx{\mathcal{N}}\left(\mu,{\frac{\sigma^2}{n}}\right)$$

  - derivation of variance

$$\text{Var}(\bar X_n)=\text{Var}\left({\frac{1}{n}}\sum_{i=1}^{n}X_i\right)={\frac{1}{n^2}}\text{Var}\left(\sum_{i=1}^{n}X_i\right)={\frac{1}{n^2}}\sum_{i=1}^{n}\text{Var}(X_i)={\frac{n\cdot\sigma^2}{n^2}}={\frac{\sigma^2}{n}}$$

  - **standard error** is the standard deviation of $\bar X_n$, which is $\sigma/\sqrt{n}$
    - what it means: error shrinks like $\sqrt n$, need $100$x samples to reduce error $10$x
  - $(1-\alpha)$ confidence interval is given by

$$\bar X_n\pm z_{1-\alpha/2}\cdot{\frac{\sigma}{\sqrt{n}}}$$

- **law of large numbers**: mean of $n$ iid RVs $X_i$ with mean $\mu$ converges to the true average

$$P(|\overline{X}_n-\mu|>\epsilon)\to0\quad\text{for any }\epsilon>0$$

  - proof from Chebyshev's (substitute $\epsilon=k\sigma$)

$$P(|{\overline{X}}_{n}-\mu|>\epsilon)\leq{\frac{\text{Var}({\overline{X}}_{n})}{\epsilon^{2}}}={\frac{\sigma^{2}}{n\epsilon^{2}}}\to0$$

# Some problem types

## First-step analysis

- what's the expected number of coin flips to get two heads in a row?
  - let
    - $E_0$ = expected flips to finish after just flipping tails (or just started)
    - $E_1$ = expected flips after just flipping heads

$$E_0=\underbrace{\frac{1}{2}(1+E_0)}_{\text{tails, start again}}+\underbrace{\frac{1}{2}(1+E_1)}_{\text{heads}}$$

$$E_1=\underbrace{\frac{1}{2}(1+E_0)}_{\text{tails, start over}}+\underbrace{\frac{1}{2}\,0}_{\text{heads again}}$$

  - then solve system of linear equations to get $E_0=6$

> ✅ First-step analysis trick: focus on the first step of the process, and express the expectations we want to solve for in terms of themselves, then solve system of linear equations.

## Counting in expectation

- how many things are expected to satisfy some property?
  - define $X_i=\mathbb 1[\text{item i has property}]$
  - define a sum of indicator random variables $N=\sum_i X_i$
  - compute $\mathbb E[X_i]=P(X_i)$ for each $i$ (use symmetry)
  - apply linearity of expectation: $\mathbb E[N]=\sum_i P(X_i)$
- examples: expected number of **fixed points** in a random permutation
  - by symmetry, position $i$ equally likely to have any item, so $P(X_i=1)=1/n$

## Max/min of $n$ RVs

- if $X_1,\ldots,X_n$ are iid with common CDF $F_X$
  - for the max $M=\max_i X_i$

$$F_M(x)=P(M\leq x)=P(\text{all }X_i\leq x)=[F_X(x)]^n$$

  - for the min $m=\min_i X_i$

$$P(m>x)=P(\text{every }X_i>m)=[1-F_X(x)]^n$$

$$F_m(x)=1-P(m>x)$$

- two different approaches to calculate expectation of $\min$ or $\max$
  - 3-step recipe: CDF → PDF → integrate
  - for non-negative RVs, can use tail formula directly $\mathbb E[X]=\int_0^{\infty}P(X>x)dx$
- let's do this for $X_i\sim$Uniform$[0,1]$
  - $\max M=\max_i X_i$
    - CDF: $F_M(x)=\prod_i P(X_i\leq x)=x^n$
    - PDF: $f_M(x)=\frac{d}{dx}x^n=nx^{n-1}$
    - expectation: $\mathbb E[M]=\int_0^1 x\cdot nx^{n-1}dx=n\int_0^1 x^n=\frac{n}{n+1}$
    - intuition: $n$ uniform points $X_i$ split the segment into $n+1$ segments which are all equivalently distributed (by symmetry), so each must have mean length $1/(n+1)$
    - alternatively with tail formula: $\mathbb E[M]=\int_0^1(1-x^n)dx=x-\frac{x^{n+1}}{n+1}\Big|_0^1=\frac{n}{n+1}$
  - $\min m=\min_i X_i$
    - $P(m>x)=(1-x)^n$
    - CDF: $F_m(x)=P(m\leq x)=1-P(m\geq x)=1-(1-x)^n$
    - PDF: $f_m(x)=\frac{d}{dx}F_m(x)=n(1-x)^{n-1}$
    - expectation: $\mathbb E[m]=\int_0^1 x\cdot n(1-x)^{n-1}dx=\frac{1}{n+1}$
- for $X_i\sim$Exponential$(\lambda)$
  - min
    - $P(m>x)=[P(X_i>x)]^n=e^{-n\lambda x}$
    - $\mathbb E[m]=\int_0^{\infty}e^{-n\lambda x}dx=\frac{1}{n\lambda}$
  - max is a lot more complicated, but good to know that it grows logarithmically with $n$
- for $X_i\sim\mathcal N(0,1)$, $\mathbb E[\max_i X_i]\approx\sqrt{2\ln n}$ (this is pretty difficult to derive)

# Combinatorics

- **combination**: choosing $k$ items from $n$, order doesn't matter

$$\binom{n}{k}=\frac{n!}{k!(n-k)!}=\frac{P(n,k)}{k!}$$

  - the relation to permutation: every set of $k$ items can be arranged in $k!$ orders, and combinations collapse the orderings into one
- **permutation**: arranging $k$ items from $n$, order matters

$$P(n,k)=\frac{n!}{(n-k)!}$$

- number of ways to arrange $n$ objects when there are $n_1$ of type 1, $n_2$ of type 2, …

$$\binom{n}{n_1,\dots,n_m}=\frac{n!}{n_1!\dotsb n_m!}$$

  - permute all $n$ objects, then divide out the rearrangements of items within each type

（接上：组合数学）

- a **derangement** is a permutation with no fixed points
- **combination with replacement**: choosing $k$ items from $n$ types with replacement

$$\binom{n+k-1}{k}$$

  - this is the **stars and bars** argument: each choice can be represented as a sequence of stars and bars, where there are $k$ stars and $n-1$ bars – this is choosing $k$ placements of stars from $n+k-1$ positions
- how many ways are there to write non-negative integer $n$ has sum of $k$ non-negative integers?
  - visualize $n$ stars with $k-1$ bars among them

| order matters? | replacement? | count |
| --- | --- | --- |
| yes | no | $\frac{n!}{(n-k)!}$ (permutations) |
| no | no | $\binom{n}{k}$ (combinations) |
| yes | yes | $n^k$ |
| no | yes | $\binom{n+k-1}{k}$ (stars and bars) |

- **inclusion-exclusion** principle

$$P(A\cup B)=P(A)+P(B)-P(A\cap B)$$

$$P(A\cup B\cup C)=P(A)+P(B)+P(C)-P(A\cap B)-P(B\cap C)-P(A\cap B\cap C)$$

  - as counts

$$|A\cup B\cup C|=|A|+|B|+|C|-|A\cap B|-|A\cap C|-|B\cap C|+|A\cap B\cap C|$$

# Markov chains

- **Markov chain** is a sequence of RVs $X_0,X_1,X_2,\dots$ taking values in some state space $S$ satisfying the **Markov property**: the future depends only on the present
- for finite state chains, the chain is described by the **transition matrix** $P$ where $P_{ij}=P(X_{t+1}=j\mid X_t=i)$
  - rows sum to 1 and are probability distributions over next-states
- if $\pi_t$ is the distribution over states at time $t$, then $\pi_{t+1}=\pi_t P$
- a **stationary distribution** satisfies $\pi=\pi P$
  - under mild conditions, the chain has a unique stationary distribution and $\pi_t\to\pi$ regardless of where we started
- the expected return time is the **reciprocal** of its stationary probability

$$\mathbb E[\text{return to state }i]=\frac{1}{\pi_i}$$

- general principle to solve for expected absorption time
  - use first-step analysis to express the recurrence
    - expected time $h_j$ to absorption from state $j$
    - $h_A=0$ for target absorbing state $A$

$$h_i=1+\sum_j P_{ij}h_j$$

    - probability of absorption $q_j$ from state $j$
    - $q_A=1$ for desired absorbing state $A$, and $q_B=0$ for all other absorbing states $B$

$$q_i=\sum_j P_{ij}q_j$$

  - solve recurrence
    - for small chains, can just solve the linear system
    - for chains with structure, guess a closed form and verify
      - to make an educated guess, compute the answers for small cases and try to generalize
      - think about what makes sense in terms of function shape
    - fit boundary conditions
- handy techniques
  - symmetry can allow you to collapse equivalent states
  - for a random walk on a circle, easier to reparameterize $h_i$ to be expected hitting time from a particular distance rather than from a particular state
  - use **stationary distribution** to provide shortcuts for expected hitting time
    - on a graph where every node looks the same (cycles, complete graphs, etc.), expected hitting time $h$ between adjacent vertices is always $n-1$ where $n$ is the number of nodes
      - this is because $\mathbb E[\text{return to self}]=n$ (since stationary distribution has $1/n$ everywhere)
      - and $\mathbb E[\text{return to self}]=1+h$ because each of the $d$ adjacent nodes contributes $\frac{1}{d}h$

# Puzzles

## Birthday problem

- how many people $n$ does it take so that the probability of two people having the same birthday exceeds 50%?
- $P(\text{collision})=1-P(\text{all unique birthdays})$
- let $d$ be number of days in the year and $n$ the number of people

$$P(\text{all unique birthdays})=\frac{d}{d}\cdot\frac{d-1}{d}\cdot\frac{d-2}{d}\cdots\frac{d-n+1}{d}=\frac{d!}{(d-n)!\cdot d^n}$$

## Gambler's ruin

- you start with $\$a$ dollars
  - win $\$1$ with probability $p$ and lose $\$1$ with probability $1-p$
  - stop when you reach $\$N$ (you win!) or $\$0$ (you're ruined)
  - probability you reach $\$N$?
- let $P_i$ be probability of reaching $N$ starting from $i$
  - boundary conditions: $P_0=0, P_N=1$
- for any starting amount $0<i<N$,

$$q_i=p\cdot q_{i+1}+(1-p)\cdot q_{i-1}$$

  - second-order linear recurrence
- when the game is fair, the probability of hitting $N$ when starting at $a$ is simply $a/N$
  - approach 1: express expected value at stopping time in two different ways
    - let $T$ be the stop time, and $W_T$ be the wealth at time $T$
      - we know $\mathbb E[W_T]=a$ (no drift)
      - we also know $W_T$ is either $0$ or $N$, so

$$\mathbb E[W_T]=q_a\cdot N+(1-q_a)\cdot0=q_a N$$

    - combining these, we get $q_a=a/N$
  - approach 2: solve recurrence

$$q_i=\frac{1}{2}q_i+\frac{1}{2}q_{i-1}\implies q_{i+1}-q_i=q_i-q_{i-1}$$

    - all consecutive differences are equal → $q$ is a linear function of $i$
    - with $q_0=0$ and $q_N=1$, we get $q_i=i/N$
  - this is a random walk with absorbing barriers at $0$ and $N$
- in general, a **martingale** is a sequence of RVs $M_0,\ldots,M_n$ such that $\mathbb E[M_i]=\mathbb E[M_n]$
- the expected absorption time $h_a$ is $a(N-a)$
  - this is tricky to derive, but easy to verify by making sure it satisfies the recursive relationships given by first-step analysis

## Random walk

- simple symmetric **random walk**: let $S_n=X_1+\dots+X_n$ where each $X_i=\pm1$ with equal probability
  - obviously $\mathbb E[S_n]=\mathbb E[X_i]=0$
  - each $X_i=\pm1$ has variance $\mathbb E[X_i^2]-(\mathbb E[X_i])^2=1-0=1$

$$\text{Var}(S_n)=\sum_{i=1}^{n}\text{Var}(X_i)=n$$

  - typical distance from origin (generally measured by standard deviation) grows like $\sqrt n$
- in the biased case ($X_i=1$ with $p$, else $X_i=-1$)

$$\mathbb E[S_n]=n(p\cdot1+(1-p)\cdot(-1))=n(2p-1)$$

## Coupon collector

- $n$ different types of coupons. every time you buy a cereal box, you get one coupon uniformly at random. in expectation, how much time does it take to collect all $n$ coupons?
  - each phase $X_k\sim$Geom$((n-k)/n)$
  - the expected length of phase $k$ is

$$\mathbb E[X_k]=\frac{n}{n-k}$$

  - total expected time

$$\mathbb E[X]=\sum_{k=0}^{n-1}\mathbb E[X_k]=\sum_{k=0}^{n-1}\frac{n}{n-k}=n\sum_{j=1}^{n}\frac{1}{j}=n\cdot H_n$$

    - where $H_n=1+1/2+1/3+\dots+1/n$ is the $n$-th **harmonic number**
  - final phase takes $n$ boxes in expectation!

## Information-theoretic puzzles

- count the information
- if we need to distinguish $k$ possibilities and each test has $m$ outcomes, then we need at least $\lceil\log_m k\rceil$ tests
  - this is a lower bound
  - whether tests exist to achieve this bound depends on the constraints of the problem
- 9 coins, one is heavier, 2 weighings
  - divide into 3 groups of 3
    - left side heavy → counterfeit on left side
    - right side heavy → counterfeit on right side
    - balanced → counterfeit in the left-out set
  - then weigh 2 of the remaining 3

## Reservoir sampling

- we receive a stream of items: $x_1,x_2,x_3,\dots$ at the end of the stream, return a single item uniformly at random from all the items seen (length $n$ of stream is not known ahead of time)
- the algorithm
  - keep the first item: $s=x_1$
  - for $n=2,3,4,\dots$
    - with probability $1/n$, replace $s$ with $x_n$
- proof of correctness by induction
  - claim: after processing $n$ items, each of $x_1,\dots,x_n$ is stored in $s$ with probability $1/n$
    - base case ($n=1$): trivially, $s=x_1$ with probability $1=1/1$
    - inductive step: strict inductive hypothesis: our last item was chosen uniformly
      - $s=x_n$ (replace): this happens with probability $1/n$
      - $s=x_k$ for some $k<n$ (keep): this requires (1) $s=x_k$ after step $n-1$, which has probability $1/(n-1)$ by induction, and that (2) we rejected $x_n$, which has probability $(n-1)/n$

$$\frac{1}{n-1}\cdot\frac{n-1}{n}=\frac{1}{n}$$

  - thus every $x_k$ is in the reservoir with probability $1/n$ after $n$ steps

## Secretary problem

- problem: $n$ candidates interleaved in random order, after each interview, decide immediately to hire or reject that candidate. our goal is to maximize probability of hiring the best candidate. we use the strategy: reject the first $k$ candidates, then hire the next candidate better than all the first $k$ (if no such candidate appears, hire the last one). what is the optimal $k$?
- let $P(k)$ = probability that strategy with threshold $k$ hires the best candidate, let $b$ be the position of the best candidate
  - for the strategy to succeed:
    1. $b>k$: best candidate appears after the first $k$
    2. the best candidate among positions $1,\dots,b-1$ is in $1,\dots,k$ (otherwise we hire someone before getting to $b$)

（接上：Secretary problem 求解）

$$P(k)=\sum_{b=k+1}^{n}P(\text{best candidate at }b)P(\text{best among }1,...,b-1\text{ appears}\leq k)$$

$$=\sum_{b=k+1}^{n}\frac{1}{n}\frac{k}{b-1}$$

$$=\frac{k}{n}\sum_{b=k+1}^{n}\frac{1}{b-1}$$

$$=\frac{k}{n}\sum_{m=k}^{n-1}\frac{1}{m}$$

$$=\frac{k}{n}(H_{n-1}-H_{k-1})$$

$$\approx\frac{k}{n}(\log(n-1)-\log(k-1))$$

$$=\frac{k}{n}\log\frac{n}{k}$$

$$=r\log\frac{1}{r}$$

$$=-r\log r$$

（右侧边注：$H_m\approx\log m$ for large $n$；换元处标注 $\text{let }r=\frac{k}{n}$）

  - taking the derivative, we see $-\frac{d}{dr}r\log r=-\log r-1$
    - setting this to 0, we get $\log r=-1\implies r=\frac{1}{e}\implies k=\frac{n}{e}$

## Monty Hall

- let $A=\{1,2,\dots,n\}$ represent the car's location and $B$ be the set of doors Monty opens
- we pick door $c$ (without loss of generality)
- we want $P(A=j\mid B)$ for each unopened door $j$
- Bayes Theorem: $P(A=j\mid B)=\frac{P(B\mid A=j)P(A=j)}{P(B)}$
  - comparing $P(A=j\mid B)$ for different $j$ reduces to comparing $P(B\mid A=j)$, because the prior $P(A=j)=1/n$ is the same for all $j$ and the denominator is shared
- case 1: Monty opens $n-2$ doors, leaving our door and one other
  - denote with index $r$ the one remaining door
  - $P(B\mid A=r)=1/(n-1)$ because when your chosen door has the car, there are $n-1$ choices (i.e., $n-1$ choose $n-2$) for which door to leave closed
  - $P(B\mid A=r)=1$ because Monty is completely constrained in which set of doors he can open
  - $P(B\mid A=k)=0$ for all opened doors $k\in B$
  - so the posteriors $P(A=r\mid B)\propto 1/(n-1)$, $P(A=r\mid B)\propto 1$
  - normalizing gives $P(A=r\mid B)=1/n$ and $P(A=r\mid B)=(n-1)/n$
- case 2: Monty opens only 1 door
  - let $R$ denote an arbitrary one of the remaining doors
  - $P(B\mid A=t)=1/(n-1)$ because there are $n-1$ choices for which door to open
  - $P(B\mid A=r)=1/(n-2)$ because there are $n-2$ choices for which door to open (every door except door $c$ and door $A$ is an option)
  - $P(B\mid A=k)=0$ where $R=\{k\}$
  - so the posteriors $P(A=t\mid B)\propto 1/(n-1)$ and $P(A=r\mid B)\propto 1/(n-2)$
  - so the remaining doors are each slightly more likely
- in general, the sum of probabilities of opened doors ($k/n$ where $k$ is the number of opened doors) is distributed onto the remaining unopened, unchosen doors (of which there are $n-k-1$)

## Longest run of heads

- we flip a fair coin $n$ times and let $L_n$ denote the longest run of consecutive heads
- there is no clean closed form for $\mathbb E[L_n]$, but a well-known approximation is $\mathbb E[L_n]\approx\log_2(n)$

## Throwing balls into bins

- if we throw $n$ balls into $m$ bins uniformly, about 37% are empty, 37% get one ball, and the rest get two or more

## Path counting

- how many paths from (0,0) to (m,n) on a grid, where each step moves one unit right or one unit up?
  - every path consists of exactly $m$ right-steps and $n$ up-steps
  - a path is uniquely specified by choosing which of the $m+n$ steps are right-steps
  - answer is $\binom{m+n}{m}$
- how many of these paths pass through (a,b)?
  - multiply number of unique paths to (a,b) and number of unique paths from (a,b) to (m,n)

$$\binom{a+b}{a}\cdot\binom{(m-a)+(n-b)}{m-a}$$

# Statistics

## Maximum likelihood estimation

- given data $x_1,\dots,x_n$ drawn from some distribution with unknown parameter $\theta$
- **maximum likelihood estimation**: pick the $\hat\theta$ that makes the observed data most probable

$$L(\theta)=\prod_{i=1}^{n}p(x_i;\theta)$$

$$\hat\theta_\text{MLE}=\arg\max_\theta L(\theta)$$

- log-likelihood trick: take log to turn product into sum

$$\ell(\theta)=\log L(\theta)=\sum_{i=1}^{n}\log p(x_i;\theta)$$

- differentiate with respect to $\theta$, set to zero, and solve for $\hat\theta$
  - check that it's a max and not a min by checking second derivative of endpoints
- **example**: $x_1,\dots,x_n$ are iid from Bernoulli($p$)
  - PMF: $P(X=x;p)=p^x(1-p)^{1-x}$
  - log likelihood

$$\ell(\lambda)=\sum_i\log p^x(1-p)^{1-x}=\sum_i x\log p+(1-x)\log(1-p)$$

    - let $k$ denote the number of 1s

$$\ell(\lambda)=k\log p+(n-k)\log(1-p)$$

  - derivative

$$\ell'(\lambda)=\frac{k}{p}-\frac{n-k}{1-p}=0\implies p=\frac{k}{n}$$

    - MLE is the sample frequency
- **example**: $x_1,\dots,x_n$ are iid from Exp($\lambda$)
  - PDF: $f(x;\lambda)=\lambda e^{-\lambda x}$
  - log likelihood

$$\ell(\lambda)=\sum_i\log[\lambda e^{-\lambda x_i}]=\sum_i\log\lambda-\lambda x_i=n\log\lambda-\lambda\sum_i x_i$$

  - derivative

$$\ell'(\lambda)=\frac{n}{\lambda}-\sum_i x_i=0\implies\hat\lambda_\text{MLE}=\frac{n}{\sum_i x_i}$$

    - MLE is the reciprocal of the sample mean
      - this makes sense because Exp($\lambda$) has mean $1/\lambda$
- **example**: $x_1,\dots,x_n$ are iid from Uniform($0,\theta$)
  - PDF: $f(x;\theta)=1/\theta$
  - likelihood

$$L(\theta)=\begin{cases}\frac{1}{\theta^n}&\text{if }\max x_i\leq\theta\\ 0&\text{otherwise}\end{cases}$$

    - this is maximized for the smallest value of $\theta$ that satisfies $\theta\geq\max x_i$ so $\hat\theta=\max x_i$
- models are trained to minimize CE loss $=-\frac{1}{N}\sum_i\log p(x_i)$ which is the same as maximizing log likelihood $=\sum_i\log p(x_i)$
  - in other words, models are trained to make observed data as probable as possible

## Bias & variance

- true parameter $\theta$, estimator $\hat\theta$
  - the estimator is a function of the data, and since the data is random $\to\hat\theta$ is an RV
- **bias**: how close is $\hat\theta$ to $\theta$ on average?

$$\text{bias}(\hat{\theta})=\mathbb{E}[\hat{\theta}]-\theta$$

- **variance**: how much does $\hat{\theta}$ depend on the data distribution?

$$\text{var}(\hat{\theta})=\mathbb{E}[(\hat{\theta}-\mathbb{E}[\hat{\theta}])^2]$$

- **mean squared error**

$$\text{MSE}(\hat{\theta})=\mathbb{E}[(\hat{\theta}-\theta)^2]$$

  - decomposes into bias^2 + variance

$$\text{MSE}(\hat{\theta})=\text{Bias}(\hat{\theta})^2+\text{Var}(\hat{\theta})$$

# Math things

## Taylor series

- a **Taylor series** approximates a function using a polynomial, using only information about the function at a single point

$$f(x)\sim\sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(x-a)^{n}$$

- a series around $a=0$ is called the **Maclaurin series**
- the Taylor series of $e^x$ around $x=0$ is

$$e^{x}=1+x+\frac{x^{2}}{2!}+\frac{x^{3}}{3!}+\dots=\sum_{n=0}^{\infty}\frac{x^{n}}{n!}$$

  - has infinite radius of convergence, so it converges to $e^x$ everywhere
- suppose we want to approximate $f(x)$ near $x=0$
  - 0th order approximation: $f(x)\approx f(0)$
    - horizontal line at $f(0)$
  - 1st order approximation: $f(x)\approx f(0)+f'(0)x$
    - tangent line to $f$ at $x=0$, matching $f$ in both value and slope at the origin
  - 2nd order approximation: $f(x)\approx f(0)+f'(0)x+\frac{f''(0)}{2!}x^{2}$
    - parabola that matches $f$ in value, slope, and curvature at 0
  - each new term adds another derivative of information, letting the polynomial hug the function more tightly near $x=0$

## Derivative rules

- memorize these

$$\frac{d}{dx}x^{n}=nx^{n-1}$$

$$\frac{d}{dx}e^{x}=e^{x}$$

$$\frac{d}{dx}\log x=\frac{1}{x}$$

$$\frac{d}{dx}\sin x=\cos x$$

$$\frac{d}{dx}\cos x=-\sin x$$

- composition rules

$$\frac{d}{dx}f(g(x))=f'(g(x))\cdot g'(x)\qquad\text{chain rule}$$

$$\frac{d}{dx}f(x)g(x)=f'(x)g(x)+f(x)g'(x)\qquad\text{product rule}$$

（接上：composition rules 余下）

$$\frac{d}{dx}\frac{f(x)}{g(x)}=\frac{f'(x)g(x)-f(x)g'(x)}{g(x)^2}\qquad\text{quotient rule}$$

## Handy equations

- the $n$-th harmonic number $H_n=\sum_{k=1}^{n}\frac{1}{k}\approx\log n$ for large $n$
- log mean $\sqrt{xy}\leq\frac{x+y}{2}$ for $x,y>0$
- some identities for $e$

$$e=\lim_{n\to\infty}\left(1+\frac{1}{n}\right)^{n}$$

  - replace $n$ with $-n$

$$\frac{1}{e}=\lim_{n\to\infty}\left(1-\frac{1}{n}\right)^{n}$$

  - replace $n$ with $n/x$

$$e^{x}=\lim_{n\to\infty}\left(1+\frac{x}{n}\right)^{n}$$

- exponential series

$$e=\sum_{k=0}^{\infty}\frac{1}{k!}\qquad e^{x}=\sum_{n=0}^{\infty}\frac{x^{n}}{n!}\qquad e^{-1}=1-\frac{1}{1!}+\frac{1}{2!}-\frac{1}{3!}+\dots$$

〔快照尾注：FireShot 捕获在"exponential series"一行后截止。原页面此节后续（如有）未包含在本快照内；如日后补全，将作为新增快照追加，不改写本节。〕
