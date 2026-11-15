<!-- 识图转写：seg-02、seg-03（原长图 y=3300–6750，与上段重叠部分已去重），2026-11-15，来源同前 -->

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
