<!-- 识图转写：seg-00、seg-01（原长图 y=0–3450，重叠段已去重），2026-11-15，来源 "FireShot Capture 003 - math - [alisawuffles.notion.site].pdf"（无文字层，主会话逐条识图） -->

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
