<!-- 识图转写：seg-04、seg-05（原长图 y=6600–10050，与上段重叠部分已去重），2026-11-15，来源同前 -->

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
