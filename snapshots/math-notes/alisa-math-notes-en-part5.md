<!-- 识图转写：seg-08、seg-09（原长图 y=13200–16650，重叠部分已去重），2026-11-15，来源同前 -->

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
