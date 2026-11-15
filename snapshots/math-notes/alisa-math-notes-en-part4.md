<!-- 识图转写：seg-06、seg-07（原长图 y=9900–13350，重叠部分已去重），2026-11-15，来源同前 -->

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
