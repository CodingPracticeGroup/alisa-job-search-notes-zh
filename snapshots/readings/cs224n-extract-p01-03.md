<!-- 识图转写：第 1-3 页，2026-11-15，来源 cs224n-self-attention-transformers-2023-draft.pdf，转写方式：主会话逐页识图 -->

# [draft] Note 10: Self-Attention & Transformers

> 边注脚标：¹ Course Instructors: Christopher Manning, John Hewitt  ² Author: John Hewitt, johnhew@cs.stanford.edu

CS 224n: Natural Language Processing with Deep Learning

*Winter 2023*

【图（第 1 页，右侧栏）：Transformer Encoder-Decoder 结构图（纵向流程图，箭头自下而上）。左侧编码器等柱：`Encoder Inputs`（蓝）→ `Embeddings`（蓝）→ `Add Positional Embeddings`（绿）→ 进入 Block 框：`Multi-Head Attention`（红）→ `Add & Norm`（黄）→ `Feed-Forward`（青）→ `Add & Norm`（黄）；Block 旁标注 "Repeat: for number of encoder blocks"，并有一条长线从编码器输出绕到右侧解码器块的 Attention 输入。右侧解码器等柱（底向上）：`Decoder Inputs`（蓝）→ `Embeddings`（蓝）→ `Add Positional Embeddings`（绿）→ Block 框：`Masked Multi-Head Attention`（红）→ `Add & Norm`（黄）→ `Multi-Head Attention`（红，接收来自编码器输出的箭头，旁注 "Repeat: for number of decoder blocks"）→ `Add & Norm`（黄）→ `Feed-Forward`（青）→ `Add & Norm`（黄）→ `Linear`（灰）→ `Softmax`（灰）→ `Probabilities`；Masked 块旁注 "Attend only to output of last Encoder Block"。图题：*Transformer Encoder-Decoder*】

**Summary.**  This note motivates moving away from recurrent architectures in NLP, introduces self-attention, and builds a minimal self-attention-based neural architecture. Finally, it dives into the details of the Transformer architecture, a self-attention-based architecture that as of 2023 is ubiquitous in NLP research.

## 1  Neural architectures and their properties

Progress in natural language processing is often accelerated by general-purpose techniques that perform better than earlier methods across a wide range of settings. Sometimes, a new idea is proposed to solve old problems; other times, old techniques become newly relevant as data or computation power becomes newly available. A few examples of these include hidden Markov models [Baum and Petrie, 1966], conditional random fields [Lafferty et al., 2001], recurrent neural networks [Rumelhart et al., 1985], convolutional neural networks [LeCun et al., 1989], and support vector machines [Cortes and Vapnik, 1995].

In this section, we'll discuss a bit about the neural modeling approaches we've discussed in Cs 224n so far, and how their limitations (and changes in the world) inspired the modern (as of 2023) zeitgeist of self-attention and Transformer-based architectures.

### 1.1  Notation and basics

Let $\mathbf{w}_{1:n}$ be a sequence, where each $\mathbf{w}_i \in \mathcal{V}$, a finite vocabulary. We'll also overload $\mathbf{w}_{1:n}$ to be a matrix of one-hot vectors, $\mathbf{w}_{1:n} \in \mathbb{R}^{n \times |\mathcal{V}|}$. We'll use $\mathbf{w} \in \mathcal{V}$ to represent an arbitrary vocabulary element, and $\mathbf{w}_i \in \mathcal{V}$ to pick out a specific indexed element of a sequence $\mathbf{w}_{1:n}$. We'll use the notation,

$$\mathbf{w}_t \sim \text{softmax}(f(\mathbf{w}_{1:t-1})), \tag{1}$$

to mean that under a model, "$w_t$ is drawn from" the probability distribution defined by the right-hand-side of the tilde, $\sim$. So in this case, $f(\mathbf{w}_{1:t-1})$ should be in $\mathbb{R}^{|\mathcal{V}|}$. When we use the $softmax$ function (as above), we'll use it without direct reference to the dimension

[页 2]

being normalized over, and it should be interpreted as follows. If $A$ is a tensor of shape $\mathbb{R}^{\ell \times d}$, the softmax is computed as follows:

$$\text{softmax}(A)_{i,j} = \frac{\exp A_{i,j}}{\sum_{j'=1}^{d} \exp A_{i,j'}}, \tag{2}$$

for all $i \in {1,\ldots,\ell}$, $j \in {1,\ldots,d}$, and similarly for tensors of more than two axes. That is, if we had a tensor $B \in \mathbb{R}^{m,\ell,d}$, we would define the softmax over the last dimension, similarly. At the risk of being verbose, we'll write it out:

$$\text{softmax}(B)_{q,i,j} = \frac{\exp B_{q,i,j}}{\sum_{j'=1}^{d} \exp B_{q,i,j'}}. \tag{3}$$

In all of our methods, we'll assume an *embedding matrix*, $E \in \mathbb{R}^{d \times |\mathcal{V}|}$, mapping from the vocabulary space to the *hidden dimensionality* $d$, written as $E\mathbf{x} \in \mathbb{R}^{d}$.

> 边注（第 2 页右侧栏）：Embedding definition

The embedding $E\mathbf{w}_i$ of a token in sequence $\mathbf{w}_{1:n}$ is what's known as a **non-contextual** representation; despite $\mathbf{w}_i$ appearing in a sequence, the representation $E\mathbf{w}_i$ is independent of context. Since we'll almost always be working on the embedded version of $\mathbf{w}_{1:n}$, we'll let $\mathbf{x} = E\mathbf{w}$, and $\mathbf{x}_{1:n} = \mathbf{w}_{1:n}E^{\top} \in \mathbb{R}^{n \times d}$. An overarching goal of the methods discussed in this note is to develop strong **contextual** representations of tokens; that is, a representation $\mathbf{h}_i$ that represents $\mathbf{w}_i$ but is a function of the entire sequence $\mathbf{x}_{1:n}$ (or a prefix $\mathbf{x}_{1:i}$, as in the case of language modeling.).

> 边注（第 2 页右侧栏）：A non-contextual representation of a token $\mathbf{x}_i$ of sequence $\mathbf{x}_{1:n}$ depends only on the identity of $\mathbf{x}_i$; a contextual representation of $\mathbf{x}_i$ depends on the entire sequence (or a prefix $\mathbf{x}_{1:i}$.)

### 1.2  The default *circa* 2017: recurrent neural networks

General-purpose modeling techniques and representations have a long history in NLP, with individual techniques falling in and out of favor. Word embeddings, for example, have a much longer history than the word2vec embeddings we studied in the first few lectures [Schütze, 1992]. Likewise, recurrent neural networks have a long and non-monotonic history in modeling problems [Elman, 1990, Bengio et al., 2000]. By 2017, however, the basic strategy to solve a natural language processing task was to begin with a recurrent neural network.

We've gone over RNNs earlier in the course, but the general form bears repeating here. A simple form of RNN is as follows:

【图（第 2 页，公式 (4) 上方）：一条蓝色弯曲箭头从上方的蓝色标注文字 "Dependence on the sequence index" 指向公式中的 $\mathbf{h}_{t-1}$（该项以淡蓝色底纹标出）。】

$$\mathbf{h}_t = \sigma(W\mathbf{h}_{t-1} + U\mathbf{x}_t), \tag{4}$$

where $\mathbf{h}_t \in \mathbb{R}^{d}$, $U \in \mathbb{R}^{d \times d}$, and $W \in \mathbb{R}^{d \times d}$. By 2017, the intuition was that there were *twofold* issues with the recurrent neural network form, and they both had to do with the the dependence on the

[页 3]

sequence index (often called the dependence on "time") highlighted in Equation 4.

***Parallelization issues with dependence on the sequence index.***  Modern graphics processing units (GPUs) are excellent at crunching through a lot of simple operations (like addition) in *parallel*. For example, when I have a matrix $A \in \mathbb{R}^{n \times k}$ and a matrix $B \in \mathbb{R}^{k \times d}$, a GPU is just blazing fast at computing $AB \in \mathbb{R}^{n \times d}$. The constraint of the operations occuring in parallel, however, is crucial – when computing $AB$ the simplest way, I'm performing a bunch of multiplies and then a bunch of sums, most of which don't depend on the output of each other. However, in a recurrent neural network, when I compute

$$\mathbf{h}_2 = \sigma(W\mathbf{h}_1 + U\mathbf{x}_2), \tag{5}$$

I can't compute $h_2$ until I know the value of $\mathbf{h}_1$, so we can write it out as

$$\mathbf{h}_2 = \sigma(W(\sigma(W\mathbf{h}_0 + U\mathbf{x}_1) + U\mathbf{x}_2). \tag{6}$$

Likewise if I wanted to compute $\mathbf{h}_3$, I can't compute it until I know $\mathbf{h}_2$, which I can't compute until I know $\mathbf{h}_1$, etc. Visually, this looks like Figure 1. As the sequence gets longer, **there is only so much I can parallelize the computation of the network on a GPU** because of the number of serial dependencies. (Serial meaning one-after-the-other.)

【图（第 3 页，Figure 1）：RNN 沿时间展开的水平链式图。上排 5 个方框依次标 1、2、3、4、5，左端有入箭头，框间以右向箭头相连（第 1 个框到第 2 个框的连线为加粗黑色，其余为细灰线），末端有出箭头；每个上排框下方有竖线连到下排对应的 5 个方框（均标 0），每个下排框下方有向下引线，引线下方依次是词 "Zuko made his uncle tea"。】
**Figure 1:** A RNN unrolled in time. The rectangles are intermediate states of the RNN (e.g., the first row is the embedding layer, and the second row is the RNN hidden state at each time step) and the number in the rectangle is the number of serial operations that need to be performed before this intermediate state can be computed

As GPUs (and later, other accelerators like Tensor Processing Units (TPUs)) became more powerful and researchers wanted to take fuller advantage of them, this dependence in time became untenable.

***Linear interaction distance.***  A related issue with RNNs is the difficulty with which distant tokens in a sequence can *interact* with each other. By interact, we mean that the presence of one token (already observed in the past) gainfully affects the processing of another token. For example, in the sentence

The chef₁ who ran out of blackberries and went to the stores₅ is₁
