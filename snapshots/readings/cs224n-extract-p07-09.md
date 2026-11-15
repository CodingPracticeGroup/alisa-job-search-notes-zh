<!-- 识图转写：第 7-9 页，2026-11-15，来源 cs224n-self-attention-transformers-2023-draft.pdf，转写方式：主会话逐页识图 -->

[页 7]

We then simply add embedded representation of the position of a word to its word embedding:

$$\bar{x}_i = P_i + \mathbf{x}_i \tag{12}$$

and perform self-attention as we otherwise would. Now, the self-attention operation can use the embedding $P_i$ to look at the word at position $i$ differently than if that word were at position $j$. This is done, e.g., in the BERT paper [Devlin et al., 2019] (which we go over later in the course.)

***Position representation through changing $\mathbf{q}$ directly.***  Instead of changing the input representation, another thing we can do is change the form of self-attention to have a built-in notion of position. One intuition is that all else held equal, self-attention should look at "nearby" words more than "far" words. Attention with Linear Biases [Press et al., 2022] is one implementation of this idea. One implementation of this would be as follows:

$$\alpha_i = \text{softmax}\left(\mathbf{k}_{1:n}\mathbf{q}_i + [-\dot\iota, \dots, -1, 0, -1, \dots, -(n-i)]\right), \tag{13}$$

where $\mathbf{k}_{1:n}\mathbf{q}_i \in \mathbb{R}^n$ are the original attention scores, and the bias we add makes attention focus more on nearby words than far away words, all else held equal. In some sense, it's odd that this works; but interesting!

### 2.3  Elementwise nonlinearity

Imagine if we were to stack self-attention layers. Would this be sufficient for a replacement for stacked LSTM layers? Intuitively, there's one thing that's missing: the elementwise nonlinearities that we've come to expect in standard deep learning architectures. In fact, if we stack two self-attention layers, we get something that looks a lot like a single self-attention layer:

$$o_i = \sum_{j=1}^{n} \alpha_{ij}V^{(2)}\left(\sum_{k=1}^{n} \alpha_{jk}V^{(1)}\mathbf{x}_k\right) \tag{14}$$

$$= \sum_{k=1}^{n}\left(\alpha_{jk}\sum_{j=1}^{n}\alpha_{ij}\right)V^{(2)}V^{(1)}\mathbf{x}_k \tag{15}$$

$$= \sum_{k=1}^{n}\alpha^*_{ij}V^*\mathbf{x}_k, \tag{16}$$

where $\alpha^*_{ij} = \left(\alpha_{jk}\sum_{i=1}^{n}\alpha_{ij}\right)$, and $V^* = V^{(2)}V^{(1)}$. So, this is just a linear combination of a linear transformation of the input, much like a single layer of self-attention! Is this good enough?³

> 脚注 ³（第 7 页右下角）：This question ends up having a nuanced answer that's out-of-scope for this note; ask me if you're interested in knowing more!

[页 8]

In practice, after a layer of self-attention, it's common to apply feed-forward network independently to each word representation:

$$h_{\text{FF}} = W_2\ \text{ReLU}(W_1 h_{\text{self-attention}} + b_1) + b_2, \tag{17}$$

where often, $W_1 \in \mathbb{R}^{5d \times d}$, and $W_2 \in \mathbb{R}^{d \times 5d}$. That is, the hidden dimension of the feed-forward network is substantially larger than the hidden dimension of the network, $d$—this is done because this matrix multiply is an efficiently parallelizable operation, so it's an efficient place to put a lot of computation and parameters.

### 2.4  Future masking

When performing language modeling like we've seen in this course (often called autoregressive modeling), we predict a word given all words so far:

$$\mathbf{w}_t \sim \text{softmax}(f(\mathbf{w}_{1:t-1})). \tag{18}$$

where $f$ is function to map a sequence to a vector in $\mathbb{R}^{|\mathcal{V}|}$.

One crucial aspect of this process is that we can't look at the future when predicting it—otherwise the problem becomes trivial. This idea is built-in to unidirectional RNNs. If we want to use an RNN for the function $f$, we can use the hidden state for word $w_{t-1}$:

$$\mathbf{w}_t \sim \text{softmax}(\mathbf{h}_{t-1}E) \tag{19}$$

$$h_{t-1} = \sigma(W\mathbf{h}_{t-2} + U\mathbf{x}_{t-1}), \tag{20}$$

and by the rollout of the RNN, we haven't looked at the future. (In this case, the future is all the words $\mathbf{w}_t,\ldots,\mathbf{w}_n$.)

In a Transformer, there's nothing explicit in the self-attention weight $\alpha$ that says not to look at indices $j > i$ when representing token $i$. In practice, we enforce this constraint simply adding a large negative constant to the input to the softmax (or equivalently, setting $\alpha_{ij} = 0$ where $j > i$).⁴

$$\alpha_{ij,\text{masked}} = \begin{cases} \alpha_{ij} & j \le i \\ 0 & \text{otherwise} \end{cases} \tag{21}$$

In a diagram, it looks like Figure 3.

> 脚注 ⁴（第 8 页右侧栏）：It might seem like one should use $-\infty$ as the constant, to "really" ensure that you can't see the future. However, this is not done; a modest constant within even the float range of the 'float16' encoding is used instead, like $-10^5$. Using infinity can lead to NaNs and it's sort of undefined how each library should treat infinite inputs, so we tend to avoid using it. And because of finite precision, a large enough negative constant will still set the attention weight to exactly zero.

### 2.5  Summary of a minimal self-attention architecture

Our minimal self-attention architecture has (1) the self-attention operation, (2) position representations, (3) elementwise nonlinearities, and (4) future masking (in the context of language modeling.)

[页 9]

【图（第 9 页右上，Figure 3）：自回归未来掩码示意表。表头横排为词 Zuko、made、his、uncle、tea；左侧竖排同为 Zuko、made、his、uncle、tea。网格为 5×5 空白方框：每行中"当前词及之前"的方框为空白，"之后"的方框填 $-\infty$（即上三角全为 $-\infty$，对角线及以下为空白）。】
**Figure 3:** Diagram of autoregressive future masking in self-attention. Words in each row have words in the future masked out (e.g., "Zuko" can only attend to "Zuko", while "made" can attend to "Zuko" and "made".)

Intuitively, these are the biggest components to understand. However, as of 2023, by far the most-used architecture in NLP is called the *Transformer*, introduced by [Vaswani et al., 2017], and it contains a number of components that end up being quite important. So now we'll get into the details of that architecture.

## 3  The Transformer

The Transformer is an architecture based on self-attention that consists of stacked *Blocks*, each of which contains self-attention and feed-forward layers, and a few other components we'll discuss. If you'd like to take a peek for intuition, we have a diagram of a Transformer language model architecture in Figure 4. The components we haven't gone over are **multi-head self-attention**, **layer normalization**, **residual connections**, and **attention scaling**—and of course, we'll discuss how these components are combined to form the Transformer.

【图（第 9 页右中，Figure 4）：Transformer Decoder（仅解码器）纵向流程图，箭头自下而上：`Decoder Inputs`（文字）→ `Embeddings`（蓝绿）→ `Add Position Embeddings`（绿）→ Block 框内：`Masked Multi-Head Attention`（红，底部有两个弧形输入箭头汇入）→ `Add & Norm`（黄）→ `Feed-Forward`（青）→ `Add & Norm`（黄）；Block 左侧竖排注 "Repeat: for number of encoder blocks"，Block 右下角标 "Block"；顶部 → `Linear`（灰）→ `Softmax`（灰）→ *Probabilities*。图题下方文字：*Transformer Decoder*】
**Figure 4:** Diagram of the Transformer Decoder (without corresponding Encoder, and so no cross-attention.

### 3.1  Multi-head Self-Attention

Intuitively, a single call of self-attention is best at picking out a *single* value (on average) from the input value set. It does so softly, by averaging over all of the values, but it requires a balancing game in the key-query dot products in order to carefully average two or more things. In Assignment 5, you'll work through a bit of this intuition more carefully. What we'll present now, **multi-head self-attention**, intuitively applies self-attention multiple times at once, each with different key, query, and value transformations of the same input, and then combines the outputs.

For an integer number of heads $k$, we define matrices $K^{(\ell)}, Q^{(\ell)}, V^{(\ell)} \in \mathbb{R}^{d \times d/k}$ for $\ell$ in $\{1,\ldots,k\}$. (We'll see why we have the dimensionality reduction to $d/k$ soon.) These our the key, query, and value matrices for each head. Correspondingly, we get keys, queries, and values $\mathbf{k}_{1:n}^{(\ell)}, \mathbf{q}_{1:n}^{(\ell)}, \mathbf{v}_{1:n}^{(\ell)}$ as in single-head self-attention.
