<!-- 识图转写：第 10-12 页，2026-11-15，来源 cs224n-self-attention-transformers-2023-draft.pdf，转写方式：主会话逐页识图 -->

[页 10]

We then perform self-attention with each head:

$$\mathbf{h}_i^{(\ell)} = \sum_{j=1}^{n} \alpha_{ij}^{(\ell)}\mathbf{v}_j^{(\ell)} \tag{22}$$

$$\alpha_{ij}^{(\ell)} = \frac{\exp(\mathbf{q}_i^{(\ell)\top}\mathbf{k}_j^{(\ell)})}{\sum_{j'=1}^{n}\exp(\mathbf{q}_i^{(\ell)\top}\mathbf{k}_{j'}^{(\ell)})} \tag{23}$$

Note that the output $\mathbf{h}_i^{(\ell)}$ of each head is in reduced dimension $d/k$. Finally, we define the output of multi-head self-attention as a linear transformation of the concatenation of the head outputs, letting $O \in \mathbb{R}^{d \times d}$:

$$\mathbf{h}_i = O\left[\mathbf{v}_i^{(1)};\cdots;\mathbf{v}_i^{(k)}\right], \tag{24}$$

where we concatenate the head outputs each of dimensionality $d \times d/k$ at their second axis, such that their concatenation has dimension $d \times d$.

***Sequence-tensor form.***  To understand why we have the reduced dimension of each head output, it's instructive to get a bit closer to how multi-head self-attention is implemented in code. In practice, **multi-head self-attention is no more expensive than single-head** due to the low-rankness of the transformations we apply.

For a single head, recall that $\mathbf{x}_{1:n}$ is a matrix in $\mathbb{R}^{n \times d}$. Then we can compute our value vectors as a matrix as $\mathbf{x}_{1:n}V$, and likewise our keys and queries $\mathbf{x}_{1:n}K$ and $\mathbf{x}_{1:n}Q$, all matrices in $\mathbb{R}^{n \times d}$. To compute self-attention, we can compute our weights in matrix operations:

$$\alpha = \text{softmax}(\mathbf{x}_{1:n}QK^\top\mathbf{x}_{1:n}^\top) \in \mathbb{R}^{n \times n} \tag{25}$$

and then compute the self-attention operation for all $\mathbf{x}_{1:n}$ via:

$$\mathbf{h}_{1:n} = \text{softmax}(\mathbf{x}_{1:n}QK^\top\mathbf{x}_{1:n}^\top)\mathbf{x}_{1:n}V \in \mathbb{R}^{n \times d}. \tag{26}$$

Here's a diagram showing the matrix ops:

【图（第 10 页中部）：矩阵运算示意。左侧 "softmax" 字样与一对大括号；括号内：竖条标注 $n$，一个青绿色竖矩形标 $x_{1:n}Q$（底部有水平双箭头标注宽度 $-d-$）；其右一个粉色横矩形标 $(x_{1:n}K)^T$。大括号右侧为等号；等号右为一个青绿色大正方形标 $\boldsymbol{\alpha}$，右侧竖标注 $n$，下方横标注 $\overline{\ \ n\ }$。】

When we perform multi-head self-attention in this matrix form, we first reshape $\mathbf{x}_{1:n}Q$, $\mathbf{x}_{1:n}K$, and $\mathbf{x}_{1:n}V$ each into a matrix of shape $\mathbb{R}^{n,k,d/k}$, splitting the model dimensionality into two axes, for the number of heads and the number of dimensions per head. We can

[页 11]

then transpose the matrices to $\mathbb{R}^{k,n,d/k}$, which intuitively should look like $k$ sequences of length $n$ and dimensionality $d/k$. This allows us to perform the batched softmax operation in parallel across the heads, using the number of heads kind of like a **batch** axis (and indeed in practice we'll also have a separate batch axis.) So, the total computation (except the last linear transformation to combine the heads) is the same, just distributed across the (each lower-rank) heads. Here's a diagram like the single-head diagram, demonstrating how the multi-head operation ends up much like the single-head operation:

【图（第 11 页中部）：多头矩阵运算示意，结构与第 10 页单头图对应。左侧 "softmax" 与左大括号（竖标注 $n$）；括号内：$k$ 个并排的竖条束（蓝、粉、黄绿、青等多色叠加），顶部标注 reshape$_{(x_{1:n}Q)}$，底部有 $k$ 个小上箭头指向竖条、下方标注 $d/k$；右侧为 $k$ 个横条束（粉色为主，顶部标注 reshape$_{(x_{1:n}K)^T}$）。等号右侧：一个蓝色大正方形标 $\boldsymbol{\alpha}$，其后叠着青、红等色方片（层叠效果），右上角斜向标注 $k$，右侧竖标注 $n$，下方横标注 $\overline{\ \ n\ }$。】

### 3.2  Layer Norm

One important learning aid in Transformers is *layer normalization* [Ba et al., 2016]. The intuition of layer norm is to reduce uninformative variation in the activations at a layer, providing a more stable input to the next layer. Further work shows that this may be most useful not in normalizing the forward pass, but actually in improving gradients in the backward pass [Xu et al., 2019].

To do this, layer norm (1) computes statistics across the activations at a layer to estimate the mean and variance of the activations, and (2) normalizes the activations with respect to those estimates, while (3) optionally learning (as parameters) an elementwise additive bias and multiplicative gain by which to sort of de-normalize the activations in a predictable way. The third part seems not to be crucial, and may even be harmful [Xu et al., 2019], so we omit it in our presentation.

One question to ask when understanding how layer norm affects a network is, "computing statistics over *what*?" That is, what constitutes a layer? In Transformers, the answer is always that statistics computed independently for a single index into the sequence length (and a single example in the batch) and shared across the $d$ hidden dimensions. Put another way, the statistics for the token at index $i$ won't affect the token at index $j \ne i$.

So, we compute the statistics for a single index $i \in \{1,\ldots,n\}$ as

$$\hat{\mu}_i = \frac{1}{d}\sum_{j=1}^{d}\mathbf{h}_{ij} \qquad \hat{\sigma}_i = \sqrt{\frac{1}{d}\sum_{j=1}^{d}(\mathbf{h}_{ij}-\mu_i)^2}, \tag{27}$$

[页 12]

where (as a reminder), $\hat{\mu}_i$ and $\hat{\sigma}_i$ are scalars, and we compute the layer norm as

$$\text{LN}(\mathbf{h}_i) = \frac{\mathbf{h}_i - \hat{\mu}_i}{\hat{\sigma}_i}, \tag{28}$$

where we've broadcasted the $\hat{\mu}_i$ and $\hat{\sigma}_i$ across the $d$ dimensions of $\mathbf{h}_i$. Layer normalization is a great tool to have in your deep learning toolbox more generally.

### 3.3  Residual Connections

Residual connections simply add the *input* of a layer to the *output* of that layer:

$$f_{\text{residual}}(\mathbf{h}_{1:n}) = f(\mathbf{h}_{1:n}) + \mathbf{h}_{1:n}, \tag{29}$$

the intuition being that (1) the gradient flow of the identity function is *great* (the local gradient is 1 everywhere!) so the connection allows for learning much deeper networks, and (2) it is easier to learn the difference of a function from the identity function than it is to learn the function from scratch. As simple as these seem, they're massively useful in deep learning, not just in Transformers!

***Add & Norm.***  In the Transformer diagrams you'll see, including Figure 4, the application of layer normalization and residual connection are often combined in a single visual block labeled *Add & Norm*. Such a layer might look like:

$$\mathbf{h}_{\text{pre-norm}} = f(\text{LN}(\mathbf{h})) + \mathbf{h}, \tag{30}$$

where $f$ is either a feed-forward operation or a self-attention operation, (this is known as *pre-normalization*), or like:

$$\mathbf{h}_{\text{post-norm}} = \text{LN}(f(\mathbf{h}) + \mathbf{h}), \tag{31}$$

which is known as *post-normalization*. It turns out that the gradients of **pre-normalization** are much better at initialization, leading to much faster training [Xiong et al., 2020].

### 3.4  Attention logit scaling

Another trick introduced in [Vaswani et al., 2017] they dub *scaled dot product attention*. The dot product part comes from the fact that we're computing dot products $\mathbf{q}_i^\top k_i$. The intuition of scaling is that, when the dimensionality $d$ of the vectors we're dotting grows large, the dot product of even random vectors (e.g., at initialization) grows roughly as $\sqrt{d}$. So, we normalize the dot products by $\sqrt{d}$ to stop this scaling:

$$\alpha = \text{softmax}(\frac{\mathbf{x}_{1:n}QK^\top\mathbf{x}_{1:n}^\top}{\sqrt{d}}) \in \mathbb{R}^{n \times n} \tag{32}$$
