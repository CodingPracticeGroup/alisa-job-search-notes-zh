<!-- 识图转写：第 4-6 页，2026-11-15，来源 cs224n-self-attention-transformers-2023-draft.pdf，转写方式：主会话逐页识图 -->

[页 4]

the number of intermediate computations—matrix multiplies and nonlinearities, for example—that separate *chef* from *is* scales with the number of words between them. We visualize this in Figure 2.

【图（第 4 页，Figure 2）：RNN 沿时间展开的水平链式图，与 Figure 1 同构但上排方框自左至右标 5、4、3、2、1（数字递减），框间左向箭头连接，右端有出箭头，其中最后一段（1 右侧）为加粗黑色、其余为细灰线；下排 5 个方框中前 4 个为空、最右一个标 0；下排引线下方依次为词 "Zuko made his uncle tea"。与 Figure 1 的差别：此处方框中的数字表示"从该词信息到达某个中间状态所需的运算数"。】
**Figure 2:** A RNN unrolled in time. The rectangles are intermediate states of the RNN (e.g., the first row is the embedding layer, and the second row is the RNN hidden state at each time step) and the number in the rectangle is roughly the number of operations separating lexical information of the word *tea* from each intermediate state.

Intuitively, researchers believe there's an issue with linear interaction distance because it can be difficult for networks to precisely "recall" the presence of a word when a large number of operations occur after observing that word. This can make it difficult to learn how distant words should impact the representation of the current word.

This notion of **direct interaction between elements** of a sequence might remind you of the attention mechanism [Bahdanau et al., 2014] in machine translation. In that context, while generating a translation, we learned how to look back into the source sequence once per token of the translation. In this note, we'll present an entire replacement for recurrent neural networks just based on attention. This will solve both the parallelization issues and the linear interaction distance issues with recurrent neural networks.

## 2  A minimal self-attention architecture

Attention, broadly construed, is a method for taking a query, and softly looking up information in a key-value store by picking the value(s) of the key(s) most like the query. By "picking" and "most like," we mean averaging overall values, putting more weight on those which correspond to the keys more like the query. In **self-attention**, we mean that we use the same elements to help us define the querys as we do the keys and values.

In this section, we'll discuss how to develop contextual representations with methods wherein the main mechanism for contextualization is not recurrence, but attention.

### 2.1  The key-query-value self-attention mechanism

There are many forms of self-attention; the form we'll discuss here is currently the most popular. It's called key-query-value self-attention.

[页 5]

Consider a token $\mathbf{x}_i$ in the sequence $\mathbf{x}_{1:n}$. From it, we define a query $\mathbf{q}_i = Q\mathbf{x}_i$, for matrix $Q \in \mathbb{R}^{d \times d}$. Then, for each token in the sequence $\mathbf{x}_j \in \{\mathbf{x}_1,\ldots,\mathbf{x}_n\}$, we define both a key and a value similarly, with two other weight matrices: $\mathbf{k}_j = K\mathbf{x}_j$, and $\mathbf{v}_j = V\mathbf{x}_j$ for $K \in \mathbb{R}^{d \times d}$ and $V \in \mathbb{R}^{d \times d}$.

Our contextual representation $\mathbf{h}_i$ of $\mathbf{x}_i$ is a linear combination (that is, a weighted sum) of the values of the sequence,

$$\mathbf{h}_i = \sum_{j=1}^{n} \alpha_{ij}\mathbf{v}_j, \tag{7}$$

where the weights, these $\alpha_{ij}$ control the strength of contribution of each $\mathbf{v}_j$. Going back to our key-value store analogy, the $\alpha_{ij}$ softly-selects what data to look up. We define these weights by computing the affinities between the keys and the query, $\mathbf{q}_i^\top k_j$, and then computing the softmax over the sequence:

$$\alpha_{ij} = \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_j)}{\sum_{j'=1}^{n}\exp(\mathbf{q}_i^\top \mathbf{k}_{j'})} \tag{8}$$

Intuitively, what we've done by this operation is take our element $\mathbf{x}_i$ and look in its own sequence $\mathbf{x}_{1:n}$ to figure out what information (in an informal sense,) from what other tokens, should be used in representing $\mathbf{x}_i$ in context. The use of matrices $K, Q, V$ intuitively allow us to use different views of the $\mathbf{x}_i$ for the different roles of key, query, and value. We perform this operation to build $\mathbf{h}_i$ for all $i \in \{1,\ldots,n\}$.

【图（第 5 页，"Self-Attention" 示意，分上下两部分，中间以横线分隔）：
上半：公式 $\mathbf{h}_i = \sum \mathbf{a}_{ij}\ \mathbf{v}_j$，旁注 "(weighted average)"；左侧标 "Self-Attention"；公式下方有两条箭头分别从 "scalar"（指向一个绿色小方块）与 "vector"（指向一个粉色高矩形）指出，示意权重标量乘以值向量。
下半：右侧一排 4 个蓝色竖条，下标 $k_{1:n}=\mathrm{Kx}_{1:n}$、标注 "(key)"，每根蓝条顶端有一个紫色小圆点；右上一个棕色竖条标注 $\mathbf{q}_i=\mathrm{Qx}_i$ (query)，从其底部引出多条弧形线连到各紫色圆点，旁注 "$\mathbf{q}_i^\top k_j \rightarrow$ softmax"；各紫色圆点再以虚线弧连到左侧一排绿色竖条，绿色竖条整体标注 $\alpha$ 与 "(weights)"（高度不等，示意权重大小）；左下一排 4 个粉色竖条，下标 $\mathbf{v}_{1:n}=\mathrm{Vx}_{1:n}$、标注 "(value)"。条形个数与文中示例序列长度不一致，为示意图。】

[页 6]

### 2.2  Position representations

Consider the sequence *the oven cooked the bread so*. This is a different sequence than *the bread cooked the oven so*, as you might guess. The former sentence has us making delicious bread, and the latter we might interpret as the bread somehow breaking the oven. In a recurrent neural network, the order of the sequence defines the order of the rollout, so the two sequences have different representations. In the self-attention operation, there's no built-in notion of order.

> 边注（第 6 页右侧栏）：The self-attention operation has **no built-in notion of the sequence order**.

To see this, let's take a look at self-attention on this sequence. We have a set of vectors $\mathbf{x}_{1:n}$ for the *oven cooked the bread so*, which we can write as

$$\mathbf{x}_{1:n} = [\mathbf{x}_{\text{the}}; \mathbf{x}_{\text{oven}}; \mathbf{x}_{\text{cooked}}; \mathbf{x}_{\text{the}}; \mathbf{x}_{\text{bread}}; \mathbf{x}_{\text{so}}] \in \mathbb{R}^{5 \times d} \tag{9}$$

【转写注：原稿此处即如此——列出 6 个 token 但维度写 $\mathbb{R}^{5\times d}$，且后文 $\alpha \in \mathbb{R}^5$；疑为原稿笔误，照录。】

As an example, consider performing self-attention to represent the word *so* in context. The weights over the context are as follows, recalling that $\mathbf{q}_i = Q\mathbf{x}_i$ for all words, and $\mathbf{k}_j = K\mathbf{x}_j$ likewise:

$$\alpha_{\text{so}} = \text{softmax}\left(\left[\mathbf{q}_{\text{so}}^\top \mathbf{k}_{\text{the}}; \mathbf{q}_{\text{so}}^\top \mathbf{k}_{\text{oven}}; \mathbf{q}_{\text{so}}^\top \mathbf{k}_{\text{cooked}}; \mathbf{q}_{\text{so}}^\top \mathbf{k}_{\text{the}}; \mathbf{q}_{\text{so}}^\top \mathbf{k}_{\text{bread}}; \mathbf{q}_{\text{so}}^\top \mathbf{k}_{\text{so}}\right]\right) \tag{10}$$

So, the weight $\alpha_{\text{so},0}$, the amount that we look up the first word, (by writing out the softmax) is,

$$\alpha_{\text{so},0} = \frac{\exp(\mathbf{q}_{\text{so}}^\top \mathbf{k}_{\text{the}})}{\exp(\mathbf{q}_{\text{so}}^\top \mathbf{k}_{\text{the}}) + \cdots + \exp(\mathbf{q}_{\text{so}}^\top \mathbf{k}_{\text{bread}})}. \tag{11}$$

So, $\alpha \in \mathbb{R}^5$ are our weights, and we compute the weighted average in Equation 7 with these weights to compute $\mathbf{h}_{\text{so}}$.

For the reordered sentence *the bread cooked the oven*, you can compute that $\alpha_{\text{so},0}$ is identical. The numerator hasn't changed, and the denominator hasn't changed; we've just rearranged terms in the sum. Likewise for $\alpha_{\text{so,bread}}$ and $\alpha_{\text{so,oven}}$, you can compute that they too are identical independent of the ordering of the sequence. This all comes back down to the two facts that (1) the representation of $\mathbf{x}$ is not position-dependent; it's just $E\mathbf{w}$ for whatever word $\mathbf{w}$, and (2) there's no dependence on position in the self-attention operations.

> 边注（第 6 页右下角）：Non-contextual embedded words $\mathbf{x}_i = E\mathbf{w}_i$ have no dependence on the word's position in a sequence $\mathbf{w}_{1:n}$; only on the identity of the word in $\mathcal{V}$.

***Position representation through learned embeddings.***  To represent position in self-attention, you either need to (1) use vectors that are already position-dependent as inputs, or (2) change the self-attention operation itself. One common solution is a simple implementation of (1). We posit a new parameter matrix, $P \in \mathbb{R}^{N \times d}$, where $N$ is the *maximum length of any sequence* that your model will be able to process.
