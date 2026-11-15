<!-- 识图转写：第 13-15 页，2026-11-15，来源 cs224n-self-attention-transformers-2023-draft.pdf，转写方式：主会话逐页识图 -->

[页 13]

### 3.5  Transformer Encoder

A Transformer Encoder takes a single sequence $\mathbf{w}_{1:n}$, and performs no future masking. It embeds the sequence with $E$ to make $\mathbf{x}_{1:n}$, adds the position representation, and then applies a stack of independently parameterized *Encoder Blocks*, each of which consisting of (1) multi-head attention and Add & Norm, and (2) feed-forward and Add & Norm. So, the output of each Block is the input to the next. Figure 5 presents this.

In the case that one wants probabilities out of the tokens of a Transformer Encoder (as in masked language modeling for BERT [Devlin et al., 2019], which we'll cover later), one applies a linear transformation to the output space followed by a softmax.

***Uses of the Transformer Encoder.***  A Transformer Encoder is great in contexts where you aren't trying to generate text autoregressively (there's no masking in the encoder so each position index can see the whole sequence,) and want strong representations for the whole sequence (again, possible because even the first token can see the whole future of the sequence when building its representation.)

【图（第 13 页右侧，Figure 5）：Transformer Encoder 纵向流程图，箭头自下而上：`Encoder Inputs`（文字）→ `Embeddings`（蓝绿）→ `Add Position Embeddings`（绿）→ Block 框内：`Multi-Head Attention`（红，底部三个弧形输入箭头汇入）→ `Add & Norm`（黄）→ `Feed-Forward`（青）→ `Add & Norm`（黄）；Block 左侧竖排注 "Repeat: for number of encoder blocks"，Block 右下角标 "Block"；顶部 → `Linear`（灰）→ `Softmax`（灰）→ *Probabilities*。图题下方斜体字：*Transformer Encoder*】
**Figure 5:** Diagram of the Transformer Encoder.

### 3.6  Transformer Decoder

To build a Transformer autoregressive language model, one uses a Transformer Decoder. These differ from Transformer Encoders simply by using future masking at each application of self-attention. This ensures that the informational constraint (no cheating by looking at the future!) holds throughout the architecture. We show a diagram of this architecture in Figure 4. Famous examples of this are GPT-2 [Radford et al., 2019], GPT-3 [Brown et al., 2020] and BLOOM [Workshop et al., 2022].

### 3.7  Transformer Encoder-Decoder

A Transformer encoder-decoder takes as input two sequences. Figure 6 shows the whole encoder-decoder structure. The first sequence $\mathbf{x}_{1:n}$ is passed through a Transformer Encoder to build contextual representations. The second sequence $\mathbf{y}_{1:m}$ is encoded through a modified Transformer Decoder architecture in which **cross-attention** (which we haven't yet defined!) is applied from the encoded representation of $\mathbf{y}_{1:m}$ to the output of the Encoder. So, let's take a quick detour to discuss cross-attention; it's not too different from what we've already seen.

[页 14]

***Cross-Attention.***  Cross-attention uses one sequence to define the keys and values of self-attention, and another sequence to define the queries. You might think, hey wait, isn't that just what attention always was before we got into this self-attention business? Yeah, pretty much. So if

$$\mathbf{h}^{(x)}_{1:n} = \text{TransformerEncoder}(\mathbf{w}_{1:n}), \tag{33}$$

and we have some intermediate representation $\mathbf{h}^{(y)}$ of sequence $\mathbf{y}_{1:m}$, then we let the queries come from the decoder (the $\mathbf{h}^{(y)}$ sequence) while the keys and values come from the encoder:

$$\mathbf{q}_i = Q\mathbf{h}^{(y)}_i \qquad\qquad i \in \{1,\ldots,m\} \tag{34}$$

$$\mathbf{k}_j = K\mathbf{h}^{(x)}_j \qquad\qquad j \in \{1,\ldots,n\} \tag{35}$$

$$\mathbf{v}_j = V\mathbf{h}^{(x)}_j \qquad\qquad j \in \{1,\ldots,n\}, \tag{36}$$

and compute the attention on $\mathbf{q}, \mathbf{k}, \mathbf{v}$ as we defined for self-attention. Note in Figure 6 that in the Transformer Encoder-Decoder, cross-attention always applies to the output of the Transformer encoder.

***Uses of the encoder-decoder.***  An encoder-decoder is used when we'd like bidirectional context on something (like an article to summarize) to build strong representations (i.e., each token can attend to all other tokens), but then generate an output according to an autoregressive decomposition as we can with a decoder. While such an architecture has been found to provide better performance than decoder-only models at modest scale [Raffel et al., 2020], it involves splitting parameters between encoder and decoder, and most of the largest Transformers are decoder-only.

*.1*

# References

[Ba et al., 2016] Ba, J. L., Kiros, J. R., and Hinton, G. E. (2016). Layer normalization. *arXiv preprint arXiv:1607.06450*.

[Bahdanau et al., 2014] Bahdanau, D., Cho, K., and Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473*.

[Baum and Petrie, 1966] Baum, L. E. and Petrie, T. (1966). Statistical inference for probabilistic functions of finite state markov chains. *The Annals of Mathematical Statistics*, 37(6):1554–1563.

[Bengio et al., 2000] Bengio, Y., Ducharme, R., and Vincent, P. (2000). A neural probabilistic language model. *Advances in neural information processing systems*, 13.

[Bengio et al., 2003] Bengio, Y., Ducharme, R., Vincent, P., and Janvin, C. (2003). A neural probabilistic language model. *J. Mach. Learn. Res.*, 3:1137–1155.

[Brown et al., 2020] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss,

[页 15]

【图（第 15 页整页，Figure 6）：Transformer Encoder-Decoder 完整结构图，双柱并列，均为箭头自下而上。
左柱（编码器）：`Encoder Inputs`（斜体文字）→ `Embeddings`（蓝绿）→ `Add Position Embeddings`（绿）→ Block 框：`Multi-Head Attention`（红，底部三弧箭头汇入）→ `Add & Norm`（黄）→ `Feed-Forward`（青）→ `Add & Norm`（黄）；Block 右侧标 "Block"，左侧有自下方绕上的弯箭头标注 "Repeat: for number of encoder blocks"。编码器顶部输出分两路：一路继续向上进入右柱的 Multi-Head Attention 块（横向长线箭头），另一路为右侧柱供 Ar。
右柱（解码器）：`Decoder Inputs`（斜体）→ `Embeddings`（蓝绿）→ `Add Position Embeddings`（绿）→ Block 框：`Masked Multi-Head Attention`（红，三弧箭头汇入）→ `Add & Norm`（黄）→ `Multi-Head Attention`（红，接收编码器输出的横长箭头，旁注 "Repeat: for number of decoder blocks."）→ `Add & Norm`（黄）→ `Feed-Forward`（青，旁注 "Attend only to output of last Encoder Block."）→ `Add & Norm`（黄）→ `Linear`（灰）→ `Softmax`（灰）→ *Probabilities*。】
**Figure 6:** A Transformer encoder-decoder.
