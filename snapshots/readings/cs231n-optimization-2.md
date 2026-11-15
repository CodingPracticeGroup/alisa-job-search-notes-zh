<div class="site-header">

<a href="https://cs231n.github.io" class="site-title">CS231n Deep
Learning for Computer Vision</a>
<a href="http://cs231n.stanford.edu/" class="site-link">Course
Website</a>

</div>

<div class="page-content">

<div class="wrap">

<div class="post">

<div class="post-header">

# 

</div>

Table of Contents:

- [Introduction](#intro)
- [Simple expressions, interpreting the gradient](#grad)
- [Compound expressions, chain rule, backpropagation](#backprop)
- [Intuitive understanding of backpropagation](#intuitive)
- [Modularity: Sigmoid example](#sigmoid)
- [Backprop in practice: Staged computation](#staged)
- [Patterns in backward flow](#patterns)
- [Gradients for vectorized operations](#mat)
- [Summary](#summary)

<span id="intro"></span>

### Introduction

**Motivation**. In this section we will develop expertise with an
intuitive understanding of **backpropagation**, which is a way of
computing gradients of expressions through recursive application of
**chain rule**. Understanding of this process and its subtleties is
critical for you to understand, and effectively develop, design and
debug neural networks.

**Problem statement**. The core problem studied in this section is as
follows: We are given some function \\f(x)\\ where \\x\\ is a vector of
inputs and we are interested in computing the gradient of \\f\\ at \\x\\
(i.e. \\\nabla f(x)\\ ).

**Motivation**. Recall that the primary reason we are interested in this
problem is that in the specific case of neural networks, \\f\\ will
correspond to the loss function ( \\L\\ ) and the inputs \\x\\ will
consist of the training data and the neural network weights. For
example, the loss could be the SVM loss function and the inputs are both
the training data \\(x_i,y_i), i=1 \ldots N\\ and the weights and biases
\\W,b\\. Note that (as is usually the case in Machine Learning) we think
of the training data as given and fixed, and of the weights as variables
we have control over. Hence, even though we can easily use
backpropagation to compute the gradient on the input examples \\x_i\\,
in practice we usually only compute the gradient for the parameters
(e.g. \\W,b\\) so that we can use it to perform a parameter update.
However, as we will see later in the class the gradient on \\x_i\\ can
still be useful sometimes, for example for purposes of visualization and
interpreting what the Neural Network might be doing.

If you are coming to this class and you’re comfortable with deriving
gradients with chain rule, we would still like to encourage you to at
least skim this section, since it presents a rarely developed view of
backpropagation as backward flow in real-valued circuits and any
insights you’ll gain may help you throughout the class.

<span id="grad"></span>

### Simple expressions and interpretation of the gradient

Lets start simple so that we can develop the notation and conventions
for more complex expressions. Consider a simple multiplication function
of two numbers \\f(x,y) = x y\\. It is a matter of simple calculus to
derive the partial derivative for either input:

\\f(x,y) = x y \hspace{0.5in} \rightarrow \hspace{0.5in} \frac{\partial
f}{\partial x} = y \hspace{0.5in} \frac{\partial f}{\partial y} = x\\

**Interpretation**. Keep in mind what the derivatives tell you: They
indicate the rate of change of a function with respect to that variable
surrounding an infinitesimally small region near a particular point:

\\\frac{df(x)}{dx} = \lim\_{h\\ \to 0} \frac{f(x + h) - f(x)}{h}\\

A technical note is that the division sign on the left-hand side is,
unlike the division sign on the right-hand side, not a division.
Instead, this notation indicates that the operator \\ \frac{d}{dx} \\ is
being applied to the function \\f\\, and returns a different function
(the derivative). A nice way to think about the expression above is that
when \\h\\ is very small, then the function is well-approximated by a
straight line, and the derivative is its slope. In other words, the
derivative on each variable tells you the sensitivity of the whole
expression on its value. For example, if \\x = 4, y = -3\\ then \\f(x,y)
= -12\\ and the derivative on \\x\\ \\\frac{\partial f}{\partial x} =
-3\\. This tells us that if we were to increase the value of this
variable by a tiny amount, the effect on the whole expression would be
to decrease it (due to the negative sign), and by three times that
amount. This can be seen by rearranging the above equation ( \\ f(x + h)
= f(x) + h \frac{df(x)}{dx} \\ ). Analogously, since \\\frac{\partial
f}{\partial y} = 4\\, we expect that increasing the value of \\y\\ by
some very small amount \\h\\ would also increase the output of the
function (due to the positive sign), and by \\4h\\.

> The derivative on each variable tells you the sensitivity of the whole
> expression on its value.

As mentioned, the gradient \\\nabla f\\ is the vector of partial
derivatives, so we have that \\\nabla f = \[\frac{\partial f}{\partial
x}, \frac{\partial f}{\partial y}\] = \[y, x\]\\. Even though the
gradient is technically a vector, we will often use terms such as *“the
gradient on x”* instead of the technically correct phrase *“the partial
derivative on x”* for simplicity.

We can also derive the derivatives for the addition operation:

\\f(x,y) = x + y \hspace{0.5in} \rightarrow \hspace{0.5in}
\frac{\partial f}{\partial x} = 1 \hspace{0.5in} \frac{\partial
f}{\partial y} = 1\\

that is, the derivative on both \\x,y\\ is one regardless of what the
values of \\x,y\\ are. This makes sense, since increasing either \\x,y\\
would increase the output of \\f\\, and the rate of that increase would
be independent of what the actual values of \\x,y\\ are (unlike the case
of multiplication above). The last function we’ll use quite a bit in the
class is the *max* operation:

\\f(x,y) = \max(x, y) \hspace{0.5in} \rightarrow \hspace{0.5in}
\frac{\partial f}{\partial x} = \mathbb{1}(x \>= y) \hspace{0.5in}
\frac{\partial f}{\partial y} = \mathbb{1}(y \>= x)\\

That is, the (sub)gradient is 1 on the input that was larger and 0 on
the other input. Intuitively, if the inputs are \\x = 4,y = 2\\, then
the max is 4, and the function is not sensitive to the setting of \\y\\.
That is, if we were to increase it by a tiny amount \\h\\, the function
would keep outputting 4, and therefore the gradient is zero: there is no
effect. Of course, if we were to change \\y\\ by a large amount (e.g.
larger than 2), then the value of \\f\\ would change, but the
derivatives tell us nothing about the effect of such large changes on
the inputs of a function; They are only informative for tiny,
infinitesimally small changes on the inputs, as indicated by the
\\\lim\_{h \rightarrow 0}\\ in its definition.

<span id="backprop"></span>

### Compound expressions with chain rule

Lets now start to consider more complicated expressions that involve
multiple composed functions, such as \\f(x,y,z) = (x + y) z\\. This
expression is still simple enough to differentiate directly, but we’ll
take a particular approach to it that will be helpful with understanding
the intuition behind backpropagation. In particular, note that this
expression can be broken down into two expressions: \\q = x + y\\ and
\\f = q z\\. Moreover, we know how to compute the derivatives of both
expressions separately, as seen in the previous section. \\f\\ is just
multiplication of \\q\\ and \\z\\, so \\\frac{\partial f}{\partial q} =
z, \frac{\partial f}{\partial z} = q\\, and \\q\\ is addition of \\x\\
and \\y\\ so \\ \frac{\partial q}{\partial x} = 1, \frac{\partial
q}{\partial y} = 1 \\. However, we don’t necessarily care about the
gradient on the intermediate value \\q\\ - the value of \\\frac{\partial
f}{\partial q}\\ is not useful. Instead, we are ultimately interested in
the gradient of \\f\\ with respect to its inputs \\x,y,z\\. The **chain
rule** tells us that the correct way to “chain” these gradient
expressions together is through multiplication. For example,
\\\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q}
\frac{\partial q}{\partial x} \\. In practice this is simply a
multiplication of the two numbers that hold the two gradients. Lets see
this with an example:

<div class="language-python highlighter-rouge">

<div class="highlight">

``` highlight
# set some inputs
x = -2; y = 5; z = -4

# perform the forward pass
q = x + y # q becomes 3
f = q * z # f becomes -12

# perform the backward pass (backpropagation) in reverse order:
# first backprop through f = q * z
dfdz = q # df/dz = q, so gradient on z becomes 3
dfdq = z # df/dq = z, so gradient on q becomes -4
dqdx = 1.0
dqdy = 1.0
# now backprop through q = x + y
dfdx = dfdq * dqdx  # The multiplication here is the chain rule!
dfdy = dfdq * dqdy  
```

</div>

</div>

We are left with the gradient in the variables `[dfdx,dfdy,dfdz]`, which
tell us the sensitivity of the variables `x,y,z` on `f`!. This is the
simplest example of backpropagation. Going forward, we will use a more
concise notation that omits the `df` prefix. For example, we will simply
write `dq` instead of `dfdq`, and always assume that the gradient is
computed on the final output.

This computation can also be nicely visualized with a circuit diagram:

<div class="fig figleft fighighlight">

![](data:image/svg+xml;base64,PHN2ZyBzdHlsZT0ibWF4LXdpZHRoOiA0MjBweCIgdmlld2JveD0iMCAwIDQyMCAyMjAiPjxkZWZzPjxtYXJrZXIgaWQ9ImFycm93aGVhZCIgcmVmeD0iNiIgcmVmeT0iMiIgbWFya2Vyd2lkdGg9IjYiIG1hcmtlcmhlaWdodD0iNCIgb3JpZW50PSJhdXRvIj48cGF0aCBkPSJNIDAsMCBWIDQgTDYsMiBaIiAvPjwvbWFya2VyPjwvZGVmcz48bGluZSB4MT0iNDAiIHkxPSIzMCIgeDI9IjExMCIgeTI9IjMwIiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSI0NSIgeT0iMjQiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9ImdyZWVuIj4tMjwvdGV4dD48dGV4dCB4PSI0NSIgeT0iNDciIGZvbnQtc2l6ZT0iMTYiIGZpbGw9InJlZCI+LTQ8L3RleHQ+PHRleHQgeD0iMzUiIHk9IjI0IiBmb250LXNpemU9IjE2IiB0ZXh0LWFuY2hvcj0iZW5kIiBmaWxsPSJibGFjayI+eDwvdGV4dD48bGluZSB4MT0iNDAiIHkxPSIxMDAiIHgyPSIxMTAiIHkyPSIxMDAiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjQ1IiB5PSI5NCIgZm9udC1zaXplPSIxNiIgZmlsbD0iZ3JlZW4iPjU8L3RleHQ+PHRleHQgeD0iNDUiIHk9IjExNyIgZm9udC1zaXplPSIxNiIgZmlsbD0icmVkIj4tNDwvdGV4dD48dGV4dCB4PSIzNSIgeT0iOTQiIGZvbnQtc2l6ZT0iMTYiIHRleHQtYW5jaG9yPSJlbmQiIGZpbGw9ImJsYWNrIj55PC90ZXh0PjxsaW5lIHgxPSI0MCIgeTE9IjE3MCIgeDI9IjExMCIgeTI9IjE3MCIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIj48L2xpbmU+PHRleHQgeD0iNDUiIHk9IjE2NCIgZm9udC1zaXplPSIxNiIgZmlsbD0iZ3JlZW4iPi00PC90ZXh0Pjx0ZXh0IHg9IjQ1IiB5PSIxODciIGZvbnQtc2l6ZT0iMTYiIGZpbGw9InJlZCI+MzwvdGV4dD48dGV4dCB4PSIzNSIgeT0iMTY0IiBmb250LXNpemU9IjE2IiB0ZXh0LWFuY2hvcj0iZW5kIiBmaWxsPSJibGFjayI+ejwvdGV4dD48bGluZSB4MT0iMjEwIiB5MT0iNjUiIHgyPSIyODAiIHkyPSI2NSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIj48L2xpbmU+PHRleHQgeD0iMjE1IiB5PSI1OSIgZm9udC1zaXplPSIxNiIgZmlsbD0iZ3JlZW4iPjM8L3RleHQ+PHRleHQgeD0iMjE1IiB5PSI4MiIgZm9udC1zaXplPSIxNiIgZmlsbD0icmVkIj4tNDwvdGV4dD48dGV4dCB4PSIyMDUiIHk9IjU5IiBmb250LXNpemU9IjE2IiB0ZXh0LWFuY2hvcj0iZW5kIiBmaWxsPSJibGFjayI+cTwvdGV4dD48Y2lyY2xlIGN4PSIxNzAiIGN5PSI2NSIgZmlsbD0id2hpdGUiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSIgcj0iMjAiPjwvY2lyY2xlPjx0ZXh0IHg9IjE3MCIgeT0iNzAiIGZvbnQtc2l6ZT0iMjAiIGZpbGw9ImJsYWNrIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj4rPC90ZXh0PjxsaW5lIHgxPSIxMTAiIHkxPSIzMCIgeDI9IjE1MCIgeTI9IjY1IiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIG1hcmtlci1lbmQ9InVybCgjYXJyb3doZWFkKSI+PC9saW5lPjxsaW5lIHgxPSIxMTAiIHkxPSIxMDAiIHgyPSIxNTAiIHkyPSI2NSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48bGluZSB4MT0iMTkwIiB5MT0iNjUiIHgyPSIyMTAiIHkyPSI2NSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48bGluZSB4MT0iMzgwIiB5MT0iMTE3IiB4Mj0iNDUwIiB5Mj0iMTE3IiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSIzODUiIHk9IjExMSIgZm9udC1zaXplPSIxNiIgZmlsbD0iZ3JlZW4iPi0xMjwvdGV4dD48dGV4dCB4PSIzODUiIHk9IjEzNCIgZm9udC1zaXplPSIxNiIgZmlsbD0icmVkIj4xPC90ZXh0Pjx0ZXh0IHg9IjM3NSIgeT0iMTExIiBmb250LXNpemU9IjE2IiB0ZXh0LWFuY2hvcj0iZW5kIiBmaWxsPSJibGFjayI+ZjwvdGV4dD48Y2lyY2xlIGN4PSIzNDAiIGN5PSIxMTciIGZpbGw9IndoaXRlIiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIHI9IjIwIj48L2NpcmNsZT48dGV4dCB4PSIzNDAiIHk9IjEyNyIgZm9udC1zaXplPSIyMCIgZmlsbD0iYmxhY2siIHRleHQtYW5jaG9yPSJtaWRkbGUiPio8L3RleHQ+PGxpbmUgeDE9IjI4MCIgeTE9IjY1IiB4Mj0iMzIwIiB5Mj0iMTE3IiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIG1hcmtlci1lbmQ9InVybCgjYXJyb3doZWFkKSI+PC9saW5lPjxsaW5lIHgxPSIxMTAiIHkxPSIxNzAiIHgyPSIzMjAiIHkyPSIxMTciIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSIgbWFya2VyLWVuZD0idXJsKCNhcnJvd2hlYWQpIj48L2xpbmU+PGxpbmUgeDE9IjM2MCIgeTE9IjExNyIgeDI9IjM4MCIgeTI9IjExNyIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48L3N2Zz4=)

<div class="figcaption">

The real-valued *"circuit"* on left shows the visual representation of
the computation. The **forward pass** computes values from inputs to
output (shown in green). The **backward pass** then performs
backpropagation which starts at the end and recursively applies the
chain rule to compute the gradients (shown in red) all the way to the
inputs of the circuit. The gradients can be thought of as flowing
backwards through the circuit.

</div>

<div style="clear:both;">

</div>

</div>

<span id="intuitive"></span>

### Intuitive understanding of backpropagation

Notice that backpropagation is a beautifully local process. Every gate
in a circuit diagram gets some inputs and can right away compute two
things: 1. its output value and 2. the *local* gradient of its output
with respect to its inputs. Notice that the gates can do this completely
independently without being aware of any of the details of the full
circuit that they are embedded in. However, once the forward pass is
over, during backpropagation the gate will eventually learn about the
gradient of its output value on the final output of the entire circuit.
Chain rule says that the gate should take that gradient and multiply it
into every gradient it normally computes for all of its inputs.

> This extra multiplication (for each input) due to the chain rule can
> turn a single and relatively useless gate into a cog in a complex
> circuit such as an entire neural network.

Lets get an intuition for how this works by referring again to the
example. The add gate received inputs \[-2, 5\] and computed output 3.
Since the gate is computing the addition operation, its local gradient
for both of its inputs is +1. The rest of the circuit computed the final
value, which is -12. During the backward pass in which the chain rule is
applied recursively backwards through the circuit, the add gate (which
is an input to the multiply gate) learns that the gradient for its
output was -4. If we anthropomorphize the circuit as wanting to output a
higher value (which can help with intuition), then we can think of the
circuit as “wanting” the output of the add gate to be lower (due to
negative sign), and with a *force* of 4. To continue the recurrence and
to chain the gradient, the add gate takes that gradient and multiplies
it to all of the local gradients for its inputs (making the gradient on
both **x** and **y** 1 \* -4 = -4). Notice that this has the desired
effect: If **x,y** were to decrease (responding to their negative
gradient) then the add gate’s output would decrease, which in turn makes
the multiply gate’s output increase.

Backpropagation can thus be thought of as gates communicating to each
other (through the gradient signal) whether they want their outputs to
increase or decrease (and how strongly), so as to make the final output
value higher.

<span id="sigmoid"></span>

### Modularity: Sigmoid example

The gates we introduced above are relatively arbitrary. Any kind of
differentiable function can act as a gate, and we can group multiple
gates into a single gate, or decompose a function into multiple gates
whenever it is convenient. Lets look at another expression that
illustrates this point:

\\f(w,x) = \frac{1}{1+e^{-(w_0x_0 + w_1x_1 + w_2)}}\\

as we will see later in the class, this expression describes a
2-dimensional neuron (with inputs **x** and weights **w**) that uses the
*sigmoid activation* function. But for now lets think of this very
simply as just a function from inputs *w,x* to a single number. The
function is made up of multiple gates. In addition to the ones described
already above (add, mul, max), there are four more:

\\f(x) = \frac{1}{x} \hspace{1in} \rightarrow \hspace{1in} \frac{df}{dx}
= -1/x^2 \\\\ f_c(x) = c + x \hspace{1in} \rightarrow \hspace{1in}
\frac{df}{dx} = 1 \\\\ f(x) = e^x \hspace{1in} \rightarrow \hspace{1in}
\frac{df}{dx} = e^x \\\\ f_a(x) = ax \hspace{1in} \rightarrow
\hspace{1in} \frac{df}{dx} = a\\

Where the functions \\f_c, f_a\\ translate the input by a constant of
\\c\\ and scale the input by a constant of \\a\\, respectively. These
are technically special cases of addition and multiplication, but we
introduce them as (new) unary gates here since we do not need the
gradients for the constants \\c,a\\. The full circuit then looks as
follows:

<div class="fig figleft fighighlight">

![](data:image/svg+xml;base64,PHN2ZyBzdHlsZT0ibWF4LXdpZHRoOiA3OTlweCIgdmlld2JveD0iMCAwIDc5OSAzMDYiPjxnIHRyYW5zZm9ybT0ic2NhbGUoMC44KSI+PGRlZnM+PG1hcmtlciBpZD0iYXJyb3doZWFkIiByZWZ4PSI2IiByZWZ5PSIyIiBtYXJrZXJ3aWR0aD0iNiIgbWFya2VyaGVpZ2h0PSI0IiBvcmllbnQ9ImF1dG8iPjxwYXRoIGQ9Ik0gMCwwIFYgNCBMNiwyIFoiIC8+PC9tYXJrZXI+PC9kZWZzPjxsaW5lIHgxPSI1MCIgeTE9IjMwIiB4Mj0iOTAiIHkyPSIzMCIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIj48L2xpbmU+PHRleHQgeD0iNTUiIHk9IjI0IiBmb250LXNpemU9IjE2IiBmaWxsPSJncmVlbiI+Mi4wMDwvdGV4dD48dGV4dCB4PSI1NSIgeT0iNDciIGZvbnQtc2l6ZT0iMTYiIGZpbGw9InJlZCI+LTAuMjA8L3RleHQ+PHRleHQgeD0iNDUiIHk9IjI0IiBmb250LXNpemU9IjE2IiB0ZXh0LWFuY2hvcj0iZW5kIiBmaWxsPSJibGFjayI+dzA8L3RleHQ+PGxpbmUgeDE9IjUwIiB5MT0iMTAwIiB4Mj0iOTAiIHkyPSIxMDAiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjU1IiB5PSI5NCIgZm9udC1zaXplPSIxNiIgZmlsbD0iZ3JlZW4iPi0xLjAwPC90ZXh0Pjx0ZXh0IHg9IjU1IiB5PSIxMTciIGZvbnQtc2l6ZT0iMTYiIGZpbGw9InJlZCI+MC4zOTwvdGV4dD48dGV4dCB4PSI0NSIgeT0iOTQiIGZvbnQtc2l6ZT0iMTYiIHRleHQtYW5jaG9yPSJlbmQiIGZpbGw9ImJsYWNrIj54MDwvdGV4dD48bGluZSB4MT0iNTAiIHkxPSIxNzAiIHgyPSI5MCIgeTI9IjE3MCIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIj48L2xpbmU+PHRleHQgeD0iNTUiIHk9IjE2NCIgZm9udC1zaXplPSIxNiIgZmlsbD0iZ3JlZW4iPi0zLjAwPC90ZXh0Pjx0ZXh0IHg9IjU1IiB5PSIxODciIGZvbnQtc2l6ZT0iMTYiIGZpbGw9InJlZCI+LTAuMzk8L3RleHQ+PHRleHQgeD0iNDUiIHk9IjE2NCIgZm9udC1zaXplPSIxNiIgdGV4dC1hbmNob3I9ImVuZCIgZmlsbD0iYmxhY2siPncxPC90ZXh0PjxsaW5lIHgxPSI1MCIgeTE9IjI0MCIgeDI9IjkwIiB5Mj0iMjQwIiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSI1NSIgeT0iMjM0IiBmb250LXNpemU9IjE2IiBmaWxsPSJncmVlbiI+LTIuMDA8L3RleHQ+PHRleHQgeD0iNTUiIHk9IjI1NyIgZm9udC1zaXplPSIxNiIgZmlsbD0icmVkIj4tMC41OTwvdGV4dD48dGV4dCB4PSI0NSIgeT0iMjM0IiBmb250LXNpemU9IjE2IiB0ZXh0LWFuY2hvcj0iZW5kIiBmaWxsPSJibGFjayI+eDE8L3RleHQ+PGxpbmUgeDE9IjUwIiB5MT0iMzEwIiB4Mj0iOTAiIHkyPSIzMTAiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjU1IiB5PSIzMDQiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9ImdyZWVuIj4tMy4wMDwvdGV4dD48dGV4dCB4PSI1NSIgeT0iMzI3IiBmb250LXNpemU9IjE2IiBmaWxsPSJyZWQiPjAuMjA8L3RleHQ+PHRleHQgeD0iNDUiIHk9IjMwNCIgZm9udC1zaXplPSIxNiIgdGV4dC1hbmNob3I9ImVuZCIgZmlsbD0iYmxhY2siPncyPC90ZXh0PjxsaW5lIHgxPSIxNzAiIHkxPSI2NSIgeDI9IjIxMCIgeTI9IjY1IiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSIxNzUiIHk9IjU5IiBmb250LXNpemU9IjE2IiBmaWxsPSJncmVlbiI+LTIuMDA8L3RleHQ+PHRleHQgeD0iMTc1IiB5PSI4MiIgZm9udC1zaXplPSIxNiIgZmlsbD0icmVkIj4wLjIwPC90ZXh0PjxjaXJjbGUgY3g9IjEzMCIgY3k9IjY1IiBmaWxsPSJ3aGl0ZSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiByPSIyMCI+PC9jaXJjbGU+PHRleHQgeD0iMTMwIiB5PSI3NSIgZm9udC1zaXplPSIyMCIgZmlsbD0iYmxhY2siIHRleHQtYW5jaG9yPSJtaWRkbGUiPio8L3RleHQ+PGxpbmUgeDE9IjkwIiB5MT0iMzAiIHgyPSIxMTAiIHkyPSI2NSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48bGluZSB4MT0iOTAiIHkxPSIxMDAiIHgyPSIxMTAiIHkyPSI2NSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48bGluZSB4MT0iMTUwIiB5MT0iNjUiIHgyPSIxNzAiIHkyPSI2NSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48bGluZSB4MT0iMTcwIiB5MT0iMjA1IiB4Mj0iMjEwIiB5Mj0iMjA1IiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSIxNzUiIHk9IjE5OSIgZm9udC1zaXplPSIxNiIgZmlsbD0iZ3JlZW4iPjYuMDA8L3RleHQ+PHRleHQgeD0iMTc1IiB5PSIyMjIiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9InJlZCI+MC4yMDwvdGV4dD48Y2lyY2xlIGN4PSIxMzAiIGN5PSIyMDUiIGZpbGw9IndoaXRlIiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIHI9IjIwIj48L2NpcmNsZT48dGV4dCB4PSIxMzAiIHk9IjIxNSIgZm9udC1zaXplPSIyMCIgZmlsbD0iYmxhY2siIHRleHQtYW5jaG9yPSJtaWRkbGUiPio8L3RleHQ+PGxpbmUgeDE9IjkwIiB5MT0iMTcwIiB4Mj0iMTEwIiB5Mj0iMjA1IiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIG1hcmtlci1lbmQ9InVybCgjYXJyb3doZWFkKSI+PC9saW5lPjxsaW5lIHgxPSI5MCIgeTE9IjI0MCIgeDI9IjExMCIgeTI9IjIwNSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48bGluZSB4MT0iMTUwIiB5MT0iMjA1IiB4Mj0iMTcwIiB5Mj0iMjA1IiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIG1hcmtlci1lbmQ9InVybCgjYXJyb3doZWFkKSI+PC9saW5lPjxsaW5lIHgxPSIyOTAiIHkxPSIxMzUiIHgyPSIzMzAiIHkyPSIxMzUiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjI5NSIgeT0iMTI5IiBmb250LXNpemU9IjE2IiBmaWxsPSJncmVlbiI+NC4wMDwvdGV4dD48dGV4dCB4PSIyOTUiIHk9IjE1MiIgZm9udC1zaXplPSIxNiIgZmlsbD0icmVkIj4wLjIwPC90ZXh0PjxjaXJjbGUgY3g9IjI1MCIgY3k9IjEzNSIgZmlsbD0id2hpdGUiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSIgcj0iMjAiPjwvY2lyY2xlPjx0ZXh0IHg9IjI1MCIgeT0iMTQwIiBmb250LXNpemU9IjIwIiBmaWxsPSJibGFjayIgdGV4dC1hbmNob3I9Im1pZGRsZSI+KzwvdGV4dD48bGluZSB4MT0iMjEwIiB5MT0iNjUiIHgyPSIyMzAiIHkyPSIxMzUiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSIgbWFya2VyLWVuZD0idXJsKCNhcnJvd2hlYWQpIj48L2xpbmU+PGxpbmUgeDE9IjIxMCIgeTE9IjIwNSIgeDI9IjIzMCIgeTI9IjEzNSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48bGluZSB4MT0iMjcwIiB5MT0iMTM1IiB4Mj0iMjkwIiB5Mj0iMTM1IiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIG1hcmtlci1lbmQ9InVybCgjYXJyb3doZWFkKSI+PC9saW5lPjxsaW5lIHgxPSI0MTAiIHkxPSIyMjIiIHgyPSI0NTAiIHkyPSIyMjIiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjQxNSIgeT0iMjE2IiBmb250LXNpemU9IjE2IiBmaWxsPSJncmVlbiI+MS4wMDwvdGV4dD48dGV4dCB4PSI0MTUiIHk9IjIzOSIgZm9udC1zaXplPSIxNiIgZmlsbD0icmVkIj4wLjIwPC90ZXh0PjxjaXJjbGUgY3g9IjM3MCIgY3k9IjIyMiIgZmlsbD0id2hpdGUiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSIgcj0iMjAiPjwvY2lyY2xlPjx0ZXh0IHg9IjM3MCIgeT0iMjI3IiBmb250LXNpemU9IjIwIiBmaWxsPSJibGFjayIgdGV4dC1hbmNob3I9Im1pZGRsZSI+KzwvdGV4dD48bGluZSB4MT0iMzMwIiB5MT0iMTM1IiB4Mj0iMzUwIiB5Mj0iMjIyIiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIG1hcmtlci1lbmQ9InVybCgjYXJyb3doZWFkKSI+PC9saW5lPjxsaW5lIHgxPSI5MCIgeTE9IjMxMCIgeDI9IjM1MCIgeTI9IjIyMiIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48bGluZSB4MT0iMzkwIiB5MT0iMjIyIiB4Mj0iNDEwIiB5Mj0iMjIyIiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIG1hcmtlci1lbmQ9InVybCgjYXJyb3doZWFkKSI+PC9saW5lPjxsaW5lIHgxPSI1MzAiIHkxPSIyMjIiIHgyPSI1NzAiIHkyPSIyMjIiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjUzNSIgeT0iMjE2IiBmb250LXNpemU9IjE2IiBmaWxsPSJncmVlbiI+LTEuMDA8L3RleHQ+PHRleHQgeD0iNTM1IiB5PSIyMzkiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9InJlZCI+LTAuMjA8L3RleHQ+PGNpcmNsZSBjeD0iNDkwIiBjeT0iMjIyIiBmaWxsPSJ3aGl0ZSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiByPSIyMCI+PC9jaXJjbGU+PHRleHQgeD0iNDkwIiB5PSIyMjciIGZvbnQtc2l6ZT0iMjAiIGZpbGw9ImJsYWNrIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj4qLTE8L3RleHQ+PGxpbmUgeDE9IjQ1MCIgeTE9IjIyMiIgeDI9IjQ3MCIgeTI9IjIyMiIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48bGluZSB4MT0iNTEwIiB5MT0iMjIyIiB4Mj0iNTMwIiB5Mj0iMjIyIiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIG1hcmtlci1lbmQ9InVybCgjYXJyb3doZWFkKSI+PC9saW5lPjxsaW5lIHgxPSI2NTAiIHkxPSIyMjIiIHgyPSI2OTAiIHkyPSIyMjIiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjY1NSIgeT0iMjE2IiBmb250LXNpemU9IjE2IiBmaWxsPSJncmVlbiI+MC4zNzwvdGV4dD48dGV4dCB4PSI2NTUiIHk9IjIzOSIgZm9udC1zaXplPSIxNiIgZmlsbD0icmVkIj4tMC41MzwvdGV4dD48Y2lyY2xlIGN4PSI2MTAiIGN5PSIyMjIiIGZpbGw9IndoaXRlIiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIHI9IjIwIj48L2NpcmNsZT48dGV4dCB4PSI2MTAiIHk9IjIyNyIgZm9udC1zaXplPSIyMCIgZmlsbD0iYmxhY2siIHRleHQtYW5jaG9yPSJtaWRkbGUiPmV4cDwvdGV4dD48bGluZSB4MT0iNTcwIiB5MT0iMjIyIiB4Mj0iNTkwIiB5Mj0iMjIyIiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIG1hcmtlci1lbmQ9InVybCgjYXJyb3doZWFkKSI+PC9saW5lPjxsaW5lIHgxPSI2MzAiIHkxPSIyMjIiIHgyPSI2NTAiIHkyPSIyMjIiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSIgbWFya2VyLWVuZD0idXJsKCNhcnJvd2hlYWQpIj48L2xpbmU+PGxpbmUgeDE9Ijc3MCIgeTE9IjIyMiIgeDI9IjgxMCIgeTI9IjIyMiIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIj48L2xpbmU+PHRleHQgeD0iNzc1IiB5PSIyMTYiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9ImdyZWVuIj4xLjM3PC90ZXh0Pjx0ZXh0IHg9Ijc3NSIgeT0iMjM5IiBmb250LXNpemU9IjE2IiBmaWxsPSJyZWQiPi0wLjUzPC90ZXh0PjxjaXJjbGUgY3g9IjczMCIgY3k9IjIyMiIgZmlsbD0id2hpdGUiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSIgcj0iMjAiPjwvY2lyY2xlPjx0ZXh0IHg9IjczMCIgeT0iMjI3IiBmb250LXNpemU9IjIwIiBmaWxsPSJibGFjayIgdGV4dC1hbmNob3I9Im1pZGRsZSI+KzE8L3RleHQ+PGxpbmUgeDE9IjY5MCIgeTE9IjIyMiIgeDI9IjcxMCIgeTI9IjIyMiIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48bGluZSB4MT0iNzUwIiB5MT0iMjIyIiB4Mj0iNzcwIiB5Mj0iMjIyIiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIG1hcmtlci1lbmQ9InVybCgjYXJyb3doZWFkKSI+PC9saW5lPjxsaW5lIHgxPSI4OTAiIHkxPSIyMjIiIHgyPSI5MzAiIHkyPSIyMjIiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9Ijg5NSIgeT0iMjE2IiBmb250LXNpemU9IjE2IiBmaWxsPSJncmVlbiI+MC43MzwvdGV4dD48dGV4dCB4PSI4OTUiIHk9IjIzOSIgZm9udC1zaXplPSIxNiIgZmlsbD0icmVkIj4xLjAwPC90ZXh0PjxjaXJjbGUgY3g9Ijg1MCIgY3k9IjIyMiIgZmlsbD0id2hpdGUiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSIgcj0iMjAiPjwvY2lyY2xlPjx0ZXh0IHg9Ijg1MCIgeT0iMjI3IiBmb250LXNpemU9IjIwIiBmaWxsPSJibGFjayIgdGV4dC1hbmNob3I9Im1pZGRsZSI+MS94PC90ZXh0PjxsaW5lIHgxPSI4MTAiIHkxPSIyMjIiIHgyPSI4MzAiIHkyPSIyMjIiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSIgbWFya2VyLWVuZD0idXJsKCNhcnJvd2hlYWQpIj48L2xpbmU+PGxpbmUgeDE9Ijg3MCIgeTE9IjIyMiIgeDI9Ijg5MCIgeTI9IjIyMiIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48L2c+PC9zdmc+)

<div class="figcaption">

Example circuit for a 2D neuron with a sigmoid activation function. The
inputs are \[x0,x1\] and the (learnable) weights of the neuron are
\[w0,w1,w2\]. As we will see later, the neuron computes a dot product
with the input and then its activation is softly squashed by the sigmoid
function to be in range from 0 to 1.

</div>

<div style="clear:both;">

</div>

</div>

In the example above, we see a long chain of function applications that
operates on the result of the dot product between **w,x**. The function
that these operations implement is called the *sigmoid function*
\\\sigma(x)\\. It turns out that the derivative of the sigmoid function
with respect to its input simplifies if you perform the derivation
(after a fun tricky part where we add and subtract a 1 in the
numerator):

\\\sigma(x) = \frac{1}{1+e^{-x}} \\\\ \rightarrow \hspace{0.3in}
\frac{d\sigma(x)}{dx} = \frac{e^{-x}}{(1+e^{-x})^2} = \left( \frac{1 +
e^{-x} - 1}{1 + e^{-x}} \right) \left( \frac{1}{1+e^{-x}} \right) =
\left( 1 - \sigma(x) \right) \sigma(x)\\

As we see, the gradient turns out to simplify and becomes surprisingly
simple. For example, the sigmoid expression receives the input 1.0 and
computes the output 0.73 during the forward pass. The derivation above
shows that the *local* gradient would simply be (1 - 0.73) \* 0.73 ~=
0.2, as the circuit computed before (see the image above), except this
way it would be done with a single, simple and efficient expression (and
with less numerical issues). Therefore, in any real practical
application it would be very useful to group these operations into a
single gate. Lets see the backprop for this neuron in code:

<div class="language-python highlighter-rouge">

<div class="highlight">

``` highlight
w = [2,-3,-3] # assume some random weights and data
x = [-1, -2]

# forward pass
dot = w[0]*x[0] + w[1]*x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot)) # sigmoid function

# backward pass through the neuron (backpropagation)
ddot = (1 - f) * f # gradient on dot variable, using the sigmoid gradient derivation
dx = [w[0] * ddot, w[1] * ddot] # backprop into x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # backprop into w
# we're done! we have the gradients on the inputs to the circuit
```

</div>

</div>

**Implementation protip: staged backpropagation**. As shown in the code
above, in practice it is always helpful to break down the forward pass
into stages that are easily backpropped through. For example here we
created an intermediate variable `dot` which holds the output of the dot
product between `w` and `x`. During backward pass we then successively
compute (in reverse order) the corresponding variables (e.g. `ddot`, and
ultimately `dw, dx`) that hold the gradients of those variables.

The point of this section is that the details of how the backpropagation
is performed, and which parts of the forward function we think of as
gates, is a matter of convenience. It helps to be aware of which parts
of the expression have easy local gradients, so that they can be chained
together with the least amount of code and effort.

<span id="staged"></span>

### Backprop in practice: Staged computation

Lets see this with another example. Suppose that we have a function of
the form:

\\f(x,y) = \frac{x + \sigma(y)}{\sigma(x) + (x+y)^2}\\

To be clear, this function is completely useless and it’s not clear why
you would ever want to compute its gradient, except for the fact that it
is a good example of backpropagation in practice. It is very important
to stress that if you were to launch into performing the differentiation
with respect to either \\x\\ or \\y\\, you would end up with very large
and complex expressions. However, it turns out that doing so is
completely unnecessary because we don’t need to have an explicit
function written down that evaluates the gradient. We only have to know
how to compute it. Here is how we would structure the forward pass of
such expression:

<div class="language-python highlighter-rouge">

<div class="highlight">

``` highlight
x = 3 # example values
y = -4

# forward pass
sigy = 1.0 / (1 + math.exp(-y)) # sigmoid in numerator   #(1)
num = x + sigy # numerator                               #(2)
sigx = 1.0 / (1 + math.exp(-x)) # sigmoid in denominator #(3)
xpy = x + y                                              #(4)
xpysqr = xpy**2                                          #(5)
den = sigx + xpysqr # denominator                        #(6)
invden = 1.0 / den                                       #(7)
f = num * invden # done!                                 #(8)
```

</div>

</div>

Phew, by the end of the expression we have computed the forward pass.
Notice that we have structured the code in such way that it contains
multiple intermediate variables, each of which are only simple
expressions for which we already know the local gradients. Therefore,
computing the backprop pass is easy: We’ll go backwards and for every
variable along the way in the forward pass
(`sigy, num, sigx, xpy, xpysqr, den, invden`) we will have the same
variable, but one that begins with a `d`, which will hold the gradient
of the output of the circuit with respect to that variable.
Additionally, note that every single piece in our backprop will involve
computing the local gradient of that expression, and chaining it with
the gradient on that expression with a multiplication. For each row, we
also highlight which part of the forward pass it refers to:

<div class="language-python highlighter-rouge">

<div class="highlight">

``` highlight
# backprop f = num * invden
dnum = invden # gradient on numerator                             #(8)
dinvden = num                                                     #(8)
# backprop invden = 1.0 / den 
dden = (-1.0 / (den**2)) * dinvden                                #(7)
# backprop den = sigx + xpysqr
dsigx = (1) * dden                                                #(6)
dxpysqr = (1) * dden                                              #(6)
# backprop xpysqr = xpy**2
dxpy = (2 * xpy) * dxpysqr                                        #(5)
# backprop xpy = x + y
dx = (1) * dxpy                                                   #(4)
dy = (1) * dxpy                                                   #(4)
# backprop sigx = 1.0 / (1 + math.exp(-x))
dx += ((1 - sigx) * sigx) * dsigx # Notice += !! See notes below  #(3)
# backprop num = x + sigy
dx += (1) * dnum                                                  #(2)
dsigy = (1) * dnum                                                #(2)
# backprop sigy = 1.0 / (1 + math.exp(-y))
dy += ((1 - sigy) * sigy) * dsigy                                 #(1)
# done! phew
```

</div>

</div>

Notice a few things:

**Cache forward pass variables**. To compute the backward pass it is
very helpful to have some of the variables that were used in the forward
pass. In practice you want to structure your code so that you cache
these variables, and so that they are available during backpropagation.
If this is too difficult, it is possible (but wasteful) to recompute
them.

**Gradients add up at forks**. The forward expression involves the
variables **x,y** multiple times, so when we perform backpropagation we
must be careful to use `+=` instead of `=` to accumulate the gradient on
these variables (otherwise we would overwrite it). This follows the
*multivariable chain rule* in Calculus, which states that if a variable
branches out to different parts of the circuit, then the gradients that
flow back to it will add.

<span id="patterns"></span>

### Patterns in backward flow

It is interesting to note that in many cases the backward-flowing
gradient can be interpreted on an intuitive level. For example, the
three most commonly used gates in neural networks (*add,mul,max*), all
have very simple interpretations in terms of how they act during
backpropagation. Consider this example circuit:

<div class="fig figleft fighighlight">

![](data:image/svg+xml;base64,PHN2ZyBzdHlsZT0ibWF4LXdpZHRoOiA0NjBweCIgdmlld2JveD0iMCAwIDQ2MCAyOTAiPjxnIHRyYW5zZm9ybT0ic2NhbGUoMSkiPjxkZWZzPjxtYXJrZXIgaWQ9ImFycm93aGVhZCIgcmVmeD0iNiIgcmVmeT0iMiIgbWFya2Vyd2lkdGg9IjYiIG1hcmtlcmhlaWdodD0iNCIgb3JpZW50PSJhdXRvIj48cGF0aCBkPSJNIDAsMCBWIDQgTDYsMiBaIiAvPjwvbWFya2VyPjwvZGVmcz48bGluZSB4MT0iNTAiIHkxPSIzMCIgeDI9IjkwIiB5Mj0iMzAiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjU1IiB5PSIyNCIgZm9udC1zaXplPSIxNiIgZmlsbD0iZ3JlZW4iPjMuMDA8L3RleHQ+PHRleHQgeD0iNTUiIHk9IjQ3IiBmb250LXNpemU9IjE2IiBmaWxsPSJyZWQiPi04LjAwPC90ZXh0Pjx0ZXh0IHg9IjQ1IiB5PSIyNCIgZm9udC1zaXplPSIxNiIgdGV4dC1hbmNob3I9ImVuZCIgZmlsbD0iYmxhY2siPng8L3RleHQ+PGxpbmUgeDE9IjUwIiB5MT0iMTAwIiB4Mj0iOTAiIHkyPSIxMDAiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjU1IiB5PSI5NCIgZm9udC1zaXplPSIxNiIgZmlsbD0iZ3JlZW4iPi00LjAwPC90ZXh0Pjx0ZXh0IHg9IjU1IiB5PSIxMTciIGZvbnQtc2l6ZT0iMTYiIGZpbGw9InJlZCI+Ni4wMDwvdGV4dD48dGV4dCB4PSI0NSIgeT0iOTQiIGZvbnQtc2l6ZT0iMTYiIHRleHQtYW5jaG9yPSJlbmQiIGZpbGw9ImJsYWNrIj55PC90ZXh0PjxsaW5lIHgxPSI1MCIgeTE9IjE3MCIgeDI9IjkwIiB5Mj0iMTcwIiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSI1NSIgeT0iMTY0IiBmb250LXNpemU9IjE2IiBmaWxsPSJncmVlbiI+Mi4wMDwvdGV4dD48dGV4dCB4PSI1NSIgeT0iMTg3IiBmb250LXNpemU9IjE2IiBmaWxsPSJyZWQiPjIuMDA8L3RleHQ+PHRleHQgeD0iNDUiIHk9IjE2NCIgZm9udC1zaXplPSIxNiIgdGV4dC1hbmNob3I9ImVuZCIgZmlsbD0iYmxhY2siPno8L3RleHQ+PGxpbmUgeDE9IjUwIiB5MT0iMjQwIiB4Mj0iOTAiIHkyPSIyNDAiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjU1IiB5PSIyMzQiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9ImdyZWVuIj4tMS4wMDwvdGV4dD48dGV4dCB4PSI1NSIgeT0iMjU3IiBmb250LXNpemU9IjE2IiBmaWxsPSJyZWQiPjAuMDA8L3RleHQ+PHRleHQgeD0iNDUiIHk9IjIzNCIgZm9udC1zaXplPSIxNiIgdGV4dC1hbmNob3I9ImVuZCIgZmlsbD0iYmxhY2siPnc8L3RleHQ+PGxpbmUgeDE9IjE3MCIgeTE9IjY1IiB4Mj0iMjEwIiB5Mj0iNjUiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjE3NSIgeT0iNTkiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9ImdyZWVuIj4tMTIuMDA8L3RleHQ+PHRleHQgeD0iMTc1IiB5PSI4MiIgZm9udC1zaXplPSIxNiIgZmlsbD0icmVkIj4yLjAwPC90ZXh0PjxjaXJjbGUgY3g9IjEzMCIgY3k9IjY1IiBmaWxsPSJ3aGl0ZSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiByPSIyMCI+PC9jaXJjbGU+PHRleHQgeD0iMTMwIiB5PSI3NSIgZm9udC1zaXplPSIyMCIgZmlsbD0iYmxhY2siIHRleHQtYW5jaG9yPSJtaWRkbGUiPio8L3RleHQ+PGxpbmUgeDE9IjkwIiB5MT0iMzAiIHgyPSIxMTAiIHkyPSI2NSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48bGluZSB4MT0iOTAiIHkxPSIxMDAiIHgyPSIxMTAiIHkyPSI2NSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48bGluZSB4MT0iMTUwIiB5MT0iNjUiIHgyPSIxNzAiIHkyPSI2NSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48bGluZSB4MT0iMTcwIiB5MT0iMjA1IiB4Mj0iMjEwIiB5Mj0iMjA1IiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSIxNzUiIHk9IjE5OSIgZm9udC1zaXplPSIxNiIgZmlsbD0iZ3JlZW4iPjIuMDA8L3RleHQ+PHRleHQgeD0iMTc1IiB5PSIyMjIiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9InJlZCI+Mi4wMDwvdGV4dD48Y2lyY2xlIGN4PSIxMzAiIGN5PSIyMDUiIGZpbGw9IndoaXRlIiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIHI9IjIwIj48L2NpcmNsZT48dGV4dCB4PSIxMzAiIHk9IjIxMCIgZm9udC1zaXplPSIyMCIgZmlsbD0iYmxhY2siIHRleHQtYW5jaG9yPSJtaWRkbGUiPm1heDwvdGV4dD48bGluZSB4MT0iOTAiIHkxPSIxNzAiIHgyPSIxMTAiIHkyPSIyMDUiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSIgbWFya2VyLWVuZD0idXJsKCNhcnJvd2hlYWQpIj48L2xpbmU+PGxpbmUgeDE9IjkwIiB5MT0iMjQwIiB4Mj0iMTEwIiB5Mj0iMjA1IiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIG1hcmtlci1lbmQ9InVybCgjYXJyb3doZWFkKSI+PC9saW5lPjxsaW5lIHgxPSIxNTAiIHkxPSIyMDUiIHgyPSIxNzAiIHkyPSIyMDUiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSIgbWFya2VyLWVuZD0idXJsKCNhcnJvd2hlYWQpIj48L2xpbmU+PGxpbmUgeDE9IjI5MCIgeTE9IjEzNSIgeDI9IjMzMCIgeTI9IjEzNSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIj48L2xpbmU+PHRleHQgeD0iMjk1IiB5PSIxMjkiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9ImdyZWVuIj4tMTAuMDA8L3RleHQ+PHRleHQgeD0iMjk1IiB5PSIxNTIiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9InJlZCI+Mi4wMDwvdGV4dD48Y2lyY2xlIGN4PSIyNTAiIGN5PSIxMzUiIGZpbGw9IndoaXRlIiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIHI9IjIwIj48L2NpcmNsZT48dGV4dCB4PSIyNTAiIHk9IjE0MCIgZm9udC1zaXplPSIyMCIgZmlsbD0iYmxhY2siIHRleHQtYW5jaG9yPSJtaWRkbGUiPis8L3RleHQ+PGxpbmUgeDE9IjIxMCIgeTE9IjY1IiB4Mj0iMjMwIiB5Mj0iMTM1IiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIG1hcmtlci1lbmQ9InVybCgjYXJyb3doZWFkKSI+PC9saW5lPjxsaW5lIHgxPSIyMTAiIHkxPSIyMDUiIHgyPSIyMzAiIHkyPSIxMzUiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSIgbWFya2VyLWVuZD0idXJsKCNhcnJvd2hlYWQpIj48L2xpbmU+PGxpbmUgeDE9IjI3MCIgeTE9IjEzNSIgeDI9IjI5MCIgeTI9IjEzNSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48bGluZSB4MT0iNDEwIiB5MT0iMTM1IiB4Mj0iNDUwIiB5Mj0iMTM1IiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSI0MTUiIHk9IjEyOSIgZm9udC1zaXplPSIxNiIgZmlsbD0iZ3JlZW4iPi0yMC4wMDwvdGV4dD48dGV4dCB4PSI0MTUiIHk9IjE1MiIgZm9udC1zaXplPSIxNiIgZmlsbD0icmVkIj4xLjAwPC90ZXh0PjxjaXJjbGUgY3g9IjM3MCIgY3k9IjEzNSIgZmlsbD0id2hpdGUiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMSIgcj0iMjAiPjwvY2lyY2xlPjx0ZXh0IHg9IjM3MCIgeT0iMTQwIiBmb250LXNpemU9IjIwIiBmaWxsPSJibGFjayIgdGV4dC1hbmNob3I9Im1pZGRsZSI+KjI8L3RleHQ+PGxpbmUgeDE9IjMzMCIgeTE9IjEzNSIgeDI9IjM1MCIgeTI9IjEzNSIgc3Ryb2tlPSJibGFjayIgc3Ryb2tlLXdpZHRoPSIxIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiPjwvbGluZT48bGluZSB4MT0iMzkwIiB5MT0iMTM1IiB4Mj0iNDEwIiB5Mj0iMTM1IiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiIG1hcmtlci1lbmQ9InVybCgjYXJyb3doZWFkKSI+PC9saW5lPjwvZz48L3N2Zz4=)

<div class="figcaption">

An example circuit demonstrating the intuition behind the operations
that backpropagation performs during the backward pass in order to
compute the gradients on the inputs. Sum operation distributes gradients
equally to all its inputs. Max operation routes the gradient to the
higher input. Multiply gate takes the input activations, swaps them and
multiplies by its gradient.

</div>

<div style="clear:both;">

</div>

</div>

Looking at the diagram above as an example, we can see that:

The **add gate** always takes the gradient on its output and distributes
it equally to all of its inputs, regardless of what their values were
during the forward pass. This follows from the fact that the local
gradient for the add operation is simply +1.0, so the gradients on all
inputs will exactly equal the gradients on the output because it will be
multiplied by x1.0 (and remain unchanged). In the example circuit above,
note that the + gate routed the gradient of 2.00 to both of its inputs,
equally and unchanged.

The **max gate** routes the gradient. Unlike the add gate which
distributed the gradient unchanged to all its inputs, the max gate
distributes the gradient (unchanged) to exactly one of its inputs (the
input that had the highest value during the forward pass). This is
because the local gradient for a max gate is 1.0 for the highest value,
and 0.0 for all other values. In the example circuit above, the max
operation routed the gradient of 2.00 to the **z** variable, which had a
higher value than **w**, and the gradient on **w** remains zero.

The **multiply gate** is a little less easy to interpret. Its local
gradients are the input values (except switched), and this is multiplied
by the gradient on its output during the chain rule. In the example
above, the gradient on **x** is -8.00, which is -4.00 x 2.00.

*Unintuitive effects and their consequences*. Notice that if one of the
inputs to the multiply gate is very small and the other is very big,
then the multiply gate will do something slightly unintuitive: it will
assign a relatively huge gradient to the small input and a tiny gradient
to the large input. Note that in linear classifiers where the weights
are dot producted \\w^Tx_i\\ (multiplied) with the inputs, this implies
that the scale of the data has an effect on the magnitude of the
gradient for the weights. For example, if you multiplied all input data
examples \\x_i\\ by 1000 during preprocessing, then the gradient on the
weights will be 1000 times larger, and you’d have to lower the learning
rate by that factor to compensate. This is why preprocessing matters a
lot, sometimes in subtle ways! And having intuitive understanding for
how the gradients flow can help you debug some of these cases.

<span id="mat"></span>

### Gradients for vectorized operations

The above sections were concerned with single variables, but all
concepts extend in a straight-forward manner to matrix and vector
operations. However, one must pay closer attention to dimensions and
transpose operations.

**Matrix-Matrix multiply gradient**. Possibly the most tricky operation
is the matrix-matrix multiplication (which generalizes all matrix-vector
and vector-vector) multiply operations:

<div class="language-python highlighter-rouge">

<div class="highlight">

``` highlight
# forward pass
W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

# now suppose we had the gradient on D from above in the circuit
dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T) #.T gives the transpose of the matrix
dX = W.T.dot(dD)
```

</div>

</div>

*Tip: use dimension analysis!* Note that you do not need to remember the
expressions for `dW` and `dX` because they are easy to re-derive based
on dimensions. For instance, we know that the gradient on the weights
`dW` must be of the same size as `W` after it is computed, and that it
must depend on matrix multiplication of `X` and `dD` (as is the case
when both `X,W` are single numbers and not matrices). There is always
exactly one way of achieving this so that the dimensions work out. For
example, `X` is of size \[10 x 3\] and `dD` of size \[5 x 3\], so if we
want `dW` and `W` has shape \[5 x 10\], then the only way of achieving
this is with `dD.dot(X.T)`, as shown above.

**Work with small, explicit examples**. Some people may find it
difficult at first to derive the gradient updates for some vectorized
expressions. Our recommendation is to explicitly write out a minimal
vectorized example, derive the gradient on paper and then generalize the
pattern to its efficient, vectorized form.

Erik Learned-Miller has also written up a longer related document on
taking matrix/vector derivatives which you might find helpful. [Find it
here](http://cs231n.stanford.edu/vecDerivs.pdf).

<span id="summary"></span>

### Summary

- We developed intuition for what the gradients mean, how they flow
  backwards in the circuit, and how they communicate which part of the
  circuit should increase or decrease and with what force to make the
  final output higher.
- We discussed the importance of **staged computation** for practical
  implementations of backpropagation. You always want to break up your
  function into modules for which you can easily derive local gradients,
  and then chain them with chain rule. Crucially, you almost never want
  to write out these expressions on paper and differentiate them
  symbolically in full, because you never need an explicit mathematical
  equation for the gradient of the input variables. Hence, decompose
  your expressions into stages such that you can differentiate every
  stage independently (the stages will be matrix vector multiplies, or
  max operations, or sum operations, etc.) and then backprop through the
  variables one step at a time.

In the next section we will start to define neural networks, and
backpropagation will allow us to efficiently compute the gradient of a
loss function with respect to its parameters. In other words, we’re now
ready to train neural nets, and the most conceptually difficult part of
this class is behind us! ConvNets will then be a small step away.

### References

- [Automatic differentiation in machine learning: a
  survey](http://arxiv.org/abs/1502.05767)

</div>

</div>

</div>

<div class="wrap">

<div class="footer-col-1 column">

- [<span class="icon github"> <img
  src="data:image/svg+xml;base64,PHN2ZyB2ZXJzaW9uPSIxLjEiIGNsYXNzPSJnaXRodWItaWNvbi1zdmciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IiB2aWV3Ym94PSIwIDAgMTYgMTYiIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDE2IDE2IiB4bWw6c3BhY2U9InByZXNlcnZlIj4KICAgICAgICAgICAgICAgIDxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBmaWxsPSIjQzJDMkMyIiBkPSJNNy45OTksMC40MzFjLTQuMjg1LDAtNy43NiwzLjQ3NC03Ljc2LDcuNzYxCiAgICAgICAgICAgICAgICBjMCwzLjQyOCwyLjIyMyw2LjMzNyw1LjMwNyw3LjM2M2MwLjM4OCwwLjA3MSwwLjUzLTAuMTY4LDAuNTMtMC4zNzRjMC0wLjE4NC0wLjAwNy0wLjY3Mi0wLjAxLTEuMzIKICAgICAgICAgICAgICAgIGMtMi4xNTksMC40NjktMi42MTQtMS4wNC0yLjYxNC0xLjA0Yy0wLjM1My0wLjg5Ni0wLjg2Mi0xLjEzNS0wLjg2Mi0xLjEzNWMtMC43MDUtMC40ODEsMC4wNTMtMC40NzIsMC4wNTMtMC40NzIKICAgICAgICAgICAgICAgIGMwLjc3OSwwLjA1NSwxLjE4OSwwLjgsMS4xODksMC44YzAuNjkyLDEuMTg2LDEuODE2LDAuODQzLDIuMjU4LDAuNjQ1YzAuMDcxLTAuNTAyLDAuMjcxLTAuODQzLDAuNDkzLTEuMDM3CiAgICAgICAgICAgICAgICBDNC44NiwxMS40MjUsMy4wNDksMTAuNzYsMy4wNDksNy43ODZjMC0wLjg0NywwLjMwMi0xLjU0LDAuNzk5LTIuMDgyQzMuNzY4LDUuNTA3LDMuNTAxLDQuNzE4LDMuOTI0LDMuNjUKICAgICAgICAgICAgICAgIGMwLDAsMC42NTItMC4yMDksMi4xMzQsMC43OTZDNi42NzcsNC4yNzMsNy4zNCw0LjE4Nyw4LDQuMTg0YzAuNjU5LDAuMDAzLDEuMzIzLDAuMDg5LDEuOTQzLDAuMjYxCiAgICAgICAgICAgICAgICBjMS40ODItMS4wMDQsMi4xMzItMC43OTYsMi4xMzItMC43OTZjMC40MjMsMS4wNjgsMC4xNTcsMS44NTcsMC4wNzcsMi4wNTRjMC40OTcsMC41NDIsMC43OTgsMS4yMzUsMC43OTgsMi4wODIKICAgICAgICAgICAgICAgIGMwLDIuOTgxLTEuODE0LDMuNjM3LTMuNTQzLDMuODI5YzAuMjc5LDAuMjQsMC41MjcsMC43MTMsMC41MjcsMS40MzdjMCwxLjAzNy0wLjAxLDEuODc0LTAuMDEsMi4xMjkKICAgICAgICAgICAgICAgIGMwLDAuMjA4LDAuMTQsMC40NDksMC41MzQsMC4zNzNjMy4wODEtMS4wMjgsNS4zMDItMy45MzUsNS4zMDItNy4zNjJDMTUuNzYsMy45MDYsMTIuMjg1LDAuNDMxLDcuOTk5LDAuNDMxeiIgLz4KICAgICAgICAgICAgICA8L3N2Zz4="
  class="github-icon-svg" /> </span>
  <span class="username">cs231n</span>](https://github.com/cs231n)
- [<span class="icon twitter"> <img
  src="data:image/svg+xml;base64,PHN2ZyB2ZXJzaW9uPSIxLjEiIGNsYXNzPSJ0d2l0dGVyLWljb24tc3ZnIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCIgdmlld2JveD0iMCAwIDE2IDE2IiBlbmFibGUtYmFja2dyb3VuZD0ibmV3IDAgMCAxNiAxNiIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSI+CiAgICAgICAgICAgICAgICA8cGF0aCBmaWxsPSIjQzJDMkMyIiBkPSJNMTUuOTY5LDMuMDU4Yy0wLjU4NiwwLjI2LTEuMjE3LDAuNDM2LTEuODc4LDAuNTE1YzAuNjc1LTAuNDA1LDEuMTk0LTEuMDQ1LDEuNDM4LTEuODA5CiAgICAgICAgICAgICAgICBjLTAuNjMyLDAuMzc1LTEuMzMyLDAuNjQ3LTIuMDc2LDAuNzkzYy0wLjU5Ni0wLjYzNi0xLjQ0Ni0xLjAzMy0yLjM4Ny0xLjAzM2MtMS44MDYsMC0zLjI3LDEuNDY0LTMuMjcsMy4yNwogICAgICAgICAgICAgICAgYzAsMC4yNTYsMC4wMjksMC41MDYsMC4wODUsMC43NDVDNS4xNjMsNS40MDQsMi43NTMsNC4xMDIsMS4xNCwyLjEyNEMwLjg1OSwyLjYwNywwLjY5OCwzLjE2OCwwLjY5OCwzLjc2NwogICAgICAgICAgICAgICAgYzAsMS4xMzQsMC41NzcsMi4xMzUsMS40NTUsMi43MjJDMS42MTYsNi40NzIsMS4xMTIsNi4zMjUsMC42NzEsNi4wOGMwLDAuMDE0LDAsMC4wMjcsMCwwLjA0MWMwLDEuNTg0LDEuMTI3LDIuOTA2LDIuNjIzLDMuMjA2CiAgICAgICAgICAgICAgICBDMy4wMiw5LjQwMiwyLjczMSw5LjQ0MiwyLjQzMyw5LjQ0MmMtMC4yMTEsMC0wLjQxNi0wLjAyMS0wLjYxNS0wLjA1OWMwLjQxNiwxLjI5OSwxLjYyNCwyLjI0NSwzLjA1NSwyLjI3MQogICAgICAgICAgICAgICAgYy0xLjExOSwwLjg3Ny0yLjUyOSwxLjQtNC4wNjEsMS40Yy0wLjI2NCwwLTAuNTI0LTAuMDE1LTAuNzgtMC4wNDZjMS40NDcsMC45MjgsMy4xNjYsMS40NjksNS4wMTMsMS40NjkKICAgICAgICAgICAgICAgIGM2LjAxNSwwLDkuMzA0LTQuOTgzLDkuMzA0LTkuMzA0YzAtMC4xNDItMC4wMDMtMC4yODMtMC4wMDktMC40MjNDMTQuOTc2LDQuMjksMTUuNTMxLDMuNzE0LDE1Ljk2OSwzLjA1OHoiIC8+CiAgICAgICAgICAgICAgPC9zdmc+"
  class="twitter-icon-svg" /> </span>
  <span class="username">cs231n</span>](https://twitter.com/cs231n)

</div>

<div class="footer-col-2 column">

</div>

<div class="footer-col-3 column">

</div>

</div>
