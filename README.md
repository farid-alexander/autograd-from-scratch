# How PyTorch's autograd works, rebuilt in 60 lines of Python

Call `loss.backward()` and PyTorch fills in `.grad` for every parameter. It can look like magic, but the mechanism is small. During the forward pass, each operation leaves a note saying how to pass a gradient back through it. `backward()` follows those notes from the loss back to the parameters.

This article builds that mechanism in about 60 lines of plain Python and checks it against PyTorch. It is adapted from Chapter 6 of my book, *The Complete Guide to PyTorch 2: From Tensors to Compiled Deep Learning* ([Kindle](https://www.amazon.com/dp/B0HR97T3B2?maas=maas_adg_FF2CFA1CAA9C9A480A5C43C36AC2F16E_afap_abs&ref_=aa_maas&tag=maas), [paperback](https://www.amazon.com/dp/B0HRP36BS4?maas=maas_adg_F18C9CB4D7B524FC5D6A506C5C99FB91_afap_abs&ref_=aa_maas&tag=maas)).

If you know Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), the idea is the same. The difference is that each piece here is named and shaped like PyTorch's own objects, so you can print PyTorch's graph and this one side by side, and every gradient is checked against PyTorch.

The code below runs top to bottom as one script. The engine needs only Python; the comparisons need PyTorch. [`small_autograd.py`](small_autograd.py) holds the engine on its own, and [`check_with_pytorch.py`](check_with_pytorch.py) compares it with PyTorch on more expressions.

## One weight, one bias, one example

Take the smallest model that still has a gradient worth computing: a prediction $\hat y = wx + b$ and the squared error $L = (\hat y - y)^2$, with $x = 2$, $y = 5$, $w = 1$ and $b = 0$. The input and target are fixed. We want the gradients with respect to $w$ and $b$.

By hand, with the chain rule:

$$\frac{\partial L}{\partial w} = 2(wx + b - y)\,x = 2(-3)(2) = -12, \qquad \frac{\partial L}{\partial b} = 2(wx + b - y) = -6$$

In PyTorch, give each intermediate result a name:

```python
import torch

x = torch.tensor(2.0)
y = torch.tensor(5.0)
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

p = w * x
y_hat = p + b
e = y_hat - y
L = e ** 2

print(L)
```

```output
tensor(9., grad_fn=<PowBackward0>)
```

The loss is 9. The interesting part is `grad_fn`: PyTorch has recorded how `L` was made.

## What the forward pass leaves behind

Every result computed from a tensor that requires a gradient gets a `grad_fn`, a **backward node** for the operation that produced it. Each node lists in `next_functions` where it will send gradients:

```python
for name, t in [("p", p), ("y_hat", y_hat), ("e", e), ("L", L)]:
    links = [type(n).__name__ if n else None
             for n, _ in t.grad_fn.next_functions]
    print(name, type(t.grad_fn).__name__, links)
```

```output
p MulBackward0 ['AccumulateGrad', None]
y_hat AddBackward0 ['MulBackward0', 'AccumulateGrad']
e SubBackward0 ['AddBackward0', None]
L PowBackward0 ['SubBackward0']
```

Each operation has one link per input, and a link is one of three things:

- **Another backward node**, when the input was itself computed. `y_hat` links to `MulBackward0` because its input `p` came from the multiplication.
- **`AccumulateGrad`**, when the input is a leaf tensor that requires a gradient, like `w` and `b`. This node adds whatever arrives into the leaf's `.grad`.
- **`None`**, when the input needs no gradient, like `x` and `y`.

![The computation L = (w · x + b − y)² drawn left to right as a graph. The parameters w = 1 and b = 0 and the data x = 2 and y = 5 feed a multiplication, an addition, a subtraction and a square, which produce p = 2, ŷ = 2, e = −3 and L = 9. Each tracked value also shows the gradient of L with respect to it: 1 at L, −6 at e, ŷ, p and b, and −12 at w; x and y are marked "no grad".](images/computational-graph.svg)

*The graph for this example, from the book. The forward pass computes values left to right; the backward pass sends gradients right to left. Parameters are green, data gray, intermediate results blue, and each blue box names the backward node PyTorch records for it.*

Nothing has been differentiated yet. The forward pass has computed the values and kept the ones the backward rules will need: the multiplication keeps `x`, because $\partial p / \partial w = x$, and the square keeps `e`, because $\partial L / \partial e = 2e$.

## What `backward()` does with it

```python
L.backward()
print(w.grad, b.grad)
```

```output
tensor(-12.) tensor(-6.)
```

The engine starts at the loss with $\partial L / \partial L = 1$ and follows the links. Each node multiplies the gradient it receives, *g*, by its local derivative and passes the result on:

| Node | Rule for an incoming gradient *g* | Sends |
|---|---|---|
| `PowBackward0`, *L* = *e*² | *g* · 2*e* toward *e* | 1 · 2(−3) = −6 |
| `SubBackward0`, *e* = *ŷ* − *y* | *g* toward *ŷ* | −6 |
| `AddBackward0`, *ŷ* = *p* + *b* | *g* toward both *p* and *b* | −6 and −6 |
| `MulBackward0`, *p* = *wx* | *g* · *x* toward *w* | −6 · 2 = −12 |

The two `AccumulateGrad` nodes store −12 in `w.grad` and −6 in `b.grad`, the values worked out by hand. Autograd is recorded local rules, applied in the right order. Now build it.

## Recording the forward pass

Start with the two kinds of node:

```python
class Node:
    def __init__(self, name, rule, inputs):
        self.name = name
        self.rule = rule    # incoming gradient -> one per input
        self.next_functions = [t.grad_fn or t.accumulator
                               for t in inputs]

class AccumulateGrad:
    name = "AccumulateGrad"
    next_functions = []

    def __init__(self, leaf):
        self.leaf = leaf

    def rule(self, g):
        old = self.leaf.grad
        self.leaf.grad = g if old is None else old + g
        return []
```

A `Node` is what `grad_fn` will point to. Its `rule` takes the incoming gradient *g* and returns one gradient per input, like a row of the table above. Its `next_functions` holds the same three kinds of link PyTorch printed: an input's own `grad_fn` if an operation produced it, its accumulator if it is a leaf that requires a gradient, and `None` otherwise. `AccumulateGrad` ends a path by adding what arrives into the leaf's `.grad`.

Next, a number type whose operations record themselves:

```python
class Scalar:
    def __init__(self, value, requires_grad=False):
        self.value = value
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self.accumulator = (AccumulateGrad(self)
                            if requires_grad else None)

    def __add__(self, other):
        return record("AddBackward", self.value + other.value,
                      [self, other], lambda g: [g, g])

    def __sub__(self, other):
        return record("SubBackward", self.value - other.value,
                      [self, other], lambda g: [g, -g])

    def __mul__(self, other):
        a, b = self.value, other.value   # saved for backward
        return record("MulBackward", a * b, [self, other],
                      lambda g: [g * b, g * a])

def record(name, value, inputs, rule):
    out = Scalar(value)
    if any(t.requires_grad for t in inputs):
        out.requires_grad = True
        out.grad_fn = Node(name, rule, inputs)
    return out
```

Each operator computes its value and hands it to `record`. If any input requires a gradient, `record` creates the node and attaches it to the result. The values a backward rule needs are saved by the `lambda` that captures them. The multiplication keeps `a` and `b`, because the derivative of *ab* is *b* with respect to *a* and *a* with respect to *b*. Addition and subtraction keep nothing, since their local derivatives are 1 and −1 whatever the inputs. Like a tensor, a `Scalar` keeps its value (`value`) apart from its history (`grad_fn`).

Run the same forward pass with it. Writing the square as `e * e` keeps the class to three operations:

```python
x, y = Scalar(2.0), Scalar(5.0)
w = Scalar(1.0, requires_grad=True)
b = Scalar(0.0, requires_grad=True)

p = w * x
y_hat = p + b
e = y_hat - y
L = e * e

for name, t in [("p", p), ("y_hat", y_hat), ("e", e), ("L", L)]:
    links = [n.name if n else None
             for n in t.grad_fn.next_functions]
    print(name, t.grad_fn.name, links)
print(L.value, w.grad)
```

```output
p MulBackward ['AccumulateGrad', None]
y_hat AddBackward ['MulBackward', 'AccumulateGrad']
e SubBackward ['AddBackward', None]
L MulBackward ['SubBackward', 'SubBackward']
9.0 None
```

The links match PyTorch's. The only difference is the square: its node links to the subtraction node twice, once for each `e` in `e * e`. And `w.grad` is still `None`. Here too, the forward pass only builds the graph.

## Running the backward pass

```python
def backward(loss):
    order, seen = [], set()
    def visit(node):               # a node's inputs go first
        if node is not None and node not in seen:
            seen.add(node)
            for nxt in node.next_functions:
                visit(nxt)
            order.append(node)
    visit(loss.grad_fn)

    grads = {loss.grad_fn: 1.0}    # dL/dL = 1
    for node in reversed(order):   # the loss's node comes first
        g = grads.pop(node)        # every arrival, summed
        for nxt, g_in in zip(node.next_functions, node.rule(g)):
            if nxt is not None:
                grads[nxt] = grads.get(nxt, 0.0) + g_in

backward(L)
print(w.grad, b.grad)
```

```output
-12.0 -6.0
```

`backward` works in two stages.

First it settles the order. `visit` follows `next_functions` from the loss's node and appends a node to `order` only after every node it sends gradients to. Reversed, the list puts each node after all the nodes that send gradients to it, an order called a **topological order**. A node must not run until everything coming to it has arrived, and this order guarantees that.

Then it walks the nodes in that order, starting from $\partial L / \partial L = 1$. When a node's turn comes, its entry in `grads` already holds the sum of all its incoming gradients. It applies its rule and adds each result to the entry of the node that result goes to. The `AccumulateGrad` nodes come last and store the totals.

The sum is where `e * e` matters. The multiplication node sends −3 along each of its two links (*g* · *b* and *g* · *a*, with *g* = 1 and *a* = *b* = −3), and `grads` adds them into −6. That is the same $\partial L / \partial e$ that PyTorch's `PowBackward0` produced in one step.

## Values used more than once

That sum is the chain rule's answer for a value that reaches the loss along several paths: add the contributions. Check it in both engines on *s* = *a* · *a* + *a* at *a* = 3, where the derivative is 2*a* + 1 = 7:

```python
a = Scalar(3.0, requires_grad=True)
s = a * a + a
backward(s)

a_t = torch.tensor(3.0, requires_grad=True)
s_t = a_t * a_t + a_t
s_t.backward()

print(a.grad, a_t.grad.item())
```

```output
7.0 7.0
```

`a` receives 3 through each input of the multiplication and 1 through the addition.

Gradients also add up across backward calls, because `AccumulateGrad` adds to whatever `.grad` already holds. Run the backward pass of the first example a second time:

```python
backward(L)
print(w.grad)
```

```output
-24.0
```

PyTorch would refuse this particular call. It frees the saved values once backward has used them, so a second `L.backward()` on the same graph raises an error unless the first call passed `retain_graph=True`. After a fresh forward pass, though, PyTorch accumulates exactly like this, which is why training loops clear the gradients with `optimizer.zero_grad()` before each backward pass.

## What PyTorch adds

The real engine does the same job at a much larger scale:

- **Tensors instead of numbers.** Each rule receives a gradient tensor and computes a vector-Jacobian product without ever building the full Jacobian matrix.
- **Hundreds of rules.** The derivative formulas are written once (most are in `tools/autograd/derivatives.yaml` in the PyTorch source), and the C++ backward nodes are generated from them.
- **A different scheduler.** Instead of a recursive search, the engine counts the links into each node and runs a node once all its gradients have arrived. It is written in C++, not Python.
- **Memory.** Saved values are freed as soon as backward has used them, which is why the second call above fails in PyTorch.
- **Safety checks.** Every tensor carries a version counter, so changing a saved tensor in place before backward raises an error instead of giving a silently wrong gradient.
- **An off switch.** Inside `torch.no_grad()` nothing is recorded, so inference spends no memory on graphs.

When you need an operation PyTorch doesn't have, `torch.autograd.Function` lets you write its forward and backward rules, much like the `lambda`s here, and plug them into the real engine.

## Try it yourself

1. Add an `exp()` method. The derivative of *eˣ* is *eˣ* itself, so the rule can reuse the output value. Compute exp(*a* · *c*) with *a* = 2 and *c* = 3 and compare `a.grad` with PyTorch's; both should be about 1210.29. This is Exercise 6.8 in the book.
2. Let `Scalar` accept plain numbers, so that `2 * w` and `w - 1` work. Python calls `__rmul__` for the first.
3. Train the model above with gradient descent: repeat the forward pass, `backward`, an update of `w.value` and `b.value`, and a reset of both gradients to `None`. The loss should fall toward 0 as 2*w* + *b* approaches 5.

## About the book

This article is adapted from Chapter 6, "Autograd: How PyTorch Computes Gradients", of *The Complete Guide to PyTorch 2: From Tensors to Compiled Deep Learning* by Farid Rahmani. The book starts from your first tensor and goes on to train image classifiers, build a small GPT-style language model from scratch, and speed training up with mixed precision, `torch.compile` and distributed training. It explains the math as you need it and has 237 exercises with answers; the ebook adds worked solutions.

- [Kindle edition on Amazon](https://www.amazon.com/dp/B0HR97T3B2?maas=maas_adg_FF2CFA1CAA9C9A480A5C43C36AC2F16E_afap_abs&ref_=aa_maas&tag=maas)
- [Paperback on Amazon](https://www.amazon.com/dp/B0HRP36BS4?maas=maas_adg_F18C9CB4D7B524FC5D6A506C5C99FB91_afap_abs&ref_=aa_maas&tag=maas)

## License

The code in this repository is under the [MIT License](LICENSE). The article text and the figure are © 2026 Farid Rahmani.
