"""A small autograd engine in plain Python, built the way PyTorch's is.

Each operation on a Scalar records a backward node holding its local rule,
the values that rule needs, and links to its inputs' nodes. backward() runs
the nodes in topological order, summing the gradients that reach each one.

See README.md for the walk-through.
"""


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

    def __repr__(self):
        return f"Scalar({self.value}, grad={self.grad})"


def record(name, value, inputs, rule):
    out = Scalar(value)
    if any(t.requires_grad for t in inputs):
        out.requires_grad = True
        out.grad_fn = Node(name, rule, inputs)
    return out


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


if __name__ == "__main__":
    # L = (w*x + b - y)^2 with x = 2, y = 5, w = 1, b = 0
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
        print(name, t.value, t.grad_fn.name, links)

    backward(L)
    print("w.grad =", w.grad, " b.grad =", b.grad)   # -12.0 -6.0
