"""Compare the small engine's gradients with PyTorch's on a few expressions.

Run:  python check_with_pytorch.py      (needs PyTorch)
"""
import random

import torch

from small_autograd import Scalar, backward

EXPRESSIONS = {
    "(w*x + b - y)^2":   lambda w, b, x, y: (w * x + b - y) * (w * x + b - y),
    "a*a + a":           lambda w, b, x, y: w * w + w,
    "(w - b) * (w + x)": lambda w, b, x, y: (w - b) * (w + x),
    "w*b*w - b*y + x":   lambda w, b, x, y: w * b * w - b * y + x,
}


def ours(f, values):
    w, b = (Scalar(v, requires_grad=True) for v in values[:2])
    x, y = (Scalar(v) for v in values[2:])
    backward(f(w, b, x, y))
    return w.grad, b.grad


def pytorch(f, values):
    w, b = (torch.tensor(v, dtype=torch.float64, requires_grad=True)
            for v in values[:2])
    x, y = (torch.tensor(v, dtype=torch.float64) for v in values[2:])
    f(w, b, x, y).backward()
    # A parameter the expression never uses gets no gradient in either.
    return tuple(None if t.grad is None else t.grad.item() for t in (w, b))


def main():
    random.seed(0)
    cases = [[1.0, 0.0, 2.0, 5.0]]
    cases += [[random.uniform(-3, 3) for _ in range(4)] for _ in range(20)]
    failures = 0
    for name, f in EXPRESSIONS.items():
        for values in cases:
            a, t = ours(f, values), pytorch(f, values)
            same = all((u is None and v is None) or
                       (u is not None and v is not None and
                        abs(u - v) <= 1e-9 * max(1.0, abs(v)))
                       for u, v in zip(a, t))
            if not same:
                failures += 1
                print(f"MISMATCH {name} at {values}: {a} vs {t}")
        print(f"{name:20} {len(cases)} cases checked")
    print("all gradients match PyTorch" if not failures
          else f"{failures} mismatches")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
