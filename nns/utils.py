import numpy as np


def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    if x.size == 0:
        return grad
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        fp = f(x)
        x[i] = old - eps
        fm = f(x)
        x[i] = old
        grad[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    a = np.asarray(analytic_grad, dtype=float)
    n = np.asarray(numeric_grad, dtype=float)
    if a.size == 0:
        return 0.0
    denom = np.maximum(np.abs(a), np.abs(n))
    denom = np.maximum(denom, tol)
    return float(np.max(np.abs(a - n) / denom))
