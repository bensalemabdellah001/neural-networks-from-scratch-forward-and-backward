import numpy as np


def make_optimizer(params, lr=1e-2, kind="sgd"):
    def _flatten(tree):
        if isinstance(tree, np.ndarray):
            return [tree]
        if isinstance(tree, dict):
            return [a for v in tree.values() for a in _flatten(v)]
        if isinstance(tree, (list, tuple)):
            return [a for v in tree for a in _flatten(v)]
        raise TypeError(f"unsupported container: {type(tree)}")

    arrays = _flatten(params)

    if kind == "sgd":
        pass
    elif kind == "momentum":
        mu = 0.9                                      # fix #4: named separately from Adam beta1
        vel = [np.zeros_like(a) for a in arrays]     # fix #5: scoped to this branch only
    elif kind == "adam":
        beta1, beta2, eps = 0.9, 0.999, 1e-8         # fix #5: scoped to this branch only
        m = [np.zeros_like(a) for a in arrays]
        v = [np.zeros_like(a) for a in arrays]
        t = [0]
    else:
        raise ValueError(f"unknown optimizer: {kind}")

    def step(grads):
        gs = _flatten(grads)
        if len(gs) != len(arrays):
            raise ValueError("grads structure does not match params")
        if kind == "sgd":
            for a, g in zip(arrays, gs):
                a -= lr * g
        elif kind == "momentum":
            for a, g, u in zip(arrays, gs, vel):
                u *= mu
                u += g
                a -= lr * u
        elif kind == "adam":
            t[0] += 1
            for a, g, mi, vi in zip(arrays, gs, m, v):
                mi *= beta1
                mi += (1 - beta1) * g
                vi *= beta2
                vi += (1 - beta2) * g * g
                a -= lr * (mi / (1 - beta1 ** t[0])) / (np.sqrt(vi / (1 - beta2 ** t[0])) + eps)

    return {"step": step}
