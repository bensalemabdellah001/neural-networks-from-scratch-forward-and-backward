import numpy as np


def initialize_weights(in_dim, out_dim, scheme="he"):
    if scheme == "he":
        std = np.sqrt(2.0 / in_dim)
    elif scheme == "xavier":
        std = np.sqrt(1.0 / in_dim)
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    W = np.random.randn(in_dim, out_dim) * std
    b = np.zeros(out_dim)
    return W, b


def make_dense(in_dim, out_dim, weight_init_fn):
    W, b = weight_init_fn(in_dim, out_dim)
    params = {"W": np.asarray(W, dtype=float), "b": np.asarray(b, dtype=float)}

    def forward(x):
        x = np.asarray(x, dtype=float)
        y = x @ params["W"] + params["b"]
        return y, x                     # cache = input (needed by backward)

    def backward(dout, cache):
        x = cache
        dout = np.asarray(dout, dtype=float)
        dx = dout @ params["W"].T       # (N, out) @ (out, in) -> (N, in)
        dW = x.T @ dout                 # (in, N)  @ (N, out) -> (in, out)
        db = dout.sum(axis=0)
        return dx, {"W": dW, "b": db}

    return {"params": params, "forward": forward, "backward": backward}


def make_activation(kind="relu"):
    if kind == "relu":
        def fwd(x):
            return np.where(x > 0, x, 0.0), x > 0   # cache = positive mask

        def bwd(dout, mask):
            return dout * mask

    elif kind == "tanh":
        def fwd(x):
            y = np.tanh(x)
            return y, y

        def bwd(dout, y):
            return dout * (1 - y ** 2)

    elif kind == "sigmoid":
        def fwd(x):
            y = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return y, y

        def bwd(dout, y):
            return dout * y * (1 - y)

    else:
        raise ValueError(f"unknown activation: {kind}")

    def forward(x):
        return fwd(np.asarray(x, dtype=float))

    def backward(dout, cache):
        return bwd(np.asarray(dout, dtype=float), cache), {}

    return {"params": {}, "forward": forward, "backward": backward}
