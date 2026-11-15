import numpy as np


def make_loss(kind="cross_entropy"):
    if kind != "cross_entropy":
        raise ValueError(f"unknown loss: {kind}")

    def loss_fn(logits, labels):
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)
        n = logits.shape[0]
        z = logits - logits.max(axis=1, keepdims=True)
        log_q = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        loss = float(-log_q[np.arange(n), labels].mean())
        d = np.exp(log_q)
        d[np.arange(n), labels] -= 1.0
        return loss, d / n

    return loss_fn
