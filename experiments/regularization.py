"""
Experiment 3 — Weight decay and input noise vs overfitting.

We deliberately overfit by using only 150 training samples and a wide
network (capacity >> data).  We then compare four training conditions:

  baseline   — plain SGD, no regularization
  wd         — L2 weight decay (λ = 0.01)
  noise      — Gaussian input noise (σ = 0.4) added to each minibatch
  wd + noise — both combined

The train-vs-val accuracy gap is the overfitting signal.  The regularized
runs close the gap without hurting training beyond what the penalty adds.

Saves: experiments/plots/regularization.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from nns import (
    forward_backward, initialize_weights,
    make_activation, make_dense, make_loss, make_optimizer, make_sequential,
)


# ---------------------------------------------------------------------------
# Data — deliberately small training set
# ---------------------------------------------------------------------------

def _load(n_train=150, seed=0):
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64)
    # stratified split: n_train train, 300 val, rest test
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, train_size=n_train, random_state=seed, stratify=y
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest
    )
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    norm = lambda a: (a - mu) / sd
    return norm(X_tr), y_tr, norm(X_va), y_va, norm(X_te), y_te


# ---------------------------------------------------------------------------
# Model — wide so it has capacity to memorise 150 samples
# ---------------------------------------------------------------------------

def _fresh(in_dim=64, seed=0):
    np.random.seed(seed)
    def init(i, o): return initialize_weights(i, o, scheme="he")
    return make_sequential([
        make_dense(in_dim, 256, init), make_activation("relu"),
        make_dense(256,    128, init), make_activation("relu"),
        make_dense(128,     10, init),
    ])


def _acc(model, x, y):
    return float(np.mean(model["forward"](x)[0].argmax(1) == y))


# ---------------------------------------------------------------------------
# Training loop with optional weight decay and input noise
# ---------------------------------------------------------------------------

def _train(X_tr, y_tr, X_va, y_va,
           epochs=300, bs=16, lr=0.05,
           weight_decay=0.0, noise_std=0.0, seed=0):
    model   = _fresh(seed=seed)
    loss_fn = make_loss()
    opt     = make_optimizer(model["params"], lr=lr, kind="sgd")
    rng     = np.random.default_rng(seed)
    n       = len(X_tr)

    tr_acc_hist, va_acc_hist = [], []

    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            xb  = X_tr[idx]
            if noise_std > 0:
                xb = xb + rng.normal(0.0, noise_std, xb.shape)
            _, grads = forward_backward(model, loss_fn, xb, y_tr[idx])
            if weight_decay > 0:
                grads = [
                    {**g, "W": g["W"] + weight_decay * p["W"]} if "W" in g else g
                    for p, g in zip(model["params"], grads)
                ]
            opt["step"](grads)

        tr_acc_hist.append(_acc(model, X_tr, y_tr))
        va_acc_hist.append(_acc(model, X_va, y_va))

    return model, tr_acc_hist, va_acc_hist


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    X_tr, y_tr, X_va, y_va, X_te, y_te = _load(n_train=150)
    print(f"  train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}")

    configs = [
        ("baseline",   "#e74c3c", dict(weight_decay=0.0, noise_std=0.0)),
        ("wd",         "#f39c12", dict(weight_decay=0.01, noise_std=0.0)),
        ("noise",      "#2ecc71", dict(weight_decay=0.0, noise_std=0.4)),
        ("wd + noise", "#3498db", dict(weight_decay=0.01, noise_std=0.4)),
    ]

    results = {}
    for label, color, kwargs in configs:
        print(f"  {label:12s} ...", end=" ", flush=True)
        model, tr, va = _train(X_tr, y_tr, X_va, y_va, **kwargs)
        test_acc = _acc(model, X_te, y_te)
        results[label] = dict(color=color, tr=tr, va=va, test=test_acc)
        print(f"train={tr[-1]:.3f}  val={va[-1]:.3f}  test={test_acc:.4f}")

    # ---- Plot ---------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for label, r in results.items():
        kw = dict(label=label, color=r["color"], linewidth=1.8)
        ax1.plot(r["tr"], **kw)
        ax2.plot(r["va"], **kw)

    ax1.set_xlabel("epoch"); ax1.set_ylabel("accuracy")
    ax1.set_title("Train accuracy\n(should overfit → 1.0 for baseline)")
    ax1.grid(alpha=0.3); ax1.legend()

    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy")
    ax2.set_title("Validation accuracy\n(regularized models should score higher)")
    ax2.grid(alpha=0.3); ax2.legend()

    fig.suptitle(
        f"Regularization — {len(y_tr)}-sample train set, overparameterised network",
        fontsize=12,
    )
    fig.tight_layout()
    out = PLOT_DIR / "regularization.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  → {out}")

    # ---- Summary table ------------------------------------------------------
    print(f"\n  {'condition':12s}  train   val    test")
    print("  " + "-" * 42)
    for label, r in results.items():
        print(f"  {label:12s}  {r['tr'][-1]:.3f}   {r['va'][-1]:.3f}   {r['test']:.4f}")

    return results


if __name__ == "__main__":
    run()
