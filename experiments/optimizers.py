"""
Experiment 1 — SGD vs SGD-with-momentum vs Adam.

Each optimizer gets its natural learning rate.  Same initialisation and
same data split across all three so the only variable is the update rule.

Story: Adam reaches 98 % val accuracy in ~20 epochs; SGD takes ~150.
They all converge to roughly the same quality; it is *speed* that differs.

Saves: experiments/plots/optimizers.png
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
# Shared helpers
# ---------------------------------------------------------------------------

def _load(seed=0):
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64)
    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.10, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=0.10/0.90, random_state=seed, stratify=y_tv)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    norm = lambda a: (a - mu) / sd
    return norm(X_tr), y_tr, norm(X_va), y_va, norm(X_te), y_te


def _fresh(in_dim=64, seed=0):
    np.random.seed(seed)
    def init(i, o): return initialize_weights(i, o, scheme="he")
    return make_sequential([
        make_dense(in_dim, 128, init), make_activation("relu"),
        make_dense(128,    64,  init), make_activation("relu"),
        make_dense(64,     10,  init),
    ])


def _acc(model, x, y):
    return float(np.mean(model["forward"](x)[0].argmax(1) == y))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(kind, lr, X_tr, y_tr, X_va, y_va, epochs=200, bs=32, seed=0):
    model   = _fresh(seed=seed)
    loss_fn = make_loss()
    opt     = make_optimizer(model["params"], lr=lr, kind=kind)
    rng     = np.random.default_rng(seed)
    n       = len(X_tr)
    tl, va  = [], []
    for _ in range(epochs):
        perm   = rng.permutation(n)
        losses = []
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            loss, grads = forward_backward(model, loss_fn, X_tr[idx], y_tr[idx])
            opt["step"](grads)
            losses.append(loss)
        tl.append(float(np.mean(losses)))
        va.append(_acc(model, X_va, y_va))
    return model, tl, va


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    X_tr, y_tr, X_va, y_va, X_te, y_te = _load()

    configs = [
        ("sgd",      "SGD",            0.05,  "#e74c3c"),
        ("momentum", "SGD + momentum", 0.015, "#2ecc71"),
        ("adam",     "Adam",           1e-3,  "#3498db"),
    ]

    results = {}
    for kind, label, lr, color in configs:
        print(f"  {label:20s} lr={lr} ...", end=" ", flush=True)
        model, tl, va = _train(kind, lr, X_tr, y_tr, X_va, y_va, epochs=200)
        test_acc = _acc(model, X_te, y_te)
        results[kind] = dict(label=label, color=color, tl=tl, va=va, test=test_acc)
        print(f"val={va[-1]:.4f}  test={test_acc:.4f}")

    # ---- Plot ---------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for r in results.values():
        ax1.plot(r["tl"], label=r["label"], color=r["color"], linewidth=1.8)
        ax2.plot(r["va"], label=r["label"], color=r["color"], linewidth=1.8)

    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss")
    ax1.set_title("Training loss"); ax1.set_yscale("log"); ax1.grid(alpha=0.3); ax1.legend()

    ax2.set_xlabel("epoch"); ax2.set_ylabel("val accuracy")
    ax2.set_title("Validation accuracy"); ax2.grid(alpha=0.3); ax2.legend()
    ax2.axhline(0.95, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    fig.suptitle("Optimizer comparison — digits dataset (1 437 train samples)", fontsize=12)
    fig.tight_layout()
    out = PLOT_DIR / "optimizers.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  → {out}")

    # ---- Results table ------------------------------------------------------
    print("\n  optimizer        val_acc  test_acc")
    print("  " + "-" * 36)
    for r in results.values():
        print(f"  {r['label']:16s} {r['va'][-1]:.4f}   {r['test']:.4f}")

    return results


if __name__ == "__main__":
    run()
