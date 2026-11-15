"""
Experiment 2 — He vs Xavier initialisation through many ReLU layers.

Two measurements:

  (a) Activation variance per layer (no training, just a forward pass).
      He init keeps variance ≈ 1 through 20 layers.
      Xavier decays by ≈ 0.5 per layer because ReLU zeroes roughly half
      the activations, and Xavier was designed for symmetric (tanh/sigmoid)
      activations where that factor does not appear.

  (b) Training loss curves on a 10-hidden-layer network trained on digits.
      Xavier-initiated networks often start with shrinking gradients and
      train slower or fail to converge.

Saves: experiments/plots/init_depth.png
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
# (a) Activation variance through depth
# ---------------------------------------------------------------------------

def activation_variance_per_layer(scheme, n_layers=20, width=128, n_samples=512, seed=0):
    """
    Forward-pass a random input through `n_layers` dense+ReLU layers and
    record the variance of the activations after each layer.
    """
    np.random.seed(seed)
    x = np.random.randn(n_samples, width)
    variances = [float(x.var())]
    for _ in range(n_layers):
        W, b = initialize_weights(width, width, scheme=scheme)
        x = np.maximum(0, x @ W + b)
        variances.append(float(x.var()))
    return variances


# ---------------------------------------------------------------------------
# (b) Training loss with deep network
# ---------------------------------------------------------------------------

def _load(seed=0):
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64)
    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.10, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=0.10/0.90, random_state=seed, stratify=y_tv)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    norm = lambda a: (a - mu) / sd
    return norm(X_tr), y_tr, norm(X_va), y_va, norm(X_te), y_te


def _deep_net(in_dim, n_classes, n_hidden=8, width=64, scheme="he", seed=0):
    np.random.seed(seed)
    def init(i, o): return initialize_weights(i, o, scheme=scheme)
    layers = [make_dense(in_dim, width, init), make_activation("relu")]
    for _ in range(n_hidden - 1):
        layers += [make_dense(width, width, init), make_activation("relu")]
    layers.append(make_dense(width, n_classes, init))
    return make_sequential(layers)


def _train_loss_curve(scheme, X_tr, y_tr, epochs=150, bs=32, seed=0):
    model   = _deep_net(X_tr.shape[1], 10, n_hidden=8, width=64, scheme=scheme, seed=seed)
    loss_fn = make_loss()
    opt     = make_optimizer(model["params"], lr=1e-3, kind="adam")
    rng     = np.random.default_rng(seed)
    n       = len(X_tr)
    history = []
    for _ in range(epochs):
        perm   = rng.permutation(n)
        losses = []
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            loss, grads = forward_backward(model, loss_fn, X_tr[idx], y_tr[idx])
            opt["step"](grads)
            losses.append(loss)
        history.append(float(np.mean(losses)))
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    # (a) Variance measurements
    print("  measuring activation variance (20 ReLU layers) ...")
    he_var  = activation_variance_per_layer("he")
    xav_var = activation_variance_per_layer("xavier")

    print(f"  He    layer 0→20: {he_var[0]:.3f} → {he_var[-1]:.3f}")
    print(f"  Xavier layer 0→20: {xav_var[0]:.3f} → {xav_var[-1]:.6f}")

    # (b) Training curves
    X_tr, y_tr, X_va, y_va, _, _ = _load()
    print("  training deep nets (He + Xavier) ...")
    he_loss  = _train_loss_curve("he",     X_tr, y_tr)
    xav_loss = _train_loss_curve("xavier", X_tr, y_tr)

    # ---- Plot ---------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    layers = list(range(len(he_var)))
    ax1.plot(layers, he_var,  label="He init",     color="#3498db", linewidth=2)
    ax1.plot(layers, xav_var, label="Xavier init", color="#e74c3c", linewidth=2)
    ax1.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, label="target ≈ 1")
    ax1.set_xlabel("layer index")
    ax1.set_ylabel("activation variance")
    ax1.set_title("Activation variance through depth\n(20 × ReLU layers, forward pass only)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(he_loss,  label="He init",     color="#3498db", linewidth=1.8)
    ax2.plot(xav_loss, label="Xavier init", color="#e74c3c", linewidth=1.8)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("train loss")
    ax2.set_title("Training loss — 8-hidden-layer ReLU network\n(Adam, same LR)")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("He vs Xavier initialisation", fontsize=12)
    fig.tight_layout()
    out = PLOT_DIR / "init_depth.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  → {out}")

    print(f"\n  final train loss — He: {he_loss[-1]:.4f}   Xavier: {xav_loss[-1]:.4f}")
    return dict(he_var=he_var, xav_var=xav_var, he_loss=he_loss, xav_loss=xav_loss)


if __name__ == "__main__":
    run()
