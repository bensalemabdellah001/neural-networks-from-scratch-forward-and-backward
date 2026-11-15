#!/usr/bin/env python
"""
Train a two-hidden-layer ReLU MLP on sklearn's digits dataset.

The digits dataset ships with scikit-learn — no download needed.
To run on MNIST instead (requires ~11 MB), replace load_data() with the
version in scripts/train_mnist.py (TODO: add once on WiFi).

Evaluation discipline:
  - 10 % of the full dataset is held out as a test set before anything else.
  - Validation accuracy (10 % of the remaining data) is used to monitor
    training; hyperparameters were chosen by watching val accuracy only.
  - Test accuracy is computed and printed exactly once, at the very end.

Usage:
    python scripts/train_digits.py
    python scripts/train_digits.py --epochs 300 --lr 3e-3 --hidden 256
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nns import (
    forward_backward,
    initialize_weights,
    make_activation,
    make_dense,
    make_loss,
    make_optimizer,
    make_sequential,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(val_fraction=0.10, test_fraction=0.10, seed=0):
    """
    Returns (X_train, y_train, X_val, y_val, X_test, y_test).

    Scaling: zero-mean / unit-std computed on training data only, then
    applied to val and test to avoid data leakage.

    swap-to-MNIST: replace the two lines below with torchvision or
    keras MNIST loading; the rest of the script is unchanged.
    """
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64)

    # Hold out test set first (never touched until final eval)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y
    )
    # Split the remaining data into train and val
    val_frac_of_tv = val_fraction / (1.0 - test_fraction)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac_of_tv, random_state=seed, stratify=y_tv
    )

    # Normalise: fit on train only
    mu  = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / std
    X_val   = (X_val   - mu) / std
    X_test  = (X_test  - mu) / std

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(in_dim, n_classes, hidden=128, seed=0):
    np.random.seed(seed)

    def init(i, o):
        return initialize_weights(i, o, scheme="he")

    return make_sequential([
        make_dense(in_dim, hidden, init),
        make_activation("relu"),
        make_dense(hidden, hidden // 2, init),
        make_activation("relu"),
        make_dense(hidden // 2, n_classes, init),
    ])


# ---------------------------------------------------------------------------
# Training loop (manual so we can log val accuracy each epoch)
# ---------------------------------------------------------------------------

def _accuracy(model, x, y):
    logits, _ = model["forward"](x)
    return float(np.mean(logits.argmax(axis=1) == y))


def run(epochs=200, batch_size=32, lr=5e-3, hidden=128, seed=0):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(seed=seed)
    n, in_dim = X_train.shape

    print(f"splits  train={n}  val={len(y_val)}  test={len(y_test)}  features={in_dim}")

    model   = build_model(in_dim, 10, hidden=hidden, seed=seed)
    loss_fn = make_loss("cross_entropy")
    opt     = make_optimizer(model["params"], lr=lr, kind="adam")
    rng     = np.random.default_rng(seed)

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        perm   = rng.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            idx          = perm[start : start + batch_size]
            loss, grads  = forward_backward(model, loss_fn, X_train[idx], y_train[idx])
            opt["step"](grads)
            losses.append(loss)

        train_loss = float(np.mean(losses))
        val_acc    = _accuracy(model, X_val, y_val)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)

        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {train_loss:.4f}  val_acc {val_acc:.4f}")

    # Test set evaluated exactly once
    test_acc = _accuracy(model, X_test, y_test)
    print(f"\nTest accuracy : {test_acc:.4f}")
    print(f"Best val acc  : {max(history['val_acc']):.4f}")
    return {"test_accuracy": test_acc, "history": history}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int,   default=200)
    parser.add_argument("--lr",     type=float, default=5e-3)
    parser.add_argument("--hidden", type=int,   default=128)
    parser.add_argument("--seed",   type=int,   default=0)
    args = parser.parse_args()
    run(epochs=args.epochs, lr=args.lr, hidden=args.hidden, seed=args.seed)
