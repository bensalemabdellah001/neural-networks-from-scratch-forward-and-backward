#!/usr/bin/env python
"""
Train on MNIST — requires ~11 MB download (torchvision).

This script is identical to train_digits.py except for load_data().
Run once on WiFi; torchvision caches the data under ~/.cache/torchvision.

Usage:
    python scripts/train_mnist.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

# Defer the import so the file is importable even without torchvision
def load_data(val_fraction=0.10, test_fraction=0.10, seed=0):
    """
    Download (once) and load MNIST.  Swap this function into train_digits.py
    to run on MNIST without changing anything else.
    """
    try:
        import torchvision
    except ImportError:
        raise ImportError("pip install torchvision  # ~11 MB download on first run")

    from sklearn.model_selection import train_test_split

    root = Path.home() / ".cache" / "torchvision" / "datasets"
    root.mkdir(parents=True, exist_ok=True)

    train_ds = torchvision.datasets.MNIST(root, train=True,  download=True)
    test_ds  = torchvision.datasets.MNIST(root, train=False, download=True)

    X_all = np.concatenate([
        train_ds.data.numpy().reshape(-1, 784).astype(np.float64),
        test_ds.data.numpy().reshape(-1, 784).astype(np.float64),
    ])
    y_all = np.concatenate([
        train_ds.targets.numpy(),
        test_ds.targets.numpy(),
    ])

    X_tv, X_test, y_tv, y_test = train_test_split(
        X_all, y_all, test_size=test_fraction, random_state=seed, stratify=y_all
    )
    val_frac_of_tv = val_fraction / (1.0 - test_fraction)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac_of_tv, random_state=seed, stratify=y_tv
    )

    mu  = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    norm = lambda a: (a - mu) / std
    return norm(X_train), y_train, norm(X_val), y_val, norm(X_test), y_test


if __name__ == "__main__":
    # Reuse train_digits.run() — only load_data differs
    import argparse
    from scripts.train_digits import run  # noqa: local import after sys.path setup

    # Monkey-patch the data loader
    import scripts.train_digits as _td
    _td.load_data = load_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,   default=20)
    parser.add_argument("--lr",     type=float, default=5e-3)
    parser.add_argument("--hidden", type=int,   default=256)
    parser.add_argument("--seed",   type=int,   default=0)
    args = parser.parse_args()
    run(epochs=args.epochs, lr=args.lr, hidden=args.hidden, seed=args.seed)
