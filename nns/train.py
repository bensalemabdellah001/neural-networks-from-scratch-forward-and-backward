import numpy as np

from .layers import make_dense, make_activation
from .losses import make_loss
from .model import make_sequential, forward_backward
from .optim import make_optimizer


def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    loss, grads = forward_backward(model, loss_fn, x_batch, y_batch)
    optimizer["step"](grads)
    return float(loss)


def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    history = []
    for _ in range(epochs):
        perm = rng.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            losses.append(train_step(model, loss_fn, optimizer, x[idx], y[idx]))
        history.append(float(np.mean(losses)))
    return history


def design_network(input_dim, num_classes, seed=0):
    def _bands(x, num_classes):
        nb = 2 * num_classes
        band = np.clip(((x[:, 0] + 1) / 2 * nb).astype(int), 0, nb - 1)
        return band % num_classes

    rng = np.random.default_rng(seed)
    n_train = 2000 * num_classes
    x_tr = rng.uniform(-1, 1, (n_train, input_dim))
    y_tr = _bands(x_tr, num_classes)
    x_ev = rng.uniform(-1, 1, (500, input_dim))
    y_ev = _bands(x_ev, num_classes)

    width = 32 * num_classes

    def init(i, o):
        return rng.standard_normal((i, o)) * np.sqrt(2.0 / i), np.zeros(o)

    model = make_sequential([
        make_dense(input_dim, width, init), make_activation("relu"),
        make_dense(width, width, init),     make_activation("relu"),
        make_dense(width, num_classes, init),
    ])
    loss_fn = make_loss()
    opt = make_optimizer(model["params"], lr=0.005, kind="adam")
    train(model, loss_fn, opt, x_tr, y_tr, epochs=150, batch_size=128, seed=seed)

    acc = float(np.mean(model["forward"](x_ev)[0].argmax(axis=1) == y_ev))
    return model, {"accuracy": acc, "x": x_ev, "y": y_ev}


def improve_generalization(baseline_model_fn, x_train, y_train, x_val, y_val, seed=0):
    loss_fn = make_loss()
    batch_size = 16

    def fresh():
        np.random.seed(seed)
        return baseline_model_fn()

    def accuracy(model, x, y):
        return float(np.mean(model["forward"](x)[0].argmax(axis=1) == y))

    base = fresh()
    train(
        base, loss_fn,
        make_optimizer(base["params"], lr=0.1, kind="sgd"),
        x_train, y_train, epochs=300, batch_size=batch_size, seed=seed,
    )
    baseline_acc = accuracy(base, x_val, y_val)

    model = fresh()
    opt = make_optimizer(model["params"], lr=0.05, kind="sgd")
    rng = np.random.default_rng(seed)
    weight_decay, noise_std, n = 0.03, 0.1, len(x_train)
    for _ in range(70):
        perm = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            xb = x_train[idx] + rng.normal(0.0, noise_std, x_train[idx].shape)
            _, grads = forward_backward(model, loss_fn, xb, y_train[idx])
            # fix #6: build new dicts instead of mutating backward's output
            penalized = [
                {**g, "W": g["W"] + weight_decay * p["W"]} if "W" in g else g
                for p, g in zip(model["params"], grads)
            ]
            opt["step"](penalized)

    predictions = model["forward"](x_val)[0].argmax(axis=1)
    return {
        "val_accuracy": float(np.mean(predictions == y_val)),
        "baseline_val_accuracy": baseline_acc,
        "predictions": predictions,
        "model": model,
    }
