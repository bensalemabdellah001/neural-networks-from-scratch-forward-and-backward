import numpy as np

from nns import (
    forward_backward,
    make_activation,
    make_dense,
    make_loss,
    make_optimizer,
    make_sequential,
    train,
    train_step,
)


def _net(seed=0):
    rng = np.random.default_rng(seed)
    def init(i, o):
        return rng.standard_normal((i, o)) * 0.1, np.zeros(o)
    return make_sequential([
        make_dense(4, 8, init),
        make_activation("relu"),
        make_dense(8, 3, init),
    ])


def test_train_step_returns_preloss():
    """Loss returned by train_step equals the loss before the parameter update."""
    model = _net()
    loss_fn = make_loss()
    opt = make_optimizer(model["params"], lr=0.01, kind="sgd")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((16, 4))
    y = rng.integers(0, 3, 16)

    # Compute loss before any update (forward_backward does not change params)
    pre_loss, _ = forward_backward(model, loss_fn, x, y)
    # train_step runs forward_backward internally, then updates params
    returned = train_step(model, loss_fn, opt, x, y)
    assert abs(returned - pre_loss) < 1e-10


def test_train_step_updates_params():
    """Parameters must actually change after train_step."""
    model = _net()
    loss_fn = make_loss()
    opt = make_optimizer(model["params"], lr=0.1, kind="sgd")
    rng = np.random.default_rng(2)
    x = rng.standard_normal((16, 4))
    y = rng.integers(0, 3, 16)

    # snapshot dense-layer weights before
    W_before = model["params"][0]["W"].copy()
    train_step(model, loss_fn, opt, x, y)
    assert not np.allclose(model["params"][0]["W"], W_before)


def test_train_history_length():
    model = _net()
    loss_fn = make_loss()
    opt = make_optimizer(model["params"], lr=0.01, kind="sgd")
    rng = np.random.default_rng(3)
    x = rng.standard_normal((32, 4))
    y = rng.integers(0, 3, 32)

    history = train(model, loss_fn, opt, x, y, epochs=10, batch_size=16)
    assert len(history) == 10
    assert all(np.isfinite(h) for h in history)


def test_train_decreases_loss():
    model = _net(seed=5)
    loss_fn = make_loss()
    opt = make_optimizer(model["params"], lr=0.05, kind="adam")
    rng = np.random.default_rng(5)
    x = rng.standard_normal((64, 4))
    y = rng.integers(0, 3, 64)

    history = train(model, loss_fn, opt, x, y, epochs=50, batch_size=32, seed=5)
    assert history[-1] < history[0]


def test_train_deterministic_with_seed():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((64, 4))
    y = rng.integers(0, 3, 64)

    def run():
        m = _net(seed=0)
        loss_fn = make_loss()
        opt = make_optimizer(m["params"], lr=0.01, kind="sgd")
        return train(m, loss_fn, opt, x, y, epochs=5, batch_size=16, seed=42)

    assert run() == run()
