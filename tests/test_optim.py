import numpy as np
import pytest

from nns import (
    forward_backward,
    make_activation,
    make_dense,
    make_loss,
    make_optimizer,
    make_sequential,
)


def _small_net(seed=0):
    rng = np.random.default_rng(seed)
    def init(i, o):
        return rng.standard_normal((i, o)) * 0.1, np.zeros(o)
    return make_sequential([
        make_dense(4, 8, init),
        make_activation("relu"),
        make_dense(8, 2, init),
    ])


@pytest.mark.parametrize("kind", ["sgd", "momentum", "adam"])
def test_optimizer_reduces_loss(kind):
    rng = np.random.default_rng(0)
    model = _small_net()
    loss_fn = make_loss()
    opt = make_optimizer(model["params"], lr=0.01, kind=kind)
    x = rng.standard_normal((32, 4))
    y = rng.integers(0, 2, 32)

    losses = []
    for _ in range(30):
        loss, grads = forward_backward(model, loss_fn, x, y)
        opt["step"](grads)
        losses.append(loss)

    assert all(np.isfinite(l) for l in losses), f"{kind}: non-finite loss"
    assert losses[-1] < losses[0], (
        f"{kind}: loss did not decrease ({losses[0]:.4f} → {losses[-1]:.4f})"
    )


def test_sgd_in_place_update():
    # SGD: new_param = param - lr * grad, nothing else
    params = {"W": np.array([1.0, 2.0, 3.0])}
    opt = make_optimizer(params, lr=0.5, kind="sgd")
    opt["step"]({"W": np.array([2.0, 4.0, 6.0])})
    assert np.allclose(params["W"], [0.0, 0.0, 0.0])


def test_optimizer_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown optimizer"):
        make_optimizer({}, lr=0.01, kind="rmsprop")


def test_optimizer_mismatched_grads_raises():
    params = [{"W": np.ones(4), "b": np.ones(2)}]
    opt = make_optimizer(params, lr=0.01, kind="sgd")
    with pytest.raises(ValueError, match="does not match"):
        opt["step"]([{"W": np.ones(4)}])   # missing 'b'
