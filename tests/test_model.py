"""
End-to-end gradient checks for the full sequential model.

For every trainable parameter p in the network we check that the analytic
gradient from forward_backward matches the numerical gradient of the loss
w.r.t. p, with all other parameters held fixed.
"""
import numpy as np
import pytest

from nns import (
    forward_backward,
    gradient_check,
    make_activation,
    make_dense,
    make_loss,
    make_sequential,
    numerical_gradient,
)

TOL = 2e-5


def _net(seed=0):
    rng = np.random.default_rng(seed)
    def init(i, o):
        return rng.standard_normal((i, o)) * 0.1, np.zeros(o)
    return make_sequential([
        make_dense(4, 8, init),
        make_activation("relu"),
        make_dense(8, 3, init),
    ])


# ---------------------------------------------------------------------------
# Shape / structural tests
# ---------------------------------------------------------------------------

def test_sequential_forward_shape():
    model = _net()
    logits, caches = model["forward"](np.random.randn(5, 4))
    assert logits.shape == (5, 3)
    assert len(caches) == 3                     # one cache entry per layer


def test_sequential_backward_shape():
    model = _net()
    x = np.random.randn(5, 4)
    logits, caches = model["forward"](x)
    dx, grads_list = model["backward"](np.random.randn(*logits.shape), caches)
    assert dx.shape == x.shape
    assert len(grads_list) == 3


def test_forward_backward_finite():
    model = _net()
    loss_fn = make_loss()
    x = np.random.randn(8, 4)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    loss, grads = forward_backward(model, loss_fn, x, y)
    assert np.isfinite(loss)
    for params, grad in zip(model["params"], grads):
        for key in params:
            assert np.all(np.isfinite(grad[key]))


# ---------------------------------------------------------------------------
# End-to-end gradient check for every parameter
# ---------------------------------------------------------------------------

def test_all_param_gradients_end_to_end():
    rng = np.random.default_rng(7)
    model = _net(seed=7)
    loss_fn = make_loss()
    x = rng.standard_normal((6, 4))
    y = rng.integers(0, 3, 6)

    _, grads = forward_backward(model, loss_fn, x, y)

    for layer_idx, (params, grad) in enumerate(zip(model["params"], grads)):
        for key in params:
            p = params[key]
            g_analytic = grad[key]
            p_save = p.copy()

            # Probe: perturb p, run full forward+loss, restore p.
            # Capture p and p_save by default arg to avoid late-binding bugs.
            def f(p_flat, p=p, p_save=p_save):
                p[:] = p_flat.reshape(p.shape)
                logits, _ = model["forward"](x)
                loss, _ = loss_fn(logits, y)
                p[:] = p_save
                return loss

            g_numeric = numerical_gradient(f, p_save.ravel().copy()).reshape(p.shape)
            p[:] = p_save  # clean state for subsequent iterations

            err = gradient_check(g_analytic, g_numeric)
            assert err < TOL, (
                f"layer {layer_idx} param '{key}': relative error {err:.2e} > {TOL}"
            )


# ---------------------------------------------------------------------------
# Deeper network — extra stress test for gradient flow
# ---------------------------------------------------------------------------

def test_deep_network_gradients():
    rng = np.random.default_rng(99)
    def init(i, o):
        return rng.standard_normal((i, o)) * 0.1, np.zeros(o)
    model = make_sequential([
        make_dense(4, 16, init), make_activation("tanh"),
        make_dense(16, 16, init), make_activation("tanh"),
        make_dense(16, 2, init),
    ])
    loss_fn = make_loss()
    x = rng.standard_normal((8, 4))
    y = rng.integers(0, 2, 8)

    _, grads = forward_backward(model, loss_fn, x, y)

    for layer_idx, (params, grad) in enumerate(zip(model["params"], grads)):
        for key in params:
            p = params[key]
            p_save = p.copy()

            def f(p_flat, p=p, p_save=p_save):
                p[:] = p_flat.reshape(p.shape)
                logits, _ = model["forward"](x)
                loss, _ = loss_fn(logits, y)
                p[:] = p_save
                return loss

            g_num = numerical_gradient(f, p_save.ravel().copy()).reshape(p.shape)
            p[:] = p_save
            err = gradient_check(grad[key], g_num)
            assert err < TOL, f"deep net layer {layer_idx} '{key}': {err:.2e}"
