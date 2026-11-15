"""
Gradient checks for every layer.

Strategy: for a layer with forward y = f(x) and backward dx = f'(x)·dout,
define a scalar g(x) = (dout * f(x)).sum() and verify
  analytic_grad  = backward(dout, cache)
  numerical_grad = numerical_gradient(g, x)
agree to within TOL.  For weight parameters we modify params[key] in place
inside the numerical-gradient probe, then restore.
"""
import numpy as np
import pytest

from nns import (
    gradient_check,
    initialize_weights,
    make_activation,
    make_dense,
    numerical_gradient,
)

TOL = 2e-5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init(rng, in_dim, out_dim):
    return rng.standard_normal((in_dim, out_dim)) * 0.1, np.zeros(out_dim)


def _numeric_param_grad(layer, x, dout, key):
    """Numerical gradient of sum(dout * forward(x)) w.r.t. params[key]."""
    p = layer["params"][key]
    p_save = p.copy()

    def f(p_flat):
        p[:] = p_flat.reshape(p.shape)
        y, _ = layer["forward"](x)
        return float((dout * y).sum())

    g = numerical_gradient(f, p_save.ravel().copy()).reshape(p.shape)
    p[:] = p_save
    return g


# ---------------------------------------------------------------------------
# Dense layer
# ---------------------------------------------------------------------------

@pytest.fixture
def dense_setup():
    rng = np.random.default_rng(0)
    B, D, H = 4, 5, 3
    layer = make_dense(D, H, lambda i, o: _init(rng, i, o))
    x    = rng.standard_normal((B, D))
    dout = rng.standard_normal((B, H))
    return layer, x, dout


def test_dense_dx(dense_setup):
    layer, x, dout = dense_setup
    y, cache = layer["forward"](x)
    dx, _ = layer["backward"](dout, cache)

    def f(x_flat):
        y, _ = layer["forward"](x_flat.reshape(x.shape))
        return float((dout * y).sum())

    dx_num = numerical_gradient(f, x.ravel().copy()).reshape(x.shape)
    assert gradient_check(dx, dx_num) < TOL


def test_dense_dW(dense_setup):
    layer, x, dout = dense_setup
    y, cache = layer["forward"](x)
    _, grads = layer["backward"](dout, cache)
    dW_num = _numeric_param_grad(layer, x, dout, "W")
    assert gradient_check(grads["W"], dW_num) < TOL


def test_dense_db(dense_setup):
    layer, x, dout = dense_setup
    y, cache = layer["forward"](x)
    _, grads = layer["backward"](dout, cache)
    db_num = _numeric_param_grad(layer, x, dout, "b")
    assert gradient_check(grads["b"], db_num) < TOL


def test_dense_no_param_leak():
    """backward must return empty grads for a param-free layer."""
    rng = np.random.default_rng(1)
    layer = make_activation("relu")
    x = np.abs(rng.standard_normal((4, 6))) + 0.1
    dout = rng.standard_normal(x.shape)
    _, empty = layer["backward"](dout, layer["forward"](x)[1])
    assert empty == {}


# ---------------------------------------------------------------------------
# Activation layers — gradient checks for all three kinds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["relu", "tanh", "sigmoid"])
def test_activation_dx(kind):
    rng = np.random.default_rng(42)
    layer = make_activation(kind)
    x = rng.standard_normal((6, 8))
    if kind == "relu":
        x = np.abs(x) + 0.1  # stay away from the kink at 0 (undefined subgradient)
    dout = rng.standard_normal(x.shape)

    y, cache = layer["forward"](x)
    dx, empty = layer["backward"](dout, cache)
    assert empty == {}

    def f(x_flat):
        y, _ = layer["forward"](x_flat.reshape(x.shape))
        return float((dout * y).sum())

    dx_num = numerical_gradient(f, x.ravel().copy()).reshape(x.shape)
    assert gradient_check(dx, dx_num) < TOL


def test_relu_zero_gradient_on_negative_inputs():
    layer = make_activation("relu")
    x = -(np.abs(np.random.randn(5, 5)) + 0.1)   # all strictly negative
    dout = np.random.randn(5, 5)
    y, cache = layer["forward"](x)
    dx, _ = layer["backward"](dout, cache)
    assert np.all(y == 0)
    assert np.all(dx == 0)


# ---------------------------------------------------------------------------
# initialize_weights
# ---------------------------------------------------------------------------

def test_initialize_weights_he_scale():
    # He init: Var(W) ≈ 2 / fan_in; use large dims so sample var is tight
    W, b = initialize_weights(2000, 100, scheme="he")
    assert W.shape == (2000, 100)
    assert b.shape == (100,) and np.all(b == 0)
    assert abs(W.var() - 2.0 / 2000) < 2e-4


def test_initialize_weights_xavier_scale():
    W, b = initialize_weights(2000, 100, scheme="xavier")
    assert abs(W.var() - 1.0 / 2000) < 1e-4


def test_initialize_weights_unknown_raises():
    with pytest.raises(ValueError, match="unknown scheme"):
        initialize_weights(4, 4, scheme="glorot")
