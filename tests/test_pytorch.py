"""
Validates our NumPy implementation against PyTorch to within 1e-6.

Design note — weight layout mismatch:
  Our make_dense stores W with shape (in_dim, out_dim):  y = x @ W + b
  PyTorch nn.Linear stores weight with shape (out_dim, in_dim): y = x @ W.T + b

So when we copy weights in we do  linear.weight ← torch(our_W.T),
and when we compare gradients out we do  dW_numpy ≈ torch_linear.weight.grad.T.

Everything runs in float64 so that numerical differences from floating-point
order-of-operations are below 1e-14; the 1e-6 tolerance is a deliberately
loose bound to guard against genuine algorithm divergence, not rounding.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
nn = torch.nn

from nns import (
    forward_backward,
    make_activation,
    make_dense,
    make_loss,
    make_sequential,
)

ATOL = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_net(weights):
    """
    Build our sequential model from a list of (W, b) pairs.
    Architecture: Dense → ReLU → … → Dense (no activation on the output layer).
    """
    layers = []
    for W, b in weights:
        layers.append(
            make_dense(
                W.shape[0], W.shape[1],
                lambda i, o, W=W, b=b: (W.copy(), b.copy()),  # default-arg capture
            )
        )
        layers.append(make_activation("relu"))
    layers.pop()  # remove the relu that was appended after the last dense layer
    return make_sequential(layers)


def _torch_net(weights):
    """
    Build the equivalent PyTorch Sequential with identical float-64 weights.

    Weight transpose: our W is (in, out); nn.Linear.weight is (out, in).
    """
    seqlist = []
    for i, (W, b) in enumerate(weights):
        linear = nn.Linear(W.shape[0], W.shape[1], dtype=torch.float64)
        with torch.no_grad():
            linear.weight.copy_(torch.from_numpy(W.T.copy()))
            linear.bias.copy_(torch.from_numpy(b.copy()))
        seqlist.append(linear)
        if i < len(weights) - 1:   # no ReLU after the output layer
            seqlist.append(nn.ReLU())
    return nn.Sequential(*seqlist)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loss_and_gradients_match_pytorch():
    """
    Single comprehensive check: same weights, same batch → loss and every
    dW/db agree between NumPy and PyTorch to within ATOL=1e-6.
    """
    rng = np.random.default_rng(42)

    weights = [
        (rng.standard_normal((4, 8)) * 0.1, rng.standard_normal(8) * 0.1),
        (rng.standard_normal((8, 6)) * 0.1, rng.standard_normal(6) * 0.1),
        (rng.standard_normal((6, 3)) * 0.1, rng.standard_normal(3) * 0.1),
    ]
    x = rng.standard_normal((16, 4))
    y = rng.integers(0, 3, 16)

    # ------------------------------------------------------------------ NumPy
    np_model = _numpy_net(weights)
    np_loss, np_grads = forward_backward(np_model, make_loss(), x, y)
    # Activation layers have empty grad dicts; filter to dense layers only
    np_dense_grads = [g for g in np_grads if "W" in g]

    # ----------------------------------------------------------------- PyTorch
    torch_net = _torch_net(weights)
    logits = torch_net(torch.from_numpy(x))
    torch_loss = nn.CrossEntropyLoss()(logits, torch.tensor(y))
    torch_loss.backward()
    torch_linears = [m for m in torch_net.modules() if isinstance(m, nn.Linear)]

    # ------------------------------------------------------------------- Loss
    loss_diff = abs(np_loss - torch_loss.item())
    assert loss_diff < ATOL, (
        f"Loss mismatch: numpy={np_loss:.10f} torch={torch_loss.item():.10f} "
        f"diff={loss_diff:.2e}"
    )

    # --------------------------------------------------------------- Gradients
    for i, (np_grad, linear) in enumerate(zip(np_dense_grads, torch_linears)):
        # dW: transpose PyTorch's (out, in) gradient back to our (in, out)
        dW_torch = linear.weight.grad.numpy().T
        err_W = float(np.max(np.abs(np_grad["W"] - dW_torch)))
        assert err_W < ATOL, f"dW mismatch at dense layer {i}: {err_W:.2e}"

        db_torch = linear.bias.grad.numpy()
        err_b = float(np.max(np.abs(np_grad["b"] - db_torch)))
        assert err_b < ATOL, f"db mismatch at dense layer {i}: {err_b:.2e}"


def test_loss_value_printed_for_reference(capsys):
    """
    Not a strict test — runs the comparison and prints actual differences
    so the magnitude is visible in `pytest -s` output.
    """
    rng = np.random.default_rng(0)
    weights = [
        (rng.standard_normal((4, 16)) * 0.1, rng.standard_normal(16) * 0.1),
        (rng.standard_normal((16, 3)) * 0.1, rng.standard_normal(3) * 0.1),
    ]
    x = rng.standard_normal((32, 4))
    y = rng.integers(0, 3, 32)

    np_model = _numpy_net(weights)
    np_loss, np_grads = forward_backward(np_model, make_loss(), x, y)
    np_dense_grads = [g for g in np_grads if "W" in g]

    torch_net = _torch_net(weights)
    logits = torch_net(torch.from_numpy(x))
    torch_loss = nn.CrossEntropyLoss()(logits, torch.tensor(y))
    torch_loss.backward()
    torch_linears = [m for m in torch_net.modules() if isinstance(m, nn.Linear)]

    print(f"\n  loss diff : {abs(np_loss - torch_loss.item()):.2e}")
    for i, (ng, lin) in enumerate(zip(np_dense_grads, torch_linears)):
        dW_err = float(np.max(np.abs(ng["W"] - lin.weight.grad.numpy().T)))
        db_err = float(np.max(np.abs(ng["b"] - lin.bias.grad.numpy())))
        print(f"  layer {i} dW: {dW_err:.2e}   db: {db_err:.2e}")

    # still assert correctness
    assert abs(np_loss - torch_loss.item()) < ATOL
