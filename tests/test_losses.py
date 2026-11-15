import numpy as np
import pytest

from nns import gradient_check, make_loss, numerical_gradient

TOL = 2e-5


def test_cross_entropy_gradient():
    rng = np.random.default_rng(0)
    n, C = 8, 4
    logits = rng.standard_normal((n, C))
    labels = rng.integers(0, C, n)
    loss_fn = make_loss("cross_entropy")

    loss, d_logits = loss_fn(logits, labels)
    assert np.isfinite(loss) and loss > 0
    assert d_logits.shape == logits.shape

    def f(flat):
        return loss_fn(flat.reshape(logits.shape), labels)[0]

    d_num = numerical_gradient(f, logits.ravel().copy()).reshape(logits.shape)
    assert gradient_check(d_logits, d_num) < TOL


def test_cross_entropy_stable_under_large_logits():
    # max-shift trick must prevent inf/nan even with saturated inputs
    logits = np.array([[1000.0, -1000.0, 0.0]])
    labels = np.array([0])
    loss, d = make_loss()(logits, labels)
    assert np.isfinite(loss)
    assert np.all(np.isfinite(d))


def test_cross_entropy_minimized_by_confident_correct_prediction():
    loss_fn = make_loss()
    labels = np.array([0])
    low_conf  = loss_fn(np.array([[2.0,  1.0, 0.0]]), labels)[0]
    high_conf = loss_fn(np.array([[10.0, 1.0, 0.0]]), labels)[0]
    assert high_conf < low_conf


def test_cross_entropy_gradient_sums_to_zero():
    # softmax gradients over classes sum to 0 for each sample
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((16, 5))
    labels = rng.integers(0, 5, 16)
    _, d = make_loss()(logits, labels)
    assert np.allclose(d.sum(axis=1), 0, atol=1e-12)


def test_cross_entropy_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown loss"):
        make_loss("mse")
