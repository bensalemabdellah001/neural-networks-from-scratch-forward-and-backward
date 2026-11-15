import numpy as np
from nns import gradient_check, numerical_gradient


def test_numerical_gradient_quadratic():
    # f(x) = sum(x^2), exact grad = 2x
    x = np.array([1.0, 2.0, 3.0])
    g = numerical_gradient(lambda x: float(np.sum(x ** 2)), x.copy())
    assert np.allclose(g, 2 * x, atol=1e-8)


def test_numerical_gradient_2d():
    # f(X) = sum(X^3), exact grad = 3X^2
    x = np.array([[1.0, 2.0], [0.5, 3.0]])
    g = numerical_gradient(lambda x: float(np.sum(x ** 3)), x.copy())
    assert np.allclose(g, 3 * x ** 2, atol=1e-5)


def test_numerical_gradient_empty():
    g = numerical_gradient(lambda x: 0.0, np.array([]))
    assert g.size == 0


def test_gradient_check_perfect_match():
    a = np.array([1.0, 2.0, 3.0])
    assert gradient_check(a, a) == 0.0


def test_gradient_check_detects_large_error():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 100.0])
    assert gradient_check(a, b) > 0.5


def test_gradient_check_handles_near_zero():
    # both near zero — should not divide by zero
    a = np.array([1e-20, 1e-20])
    b = np.array([0.0, 0.0])
    err = gradient_check(a, b)
    assert np.isfinite(err)
