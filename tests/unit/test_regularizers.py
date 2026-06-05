from __future__ import annotations

import numpy as np
import pytest
from spacecore import Context, HermitianSpace

from sdplab.regularization import (
    EntropyReg,
    EntropyRegLog,
    QuadraticReg,
    Regularizer,
    SDPRegularized,
)


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _logsumexp(x):
    x = np.asarray(x, dtype=np.float64)
    shift = np.max(x)
    return shift + np.log(np.sum(np.exp(x - shift)))


def test_public_regularizer_imports():
    """Keep the regularizer package exports stable."""
    import sdplab.regularization as regularization

    assert regularization.Regularizer is Regularizer
    assert regularization.EntropyReg is EntropyReg
    assert regularization.EntropyRegLog is EntropyRegLog
    assert regularization.QuadraticReg is QuadraticReg
    assert regularization.SDPRegularized is SDPRegularized
    assert set(regularization.__all__) == {
        "Regularizer",
        "SDPRegularized",
        "EntropyReg",
        "EntropyRegLog",
        "QuadraticReg",
    }


def test_entropy_phi_extended_domain_avoids_nonpositive_log(np_ctx):
    """Entropy is zero at 0, finite on positives, and +inf on negatives."""
    reg = EntropyReg(0.25, ctx=np_ctx)
    x = np_ctx.asarray([-1.0, 0.0, 1.0, 2.0])

    with np.errstate(divide="raise", invalid="raise"):
        phi = reg.phi(x)

    assert np.isposinf(phi[0])
    assert phi[1] == 0.0
    assert np.allclose(phi[2:], [-1.0, 2.0 * (np.log(2.0) - 1.0)])
    assert np.allclose(reg.phi_star(x), np.exp(x))
    assert np.allclose(reg.phi_star_prime(x), np.exp(x))
    assert np.allclose(reg.log_phi_star_prime(x), x)


def test_quadratic_regularizer_extended_domain_and_conjugacy(np_ctx):
    """Quadratic phi includes the nonnegative-domain indicator."""
    reg = QuadraticReg(0.5, ctx=np_ctx)
    x = np_ctx.asarray([-1.0, 0.0, 2.0])

    phi = reg.phi(x)
    assert np.isposinf(phi[0])
    assert np.allclose(phi[1:], [0.0, 2.0])
    assert np.allclose(reg.phi_star(x), [0.0, 0.0, 2.0])
    assert np.allclose(reg.phi_star_prime(x), [0.0, 0.0, 2.0])

    grid = np.linspace(0.0, 6.0, 20_001)
    for s in [-2.0, 0.0, 1.5, 4.0]:
        numerical_conjugate = np.max(s * grid - 0.5 * grid**2)
        assert np.allclose(reg.phi_star(np_ctx.asarray(s)), numerical_conjugate, atol=1e-4)


def test_quadratic_log_derivative_marks_nonpositive_entries(np_ctx):
    reg = QuadraticReg(1.0, ctx=np_ctx)
    x = np_ctx.asarray([-1.0, 0.0, 2.0])

    with np.errstate(divide="ignore", invalid="ignore"):
        log_prime = reg.log_phi_star_prime(x)

    assert np.isneginf(log_prime[0])
    assert np.isneginf(log_prime[1])
    assert np.allclose(log_prime[2], np.log(2.0))


def test_entropy_log_represents_log_trace_exp_value(np_ctx):
    eps = 0.5
    space = HermitianSpace(2, ctx=np_ctx)
    reg = EntropyRegLog(eps, space, ctx=np_ctx)
    x = np_ctx.asarray([[1.0, 0.0], [0.0, 3.0]])

    expected = eps * _logsumexp([1.0 / eps, 3.0 / eps])

    assert np.allclose(reg.legendre(x), expected)
    assert np.allclose(reg._phi_star(np_ctx.asarray([1.0 / eps, 3.0 / eps])), expected / eps)


def test_entropy_log_rejects_elementwise_phi_star(np_ctx):
    reg = EntropyRegLog(1.0, ctx=np_ctx)

    with pytest.raises(NotImplementedError, match="no elementwise phi_star"):
        reg.phi_star(np_ctx.asarray([1.0, 2.0]))


def test_entropy_log_derivatives_are_normalized_weights(np_ctx):
    reg = EntropyRegLog(1.0, ctx=np_ctx)
    x = np_ctx.asarray([1.0, 2.0, -3.0])
    expected = np.exp(_to_numpy(x) - _logsumexp(_to_numpy(x)))

    assert np.allclose(reg.phi_star_prime(x), expected)
    assert np.allclose(np.exp(reg.log_phi_star_prime(x)), expected)
    assert np.allclose(np.sum(reg.phi_star_prime(x)), 1.0)


def test_entropy_log_matrix_gradients_have_trace_one(np_ctx):
    eps = 0.75
    space = HermitianSpace(2, ctx=np_ctx)
    reg = EntropyRegLog(eps, space, ctx=np_ctx)
    x = np_ctx.asarray([[1.0, 0.0], [0.0, -0.5]])

    _, grad = reg.legendre_and_grad(x, normalized=False)
    grad_from_prime = reg.phi_star_prime_matrix(x, normalized=False)
    grad_from_normalized_prime = reg.phi_star_prime_matrix(x, normalized=True)

    assert np.allclose(np.trace(grad), 1.0)
    assert np.allclose(np.trace(grad_from_prime), 1.0)
    assert np.allclose(np.trace(grad_from_normalized_prime), 1.0)
    assert np.allclose(grad, grad_from_prime)
    assert np.allclose(grad, grad_from_normalized_prime)


def test_entropy_log_gradient_matches_finite_difference(np_ctx):
    eps = 0.6
    space = HermitianSpace(2, ctx=np_ctx)
    reg = EntropyRegLog(eps, space, ctx=np_ctx)
    x = np_ctx.asarray([[0.5, 0.0], [0.0, -0.25]])
    direction = np_ctx.asarray([[0.2, 0.0], [0.0, 0.7]])
    h = 1e-6

    _, grad = reg.legendre_and_grad(x)
    finite_diff = (reg.legendre(x + h * direction) - reg.legendre(x - h * direction)) / (2.0 * h)
    directional_derivative = np.real(np.trace(grad @ direction))

    assert np.allclose(finite_diff, directional_derivative, rtol=1e-5, atol=1e-6)


def test_regularizer_scalar_formulas_on_jax_backend(jax_ctx):
    entropy = EntropyReg(1.0, ctx=jax_ctx)
    entropy_log = EntropyRegLog(1.0, ctx=jax_ctx)
    quadratic = QuadraticReg(1.0, ctx=jax_ctx)
    x = jax_ctx.asarray([-1.0, 0.0, 2.0])

    assert np.isposinf(_to_numpy(entropy.phi(x))[0])
    assert np.allclose(_to_numpy(entropy_log.phi_star_prime(x)).sum(), 1.0)
    assert np.allclose(_to_numpy(quadratic.phi_star_prime(x)), [0.0, 0.0, 2.0])


def test_regularizer_scalar_formulas_on_torch_backend():
    torch = pytest.importorskip("torch")
    try:
        from spacecore import TorchOps
    except ImportError:
        pytest.skip("spacecore TorchOps is unavailable")

    torch_ctx = Context(TorchOps(), dtype=torch.float64)
    entropy = EntropyReg(1.0, ctx=torch_ctx)
    entropy_log = EntropyRegLog(1.0, ctx=torch_ctx)
    quadratic = QuadraticReg(1.0, ctx=torch_ctx)
    x = torch_ctx.asarray([-1.0, 0.0, 2.0])

    assert np.isposinf(_to_numpy(entropy.phi(x))[0])
    assert np.allclose(_to_numpy(entropy_log.phi_star_prime(x)).sum(), 1.0)
    assert np.allclose(_to_numpy(quadratic.phi_star_prime(x)), [0.0, 0.0, 2.0])
