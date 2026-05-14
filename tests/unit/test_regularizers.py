from __future__ import annotations

import numpy as np

from sdplab.regularization import EntropyReg, EntropyRegLog, QuadraticReg


def test_quadratic_regularizer_scalar_formulas(np_ctx):
    """Cover the quadratic phi, conjugate, derivative, and log derivative."""
    reg = QuadraticReg(0.5, ctx=np_ctx)
    x = np_ctx.asarray([-1.0, 0.0, 2.0])

    assert np.allclose(reg.phi(x), [0.5, 0.0, 2.0])
    assert np.allclose(reg.phi_star(x), [0.0, 0.0, 2.0])
    assert np.allclose(reg.phi_star_prime(x), [0.0, 0.0, 2.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        log_prime = reg.log_phi_star_prime(x)
    assert np.isneginf(log_prime[0])
    assert np.allclose(log_prime[2], np.log(2.0))


def test_entropy_regularizer_scalar_formulas(np_ctx):
    """Cover entropy formulas on positive and nonpositive eigenvalues."""
    reg = EntropyReg(0.25, ctx=np_ctx)
    x = np_ctx.asarray([0.0, 1.0, 2.0])

    with np.errstate(divide="ignore", invalid="ignore"):
        phi = reg.phi(x)
    assert np.allclose(phi, [0.0, -1.0, 2.0 * (np.log(2.0) - 1.0)])
    assert np.allclose(reg.phi_star(x), np.exp(x))
    assert np.allclose(reg.phi_star_prime(x), np.exp(x))
    assert np.allclose(reg.log_phi_star_prime(x), x)


def test_entropy_log_uses_logsumexp_conjugate(np_ctx):
    """Ensure the trace-normalized entropy variant returns logsumexp."""
    reg = EntropyRegLog(1.0, ctx=np_ctx)
    x = np_ctx.asarray([1.0, 2.0])

    expected = np.log(np.exp(1.0) + np.exp(2.0))
    assert np.allclose(reg._phi_star(x), expected)
