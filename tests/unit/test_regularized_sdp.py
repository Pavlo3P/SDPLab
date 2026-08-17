"""Tests for RegularizedSDPDualFunctional / BoundDualFunctional (spacecore 0.4.2 API).

All variables are plain space elements; eps is a per-call argument. The
functional lives on ``problem.cod`` and evaluates
``D_eps(y) = <b, y> - eps Tr psi((A^dag y - C) / eps)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from sdplab.examples import generate_max_cut
from sdplab.regularization import (
    BoundDualFunctional,
    EntropyReg,
    QuadraticReg,
    RegularizedSDPDualFunctional,
)

FD_STEP = 1e-6
FD_RTOL = 1e-4

# (regularizer class, eps, elementwise psi) pairs used throughout.
REG_CASES = [
    pytest.param(EntropyReg, 0.5, lambda s: np.exp(s), id="entropy-eps0.5"),
    pytest.param(
        QuadraticReg,
        1e-2,
        lambda s: np.maximum(s, 0.0) ** 2 / 2,
        id="quadratic-eps1e-2",
    ),
]


@pytest.fixture(scope="module")
def maxcut():
    """Deterministic Max-Cut SDP: Herm(6) domain, R^6 codomain."""
    return generate_max_cut(6, seed=0)


@pytest.fixture(scope="module")
def y6():
    """Fixed dual iterate in the Max-Cut codomain."""
    rng = np.random.default_rng(42)
    return 0.1 * rng.normal(size=6)


def central_fd_grad(value_fn, y, step=FD_STEP):
    """Central finite differences of a scalar function of a dense vector."""
    y = np.asarray(y, dtype=float)
    grad = np.zeros_like(y)
    for i in range(y.size):
        e = np.zeros_like(y)
        e[i] = step
        grad[i] = (value_fn(y + e) - value_fn(y - e)) / (2.0 * step)
    return grad


# ---------------------------------------------------------------------------
# value / grad correctness on the Max-Cut problem
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reg_cls, eps, psi", REG_CASES)
def test_value_equals_dual_objective_minus_legendre(maxcut, y6, reg_cls, eps, psi):
    """D_eps(y) = <b, y> - eps * sum psi(eig(slack) / eps), computed by hand."""
    functional = RegularizedSDPDualFunctional(maxcut, reg_cls(maxcut.dom))

    val = functional.value(y6, eps)

    slack = np.asarray(functional.slack(y6))
    assert np.allclose(slack, np.diag(y6) - np.asarray(maxcut.C.to_dense()))
    s = np.linalg.eigvalsh(slack)
    expected = float(np.dot(np.asarray(maxcut.b), y6)) - eps * float(np.sum(psi(s / eps)))

    assert np.ndim(val) == 0
    assert np.allclose(float(val), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("reg_cls, eps, psi", REG_CASES)
def test_grad_matches_central_finite_differences(maxcut, y6, reg_cls, eps, psi):
    """The Riesz gradient on the R^6 codomain matches coordinatewise central FD."""
    functional = RegularizedSDPDualFunctional(maxcut, reg_cls(maxcut.dom))

    grad = np.asarray(functional.grad(y6, eps))
    assert grad.shape == maxcut.cod.shape

    fd = central_fd_grad(lambda y: float(functional.value(y, eps)), y6)
    np.testing.assert_allclose(grad, fd, rtol=FD_RTOL, atol=1e-8)


@pytest.mark.parametrize("reg_cls, eps, psi", REG_CASES)
def test_value_and_grad_consistent_with_value_and_grad_methods(
    maxcut, y6, reg_cls, eps, psi
):
    functional = RegularizedSDPDualFunctional(maxcut, reg_cls(maxcut.dom))

    val, grad = functional.value_and_grad(y6, eps)

    assert np.allclose(float(val), float(functional.value(y6, eps)))
    np.testing.assert_allclose(
        np.asarray(grad), np.asarray(functional.grad(y6, eps)), rtol=0, atol=0
    )


# ---------------------------------------------------------------------------
# sign handling: negation goes through the spacecore functional algebra
# ---------------------------------------------------------------------------


def test_negated_bound_functional_flips_value_and_grad(maxcut, y6):
    """``-bound`` (spacecore ScaledFunctional) is the minimization view of D_eps."""
    eps = 0.5
    functional = RegularizedSDPDualFunctional(maxcut, EntropyReg(maxcut.dom))
    minus = -functional.bind(eps)

    assert np.allclose(float(minus.value(y6)), -float(functional.value(y6, eps)))
    np.testing.assert_allclose(
        np.asarray(minus.grad(y6)), -np.asarray(functional.grad(y6, eps))
    )

    val_m, grad_m = minus.value_and_grad(y6)
    val_p, grad_p = functional.value_and_grad(y6, eps)
    assert np.allclose(float(val_m), -float(val_p))
    np.testing.assert_allclose(np.asarray(grad_m), -np.asarray(grad_p))


# ---------------------------------------------------------------------------
# bind(eps) -> BoundDualFunctional
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reg_cls, eps, psi", REG_CASES)
def test_bind_matches_parent_two_argument_calls(maxcut, y6, reg_cls, eps, psi):
    functional = RegularizedSDPDualFunctional(maxcut, reg_cls(maxcut.dom))
    bound = functional.bind(eps)

    assert isinstance(bound, BoundDualFunctional)
    assert bound.base is functional
    assert float(bound.eps_val) == eps

    assert np.allclose(float(bound.value(y6)), float(functional.value(y6, eps)))
    np.testing.assert_allclose(
        np.asarray(bound.grad(y6)), np.asarray(functional.grad(y6, eps)), rtol=0, atol=0
    )

    b_val, b_grad = bound.value_and_grad(y6)
    p_val, p_grad = functional.value_and_grad(y6, eps)
    assert np.allclose(float(b_val), float(p_val))
    np.testing.assert_allclose(np.asarray(b_grad), np.asarray(p_grad), rtol=0, atol=0)


def test_bound_functional_has_no_sign_flag(maxcut):
    """The bound view reports D_eps as-is; sign flipping is not its job."""
    functional = RegularizedSDPDualFunctional(maxcut, EntropyReg(maxcut.dom))
    bound = functional.bind(0.5)
    assert not hasattr(bound, "invert_sign")
    assert not hasattr(functional, "invert_sign")


def test_bound_primal_from_dual_matches_parent(maxcut, y6):
    eps = 1e-2
    functional = RegularizedSDPDualFunctional(maxcut, QuadraticReg(maxcut.dom))
    bound = functional.bind(eps)

    np.testing.assert_allclose(
        np.asarray(bound.primal_from_dual(y6, normalized=False)),
        np.asarray(functional.primal_from_dual(y6, eps, normalized=False)),
    )


# ---------------------------------------------------------------------------
# primal recovery
# ---------------------------------------------------------------------------


def test_primal_from_dual_spectral_formula_quadratic(maxcut, y6):
    """X = V diag(psi'(s/eps)) V^dag with slack = V diag(s) V^dag (unnormalized)."""
    eps = 1e-2
    functional = RegularizedSDPDualFunctional(maxcut, QuadraticReg(maxcut.dom))

    X = np.asarray(functional.primal_from_dual(y6, eps, normalized=False))

    # A plain dom element: Hermitian matrix of the domain shape.
    assert X.shape == maxcut.dom.shape
    np.testing.assert_allclose(X, X.T.conj())
    maxcut.dom.check_member(X)

    slack = np.asarray(functional.slack(y6))
    s, V = np.linalg.eigh(slack)
    lam = np.maximum(s / eps, 0.0)  # psi'(s/eps) for QuadraticReg

    np.testing.assert_allclose(np.linalg.eigvalsh(X), lam, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(X, (V * lam) @ V.T.conj(), rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# TreeSpace (mixed cone) domain
