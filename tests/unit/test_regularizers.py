"""Tests for the spectral regularizers (spacecore 0.4.2 API).

Regularizers are constructed on a Euclidean Jordan algebra space; the
strength eps is a per-call ``val`` argument everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest
from spacecore import ElementwiseJordanSpace, HermitianSpace, TreeSpace

from sdplab import regularization
from sdplab.regularization import EntropyReg, QuadraticReg, Regularizer


def test_public_surface():
    assert regularization.EntropyReg is EntropyReg
    assert regularization.QuadraticReg is QuadraticReg
    assert set(regularization.__all__) == {
        "Regularizer",
        "RegularizedSDPDualFunctional",
        "BoundDualFunctional",
        "EntropyReg",
        "QuadraticReg",
    }
    assert issubclass(EntropyReg, Regularizer)
    assert issubclass(QuadraticReg, Regularizer)


def test_normalized_flag_is_bounded_log_partition(np_ctx):
    """normalized=True selects the fixed-trace log-partition, exact for entropy."""
    import scipy.special
    from spacecore import HermitianSpace

    space = HermitianSpace(3, ctx=np_ctx)
    reg = EntropyReg(space)
    D = np.array([0.5, 1.0, 8.0])          # eigenvalues that overflow exp(D/eps)
    X = np_ctx.asarray(np.diag(D))
    eps = 1e-3

    # Free legendre overflows; the normalized one is the finite log-partition.
    assert np.isinf(float(reg.legendre(X, eps)))                       # separable exp sum
    leg = float(reg.legendre(X, eps, normalized=True))
    assert np.isfinite(leg)
    # eps*(logsumexp + 1): the trailing eps is what makes this the conjugate
    # of phi(t) = t(log t - 1), whose -t contributes -eps on the unit-trace face.
    assert np.isclose(leg, eps * (scipy.special.logsumexp(D / eps) + 1.0))

    # normalized gradient is the unit-trace Gibbs state softmax(D / eps).
    val, grad = reg.legendre_and_grad(X, eps, normalized=True)
    gibbs = scipy.special.softmax(D / eps)
    assert np.isclose(val, leg)
    assert np.isclose(float(np.trace(np.asarray(grad))), 1.0)
    np.testing.assert_allclose(np.linalg.eigvalsh(np.asarray(grad)), np.sort(gibbs),
                               atol=1e-12)


def test_normalized_value_and_grad_are_consistent_for_entropy(np_ctx):
    """For entropy the log-partition value and Gibbs gradient are ∇-consistent."""
    from spacecore import HermitianSpace

    space = HermitianSpace(3, ctx=np_ctx)
    reg = EntropyReg(space)
    rng = np.random.default_rng(0)
    M = rng.normal(size=(3, 3)); X = (M + M.T) / 2
    eps = 0.3

    # d/dt legendre(X + t E, eps, normalized) == <grad, E> for a Hermitian E.
    _, grad = reg.legendre_and_grad(np_ctx.asarray(X), eps, normalized=True)
    E = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    h = 1e-6
    fd = (float(reg.legendre(np_ctx.asarray(X + h * E), eps, normalized=True))
          - float(reg.legendre(np_ctx.asarray(X - h * E), eps, normalized=True))) / (2 * h)
    assert np.isclose(fd, float(np.sum(np.asarray(grad) * E)), rtol=1e-5, atol=1e-7)


def test_separable_entropy_normalized_grad_is_overflow_safe(np_ctx):
    """The normalized separable-entropy gradient stays finite at tiny eps."""
    from spacecore import HermitianSpace

    space = HermitianSpace(3, ctx=np_ctx)
    reg = EntropyReg(space)
    X = np_ctx.asarray(np.diag([0.5, 1.0, 8.0]))
    _, grad = reg.legendre_and_grad(X, 1e-6, normalized=True)
    g = np.asarray(grad)
    assert np.all(np.isfinite(g))
    assert np.isclose(float(np.trace(g)), 1.0)


# ---------------------------------------------------------------------------
# Elementwise scalar formulas
# ---------------------------------------------------------------------------


def test_entropy_scalar_formulas(np_ctx):
    reg = EntropyReg(HermitianSpace(2, ctx=np_ctx))
    x = np.array([0.5, 1.0, 2.0])

    np.testing.assert_allclose(reg.phi(x), x * (np.log(x) - 1.0))
    assert float(reg.phi(np.array(0.0))) == 0.0
    assert np.isinf(float(reg.phi(np.array(-1.0))))
    np.testing.assert_allclose(reg.phi_star(x), np.exp(x))
    np.testing.assert_allclose(reg.phi_star_prime(x), np.exp(x))
    np.testing.assert_allclose(reg.log_phi_star_prime(x), x)


def test_quadratic_scalar_formulas(np_ctx):
    reg = QuadraticReg(HermitianSpace(2, ctx=np_ctx))
    x = np.array([-1.0, 0.5, 2.0])

    # phi is t^2/2 on the nonnegative branch and +inf out of domain, so that it
    # is the Fenchel partner of phi_star; only round-off negatives (down to
    # -NEG_EIG_TOL) are pulled back to the limit phi(0) = 0.
    phi = reg.phi(x)
    np.testing.assert_allclose(phi[1:], x[1:] ** 2 / 2)
    assert np.isinf(phi[0])
    assert float(reg.phi(np.array(-1e-15))) == 0.0
    # The conjugate side keeps the cone semantics.
    np.testing.assert_allclose(reg.phi_star(x), np.maximum(x, 0.0) ** 2 / 2)
    np.testing.assert_allclose(reg.phi_star_prime(x), np.maximum(x, 0.0))
    log_prime = reg.log_phi_star_prime(x)
    np.testing.assert_allclose(log_prime[1:], np.log(x[1:]))
    assert np.isneginf(log_prime[0])


# ---------------------------------------------------------------------------
# Spectral calculus on a Hermitian space (hand-computable diagonal data)
# ---------------------------------------------------------------------------


@pytest.fixture()
def herm3(np_ctx):
    return HermitianSpace(3, ctx=np_ctx)


DIAG = np.array([0.5, 1.0, 2.0])


def test_call_is_eps_times_trace_phi(herm3, np_ctx):
    reg = EntropyReg(herm3)
    X = np_ctx.asarray(np.diag(DIAG))
    eps = 0.3

    expected = eps * float(np.sum(DIAG * (np.log(DIAG) - 1.0)))
    assert np.allclose(float(reg(X, eps)), expected)


@pytest.mark.parametrize(
    "reg_cls, psi",
    [
        (EntropyReg, lambda s: np.exp(s)),
        (QuadraticReg, lambda s: np.maximum(s, 0.0) ** 2 / 2),
    ],
)
def test_legendre_scaling_identity(herm3, np_ctx, reg_cls, psi):
    """legendre(X, val) == val * sum psi(eig(X) / val)."""
    reg = reg_cls(herm3)
    X = np_ctx.asarray(np.diag(DIAG))

    for eps in (0.25, 1.0, 3.0):
        expected = eps * float(np.sum(psi(DIAG / eps)))
        assert np.allclose(float(reg.legendre(X, eps)), expected)


@pytest.mark.parametrize(
    "reg_cls, psi_prime",
    [
        (EntropyReg, lambda s: np.exp(s)),
        (QuadraticReg, lambda s: np.maximum(s, 0.0)),
    ],
)
def test_legendre_and_grad_spectral_formula(herm3, np_ctx, reg_cls, psi_prime):
    """The gradient is psi'(X / eps) applied through the frame of X."""
    rng = np.random.default_rng(5)
    M = rng.normal(size=(3, 3))
    X = np_ctx.asarray((M + M.T) / 2)
    eps = 0.7

    reg = reg_cls(herm3)
    val, grad = reg.legendre_and_grad(X, eps)

    s, V = np.linalg.eigh(np.asarray(X))
    expected_grad = (V * psi_prime(s / eps)) @ V.T
    assert np.allclose(float(val), float(reg.legendre(X, eps)))
    np.testing.assert_allclose(np.asarray(grad), expected_grad, rtol=1e-10, atol=1e-12)


def test_normalized_gradient_has_unit_trace(herm3, np_ctx):
    """normalized=True rescales the gradient eigenvalues to unit trace."""
    rng = np.random.default_rng(6)
    M = rng.normal(size=(3, 3))
    X = np_ctx.asarray((M + M.T) / 2)
    eps = 0.5

    for reg in (EntropyReg(herm3), QuadraticReg(herm3)):
        norm = np.asarray(reg.phi_star_prime_matrix(X, eps, normalized=True))
        assert np.allclose(float(np.trace(norm)), 1.0)

    # How the unit trace is reached differs. Entropy's shift acts as a global
    # rescale, so the result is psi' over its own trace.
    ent = EntropyReg(herm3)
    raw = np.asarray(ent.phi_star_prime_matrix(X, eps, normalized=False))
    norm = np.asarray(ent.phi_star_prime_matrix(X, eps, normalized=True))
    np.testing.assert_allclose(norm, raw / np.trace(raw), rtol=1e-10, atol=1e-12)

    # Quadratic's shift moves the clip point, so it is the simplex projection:
    # unit trace, but sparse and not any multiple of psi'.
    quad = QuadraticReg(herm3)
    raw = np.asarray(quad.phi_star_prime_matrix(X, eps, normalized=False))
    norm = np.asarray(quad.phi_star_prime_matrix(X, eps, normalized=True))
    assert not np.allclose(norm, raw / np.trace(raw))


def test_quadratic_fixed_trace_conjugate_is_a_gradient_pair(herm3, np_ctx):
    """The sparsemax value and its gradient satisfy the Fenchel identity."""
    rng = np.random.default_rng(11)
    M = rng.normal(size=(3, 3))
    S = (M + M.T) / 2
    X = np_ctx.asarray(S)
    eps = 0.7
    reg = QuadraticReg(herm3)

    val, grad = reg.legendre_and_grad(X, eps, normalized=True)
    G = np.asarray(grad)

    # F(S) = <S, X*> - eps*Tr[phi(X*)] at the constrained maximizer X*.
    assert float(np.trace(G)) == pytest.approx(1.0, abs=1e-12)
    fenchel = float(np.sum(S * G)) - float(reg(np_ctx.asarray(G), eps))
    assert float(val) == pytest.approx(fenchel, abs=1e-10)

    # ...and the value's slope is that gradient.
    E = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    h = 1e-6
    fd = (float(reg.legendre(np_ctx.asarray(S + h * E), eps, True))
          - float(reg.legendre(np_ctx.asarray(S - h * E), eps, True))) / (2 * h)
    assert fd == pytest.approx(float(np.sum(G * E)), rel=1e-5, abs=1e-8)


def test_phi_star_prime_matrix_matches_gradient(herm3, np_ctx):
    rng = np.random.default_rng(7)
    M = rng.normal(size=(3, 3))
    X = np_ctx.asarray((M + M.T) / 2)
    eps = 0.4

    reg = EntropyReg(herm3)
    _, grad = reg.legendre_and_grad(X, eps, normalized=True)
    np.testing.assert_allclose(
        np.asarray(reg.phi_star_prime_matrix(X, eps, normalized=True)),
        np.asarray(grad),
    )


# ---------------------------------------------------------------------------
# Elementwise-Jordan and TreeSpace domains
# ---------------------------------------------------------------------------


def test_elementwise_domain_spectrum_is_identity(np_ctx):
    """On an orthant space the spectrum is the vector itself."""
    space = ElementwiseJordanSpace((4,), ctx=np_ctx)
    reg = QuadraticReg(space)
    x = np_ctx.asarray([-1.0, 0.5, 2.0, 3.0])
    eps = 0.5

    expected = eps * float(np.sum(np.maximum(np.asarray(x) / eps, 0.0) ** 2 / 2))
    assert np.allclose(float(reg.legendre(x, eps)), expected)

    _, grad = reg.legendre_and_grad(x, eps)
    np.testing.assert_allclose(
        np.asarray(grad), np.maximum(np.asarray(x) / eps, 0.0)
    )


def test_tree_domain_legendre_sums_over_leaves(np_ctx):
    """On a tree of (Herm(2), orthant(3)) leaves legendre is the leaf sum."""
    herm = HermitianSpace(2, ctx=np_ctx)
    orthant = ElementwiseJordanSpace((3,), ctx=np_ctx)
    tree = TreeSpace.from_leaf_spaces((herm, orthant))

    H = np.array([[1.0, 0.5], [0.5, -0.5]])
    v = np.array([0.2, -0.1, 1.5])
    X = (np_ctx.asarray(H), np_ctx.asarray(v))
    eps = 0.8

    reg = QuadraticReg(tree)
    per_leaf = (
        float(QuadraticReg(herm).legendre(X[0], eps))
        + float(QuadraticReg(orthant).legendre(X[1], eps))
    )
    assert np.allclose(float(reg.legendre(X, eps)), per_leaf)

    _, grad = reg.legendre_and_grad(X, eps)
    gH, gv = tree.flatten_tree(grad)
    s, V = np.linalg.eigh(H)
    np.testing.assert_allclose(
        np.asarray(gH), (V * np.maximum(s / eps, 0.0)) @ V.T, atol=1e-12
    )
    np.testing.assert_allclose(np.asarray(gv), np.maximum(v / eps, 0.0))


# ---------------------------------------------------------------------------
# Backend parity
# ---------------------------------------------------------------------------


def test_numpy_jax_parity_legendre_and_grad(np_ctx, jax_ctx):
    rng = np.random.default_rng(11)
    M = rng.normal(size=(4, 4))
    X = (M + M.T) / 2
    eps = 0.6

    reg_np = EntropyReg(HermitianSpace(4, ctx=np_ctx))
    reg_jax = EntropyReg(HermitianSpace(4, ctx=jax_ctx))

    val_np, grad_np = reg_np.legendre_and_grad(np_ctx.asarray(X), eps)
    val_jax, grad_jax = reg_jax.legendre_and_grad(jax_ctx.asarray(X), eps)

    assert np.allclose(float(val_np), float(val_jax))
    np.testing.assert_allclose(
        np.asarray(grad_np), np.asarray(grad_jax), rtol=1e-10, atol=1e-12
    )
