"""Regularized QOT: convergence to a closed-form optimum, per regularizer.

Every test here is a quantum optimal-transport instance -- the constraint is
``QOTConstraintOp``, so ``b`` is the one-body partial traces of the intended
optimum -- solved through the regularized dual and checked against the primal
that was used to build it.

The construction inverts the usual direction: instead of solving and hoping the
answer is right, *pick* the optimum and derive the problem from it. For a dual
point ``U`` and slack :math:`S = \\mathcal{A}^\\dagger U - H`, set
:math:`X^\\star = \\psi'(S/\\varepsilon)` and take :math:`b = \\mathcal{A}X^\\star`.
Then

    grad D_eps(U) = b - A psi'((A^dagger U - H)/eps) = 0

exactly, so ``U`` is a stationary point of a concave objective and ``X*`` is the
regularized primal optimum. Each test asserts that stationarity first (it is the
premise) and then that the solver actually travels back to ``X*`` from a cold
start.

For entropy this is the Gibbs variational principle: :math:`\\psi' = \\exp`, so
``X*`` is the Gibbs operator of the slack, and its normalized form is the Gibbs
*state*. The marginal data ``b`` is the partial traces of that state, which is
what a QOT instance constrains.

The QOT codomain is complex, so these run on the optax/JAX route --
``minimize_scipy`` requires a real codomain.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from spacecore import Context, NumpyOps

from sdplab.problem import SDPProblem
from sdplab.regularization import (
    EntropyReg,
    QuadraticReg,
    RegularizedSDPDualFunctional,
)
from sdplab.solvers import run_regularized_solver
from sdplab.special.qot import QOTConstraintOp

D_LOCAL, N_SITES, EPS = 2, 2, 1.0


@pytest.fixture(scope="module")
def jax_complex_ctx():
    """A complex128 JAX context; QOT's stacked Hermitian codomain needs one."""
    pytest.importorskip("jax")
    from spacecore import JaxOps

    return Context(JaxOps(), dtype="complex128", check_level="none")


def _instance(psi_prime, eps=EPS, d=D_LOCAL, N=N_SITES, seed=0):
    """Return ``(sdp, X_star, U)`` whose regularized optimum is ``X_star``.

    ``U`` is drawn away from zero on purpose: with ``b`` built at ``U = 0`` the
    optimum would coincide with the solver's default cold start and the test
    would pass without anything being optimized.
    """
    ctx = Context(NumpyOps(), dtype=np.complex128, check_level="none")
    dim = d**N
    rng = np.random.default_rng(seed)

    M = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    H = (M + M.conj().T) / 2                      # Hamiltonian / cost

    U = rng.normal(size=(N, d, d)) + 1j * rng.normal(size=(N, d, d))
    U = 0.3 * (U + np.conj(np.swapaxes(U, -1, -2))) / 2

    op = QOTConstraintOp(d=d, N=N, ctx=ctx)
    S = np.asarray(op.rapply(ctx.asarray(U))) - H     # dual slack at U
    X_star = psi_prime(S / eps)

    b = op.apply(ctx.asarray(X_star))                 # marginals of X_star
    sdp = SDPProblem(ctx.asarray(H), op, b, ctx=ctx)
    return sdp, X_star, ctx.asarray(U)


def _gibbs(S):
    """Unnormalized Gibbs operator ``exp(S)`` -- the entropy ``psi'``."""
    return sla.expm(S)


def _clipped(S):
    """``max(S, 0)`` spectrally -- the quadratic ``psi'``."""
    w, V = np.linalg.eigh(S)
    return (V * np.maximum(w, 0.0)) @ V.conj().T


def _simplex_projection(S):
    """The unit-trace quadratic ``psi'``: ``max(S - tau, 0)`` with ``sum = 1``.

    Reimplemented here by sorting, independently of
    :meth:`~sdplab.regularization.QuadraticReg._simplex_threshold`, so the test
    checks the solver against the mathematics rather than against itself.
    """
    w, V = np.linalg.eigh(S)
    u = np.sort(w)[::-1]
    css = np.cumsum(u)
    k = np.arange(1, u.size + 1)
    size = int(np.sum(u > (css - 1.0) / k))
    tau = (css[size - 1] - 1.0) / size
    return (V * np.maximum(w - tau, 0.0)) @ V.conj().T


def _assert_stationary(sdp, reg_cls, U, normalized):
    """The constructed ``U`` must be a stationary point of the dual."""
    F = RegularizedSDPDualFunctional(sdp, reg_cls(sdp.dom))
    g = np.asarray(F.grad(U, EPS, normalized))
    assert np.linalg.norm(g) < 1e-10


def test_entropy_recovers_the_gibbs_operator(jax_complex_ctx):
    """Entropy at eps=1: the optimum is exp(slack), recovered from its marginals."""
    optax = pytest.importorskip("optax")
    sdp, X_star, U = _instance(_gibbs)
    _assert_stationary(sdp, EntropyReg, U, normalized=False)

    sdp_j = sdp.convert(jax_complex_ctx)
    F = RegularizedSDPDualFunctional(sdp_j, EntropyReg(sdp_j.dom))
    result = run_regularized_solver(
        F.bind(EPS), opt=optax.lbfgs(), max_iter=5000, tol=1e-10, verbose=0
    )

    assert result.converged is True
    X = np.asarray(F.primal_from_dual(result.dual, EPS, normalized=False))
    np.testing.assert_allclose(X, X_star, atol=1e-8)


def test_quadratic_recovers_the_clipped_slack(jax_complex_ctx):
    """Quadratic at eps=1: the optimum is max(slack, 0), spectrally."""
    optax = pytest.importorskip("optax")
    sdp, X_star, U = _instance(_clipped)
    _assert_stationary(sdp, QuadraticReg, U, normalized=False)

    sdp_j = sdp.convert(jax_complex_ctx)
    F = RegularizedSDPDualFunctional(sdp_j, QuadraticReg(sdp_j.dom))
    result = run_regularized_solver(
        F.bind(EPS), opt=optax.lbfgs(), max_iter=5000, tol=1e-10, verbose=0
    )

    assert result.converged is True
    X = np.asarray(F.primal_from_dual(result.dual, EPS, normalized=False))
    np.testing.assert_allclose(X, X_star, atol=1e-8)


def test_entropy_normalized_recovers_the_gibbs_state(jax_complex_ctx):
    """The fixed-trace form: the optimum is the unit-trace Gibbs state.

    ``normalized=True`` selects the fixed-trace conjugate
    ``eps*(logsumexp(S/eps) + 1)``, whose gradient *is* the Gibbs state, so the
    value and gradient are a genuine pair and a line-searching method works.
    """
    optax = pytest.importorskip("optax")

    def normalized_gibbs(S):
        G = sla.expm(S)
        return G / np.trace(G).real

    sdp, X_star, U = _instance(normalized_gibbs)
    assert float(np.real(np.trace(X_star))) == pytest.approx(1.0)
    _assert_stationary(sdp, EntropyReg, U, normalized=True)

    sdp_j = sdp.convert(jax_complex_ctx)
    F = RegularizedSDPDualFunctional(sdp_j, EntropyReg(sdp_j.dom))
    result = run_regularized_solver(
        F.bind(EPS, normalized=True),
        opt=optax.lbfgs(), max_iter=5000, tol=1e-10, verbose=0,
    )

    assert result.converged is True
    X = np.asarray(F.primal_from_dual(result.dual, EPS, normalized=True))
    assert float(np.real(np.trace(X))) == pytest.approx(1.0, abs=1e-8)
    np.testing.assert_allclose(X, X_star, atol=1e-7)


def test_quadratic_normalized_recovers_the_simplex_projection(jax_complex_ctx):
    """The fixed-trace quadratic form: the optimum is the simplex projection.

    The counterpart of the Gibbs-state test. Here the trace multiplier moves the
    clip point rather than rescaling, so the optimum is sparsemax of the slack
    spectrum -- unit trace but genuinely low rank, unlike the full-rank Gibbs
    state. Value and gradient come from the same threshold, so L-BFGS applies.
    """
    optax = pytest.importorskip("optax")

    sdp, X_star, U = _instance(_simplex_projection)
    assert float(np.real(np.trace(X_star))) == pytest.approx(1.0)
    rank = int(np.sum(np.linalg.eigvalsh(X_star) > 1e-10))
    assert rank < X_star.shape[0]                    # the point of sparsemax
    _assert_stationary(sdp, QuadraticReg, U, normalized=True)

    sdp_j = sdp.convert(jax_complex_ctx)
    F = RegularizedSDPDualFunctional(sdp_j, QuadraticReg(sdp_j.dom))
    result = run_regularized_solver(
        F.bind(EPS, normalized=True),
        opt=optax.lbfgs(), max_iter=5000, tol=1e-10, verbose=0,
    )

    assert result.converged is True
    X = np.asarray(F.primal_from_dual(result.dual, EPS, normalized=True))
    assert float(np.real(np.trace(X))) == pytest.approx(1.0, abs=1e-8)
    np.testing.assert_allclose(X, X_star, atol=1e-7)
