"""Integration tests for :func:`sdplab.solvers.run_regularized_solver` dispatch.

The dispatcher consumes a :class:`BoundDualFunctional` -- ε and the
normalization travel with the functional -- routes to spacecore's
``minimize_scipy``/``minimize_optax``, and reports the MAXIMIZED dual value
``D_eps`` in ``final_loss``/``loss_history``. All duals and primals are plain
space elements.
"""

from __future__ import annotations

import numpy as np
import pytest

from sdplab.examples import generate_max_cut, generate_random_qot
from sdplab.regularization import (
    QuadraticReg,
    RegularizedSDPDualFunctional,
)
from sdplab.solvers import OptimizeResult, run_cvxpy_solver, run_regularized_solver

EPS = 1e-3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def maxcut_sdp():
    """Deterministic numpy-backed MaxCut(8) instance."""
    return generate_max_cut(8, seed=0)


@pytest.fixture(scope="module")
def maxcut_functional(maxcut_sdp):
    """Quadratically regularized dual functional on the numpy MaxCut."""
    return RegularizedSDPDualFunctional(maxcut_sdp, QuadraticReg(maxcut_sdp.dom))


@pytest.fixture(scope="module")
def cvxpy_objective(maxcut_sdp):
    """Reference optimal value p* from CVXPY/CLARABEL."""
    pytest.importorskip("cvxpy")
    X, _ = run_cvxpy_solver(maxcut_sdp, solver="CLARABEL")
    return float(maxcut_sdp.primal_objective(X))


@pytest.fixture(scope="module")
def scipy_result(maxcut_functional):
    """Shared scipy-route solve on the numpy MaxCut functional."""
    pytest.importorskip("scipy")
    return run_regularized_solver(
        maxcut_functional.bind(EPS), method="scipy", tol=1e-10, verbose=0
    )


@pytest.fixture(scope="module")
def jax_maxcut_functional(jax_ctx):
    """The same MaxCut(8) functional on the JAX backend."""
    sdp = generate_max_cut(8, seed=0, ctx=jax_ctx)
    return RegularizedSDPDualFunctional(sdp, QuadraticReg(sdp.dom))


# ---------------------------------------------------------------------------
# (a) scipy route on numpy MaxCut
# ---------------------------------------------------------------------------


def test_scipy_route_converges_to_cvxpy_objective(scipy_result, cvxpy_objective):
    assert isinstance(scipy_result, OptimizeResult)
    assert scipy_result.converged is True
    # final_loss is the MAXIMIZED dual value D_eps: negative here, close to p*.
    assert scipy_result.final_loss < 0.0
    assert cvxpy_objective < 0.0
    assert scipy_result.final_loss == pytest.approx(cvxpy_objective, rel=0.05)


def test_scipy_route_raw_is_scipy_optimize_result(scipy_result):
    scipy_optimize = pytest.importorskip("scipy.optimize")
    assert isinstance(scipy_result.raw, scipy_optimize.OptimizeResult)
    # spacecore.minimize_scipy attaches the decoded space element to the result.
    assert hasattr(scipy_result.raw, "x_element")
    assert scipy_result.dual.shape == (8,)


# ---------------------------------------------------------------------------
# (b) optax route on jax MaxCut with L-BFGS
# ---------------------------------------------------------------------------


def test_optax_lbfgs_route_matches_scipy_route(jax_maxcut_functional, scipy_result):
    optax = pytest.importorskip("optax")
    from spacecore import OptaxResult

    result = run_regularized_solver(
        jax_maxcut_functional.bind(EPS),
        method="optax",
        opt=optax.lbfgs(),
        max_iter=2000,
        tol=1e-8,
        verbose=0,
    )
    assert result.converged is True
    assert result.final_loss == pytest.approx(scipy_result.final_loss, rel=1e-3)
    assert result.loss_history is not None and len(result.loss_history) > 0
    assert result.grad_norm_history is not None
    assert len(result.grad_norm_history) == len(result.loss_history)
    assert isinstance(result.raw, OptaxResult)


# ---------------------------------------------------------------------------
# (c) method=None auto-dispatch by backend family
# ---------------------------------------------------------------------------


def test_auto_dispatch_numpy_uses_scipy(maxcut_functional):
    scipy_optimize = pytest.importorskip("scipy.optimize")
    result = run_regularized_solver(maxcut_functional.bind(EPS), verbose=0)
    assert isinstance(result.raw, scipy_optimize.OptimizeResult)


def test_auto_dispatch_jax_uses_optax(jax_maxcut_functional):
    pytest.importorskip("optax")
    from spacecore import OptaxResult

    # Only the dispatch target matters here; keep the run short.
    result = run_regularized_solver(
        jax_maxcut_functional.bind(EPS), max_iter=10, tol=1e-8, verbose=0
    )
    assert isinstance(result.raw, OptaxResult)


# ---------------------------------------------------------------------------
# (d) the bound functional carries eps and the normalization
# ---------------------------------------------------------------------------


def test_bound_eps_is_what_gets_solved(maxcut_functional, scipy_result):
    """A different bound eps gives a different solve, from the same base."""
    other = run_regularized_solver(
        maxcut_functional.bind(1e-1), method="scipy", tol=1e-10, verbose=0
    )
    assert other.converged is True
    assert other.final_loss != pytest.approx(scipy_result.final_loss, rel=1e-6)


def test_bound_normalization_reaches_the_solver(maxcut_functional):
    """``normalized`` travels with the functional rather than being re-passed."""
    from sdplab.regularization import EntropyReg

    entropy = RegularizedSDPDualFunctional(
        maxcut_functional.problem, EntropyReg(maxcut_functional.problem.dom)
    )
    bound = entropy.bind(EPS, normalized=True)
    assert bound.normalized is True
    # The gradient is the unit-trace recovery, so it differs from the free one.
    y = bound.domain.zeros()
    free = np.asarray(bound.base.grad(y, EPS, False))
    norm = np.asarray(bound.base.grad(y, EPS, True))
    assert not np.allclose(free, norm)


# ---------------------------------------------------------------------------
# (e) error handling
# ---------------------------------------------------------------------------


def test_unbound_functional_raises_type_error(maxcut_functional):
    with pytest.raises(TypeError, match="BoundDualFunctional"):
        run_regularized_solver(maxcut_functional, verbose=0)


def test_unknown_method_raises_value_error(maxcut_functional):
    with pytest.raises(ValueError, match="Unknown method"):
        run_regularized_solver(maxcut_functional.bind(EPS), method="bogus", verbose=0)


def test_plain_problem_raises_type_error(maxcut_sdp):
    with pytest.raises(TypeError, match="BoundDualFunctional"):
        run_regularized_solver(maxcut_sdp, verbose=0)


# ---------------------------------------------------------------------------
# (f) complex codomain is rejected on the scipy route
# ---------------------------------------------------------------------------


def test_scipy_route_rejects_complex_qot_codomain():
    """QOT's stacked Hermitian codomain is complex; minimize_scipy needs real.

    The realifying adapter that used to bridge this is gone, so the dispatcher
    refuses up front instead of failing inside SciPy. The optax route on a JAX
    backend is the supported path for a complex codomain.
    """
    pytest.importorskip("scipy")
    sdp, _gamma = generate_random_qot(2, 2, (0.7, 0.3), seed=0)
    F = RegularizedSDPDualFunctional(sdp, QuadraticReg(sdp.dom))

    assert getattr(sdp.cod, "field", "real") == "complex"
    with pytest.raises(NotImplementedError, match="real codomain"):
        run_regularized_solver(F.bind(EPS), method="scipy", verbose=0)
