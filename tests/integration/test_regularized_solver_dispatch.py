from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from spacecore import DenseLinOp, HermitianSpace

from sdplab.regularization import QuadraticReg, SDPRegularized
from sdplab.sdp import SDPDenseProblem
from sdplab.solvers import ConvergenceInfo
from _regularized import run_regularized_solver


def make_trace_regularized_sdp(ctx):
    """Build the shared one-constraint regularized SDP fixture."""
    dom = HermitianSpace(2, ctx=ctx)
    A = DenseLinOp(ctx.asarray([np.eye(2)]), dom, ctx=ctx)
    C = ctx.asarray([[1.0, 0.0], [0.0, 2.0]])
    b = ctx.asarray([1.0])
    sdp = SDPDenseProblem(C, A, b, tau=1.0, ctx=ctx)
    return SDPRegularized(sdp, QuadraticReg(1.0, ctx=ctx))


@dataclass
class FakeSciPyResult:
    """Minimal scipy.optimize.minimize result shape used by the dispatcher."""

    x: np.ndarray
    success: bool = True


def test_numpy_regularized_solver_launches_scipy_minimize(np_ctx, monkeypatch):
    """Integration check: NumPy backend dispatches to scipy.optimize.minimize."""
    scipy = pytest.importorskip("scipy.optimize")
    reg_sdp = make_trace_regularized_sdp(np_ctx)
    calls = []

    def fake_minimize(fun, x0, **kwargs):
        calls.append({"x0": x0.copy(), "kwargs": kwargs})
        fun(x0)
        return FakeSciPyResult(x=x0)

    monkeypatch.setattr(scipy, "minimize", fake_minimize)
    info = run_regularized_solver(reg_sdp, max_iter=3, tol=1e-5, method="BFGS")

    assert len(calls) == 1
    assert calls[0]["kwargs"]["method"] == "BFGS"
    assert calls[0]["kwargs"]["options"] == {"maxiter": 3}
    assert np.allclose(info.dual.y, [0.0])
    assert info.tol_reached is True


def test_jax_regularized_solver_launches_optax(jax_ctx, monkeypatch):
    """Integration check: JAX backend dispatches to run_optax_solver."""
    optax = pytest.importorskip("optax")
    reg_sdp = make_trace_regularized_sdp(jax_ctx)
    opt = optax.sgd(0.1)
    calls = []

    def fake_run_optax_solver(problem, init_dual, **kwargs):
        calls.append({"problem": problem, "init_dual": init_dual, "kwargs": kwargs})
        return ConvergenceInfo(dual=init_dual, tol_reached=True)

    monkeypatch.setattr("_regularized.run_optax_solver", fake_run_optax_solver)
    info = run_regularized_solver(reg_sdp, opt=opt, max_iter=4, tol=1e-4)

    assert len(calls) == 1
    assert calls[0]["problem"] is reg_sdp
    assert calls[0]["kwargs"]["opt"] is opt
    assert calls[0]["kwargs"]["max_iter"] == 4
    assert calls[0]["kwargs"]["tol"] == 1e-4
    assert info.tol_reached is True
