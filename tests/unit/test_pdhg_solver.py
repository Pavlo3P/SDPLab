from __future__ import annotations

import numpy as np
import pytest
from spacecore import DenseLinOp, HermitianSpace

from sdplab.sdp import SDPDenseProblem
from sdplab.solvers import OptimizeResult
from sdplab.solvers.pdhg import run_pdhg_solver


def make_trace_sdp(ctx):
    dom = HermitianSpace(2, ctx=ctx)
    A = DenseLinOp(ctx.asarray([np.eye(2)]), dom, ctx=ctx)
    C = ctx.asarray([[1.0, 0.0], [0.0, 2.0]])
    b = ctx.asarray([1.0])
    return SDPDenseProblem(C, A, b, tau=1.0, ctx=ctx)


def test_pdhg_solver_returns_optimize_result(np_ctx):
    sdp = make_trace_sdp(np_ctx)

    result = run_pdhg_solver(
        sdp,
        tau=0.2,
        sigma=0.2,
        max_iter=5,
        tol=1e-12,
        verbose=0,
    )

    assert isinstance(result, OptimizeResult)
    assert result.primal is not None
    assert result.dual is not None
    assert result.num_iters == 5
    assert result.converged is False
    assert len(result.loss_history) == 5
    assert len(result.grad_norm_history) == 5
    assert len(result.step_times) == 5


def test_pdhg_solver_rejects_jit_mode_without_jax_backend(np_ctx):
    sdp = make_trace_sdp(np_ctx)

    with pytest.raises(ValueError, match="jit=True"):
        run_pdhg_solver(
            sdp,
            tau=0.2,
            sigma=0.2,
            max_iter=1,
            tol=1e-6,
            jit=True,
            verbose=0,
        )


def test_pdhg_solver_supports_jitted_step_on_jax_backend(jax_ctx):
    sdp = make_trace_sdp(jax_ctx)

    result = run_pdhg_solver(
        sdp,
        tau=0.2,
        sigma=0.2,
        max_iter=2,
        tol=1e-12,
        jit=True,
        verbose=0,
    )

    assert isinstance(result, OptimizeResult)
    assert result.primal is not None
    assert result.dual is not None
    assert result.num_iters == 2
