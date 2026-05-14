from __future__ import annotations

import numpy as np
from spacecore import DenseLinOp, HermitianSpace

from sdplab.regularization import EntropyReg, QuadraticReg, SDPRegularized
from sdplab.sdp import SDPDenseProblem


def make_trace_sdp(ctx):
    """Build a small diagonal SDP for regularization checks."""
    dom = HermitianSpace(2, ctx=ctx)
    A = DenseLinOp(ctx.asarray([np.eye(2)]), dom, ctx=ctx)
    C = ctx.asarray([[1.0, 0.0], [0.0, 2.0]])
    b = ctx.asarray([1.0])
    return SDPDenseProblem(C, A, b, tau=1.0, ctx=ctx)


def test_regularized_primal_objective_adds_penalty(np_ctx):
    """Ensure P_eps(X) equals the linear objective plus eps phi(X)."""
    sdp = make_trace_sdp(np_ctx)
    reg_sdp = SDPRegularized(sdp, QuadraticReg(0.2, ctx=np_ctx))
    primal = sdp.primal_from_array(np_ctx.asarray([[0.25, 0.0], [0.0, 0.75]]))

    assert np.allclose(reg_sdp.primal_objective_reg(primal), 1.8125)


def test_regularized_dual_objective_subtracts_legendre_term(np_ctx):
    """Ensure D_eps(y) equals the dual objective minus eps phi*(slack / eps)."""
    sdp = make_trace_sdp(np_ctx)
    reg_sdp = SDPRegularized(sdp, QuadraticReg(1.0, ctx=np_ctx))
    dual = sdp.dual_from_array(np_ctx.asarray([3.0]))

    assert np.allclose(reg_sdp.dual_objective_reg(dual), 0.5)


def test_regularized_primal_from_dual_normalizes_entropy_weights(np_ctx):
    """Check entropy recovery creates normalized Gibbs weights."""
    sdp = make_trace_sdp(np_ctx)
    reg_sdp = SDPRegularized(sdp, EntropyReg(1.0, ctx=np_ctx))
    dual = sdp.dual_from_array(np_ctx.asarray([3.0]))

    primal = reg_sdp.primal_from_dual(dual, normalized=True)
    expected = np.diag([np.exp(2.0), np.exp(1.0)])
    expected = expected / np.trace(expected)
    assert np.allclose(primal.X, expected)
