from __future__ import annotations

import numpy as np
from spacecore import DenseLinOp, HermitianSpace

from sdplab.sdp import SDPDenseProblem


def make_trace_sdp(ctx):
    """Build a two-by-two SDP whose only equality is trace(X) = 1."""
    dom = HermitianSpace(2, ctx=ctx)
    A_data = ctx.asarray([np.eye(2)])
    A = DenseLinOp(A_data, dom, ctx=ctx)
    C = ctx.asarray([[1.0, 0.0], [0.0, 2.0]])
    b = ctx.asarray([1.0])
    return SDPDenseProblem(C, A, b, tau=1.0, ctx=ctx)


def test_dense_sdp_primal_and_dual_objectives(np_ctx):
    """Check objective values for simple primal and dual variables."""
    sdp = make_trace_sdp(np_ctx)
    primal = sdp.primal_from_array(np_ctx.asarray([[0.25, 0.0], [0.0, 0.75]]))
    dual = sdp.dual_from_array(np_ctx.asarray([3.0]))

    assert np.allclose(sdp.primal_objective(primal), 1.75)
    assert np.allclose(sdp.dual_objective(dual), 3.0)


def test_dense_sdp_linear_maps_and_dual_slack_spectrum(np_ctx):
    """Check A, A^T, and eigendecomposition of A^T y - C."""
    sdp = make_trace_sdp(np_ctx)
    primal = sdp.primal_from_array(np_ctx.asarray([[0.25, 0.0], [0.0, 0.75]]))
    dual = sdp.dual_from_array(np_ctx.asarray([3.0]))

    assert np.allclose(sdp.A_apply(primal).y, [1.0])
    assert np.allclose(sdp.AT_apply(dual).X, 3.0 * np.eye(2))
    eigvals, _ = sdp.dual_constr_eig_decomp(dual)
    assert np.allclose(eigvals, [1.0, 2.0])


def test_dense_sdp_recovers_primal_from_eigendecomposition(np_ctx):
    """Verify eigenpair recovery returns V diag(lambda) V^T."""
    sdp = make_trace_sdp(np_ctx)
    eigvals = np_ctx.asarray([0.25, 0.75])
    eigvecs = np_ctx.asarray([[1.0, 0.0], [0.0, 1.0]])

    primal = sdp.primal_from_eigendecomp(eigvals, eigvecs)
    assert np.allclose(primal.X, [[0.25, 0.0], [0.0, 0.75]])
