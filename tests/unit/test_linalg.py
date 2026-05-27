"""Tests for the matrix utilities in sdplab.linalg."""

from __future__ import annotations

import numpy as np

from sdplab.linalg import kron_all, log_trace_exp


def test_kron_all_returns_left_folded_kronecker_product(np_ctx):
    """Check a basic matrix utility against NumPy's kron."""
    a = np_ctx.asarray([[1.0, 2.0], [3.0, 4.0]])
    b = np_ctx.asarray([[0.0, 5.0], [6.0, 7.0]])
    c = np_ctx.asarray([[1.0]])

    assert np.allclose(kron_all(np_ctx, [a, b, c]), np.kron(np.kron(a, b), c))


def test_log_trace_exp_matches_eigenvalue_logsumexp(np_ctx):
    """Check log(trace(exp(X))) for a diagonal Hermitian matrix."""
    from spacecore import HermitianSpace

    space = HermitianSpace(2, ctx=np_ctx)
    x = np_ctx.asarray([[1.0, 0.0], [0.0, 2.0]])

    assert np.allclose(log_trace_exp(space, x), np.log(np.exp(1.0) + np.exp(2.0)))
