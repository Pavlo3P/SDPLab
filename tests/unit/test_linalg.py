from __future__ import annotations

import numpy as np
from spacecore import VectorSpace

from sdplab.linalg import kron_all, log_trace_exp, power_method


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


def test_power_method_returns_dominant_direction(np_ctx):
    """Check power iteration converges to the dominant eigenvector."""
    space = VectorSpace((2,), ctx=np_ctx)
    matrix = np_ctx.asarray([[3.0, 0.0], [0.0, 1.0]])

    vec = power_method(space, lambda x: matrix @ x, np_ctx.asarray([1.0, 1.0]), n_iter=20)
    assert np.allclose(np.abs(vec), [1.0, 0.0], atol=1e-3)
