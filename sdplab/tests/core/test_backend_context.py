import numpy as np
import pytest

from qotlib.core.backend import BackendContext, NumpyOps


def test_backend_context_assert_dense_and_asarray():
    """Validate dtype sanitation plus dense assertions for BackendContext."""
    ops = NumpyOps()
    ctx = BackendContext(ops=ops, dtype=np.float64)

    arr = ctx.asarray([1, 2, 3])
    assert ops.is_dense(arr)
    assert arr.dtype == np.dtype(np.float64)
    assert ctx.assert_dense(arr) is arr

    with pytest.raises(TypeError):
        ctx.assert_dense([1, 2, 3])


def test_backend_context_sparse_checks():
    """Ensure sparse checks respect allow_sparse and type validation."""
    ops = NumpyOps()
    sp = pytest.importorskip("scipy").sparse
    sparse = sp.csr_matrix(np.eye(2))

    ctx = BackendContext(ops=ops, allow_sparse=False)
    with pytest.raises(TypeError):
        ctx.assert_sparse(sparse)

    ctx_allow = BackendContext(ops=ops, allow_sparse=True)
    assert ctx_allow.assert_sparse(sparse) is sparse

    with pytest.raises(TypeError):
        ctx_allow.assert_sparse(np.eye(2))
