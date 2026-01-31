import numpy as np
import pytest

from qotlib.core.backend import BackendContext, NumpyOps
from qotlib.core.low_rank import LowRankMatrix, LowRankHermitianMatrixSpace


def _make_low_rank():
    ops = NumpyOps()
    ctx = BackendContext(ops=ops, dtype=np.float64)
    eigvals = ops.asarray([1.0, 2.0])
    eigvecs = ops.asarray([[1.0, 0.0], [0.0, 1.0]])
    return ctx, eigvals, eigvecs


def test_low_rank_matrix_properties_and_ops():
    """Validate low-rank matrix properties, conversions, and algebra."""
    ctx, eigvals, eigvecs = _make_low_rank()
    mat = LowRankMatrix(ctx=ctx, max_rank=2, eigvals=eigvals, eigvecs=eigvecs)

    assert mat.r == 2
    assert mat.dim == 2
    assert mat.shape == (2, 2)

    dense = mat.to_dense()
    assert np.allclose(dense, np.diag([1.0, 2.0]))

    vec = ctx.ops.asarray([1.0, -1.0])
    assert np.allclose(mat.matvec(vec), dense @ vec)

    assert np.isclose(mat.inner(mat), np.vdot(dense, dense))
    assert np.isclose(mat.l2_norm(), np.sqrt(np.vdot(eigvals, eigvals)))
    assert np.isclose(mat.trace(), eigvals.sum())

    mat_T = mat.T
    assert np.allclose(mat_T.to_dense(), dense.T.conj())

    mat_conj = mat.conj()
    assert np.allclose(mat_conj.to_dense(), dense.conj())

    children, aux = mat.tree_flatten()
    rebuilt = LowRankMatrix.tree_unflatten(aux, children)
    assert np.allclose(rebuilt.to_dense(), dense)


def test_low_rank_matrix_shape_validation():
    """Ensure low-rank matrix shape validation rejects bad eigvecs."""
    ctx, eigvals, _ = _make_low_rank()
    with pytest.raises(TypeError):
        LowRankMatrix(ctx=ctx, max_rank=2, eigvals=eigvals, eigvecs=ctx.ops.asarray([[1.0], [0.0]]))


def test_low_rank_hermitian_matrix_space_ops():
    """Check low-rank Hermitian space operations and conversions."""
    ctx, eigvals, eigvecs = _make_low_rank()
    space = LowRankHermitianMatrixSpace(ctx=ctx, shape=(2, 2), max_rank=2, n=2)

    x = LowRankMatrix(ctx=ctx, max_rank=2, eigvals=eigvals, eigvecs=eigvecs)
    y = LowRankMatrix(ctx=ctx, max_rank=2, eigvals=ctx.ops.asarray([0.5, 1.5]), eigvecs=eigvecs)

    zeros = space.zeros()
    assert np.allclose(zeros.to_dense(), np.zeros((2, 2)))

    added = space.add(x, y)
    assert np.allclose(added, x.to_dense() + y.to_dense())

    scaled = space.scale(2.0, x)
    assert np.allclose(scaled.to_dense(), 2.0 * x.to_dense())

    assert np.isclose(space.inner(x, y), x.inner(y))

    eigvals_out, eigvecs_out = space.eigh(x)
    assert np.allclose(eigvals_out, eigvals)
    assert np.allclose(eigvecs_out, eigvecs)

    flat = space.flatten(x)
    assert flat.shape == (4,)

    unflat = space.unflatten(flat)
    assert isinstance(unflat, LowRankMatrix)
    assert unflat.shape == (2, 2)

    with pytest.raises(TypeError):
        space.scale(1j, x)
