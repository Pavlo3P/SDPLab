import numpy as np
import pytest

from qotlib.core.backend import BackendContext, NumpyOps
from qotlib.core.space import DenseVectorSpace, DenseHermitianMatrixSpace


def test_dense_vector_space_ops():
    """Validate vector space operations, checks, and flattening."""
    ops = NumpyOps()
    ctx = BackendContext(ops=ops, dtype=np.float64)
    space = DenseVectorSpace(ctx=ctx, shape=(3,), n=3)

    zeros = space.zeros()
    assert zeros.shape == (3,)
    assert np.allclose(zeros, 0.0)

    x = ops.asarray([1.0, 2.0, 3.0])
    y = ops.asarray([0.5, -1.0, 2.0])

    assert np.allclose(space.add(x, y), x + y)
    assert np.allclose(space.scale(2.0, x), 2.0 * x)
    assert np.isclose(space.inner(x, y), np.vdot(x, y))
    assert np.isclose(space.norm(x), np.sqrt(np.real(np.vdot(x, x))))

    flat = space.flatten(x)
    assert np.allclose(flat, x)

    unflat = space.unflatten(flat)
    assert np.allclose(unflat, x)

    with pytest.raises(TypeError):
        space.check_member(ops.asarray([[1.0, 2.0, 3.0]]))

    with pytest.raises(TypeError):
        space.eigh(x)


def test_dense_hermitian_matrix_space_ops():
    """Validate Hermitian matrix space operations and invariants."""
    ops = NumpyOps()
    ctx = BackendContext(ops=ops, dtype=np.float64)
    space = DenseHermitianMatrixSpace(ctx=ctx, n=2, atol=1e-8)

    X = ops.asarray([[1.0, 2.0], [2.0, 3.0]])
    assert space.is_hermitian(X)

    sym = space.symmetrize(ops.asarray([[1.0, 2.0], [3.0, 4.0]]))
    assert space.is_hermitian(sym)

    zeros = space.zeros()
    assert zeros.shape == (2, 2)
    assert np.allclose(zeros, 0.0)

    Y = ops.asarray([[0.5, 1.0], [1.0, 0.5]])
    assert np.allclose(space.add(X, Y), X + Y)
    assert np.allclose(space.scale(2.0, X), 2.0 * X)
    assert np.isclose(space.inner(X, Y), np.vdot(X, Y))

    w, U = space.eigh(X)
    assert np.allclose(X, U @ np.diag(w) @ U.T.conj())

    flat = space.flatten(X)
    assert flat.shape == (4,)
    unflat = space.unflatten(flat)
    assert space.is_hermitian(unflat)

    with pytest.raises(TypeError):
        space.check_member(ops.asarray([[1.0, 2.0], [0.0, 1.0]]))

    with pytest.raises(TypeError):
        space.scale(1j, X)
