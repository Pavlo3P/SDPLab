import numpy as np
import pytest

from qotlib.core.backend import NumpyOps


@pytest.fixture()
def ops():
    return NumpyOps()


def test_numpy_ops_dense_and_constants(ops):
    """Check dense/sparse detection and core constants for NumPy ops."""
    arr = np.array([1.0, 2.0])
    assert ops.is_dense(arr)
    assert not ops.is_sparse(arr)
    assert np.isinf(ops.inf)
    assert np.isnan(ops.nan)
    assert np.isclose(ops.pi, np.pi)
    assert np.isclose(ops.e, np.e)
    assert ops.eps > 0


def test_numpy_ops_array_creation(ops):
    """Verify array creation helpers produce expected shapes and values."""
    arr = ops.asarray([1, 2, 3], dtype=np.float64)
    assert arr.dtype == np.dtype(np.float64)

    empty = ops.empty((2, 3))
    assert empty.shape == (2, 3)

    zeros = ops.zeros((2, 2), dtype=np.float64)
    assert np.allclose(zeros, 0.0)

    full = ops.full((2, 2), fill_value=7)
    assert np.allclose(full, 7)

    eye = ops.eye(2)
    assert np.allclose(eye, np.eye(2))


def test_numpy_ops_shape_and_stack(ops):
    """Exercise reshape, transpose, ravel, and stacking semantics."""
    arr = ops.asarray([[1, 2], [3, 4]])
    raveled = ops.ravel(arr)
    assert raveled.shape == (4,)

    reshaped = ops.reshape(raveled, (2, 2))
    assert np.allclose(reshaped, arr)

    transposed = ops.transpose(arr)
    assert np.allclose(transposed, arr.T)

    stacked = ops.stack([arr, arr], axis=0)
    assert stacked.shape == (2, 2, 2)


def test_numpy_ops_complex_helpers(ops):
    """Validate conjugate/real/imag helpers for complex arrays."""
    z = ops.asarray([1 + 2j, 3 - 4j])
    assert np.allclose(ops.conj(z), np.conj(z))
    assert np.allclose(ops.real(z), np.real(z))
    assert np.allclose(ops.imag(z), np.imag(z))


def test_numpy_ops_numeric_reductions(ops):
    """Check reductions, ordering ops, and basic elementwise helpers."""
    arr = ops.asarray([1.0, -2.0, 3.0])
    assert np.allclose(ops.abs(arr), np.abs(arr))
    assert np.allclose(ops.sign(arr), np.sign(arr))
    assert np.allclose(ops.sqrt(ops.abs(arr)), np.sqrt(np.abs(arr)))

    mat = ops.asarray([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(ops.sum(mat), np.sum(mat))
    assert np.allclose(ops.prod(mat), np.prod(mat))
    assert np.allclose(ops.trace(mat), np.trace(mat))

    assert np.allclose(ops.argsort(arr), np.argsort(arr))
    assert np.allclose(ops.sort(arr), np.sort(arr))
    assert np.allclose(ops.argmin(arr), np.argmin(arr))
    assert np.allclose(ops.argmax(arr), np.argmax(arr))


def test_numpy_ops_linear_algebra(ops):
    """Validate linear algebra ops such as vdot, matmul, and eigendecomp."""
    a = ops.asarray([[1.0, 2.0], [3.0, 4.0]])
    b = ops.asarray([[5.0, 6.0], [7.0, 8.0]])
    v = ops.asarray([1.0, -1.0])

    assert np.allclose(ops.vdot(v, v), np.vdot(v, v))
    assert np.allclose(ops.matmul(a, b), a @ b)
    assert np.allclose(ops.kron(a, b), np.kron(a, b))
    assert np.allclose(ops.einsum("ij,j->i", a, v), np.einsum("ij,j->i", a, v))

    herm = ops.asarray([[2.0, 0.0], [0.0, 3.0]])
    w, U = ops.eigh(herm)
    assert np.allclose(herm, U @ np.diag(w) @ U.T.conj())


def test_numpy_ops_exponentials_and_comparisons(ops):
    """Ensure exp/log/maximum/minimum/where behave as NumPy does."""
    arr = ops.asarray([0.0, 1.0])
    assert np.allclose(ops.exp(arr), np.exp(arr))
    assert np.allclose(ops.log(ops.exp(arr)), np.log(np.exp(arr)))

    x = ops.asarray([1.0, 3.0])
    y = ops.asarray([2.0, 2.0])
    assert np.allclose(ops.maximum(x, y), np.maximum(x, y))
    assert np.allclose(ops.minimum(x, y), np.minimum(x, y))

    condition = ops.asarray([True, False])
    assert np.allclose(ops.where(condition, x, y), np.where(condition, x, y))


def test_numpy_ops_sparse_and_logsumexp(ops):
    """Cover sparse matmul and logsumexp with SciPy when available."""
    sp = pytest.importorskip("scipy")
    sparse = sp.sparse.csr_matrix(np.eye(2))
    dense = ops.asarray([1.0, 2.0])

    assert ops.is_sparse(sparse)
    assert np.allclose(ops.sparse_matmul(sparse, dense), sparse @ dense)

    logsum = ops.logsumexp(ops.asarray([[0.0, 1.0]]), axis=1)
    expected = sp.special.logsumexp(np.array([[0.0, 1.0]]), axis=1)
    assert np.allclose(logsum, expected)
