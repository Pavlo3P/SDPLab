import numpy as np

def test_numpy_sparse_matmul(np_ops):
    """Validate sparse matmul using SciPy sparse matrices."""
    sp = np_ops._require_scipy()

    A = sp.sparse.csr_matrix(np.array([[1., 0.], [2., 3.]]))
    b = np.array([4., 5.])
    y = np_ops.sparse_matmul(A, b)
    assert np.allclose(y, A @ b)
