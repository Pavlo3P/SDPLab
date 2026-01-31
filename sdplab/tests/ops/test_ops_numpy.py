import numpy as np

def test_numpy_ops_creation(np_ops):
    """Check array creation helpers (asarray, zeros, full, eye)."""
    x = np_ops.asarray([[1, 2], [3, 4]], dtype=np.float32, order="C")
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float32

    z = np_ops.zeros((2, 3), dtype=np.float64, order="F")
    assert z.shape == (2, 3)
    assert z.dtype == np.float64
    assert np.all(z == 0.)

    f = np_ops.full(5, 7, dtype=np.int32)
    assert f.shape == (5,)
    assert np.all(f == 7)

    e = np_ops.eye(3, dtype=np.float32)
    assert e.shape == (3, 3)
    assert np.allclose(e, np.eye(3, dtype=np.float32))

def test_numpy_ops_reductions(np_ops):
    """Validate sum/prod reductions with axes and keepdims."""
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    s = np_ops.sum(a, axis=(1, 2), keepdims=True)
    assert s.shape == (2, 1, 1)
    assert np.allclose(s, np.sum(a, axis=(1, 2), keepdims=True))

    p = np_ops.prod(a + 1, axis=2)
    assert np.allclose(p, np.prod(a + 1, axis=2))

def test_numpy_ops_linalg(np_ops):
    """Cover matmul and vdot for NumPy backend ops."""
    a = np.array([[1., 2.], [3., 4.]], dtype=np.float64)
    b = np.array([[5.], [6.]], dtype=np.float64)
    m = np_ops.matmul(a, b)
    assert np.allclose(m, a @ b)

    v = np_ops.vdot(np.array([1+2j, 3+4j]), np.array([5+6j, 7+8j]))
    assert np.allclose(v, np.vdot(np.array([1+2j, 3+4j]), np.array([5+6j, 7+8j])))

def test_numpy_ops_eigh_stacked(np_ops):
    """Ensure stacked eigendecomposition returns expected shapes."""
    # stacked Hermitian matrices: (..., M, M)
    x = np.zeros((3, 2, 2), dtype=np.float64)
    x[:, 0, 0] = [1.0, 2.0, 3.0]
    x[:, 1, 1] = [4.0, 5.0, 6.0]
    w, v = np_ops.eigh(x, UPLO="L")
    assert w.shape == (3, 2)
    assert v.shape == (3, 2, 2)


def test_numpy_ops_loops(np_ops):
    """Validate NumPy loop helpers mirror jax.lax semantics."""
    init = np_ops.asarray(0, dtype=np.int64)

    def body_fun(i, val):
        return val + i

    out = np_ops.fori_loop(0, 5, body_fun, init)
    assert np.allclose(out, np.array(10, dtype=np.int64))

    def cond_fun(val):
        return val < 10

    def while_body(val):
        return val + 3

    out_while = np_ops.while_loop(cond_fun, while_body, init)
    assert np.allclose(out_while, np.array(12, dtype=np.int64))
