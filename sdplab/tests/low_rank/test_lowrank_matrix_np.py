import numpy as np
import pytest


def _make_row_orthonormal_eigvecs_np(rng: np.random.Generator, r: int, n: int) -> np.ndarray:
    """
    Return V with shape (n, r) with orthonormal columns: V^H V = I_r.
    Construct via QR on an (n, r) random complex matrix.
    """
    A = rng.standard_normal((n, r)) + 1j * rng.standard_normal((n, r))
    Q, _ = np.linalg.qr(A)  # Q: (n, r)
    V = np.conj(Q)          # (n, r) (matches your JAX helper style)
    I = V.T.conj() @ V
    assert np.allclose(I, np.eye(r), atol=1e-10)
    return V


def _dense_from_factors_row(eigvals: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    """
    Dense materialization for column-eigvec convention (n, r):
      X = V diag(s) V^H
    """
    return (eigvecs * eigvals) @ eigvecs.T.conj()


@pytest.mark.parametrize("n,r", [(6, 2), (9, 4)])
def test_numpy_matvec_roundtrip(n: int, r: int):
    """Validate matvec matches dense materialization for NumPy."""
    from qotlib.core.backend import BackendContext
    from qotlib.core.backend.numpy import NumpyOps
    from qotlib.core.low_rank import LowRankMatrix

    rng = np.random.default_rng(0)
    ctx = BackendContext(NumpyOps())

    V = _make_row_orthonormal_eigvecs_np(rng, r=r, n=n)
    s = rng.standard_normal((r,)).astype(np.float64)
    x = rng.standard_normal((n,)) + 1j * rng.standard_normal((n,))

    X = LowRankMatrix(ctx=ctx, max_rank=r, eigvals=s, eigvecs=V)

    y = X.matvec(x)

    dense = _dense_from_factors_row(s, V)
    y_ref = dense @ x

    assert y.shape == (n,)
    assert np.allclose(y, y_ref, atol=1e-10)


@pytest.mark.parametrize("n,r", [(5, 1), (8, 3)])
def test_numpy_to_dense(n: int, r: int):
    """Ensure to_dense matches explicit factor reconstruction."""
    from qotlib.core.backend import BackendContext
    from qotlib.core.backend.numpy import NumpyOps
    from qotlib.core.low_rank import LowRankMatrix

    rng = np.random.default_rng(10)
    ctx = BackendContext(NumpyOps())

    V = _make_row_orthonormal_eigvecs_np(rng, r=r, n=n)
    s = rng.standard_normal((r,)).astype(np.float64)

    X = LowRankMatrix(ctx=ctx, max_rank=r, eigvals=s, eigvecs=V)

    dense = X.to_dense()
    dense_ref = _dense_from_factors_row(s, V)

    assert dense.shape == (n, n)
    assert np.allclose(dense, dense_ref, atol=1e-10)


@pytest.mark.parametrize("n,rx,ry", [(8, 2, 3), (10, 4, 4)])
def test_numpy_inner_matches_trace(n: int, rx: int, ry: int):
    """Confirm low-rank inner matches dense trace identity."""
    from qotlib.core.backend import BackendContext
    from qotlib.core.backend.numpy import NumpyOps
    from qotlib.core.low_rank import LowRankMatrix

    rng = np.random.default_rng(20)
    ctx = BackendContext(NumpyOps())

    Vx = _make_row_orthonormal_eigvecs_np(rng, r=rx, n=n)
    Vy = _make_row_orthonormal_eigvecs_np(rng, r=ry, n=n)
    sx = rng.standard_normal((rx,)).astype(np.float64)
    sy = rng.standard_normal((ry,)).astype(np.float64)

    X = LowRankMatrix(ctx=ctx, max_rank=rx, eigvals=sx, eigvecs=Vx)
    Y = LowRankMatrix(ctx=ctx, max_rank=ry, eigvals=sy, eigvecs=Vy)

    got = X.inner(Y)

    dense_X = _dense_from_factors_row(sx, Vx)
    dense_Y = _dense_from_factors_row(sy, Vy)
    ref = np.vdot(dense_X, dense_Y)  # tr(X^H Y) == vdot(X, Y)

    assert np.allclose(got, ref, atol=1e-10)
