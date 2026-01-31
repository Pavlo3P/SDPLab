import numpy as np
import pytest


def _jax_import():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _make_row_orthonormal_eigvecs_jax(key, r: int, n: int, jax, jnp):
    """
    Return V with shape (r, n) and orthonormal rows: V^H V = I_r.
    Construct Q with orthonormal columns (n, r) and set V = Q^H.
    """
    key1, key2 = jax.random.split(key)
    A = jax.random.normal(key1, (n, r)) + 1j * jax.random.normal(key2, (n, r))
    Q, _ = jnp.linalg.qr(A)  # Q: (n, r)
    V = jnp.conj(Q)  # (n, r)
    # Check outside jit
    I = V.T.conj() @ V
    assert np.allclose(np.array(I), np.eye(r), atol=1e-6)
    return V


def _dense_from_factors_row(eigvals, eigvecs):
    return (eigvecs * eigvals) @ eigvecs.T.conj()


@pytest.mark.parametrize("n,r", [(6, 2), (9, 4)])
def test_jit_matvec_pytree_roundtrip(n: int, r: int):
    """Validate JIT matvec for LowRankMatrix PyTree arguments."""
    from qotlib.core.backend import BackendContext
    from qotlib.core.backend.jax import JaxOps
    from qotlib.core.low_rank import LowRankMatrix
    jax, jnp = _jax_import()
    
    ctx = BackendContext(JaxOps())
    key = jax.random.PRNGKey(0)
    V = _make_row_orthonormal_eigvecs_jax(key, r=r, n=n, jax=jax, jnp=jnp)
    s = jnp.asarray(jax.random.normal(jax.random.PRNGKey(1), (r,)), dtype=jnp.float64)
    x = jax.random.normal(jax.random.PRNGKey(2), (n,)) + 1j * jax.random.normal(jax.random.PRNGKey(3), (n,))

    X = LowRankMatrix(ctx=ctx, max_rank=r, eigvals=s, eigvecs=V)

    # jit with LowRankMatrix as a PyTree argument
    @jax.jit
    def f(A: LowRankMatrix, v):
        return A.matvec(v)

    y = f(X, x)

    # reference using dense materialization in JAX
    dense = _dense_from_factors_row(s, V)
    y_ref = dense @ x

    assert np.allclose(np.array(y), np.array(y_ref), atol=1e-5)


@pytest.mark.parametrize("n,r", [(5, 1), (8, 3)])
def test_jit_to_dense_pytree(n: int, r: int):
    """Ensure JIT to_dense matches explicit factor reconstruction."""
    from qotlib.core.backend import BackendContext
    from qotlib.core.backend.jax import JaxOps
    from qotlib.core.low_rank import LowRankMatrix
    jax, jnp = _jax_import()
    
    ctx = BackendContext(JaxOps())
    key = jax.random.PRNGKey(10)
    V = _make_row_orthonormal_eigvecs_jax(key, r=r, n=n, jax=jax, jnp=jnp)
    s = jnp.asarray(jax.random.normal(jax.random.PRNGKey(11), (r,)), dtype=jnp.float64)

    X = LowRankMatrix(ctx=ctx, max_rank=r, eigvals=s, eigvecs=V)

    @jax.jit
    def g(A: LowRankMatrix):
        return A.to_dense()

    dense = g(X)
    dense_ref = _dense_from_factors_row(s, V)

    assert dense.shape == (n, n)
    assert np.allclose(np.array(dense), np.array(dense_ref), atol=1e-5)


@pytest.mark.parametrize("n,rx,ry", [(8, 2, 3), (10, 4, 4)])
def test_jit_inner_matches_trace(n: int, rx: int, ry: int):
    """Confirm JIT inner matches dense trace identity."""
    from qotlib.core.backend import BackendContext
    from qotlib.core.backend.jax import JaxOps
    from qotlib.core.low_rank import LowRankMatrix
    jax, jnp = _jax_import()

    ctx = BackendContext(JaxOps())
    Vx = _make_row_orthonormal_eigvecs_jax(jax.random.PRNGKey(20), r=rx, n=n, jax=jax, jnp=jnp)
    Vy = _make_row_orthonormal_eigvecs_jax(jax.random.PRNGKey(21), r=ry, n=n, jax=jax, jnp=jnp)
    sx = jnp.asarray(jax.random.normal(jax.random.PRNGKey(22), (rx,)), dtype=jnp.float64)
    sy = jnp.asarray(jax.random.normal(jax.random.PRNGKey(23), (ry,)), dtype=jnp.float64)

    X = LowRankMatrix(ctx=ctx, max_rank=rx, eigvals=sx, eigvecs=Vx)
    Y = LowRankMatrix(ctx=ctx, max_rank=ry, eigvals=sy, eigvecs=Vy)

    @jax.jit
    def h(A: LowRankMatrix, B: LowRankMatrix):
        return A.inner(B)

    got = h(X, Y)

    dense_X = _dense_from_factors_row(sx, Vx)
    dense_Y = _dense_from_factors_row(sy, Vy)
    ref = jnp.vdot(dense_X, dense_Y)

    assert np.allclose(np.array(got), np.array(ref), atol=1e-5)
