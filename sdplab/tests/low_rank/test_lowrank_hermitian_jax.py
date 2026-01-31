from __future__ import annotations

import pytest


def _jax_import():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    return jax, jnp

from qotlib.core.backend import BackendContext
from qotlib.core.backend.jax import JaxOps
from qotlib.core.low_rank import LowRankHermitianMatrixSpace, LowRankMatrix


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def jax_jnp():
    return _jax_import()


@pytest.fixture
def ctx_jax_no_checks(jax_jnp):
    _, jnp = jax_jnp
    return BackendContext(ops=JaxOps(), dtype=jnp.complex128, enable_checks=False)


@pytest.fixture
def space(ctx_jax_no_checks):
    n, r = 12, 4
    return LowRankHermitianMatrixSpace(ctx=ctx_jax_no_checks, shape=(n, n), n=n, max_rank=r)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _col_orthonormal(jax, jnp, n: int, r: int, seed: int, dtype):
    key = jax.random.PRNGKey(seed)
    a = jax.random.normal(key, (n, r), dtype=jnp.float64)
    b = jax.random.normal(jax.random.fold_in(key, 1), (n, r), dtype=jnp.float64)
    A = (a + 1j * b).astype(dtype)
    Q, _ = jnp.linalg.qr(A)
    return Q  # (n, r), columns orthonormal


def make_lowrank(ctx: BackendContext, jax, jnp, n: int, r: int, seed: int) -> LowRankMatrix:
    V = _col_orthonormal(jax, jnp, n=n, r=r, seed=seed, dtype=ctx.dtype)
    key = jax.random.PRNGKey(seed + 123)
    s = jax.random.normal(key, (r,), dtype=V.real.dtype)  # real eigenvalues
    return LowRankMatrix(ctx=ctx, max_rank=r, eigvals=s, eigvecs=V)


def make_dense_hermitian(jax, jnp, n: int, seed: int, dtype):
    key = jax.random.PRNGKey(seed)
    a = jax.random.normal(key, (n, n), dtype=jnp.float64)
    b = jax.random.normal(jax.random.fold_in(key, 1), (n, n), dtype=jnp.float64)
    A = (a + 1j * b).astype(dtype)
    return (A + jnp.conj(A).T) / 2.0


# -----------------------------------------------------------------------------
# JIT: each method of LowRankHermitianMatrixSpace
# -----------------------------------------------------------------------------
def test_jit_zeros(space, jax_jnp):
    """Ensure zeros is JIT-compatible and returns expected factors."""
    jax, jnp = jax_jnp
    S = space

    eager = S.zeros()
    eager_children = (eager.eigvals, eager.eigvecs)

    @jax.jit
    def f():
        z = S.zeros()
        # Return children to make the test robust even if returning LowRankMatrix is problematic.
        return z.eigvals, z.eigvecs

    w, V = f()
    assert jnp.allclose(w, eager_children[0])
    assert jnp.allclose(V, eager_children[1])


def test_jit_add(space, jax_jnp):
    """Check JIT add matches eager and preserves Hermitian form."""
    jax, jnp = jax_jnp
    S = space
    x = make_lowrank(S.ctx, jax, jnp, S.n, S.max_rank, seed=10)
    y = make_lowrank(S.ctx, jax, jnp, S.n, S.max_rank, seed=11)

    eager = S.add(x, y)

    @jax.jit
    def f(a, b):
        return S.add(a, b)

    out = f(x, y)
    assert jnp.allclose(out, eager)
    assert jnp.allclose(out, jnp.conj(out).T)


def test_jit_scale(space, jax_jnp):
    """Validate JIT scale matches eager results."""
    jax, jnp = jax_jnp
    S = space
    x = make_lowrank(S.ctx, jax, jnp, S.n, S.max_rank, seed=12)
    a = 2.5

    eager = S.scale(a, x).to_dense()

    @jax.jit
    def f(alpha, A):
        # Return dense to avoid depending on returning LowRankMatrix from jit.
        return S.scale(alpha, A).to_dense()

    out = f(a, x)
    assert jnp.allclose(out, eager)


def test_jit_inner(space, jax_jnp):
    """Ensure inner product works under JIT."""
    jax, jnp = jax_jnp
    S = space
    x = make_lowrank(S.ctx, jax, jnp, S.n, S.max_rank, seed=20)
    y = make_lowrank(S.ctx, jax, jnp, S.n, S.max_rank, seed=21)

    eager = S.inner(x, y)

    @jax.jit
    def f(a, b):
        return S.inner(a, b)

    out = f(x, y)
    assert jnp.allclose(out, eager)


def test_jit_eigh(space, jax_jnp):
    """Ensure eigendecomposition works under JIT."""
    jax, jnp = jax_jnp
    S = space
    x = make_lowrank(S.ctx, jax, jnp, S.n, S.max_rank, seed=22)

    ew, eV = S.eigh(x)

    @jax.jit
    def f(a):
        return S.eigh(a)

    w, V = f(x)
    assert jnp.allclose(w, ew)
    assert jnp.allclose(V, eV)


def test_jit_flatten(space, jax_jnp):
    """Ensure flatten works under JIT and returns correct shape."""
    jax, jnp = jax_jnp
    S = space
    x = make_lowrank(S.ctx, jax, jnp, S.n, S.max_rank, seed=30)

    eager = S.flatten(x)

    @jax.jit
    def f(a):
        return S.flatten(a)

    out = f(x)
    assert jnp.allclose(out, eager)
    assert tuple(out.shape) == (S.n * S.n,)


def test_jit_unflatten(space, jax_jnp):
    """Ensure unflatten works under JIT and returns Hermitian matrices."""
    jax, jnp = jax_jnp
    S = space
    H = make_dense_hermitian(jax, jnp, S.n, seed=0, dtype=S.ctx.dtype)
    v = jnp.ravel(H)

    eager = S.unflatten(v).to_dense()

    @jax.jit
    def f(vec):
        return S.unflatten(vec).to_dense()

    out = f(v)
    assert jnp.allclose(out, eager)
    assert jnp.allclose(out, jnp.conj(out).T)


# -----------------------------------------------------------------------------
# Optional: also check LowRankMatrix methods jittability (useful in practice)
# -----------------------------------------------------------------------------
def test_jit_lowrankmatrix_methods(space, jax_jnp):
    """Check LowRankMatrix core methods under JIT."""
    jax, jnp = jax_jnp
    S = space
    x = make_lowrank(S.ctx, jax, jnp, S.n, S.max_rank, seed=100)
    y = make_lowrank(S.ctx, jax, jnp, S.n, S.max_rank, seed=101)
    v = jax.random.normal(jax.random.PRNGKey(7), (S.n,), dtype=jnp.float64) + 0j

    eager = (
        x.to_dense(),
        x.matvec(v),
        x.inner(y),
        x.l2_norm(),
        x.trace(),
        x.conj().to_dense(),
        x.T.to_dense(),
    )

    @jax.jit
    def f(A: LowRankMatrix, B: LowRankMatrix, vec):
        return (
            A.to_dense(),
            A.matvec(vec),
            A.inner(B),
            A.l2_norm(),
            A.trace(),
            A.conj().to_dense(),
            A.T.to_dense(),
        )

    out = f(x, y, v)
    for got, exp in zip(out, eager):
        assert jnp.allclose(got, exp)
