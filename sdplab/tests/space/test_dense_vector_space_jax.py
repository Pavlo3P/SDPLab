from __future__ import annotations

import pytest


def _jax_import():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    # match your earlier convention: use x64 where available
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


from qotlib.core.backend import BackendContext
from qotlib.core.backend.jax import JaxOps
from qotlib.core.space import DenseVectorSpace


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def jax_jnp():
    return _jax_import()


@pytest.fixture
def ctx_jax(jax_jnp):
    _, jnp = jax_jnp
    return BackendContext(
        ops=JaxOps(),
        dtype=jnp.float64,
        enable_checks=True,
    )


@pytest.fixture
def ctx_jax_no_checks(jax_jnp):
    _, jnp = jax_jnp
    return BackendContext(
        ops=JaxOps(),
        dtype=jnp.float64,
        enable_checks=False,
    )


@pytest.fixture
def space(ctx_jax):
    n = 7
    return DenseVectorSpace(ctx=ctx_jax, shape=(n,), n=n)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def rand_vec(jax, jnp, n: int, seed: int = 0, dtype=None):
    dtype = jnp.float64 if dtype is None else dtype
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (n,), dtype=dtype)


def rand_cvec(jax, jnp, n: int, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    a = jax.random.normal(key, (n,), dtype=jnp.float64)
    b = jax.random.normal(jax.random.fold_in(key, 1), (n,), dtype=jnp.float64)
    return (a + 1j * b).astype(jnp.complex128)


# ---------------------------------------------------------------------
# Construction & membership (eager-only)
# ---------------------------------------------------------------------
def test_construct_rejects_nonpositive_n(ctx_jax):
    """Reject vector spaces with non-positive dimension."""
    with pytest.raises(ValueError, match="n must be positive"):
        DenseVectorSpace(ctx=ctx_jax, shape=(0,), n=0)


def test_check_member_accepts_correct_shape(space, jax_jnp):
    """Accept JAX members with correct shape."""
    _, jnp = jax_jnp
    x = jnp.zeros((space.n,), dtype=space.ctx.dtype)
    space.check_member(x)  # should not raise


def test_check_member_rejects_wrong_shape(space, jax_jnp):
    """Reject JAX members with wrong shape."""
    _, jnp = jax_jnp
    x = jnp.zeros((space.n, 1), dtype=space.ctx.dtype)
    with pytest.raises(TypeError, match=r"Expected shape"):
        space.check_member(x)


def test_check_member_rejects_non_dense(space):
    """Reject non-dense members in JAX space."""
    # Whatever ctx.assert_dense considers "not dense" should raise.
    with pytest.raises(TypeError):
        space.check_member([0.0] * space.n)


# ---------------------------------------------------------------------
# Zeros and algebra (eager)
# ---------------------------------------------------------------------
def test_zeros_is_member_and_identity(space, jax_jnp):
    """Ensure zeros is a member and additive identity."""
    jax, jnp = jax_jnp
    z = space.zeros()
    x = rand_vec(jax, jnp, space.n, seed=1, dtype=space.ctx.dtype)

    space.check_member(z)
    assert jnp.allclose(space.add(x, z), x)
    assert jnp.allclose(space.add(z, x), x)


def test_add_commutative_and_associative(space, jax_jnp):
    """Check add commutativity and associativity."""
    jax, jnp = jax_jnp
    x = rand_vec(jax, jnp, space.n, seed=2, dtype=space.ctx.dtype)
    y = rand_vec(jax, jnp, space.n, seed=3, dtype=space.ctx.dtype)
    z = rand_vec(jax, jnp, space.n, seed=4, dtype=space.ctx.dtype)

    assert jnp.allclose(space.add(x, y), space.add(y, x))
    assert jnp.allclose(space.add(space.add(x, y), z), space.add(x, space.add(y, z)))


def test_scale_distributive(space, jax_jnp):
    """Check scale distributivity over addition."""
    jax, jnp = jax_jnp
    x = rand_vec(jax, jnp, space.n, seed=5, dtype=space.ctx.dtype)
    y = rand_vec(jax, jnp, space.n, seed=6, dtype=space.ctx.dtype)
    a = 2.25

    lhs = space.scale(a, space.add(x, y))
    rhs = space.add(space.scale(a, x), space.scale(a, y))
    assert jnp.allclose(lhs, rhs)


def test_axpy_matches_add_scale(space, jax_jnp):
    """Ensure axpy matches scale+add semantics."""
    jax, jnp = jax_jnp
    x = rand_vec(jax, jnp, space.n, seed=7, dtype=space.ctx.dtype)
    y = rand_vec(jax, jnp, space.n, seed=8, dtype=space.ctx.dtype)
    a = -0.75

    got = space.axpy(a, x, y)
    exp = space.add(space.scale(a, x), y)
    assert jnp.allclose(got, exp)


# ---------------------------------------------------------------------
# Inner product & norm (eager)
# ---------------------------------------------------------------------
def test_inner_matches_jnp_vdot(space, jax_jnp):
    """Confirm inner product matches JAX vdot."""
    jax, jnp = jax_jnp
    x = rand_vec(jax, jnp, space.n, seed=9, dtype=space.ctx.dtype)
    y = rand_vec(jax, jnp, space.n, seed=10, dtype=space.ctx.dtype)

    got = space.inner(x, y)
    exp = jnp.vdot(x, y)
    assert jnp.allclose(got, exp)


def test_inner_conjugate_symmetry_complex_dtype(jax_jnp):
    """Verify conjugate symmetry for complex inner products."""
    jax, jnp = jax_jnp
    ctx = BackendContext(ops=JaxOps(), dtype=jnp.complex128, enable_checks=True)
    n = 6
    S = DenseVectorSpace(ctx=ctx, shape=(n,), n=n)

    x = rand_cvec(jax, jnp, n, seed=11)
    y = rand_cvec(jax, jnp, n, seed=12)

    xy = S.inner(x, y)
    yx = S.inner(y, x)
    assert jnp.allclose(xy, jnp.conj(yx))


def test_norm_consistency(space, jax_jnp):
    """Check norm consistency with inner product."""
    jax, jnp = jax_jnp
    x = rand_vec(jax, jnp, space.n, seed=13, dtype=space.ctx.dtype)

    lhs = space.norm(x) ** 2
    rhs = jnp.real(space.inner(x, x))
    assert jnp.allclose(lhs, rhs)


# ---------------------------------------------------------------------
# Flatten / unflatten (eager)
# ---------------------------------------------------------------------
def test_flatten_is_identity(space, jax_jnp):
    """Ensure flatten returns the same JAX vector."""
    jax, jnp = jax_jnp
    x = rand_vec(jax, jnp, space.n, seed=14, dtype=space.ctx.dtype)
    v = space.flatten(x)

    # DenseVectorSpace.flatten returns x as-is
    assert v is x
    assert jnp.allclose(v, x)


def test_unflatten_reshapes(space, jax_jnp):
    """Ensure unflatten reshapes vectors correctly."""
    jax, jnp = jax_jnp
    v = rand_vec(jax, jnp, space.n, seed=15, dtype=space.ctx.dtype)
    x = space.unflatten(v)

    assert tuple(x.shape) == (space.n,)
    assert jnp.allclose(x, v)


# ---------------------------------------------------------------------
# eigh is not defined
# ---------------------------------------------------------------------
def test_eigh_raises(space, jax_jnp):
    """Ensure vector spaces reject eigendecomposition."""
    jax, jnp = jax_jnp
    x = rand_vec(jax, jnp, space.n, seed=16, dtype=space.ctx.dtype)
    with pytest.raises(TypeError, match=r"eigh is not defined for vector spaces"):
        space.eigh(x)


# ---------------------------------------------------------------------
# JIT compatibility tests (enable_checks=False)
# ---------------------------------------------------------------------
def test_jit_add_scale_inner_norm_flatten_unflatten(ctx_jax_no_checks, jax_jnp):
    """Check JIT compatibility for core vector space operations."""
    jax, jnp = jax_jnp
    n = 8
    S = DenseVectorSpace(ctx=ctx_jax_no_checks, shape=(n,), n=n)

    x = rand_vec(jax, jnp, n, seed=20, dtype=jnp.float64)
    y = rand_vec(jax, jnp, n, seed=21, dtype=jnp.float64)
    a = 1.75

    eager_add = S.add(x, y)
    eager_scale = S.scale(a, x)
    eager_inner = S.inner(x, y)
    eager_norm = S.norm(x)
    eager_flat = S.flatten(x)
    eager_unflat = S.unflatten(eager_flat)

    @jax.jit
    def f(u, v):
        return (
            S.add(u, v),
            S.scale(a, u),
            S.inner(u, v),
            S.norm(u),
            S.flatten(u),
            S.unflatten(S.flatten(u)),
        )

    out_add, out_scale, out_inner, out_norm, out_flat, out_unflat = f(x, y)

    assert jnp.allclose(out_add, eager_add)
    assert jnp.allclose(out_scale, eager_scale)
    assert jnp.allclose(out_inner, eager_inner)
    assert jnp.allclose(out_norm, eager_norm)
    assert jnp.allclose(out_flat, eager_flat)
    assert jnp.allclose(out_unflat, eager_unflat)


def test_jit_vmap_works(ctx_jax_no_checks, jax_jnp):
    """
    Optional but valuable: ensure the space ops compose with vmap.
    """
    jax, jnp = jax_jnp
    n = 5
    S = DenseVectorSpace(ctx=ctx_jax_no_checks, shape=(n,), n=n)

    X = jax.random.normal(jax.random.PRNGKey(30), (10, n), dtype=jnp.float64)
    y = jax.random.normal(jax.random.PRNGKey(31), (n,), dtype=jnp.float64)

    @jax.jit
    def batched_inner(A):
        # returns shape (10,)
        return jax.vmap(lambda u: S.inner(u, y))(A)

    out = batched_inner(X)
    # Compare to manual JAX computation
    exp = jnp.sum(jnp.conj(X) * y[None, :], axis=1)  # equals vdot row-wise
    assert jnp.allclose(out, exp)
