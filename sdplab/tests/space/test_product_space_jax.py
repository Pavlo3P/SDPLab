from __future__ import annotations

import pytest


def _jax_import():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


from qotlib.core.backend import BackendContext
from qotlib.core.backend.jax import JaxOps
from qotlib.core.space import DenseVectorSpace, DenseHermitianMatrixSpace, ProductSpace


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
        dtype=jnp.complex128,
        enable_checks=True,
    )


@pytest.fixture
def ctx_jax_no_checks(jax_jnp):
    _, jnp = jax_jnp
    return BackendContext(
        ops=JaxOps(),
        dtype=jnp.complex128,
        enable_checks=False,
    )


@pytest.fixture
def vec_space(ctx_jax):
    n = 7
    return DenseVectorSpace(ctx=ctx_jax, shape=(n,), n=n)


@pytest.fixture
def herm_space(ctx_jax):
    n = 5
    return DenseHermitianMatrixSpace(
        ctx=ctx_jax,
        n=n,
        atol=1e-12,
        rtol=0.0,
    )


@pytest.fixture
def prod_space(vec_space, herm_space):
    return ProductSpace(spaces=(vec_space, herm_space))


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def rand_cvec(jax, jnp, n: int, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    a = jax.random.normal(key, (n,), dtype=jnp.float64)
    b = jax.random.normal(jax.random.fold_in(key, 1), (n,), dtype=jnp.float64)
    return (a + 1j * b).astype(jnp.complex128)


def hermitian(jax, jnp, n: int, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    a = jax.random.normal(key, (n, n), dtype=jnp.float64)
    b = jax.random.normal(jax.random.fold_in(key, 1), (n, n), dtype=jnp.float64)
    A = (a + 1j * b).astype(jnp.complex128)
    return (A + jnp.conj(A).T) / 2.0


def rand_cvec_key(jax, jnp, key, n: int):
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (n,), dtype=jnp.float64)
    b = jax.random.normal(k2, (n,), dtype=jnp.float64)
    return (a + 1j * b).astype(jnp.complex128)


def hermitian_key(jax, jnp, key, n: int):
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (n, n), dtype=jnp.float64)
    b = jax.random.normal(k2, (n, n), dtype=jnp.float64)
    A = (a + 1j * b).astype(jnp.complex128)
    return (A + jnp.conj(A).T) / 2.0


# ---------------------------------------------------------------------
# Construction & membership (eager-only checks)
# ---------------------------------------------------------------------
def test_construct_rejects_empty(ctx_jax):
    with pytest.raises(ValueError):
        ProductSpace(spaces=())


def test_check_member_accepts_correct_tuple(prod_space, vec_space, herm_space, jax_jnp):
    jax, jnp = jax_jnp
    x = rand_cvec(jax, jnp, vec_space.n, seed=1)
    X = hermitian(jax, jnp, herm_space.n, seed=2)
    prod_space.check_member((x, X))  # should not raise


def test_check_member_rejects_non_tuple(prod_space, vec_space, jax_jnp):
    jax, jnp = jax_jnp
    x = rand_cvec(jax, jnp, vec_space.n, seed=3)
    with pytest.raises(TypeError):
        prod_space.check_member(x)


def test_check_member_rejects_wrong_arity(prod_space, vec_space, herm_space, jax_jnp):
    jax, jnp = jax_jnp
    x = rand_cvec(jax, jnp, vec_space.n, seed=4)
    X = hermitian(jax, jnp, herm_space.n, seed=5)

    with pytest.raises(ValueError):
        prod_space.check_member((x,))      # missing
    with pytest.raises(ValueError):
        prod_space.check_member((x, X, x))  # extra


def test_check_member_rejects_invalid_component(prod_space, vec_space, herm_space, jax_jnp):
    jax, jnp = jax_jnp
    x = rand_cvec(jax, jnp, vec_space.n, seed=6)
    X = hermitian(jax, jnp, herm_space.n, seed=7)
    # Break Hermitian symmetry beyond tolerance
    X = X.at[0, 1].add(1e-6)

    with pytest.raises(TypeError):
        prod_space.check_member((x, X))


# ---------------------------------------------------------------------
# Zeros & algebra (eager)
# ---------------------------------------------------------------------
def test_zeros_is_member_and_identity(prod_space, vec_space, herm_space, jax_jnp):
    jax, jnp = jax_jnp
    z = prod_space.zeros()
    prod_space.check_member(z)

    a = (
        rand_cvec(jax, jnp, vec_space.n, seed=10),
        hermitian(jax, jnp, herm_space.n, seed=11),
    )

    got1 = prod_space.add(a, z)
    got2 = prod_space.add(z, a)

    assert jnp.allclose(got1[0], a[0])
    assert jnp.allclose(got1[1], a[1])
    assert jnp.allclose(got2[0], a[0])
    assert jnp.allclose(got2[1], a[1])


def test_add_commutative_and_associative(prod_space, vec_space, herm_space, jax_jnp):
    jax, jnp = jax_jnp
    a = (rand_cvec(jax, jnp, vec_space.n, seed=12), hermitian(jax, jnp, herm_space.n, seed=13))
    b = (rand_cvec(jax, jnp, vec_space.n, seed=14), hermitian(jax, jnp, herm_space.n, seed=15))
    c = (rand_cvec(jax, jnp, vec_space.n, seed=16), hermitian(jax, jnp, herm_space.n, seed=17))

    ab = prod_space.add(a, b)
    ba = prod_space.add(b, a)
    assert jnp.allclose(ab[0], ba[0])
    assert jnp.allclose(ab[1], ba[1])

    lhs = prod_space.add(prod_space.add(a, b), c)
    rhs = prod_space.add(a, prod_space.add(b, c))
    assert jnp.allclose(lhs[0], rhs[0])
    assert jnp.allclose(lhs[1], rhs[1])


def test_scale_distributive(prod_space, vec_space, herm_space, jax_jnp):
    jax, jnp = jax_jnp
    a = (rand_cvec(jax, jnp, vec_space.n, seed=18), hermitian(jax, jnp, herm_space.n, seed=19))
    b = (rand_cvec(jax, jnp, vec_space.n, seed=20), hermitian(jax, jnp, herm_space.n, seed=21))
    alpha = 1.75

    lhs = prod_space.scale(alpha, prod_space.add(a, b))
    rhs = prod_space.add(prod_space.scale(alpha, a), prod_space.scale(alpha, b))

    assert jnp.allclose(lhs[0], rhs[0])
    assert jnp.allclose(lhs[1], rhs[1])


def test_axpy_matches_add_scale(prod_space, vec_space, herm_space, jax_jnp):
    jax, jnp = jax_jnp
    x = (rand_cvec(jax, jnp, vec_space.n, seed=22), hermitian(jax, jnp, herm_space.n, seed=23))
    y = (rand_cvec(jax, jnp, vec_space.n, seed=24), hermitian(jax, jnp, herm_space.n, seed=25))
    a = -0.75

    got = prod_space.axpy(a, x, y)
    exp = prod_space.add(prod_space.scale(a, x), y)

    assert jnp.allclose(got[0], exp[0])
    assert jnp.allclose(got[1], exp[1])


# ---------------------------------------------------------------------
# Inner & norm (eager)
# ---------------------------------------------------------------------
def test_inner_is_sum_of_component_inners(prod_space, vec_space, herm_space, jax_jnp):
    jax, jnp = jax_jnp
    x = rand_cvec(jax, jnp, vec_space.n, seed=26)
    X = hermitian(jax, jnp, herm_space.n, seed=27)
    y = rand_cvec(jax, jnp, vec_space.n, seed=28)
    Y = hermitian(jax, jnp, herm_space.n, seed=29)

    got = prod_space.inner((x, X), (y, Y))
    exp = vec_space.inner(x, y) + herm_space.inner(X, Y)
    assert jnp.allclose(got, exp)


def test_inner_conjugate_symmetry(prod_space, vec_space, herm_space, jax_jnp):
    jax, jnp = jax_jnp
    x = rand_cvec(jax, jnp, vec_space.n, seed=30)
    X = hermitian(jax, jnp, herm_space.n, seed=31)
    y = rand_cvec(jax, jnp, vec_space.n, seed=32)
    Y = hermitian(jax, jnp, herm_space.n, seed=33)

    xy = prod_space.inner((x, X), (y, Y))
    yx = prod_space.inner((y, Y), (x, X))
    assert jnp.allclose(xy, jnp.conj(yx))


def test_norm_consistency(prod_space, vec_space, herm_space, jax_jnp):
    jax, jnp = jax_jnp
    x = rand_cvec(jax, jnp, vec_space.n, seed=34)
    X = hermitian(jax, jnp, herm_space.n, seed=35)

    lhs = prod_space.norm((x, X)) ** 2
    rhs = jnp.real(prod_space.inner((x, X), (x, X)))
    assert jnp.allclose(lhs, rhs)


# ---------------------------------------------------------------------
# Flatten / unflatten (eager)
# ---------------------------------------------------------------------
def test_flatten_shape(prod_space, vec_space, herm_space, jax_jnp):
    jax, jnp = jax_jnp
    x = rand_cvec(jax, jnp, vec_space.n, seed=36)
    X = hermitian(jax, jnp, herm_space.n, seed=37)

    v = prod_space.flatten((x, X))
    assert tuple(v.shape) == (vec_space.n + herm_space.n * herm_space.n,)


def test_unflatten_roundtrip(prod_space, vec_space, herm_space, jax_jnp):
    jax, jnp = jax_jnp
    x = rand_cvec(jax, jnp, vec_space.n, seed=38)
    X = hermitian(jax, jnp, herm_space.n, seed=39)

    v = prod_space.flatten((x, X))
    y, Y = prod_space.unflatten(v)

    vec_space.check_member(y)
    herm_space.check_member(Y)

    assert jnp.allclose(y, x)
    assert jnp.allclose(Y, X)


# ---------------------------------------------------------------------
# eigh policy (undefined for ProductSpace)
# ---------------------------------------------------------------------
def test_eigh_raises(prod_space, vec_space, herm_space, jax_jnp):
    jax, jnp = jax_jnp
    x = rand_cvec(jax, jnp, vec_space.n, seed=40)
    X = hermitian(jax, jnp, herm_space.n, seed=41)

    with pytest.raises((TypeError, NotImplementedError)):
        prod_space.eigh((x, X))


# ---------------------------------------------------------------------
# JIT compatibility tests (enable_checks=False)
# ---------------------------------------------------------------------
def test_jit_add_scale_inner_norm_flatten_unflatten(ctx_jax_no_checks, jax_jnp):
    jax, jnp = jax_jnp

    n = 8
    m = 4
    V = DenseVectorSpace(ctx=ctx_jax_no_checks, shape=(n,), n=n)
    H = DenseHermitianMatrixSpace(ctx=ctx_jax_no_checks, n=m)
    P = ProductSpace(spaces=(V, H))

    x = rand_cvec(jax, jnp, n, seed=50)
    X = hermitian(jax, jnp, m, seed=51)
    y = rand_cvec(jax, jnp, n, seed=52)
    Y = hermitian(jax, jnp, m, seed=53)
    a = 1.25

    eager_add = P.add((x, X), (y, Y))
    eager_scale = P.scale(a, (x, X))
    eager_inner = P.inner((x, X), (y, Y))
    eager_norm = P.norm((x, X))
    eager_flat = P.flatten((x, X))
    eager_unflat = P.unflatten(eager_flat)

    @jax.jit
    def f(u, U, v, V_):
        # NOTE: avoid calling check_member under jit; checks are disabled anyway
        return (
            P.add((u, U), (v, V_)),
            P.scale(a, (u, U)),
            P.inner((u, U), (v, V_)),
            P.norm((u, U)),
            P.flatten((u, U)),
            P.unflatten(P.flatten((u, U))),
        )

    out_add, out_scale, out_inner, out_norm, out_flat, out_unflat = f(x, X, y, Y)

    assert jnp.allclose(out_add[0], eager_add[0])
    assert jnp.allclose(out_add[1], eager_add[1])

    assert jnp.allclose(out_scale[0], eager_scale[0])
    assert jnp.allclose(out_scale[1], eager_scale[1])

    assert jnp.allclose(out_inner, eager_inner)
    assert jnp.allclose(out_norm, eager_norm)
    assert jnp.allclose(out_flat, eager_flat)

    assert jnp.allclose(out_unflat[0], eager_unflat[0])
    assert jnp.allclose(out_unflat[1], eager_unflat[1])
    # Hermitian unflatten should remain Hermitian (DenseHermitianMatrixSpace enforces it)
    assert jnp.allclose(out_unflat[1], jnp.conj(out_unflat[1]).T)


def test_jit_vmap_inner_over_batch(ctx_jax_no_checks, jax_jnp):
    """
    Ensure ProductSpace ops compose with vmap under jit (common in optimization loops).
    """
    jax, jnp = jax_jnp

    n = 6
    m = 3
    V = DenseVectorSpace(ctx=ctx_jax_no_checks, shape=(n,), n=n)
    H = DenseHermitianMatrixSpace(ctx=ctx_jax_no_checks, n=m)
    P = ProductSpace(spaces=(V, H))

    key = jax.random.PRNGKey(60)
    keys = jax.random.split(key, 5)

    # Batch of vectors and Hermitian matrices, generated purely from keys (JIT/Vmap-safe)
    Xs = jax.vmap(lambda k: rand_cvec_key(jax, jnp, k, n))(keys)          # (5, n)
    Hs = jax.vmap(lambda k: hermitian_key(jax, jnp, k, m))(keys)          # (5, m, m)

    key_y, key_Y = jax.random.split(jax.random.PRNGKey(61))
    y = rand_cvec_key(jax, jnp, key_y, n)
    Y = hermitian_key(jax, jnp, key_Y, m)

    @jax.jit
    def batched_inner(vecs, mats):
        return jax.vmap(lambda u, U: P.inner((u, U), (y, Y)))(vecs, mats)

    out = batched_inner(Xs, Hs)

    exp = jax.vmap(lambda u, U: V.inner(u, y) + H.inner(U, Y))(Xs, Hs)
    assert jnp.allclose(out, exp)
