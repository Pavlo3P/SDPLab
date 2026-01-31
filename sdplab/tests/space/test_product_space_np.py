from __future__ import annotations

import numpy as np
import pytest

from qotlib.core.backend import BackendContext
from qotlib.core.backend.numpy import NumpyOps
from qotlib.core.space import DenseVectorSpace, DenseHermitianMatrixSpace, ProductSpace


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def ctx_np_c128():
    return BackendContext(
        ops=NumpyOps(),
        dtype=np.complex128,
        enable_checks=True,
    )


@pytest.fixture
def vec_space(ctx_np_c128):
    n = 7
    return DenseVectorSpace(ctx=ctx_np_c128, shape=(n,), n=n)


@pytest.fixture
def herm_space(ctx_np_c128):
    n = 5
    return DenseHermitianMatrixSpace(
        ctx=ctx_np_c128,
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
def rand_cvec(n: int, seed: int = 0, dtype=np.complex128):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    return (a + 1j * b).astype(dtype)


def hermitian(n: int, seed: int = 0, dtype=np.complex128):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = A.astype(dtype)
    return (A + A.conj().T) / 2


# ---------------------------------------------------------------------
# Construction & membership
# ---------------------------------------------------------------------
def test_construct_rejects_empty(ctx_np_c128):
    with pytest.raises(ValueError):
        ProductSpace(spaces=())


def test_check_member_accepts_tuple(prod_space, vec_space, herm_space):
    x = rand_cvec(vec_space.n, seed=1, dtype=vec_space.ctx.dtype)
    X = hermitian(herm_space.n, seed=2, dtype=herm_space.ctx.dtype)
    prod_space.check_member((x, X))  # should not raise


def test_check_member_rejects_non_tuple(prod_space, vec_space):
    x = rand_cvec(vec_space.n, seed=3, dtype=vec_space.ctx.dtype)
    with pytest.raises(TypeError):
        prod_space.check_member(x)  # not a tuple


def test_check_member_rejects_wrong_arity(prod_space, vec_space, herm_space):
    x = rand_cvec(vec_space.n, seed=4, dtype=vec_space.ctx.dtype)
    X = hermitian(herm_space.n, seed=5, dtype=herm_space.ctx.dtype)

    with pytest.raises(ValueError):
        prod_space.check_member((x,))          # missing component

    with pytest.raises(ValueError):
        prod_space.check_member((x, X, x))     # extra component


def test_check_member_rejects_invalid_component(prod_space, vec_space, herm_space):
    x = rand_cvec(vec_space.n, seed=6, dtype=vec_space.ctx.dtype)
    X = hermitian(herm_space.n, seed=7, dtype=herm_space.ctx.dtype)
    X[0, 1] += 1e-6  # break Hermitian property

    with pytest.raises(TypeError):
        prod_space.check_member((x, X))


# ---------------------------------------------------------------------
# Zeros and algebra
# ---------------------------------------------------------------------
def test_zeros_is_member_and_identity(prod_space, vec_space, herm_space):
    z = prod_space.zeros()
    assert isinstance(z, tuple) and len(z) == 2
    prod_space.check_member(z)

    x = rand_cvec(vec_space.n, seed=10, dtype=vec_space.ctx.dtype)
    X = hermitian(herm_space.n, seed=11, dtype=herm_space.ctx.dtype)
    a = (x, X)

    got1 = prod_space.add(a, z)
    got2 = prod_space.add(z, a)

    assert np.allclose(got1[0], a[0])
    assert np.allclose(got1[1], a[1])
    assert np.allclose(got2[0], a[0])
    assert np.allclose(got2[1], a[1])


def test_add_commutative_and_associative(prod_space, vec_space, herm_space):
    a = (
        rand_cvec(vec_space.n, seed=12, dtype=vec_space.ctx.dtype),
        hermitian(herm_space.n, seed=13, dtype=herm_space.ctx.dtype),
    )
    b = (
        rand_cvec(vec_space.n, seed=14, dtype=vec_space.ctx.dtype),
        hermitian(herm_space.n, seed=15, dtype=herm_space.ctx.dtype),
    )
    c = (
        rand_cvec(vec_space.n, seed=16, dtype=vec_space.ctx.dtype),
        hermitian(herm_space.n, seed=17, dtype=herm_space.ctx.dtype),
    )

    ab = prod_space.add(a, b)
    ba = prod_space.add(b, a)
    assert np.allclose(ab[0], ba[0])
    assert np.allclose(ab[1], ba[1])

    lhs = prod_space.add(prod_space.add(a, b), c)
    rhs = prod_space.add(a, prod_space.add(b, c))
    assert np.allclose(lhs[0], rhs[0])
    assert np.allclose(lhs[1], rhs[1])


def test_scale_distributive(prod_space, vec_space, herm_space):
    a = (
        rand_cvec(vec_space.n, seed=18, dtype=vec_space.ctx.dtype),
        hermitian(herm_space.n, seed=19, dtype=herm_space.ctx.dtype),
    )
    b = (
        rand_cvec(vec_space.n, seed=20, dtype=vec_space.ctx.dtype),
        hermitian(herm_space.n, seed=21, dtype=herm_space.ctx.dtype),
    )
    alpha = -0.75

    lhs = prod_space.scale(alpha, prod_space.add(a, b))
    rhs = prod_space.add(prod_space.scale(alpha, a), prod_space.scale(alpha, b))

    assert np.allclose(lhs[0], rhs[0])
    assert np.allclose(lhs[1], rhs[1])


def test_axpy_matches_add_scale(prod_space, vec_space, herm_space):
    x = (
        rand_cvec(vec_space.n, seed=22, dtype=vec_space.ctx.dtype),
        hermitian(herm_space.n, seed=23, dtype=herm_space.ctx.dtype),
    )
    y = (
        rand_cvec(vec_space.n, seed=24, dtype=vec_space.ctx.dtype),
        hermitian(herm_space.n, seed=25, dtype=herm_space.ctx.dtype),
    )
    a = 2.25

    got = prod_space.axpy(a, x, y)
    exp = prod_space.add(prod_space.scale(a, x), y)

    assert np.allclose(got[0], exp[0])
    assert np.allclose(got[1], exp[1])


# ---------------------------------------------------------------------
# Inner product & norm
# ---------------------------------------------------------------------
def test_inner_is_sum_of_component_inners(prod_space, vec_space, herm_space):
    x = rand_cvec(vec_space.n, seed=26, dtype=vec_space.ctx.dtype)
    X = hermitian(herm_space.n, seed=27, dtype=herm_space.ctx.dtype)

    y = rand_cvec(vec_space.n, seed=28, dtype=vec_space.ctx.dtype)
    Y = hermitian(herm_space.n, seed=29, dtype=herm_space.ctx.dtype)

    got = prod_space.inner((x, X), (y, Y))
    exp = vec_space.inner(x, y) + herm_space.inner(X, Y)

    assert np.allclose(got, exp)


def test_inner_conjugate_symmetry(prod_space, vec_space, herm_space):
    x = rand_cvec(vec_space.n, seed=30, dtype=vec_space.ctx.dtype)
    X = hermitian(herm_space.n, seed=31, dtype=herm_space.ctx.dtype)
    y = rand_cvec(vec_space.n, seed=32, dtype=vec_space.ctx.dtype)
    Y = hermitian(herm_space.n, seed=33, dtype=herm_space.ctx.dtype)

    xy = prod_space.inner((x, X), (y, Y))
    yx = prod_space.inner((y, Y), (x, X))
    assert np.allclose(xy, np.conj(yx))


def test_norm_consistency(prod_space, vec_space, herm_space):
    x = rand_cvec(vec_space.n, seed=34, dtype=vec_space.ctx.dtype)
    X = hermitian(herm_space.n, seed=35, dtype=herm_space.ctx.dtype)

    lhs = prod_space.norm((x, X)) ** 2
    rhs = np.real(prod_space.inner((x, X), (x, X)))
    assert np.allclose(lhs, rhs)


# ---------------------------------------------------------------------
# Flatten / unflatten
# ---------------------------------------------------------------------
def test_flatten_shape(prod_space, vec_space, herm_space):
    x = rand_cvec(vec_space.n, seed=36, dtype=vec_space.ctx.dtype)
    X = hermitian(herm_space.n, seed=37, dtype=herm_space.ctx.dtype)

    v = prod_space.flatten((x, X))
    assert v.shape == (vec_space.n + herm_space.n * herm_space.n,)


def test_unflatten_roundtrip(prod_space, vec_space, herm_space):
    x = rand_cvec(vec_space.n, seed=38, dtype=vec_space.ctx.dtype)
    X = hermitian(herm_space.n, seed=39, dtype=herm_space.ctx.dtype)

    v = prod_space.flatten((x, X))
    y, Y = prod_space.unflatten(v)

    vec_space.check_member(y)
    herm_space.check_member(Y)

    assert np.allclose(y, x)
    assert np.allclose(Y, X)


# ---------------------------------------------------------------------
# eigh policy
# ---------------------------------------------------------------------
def test_eigh_raises(prod_space, vec_space, herm_space):
    x = rand_cvec(vec_space.n, seed=40, dtype=vec_space.ctx.dtype)
    X = hermitian(herm_space.n, seed=41, dtype=herm_space.ctx.dtype)

    with pytest.raises((TypeError, NotImplementedError)):
        prod_space.eigh((x, X))


# ---------------------------------------------------------------------
# Checks disabled policy
# ---------------------------------------------------------------------
def test_checks_disabled_skips_membership_validation():
    ctx = BackendContext(ops=NumpyOps(), dtype=np.complex128, enable_checks=False)

    n = 4
    m = 3
    V = DenseVectorSpace(ctx=ctx, shape=(n,), n=n)
    H = DenseHermitianMatrixSpace(ctx=ctx, n=m)
    P = ProductSpace(spaces=(V, H))

    # Wrong Hermitian component (not symmetric), but checks are disabled
    x = np.zeros((n,), dtype=ctx.dtype)
    A = np.random.randn(m, m) + 1j * np.random.randn(m, m)

    out = P.add((x, A), (x, A))
    assert out[0].shape == (n,)
    assert out[1].shape == (m, m)
