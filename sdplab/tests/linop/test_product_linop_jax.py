from __future__ import annotations

import numpy as np
import pytest

from qotlib.core.backend import BackendContext
from qotlib.core.linop._dense import DenseArrayLinOp
from qotlib.core.space import DenseVectorSpace, ProductSpace

# Adjust this import to your actual module path
from qotlib.core.linop._product import BlockDiagonalLinOp, SumToSingleLinOp, StackedLinOp


def _jax_import():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _jax_ops_import():
    # Adjust this import path if your project structure differs.
    from qotlib.core.backend.jax import JaxOps  # noqa: WPS433

    return JaxOps


@pytest.fixture()
def jax_ctx():
    _jax_import()
    JaxOps = _jax_ops_import()
    return BackendContext(
        ops=JaxOps(),
        dtype=np.float64,
        allow_sparse=True,
        enable_checks=True,
    )


@pytest.fixture()
def jax_ctx_no_checks():
    _jax_import()
    JaxOps = _jax_ops_import()
    return BackendContext(
        ops=JaxOps(),
        dtype=np.float64,
        allow_sparse=True,
        enable_checks=False,
    )


@pytest.fixture()
def spaces(jax_ctx):
    # domain blocks
    x1 = DenseVectorSpace(ctx=jax_ctx, shape=(5,), n=5)
    x2 = DenseVectorSpace(ctx=jax_ctx, shape=(3,), n=3)

    # codomain blocks
    y1 = DenseVectorSpace(ctx=jax_ctx, shape=(4,), n=4)
    y2 = DenseVectorSpace(ctx=jax_ctx, shape=(2,), n=2)

    # single spaces
    x = DenseVectorSpace(ctx=jax_ctx, shape=(6,), n=6)
    y = DenseVectorSpace(ctx=jax_ctx, shape=(7,), n=7)

    return x1, x2, y1, y2, x, y


# ---------------------------------------------------------------------
# BlockDiagonalLinOp tests:  (X1×X2) -> (Y1×Y2)
# ---------------------------------------------------------------------
def test_blockdiag_apply_rapply_correctness(spaces):
    jax, jnp = _jax_import()
    x1, x2, y1, y2, _, _ = spaces

    dom = ProductSpace(spaces=(x1, x2))
    cod = ProductSpace(spaces=(y1, y2))

    key = jax.random.PRNGKey(0)
    k1, k2, kx1, kx2, ky1, ky2 = jax.random.split(key, 6)

    A1 = jax.random.normal(k1, (y1.n, x1.n), dtype=jnp.float64)
    A2 = jax.random.normal(k2, (y2.n, x2.n), dtype=jnp.float64)

    op1 = DenseArrayLinOp(dom=x1, cod=y1, A=A1)
    op2 = DenseArrayLinOp(dom=x2, cod=y2, A=A2)

    A = BlockDiagonalLinOp(dom=dom, cod=cod, ops=(op1, op2))

    x = (jax.random.normal(kx1, (x1.n,), dtype=jnp.float64),
         jax.random.normal(kx2, (x2.n,), dtype=jnp.float64))
    y = A.apply(x)

    assert y[0].shape == (y1.n,)
    assert y[1].shape == (y2.n,)
    assert np.allclose(np.asarray(y[0]), np.asarray(A1 @ x[0]))
    assert np.allclose(np.asarray(y[1]), np.asarray(A2 @ x[1]))

    z = (jax.random.normal(ky1, (y1.n,), dtype=jnp.float64),
         jax.random.normal(ky2, (y2.n,), dtype=jnp.float64))
    xt = A.rapply(z)

    assert xt[0].shape == (x1.n,)
    assert xt[1].shape == (x2.n,)
    assert np.allclose(np.asarray(xt[0]), np.asarray(A1.T.conj() @ z[0]))
    assert np.allclose(np.asarray(xt[1]), np.asarray(A2.T.conj() @ z[1]))


def test_blockdiag_adjoint_property(spaces):
    jax, jnp = _jax_import()
    x1, x2, y1, y2, _, _ = spaces

    dom = ProductSpace(spaces=(x1, x2))
    cod = ProductSpace(spaces=(y1, y2))

    key = jax.random.PRNGKey(1)
    k1, k2, kx1, kx2, ky1, ky2 = jax.random.split(key, 6)

    A1 = jax.random.normal(k1, (y1.n, x1.n), dtype=jnp.float64)
    A2 = jax.random.normal(k2, (y2.n, x2.n), dtype=jnp.float64)

    op1 = DenseArrayLinOp(dom=x1, cod=y1, A=A1)
    op2 = DenseArrayLinOp(dom=x2, cod=y2, A=A2)

    A = BlockDiagonalLinOp(dom=dom, cod=cod, ops=(op1, op2))

    x = (jax.random.normal(kx1, (x1.n,), dtype=jnp.float64),
         jax.random.normal(kx2, (x2.n,), dtype=jnp.float64))
    y = (jax.random.normal(ky1, (y1.n,), dtype=jnp.float64),
         jax.random.normal(ky2, (y2.n,), dtype=jnp.float64))

    lhs = cod.inner(A.apply(x), y)
    rhs = dom.inner(x, A.rapply(y))
    assert np.allclose(np.asarray(lhs), np.asarray(rhs))


# ---------------------------------------------------------------------
# JIT for BlockDiagonalLinOp (enable_checks=False)
# ---------------------------------------------------------------------
def test_blockdiag_is_jittable(jax_ctx_no_checks):
    jax, jnp = _jax_import()

    x1 = DenseVectorSpace(ctx=jax_ctx_no_checks, shape=(5,), n=5)
    x2 = DenseVectorSpace(ctx=jax_ctx_no_checks, shape=(3,), n=3)
    y1 = DenseVectorSpace(ctx=jax_ctx_no_checks, shape=(4,), n=4)
    y2 = DenseVectorSpace(ctx=jax_ctx_no_checks, shape=(2,), n=2)

    dom = ProductSpace(spaces=(x1, x2))
    cod = ProductSpace(spaces=(y1, y2))

    key = jax.random.PRNGKey(2)
    k1, k2, kx1, kx2 = jax.random.split(key, 4)

    A1 = jax.random.normal(k1, (y1.n, x1.n), dtype=jnp.float64)
    A2 = jax.random.normal(k2, (y2.n, x2.n), dtype=jnp.float64)

    op1 = DenseArrayLinOp(dom=x1, cod=y1, A=A1)
    op2 = DenseArrayLinOp(dom=x2, cod=y2, A=A2)
    A = BlockDiagonalLinOp(dom=dom, cod=cod, ops=(op1, op2))

    x = (jax.random.normal(kx1, (x1.n,), dtype=jnp.float64),
         jax.random.normal(kx2, (x2.n,), dtype=jnp.float64))

    y_eager = A.apply(x)

    f = jax.jit(lambda op_, x_: op_.apply(x_))
    y_jit = f(A, x)

    assert np.allclose(np.asarray(y_jit[0]), np.asarray(y_eager[0]))
    assert np.allclose(np.asarray(y_jit[1]), np.asarray(y_eager[1]))


# ---------------------------------------------------------------------
# SumToSingleLinOp tests:  (X1×X2) -> Y
# ---------------------------------------------------------------------
def test_sum_to_single_apply_rapply_correctness(spaces):
    jax, jnp = _jax_import()
    x1, x2, _, _, _, y = spaces

    dom = ProductSpace(spaces=(x1, x2))
    cod = y

    key = jax.random.PRNGKey(3)
    k1, k2, kx1, kx2, ky = jax.random.split(key, 5)

    A1 = jax.random.normal(k1, (cod.n, x1.n), dtype=jnp.float64)
    A2 = jax.random.normal(k2, (cod.n, x2.n), dtype=jnp.float64)

    op1 = DenseArrayLinOp(dom=x1, cod=cod, A=A1)
    op2 = DenseArrayLinOp(dom=x2, cod=cod, A=A2)

    A = SumToSingleLinOp(dom=dom, cod=cod, ops=(op1, op2))

    x = (jax.random.normal(kx1, (x1.n,), dtype=jnp.float64),
         jax.random.normal(kx2, (x2.n,), dtype=jnp.float64))
    out = A.apply(x)

    assert out.shape == (cod.n,)
    assert np.allclose(np.asarray(out), np.asarray(A1 @ x[0] + A2 @ x[1]))

    u = jax.random.normal(ky, (cod.n,), dtype=jnp.float64)
    xt = A.rapply(u)

    assert xt[0].shape == (x1.n,)
    assert xt[1].shape == (x2.n,)
    assert np.allclose(np.asarray(xt[0]), np.asarray(A1.T.conj() @ u))
    assert np.allclose(np.asarray(xt[1]), np.asarray(A2.T.conj() @ u))


def test_sum_to_single_adjoint_property(spaces):
    jax, jnp = _jax_import()
    x1, x2, _, _, _, y = spaces

    dom = ProductSpace(spaces=(x1, x2))
    cod = y

    key = jax.random.PRNGKey(4)
    k1, k2, kx1, kx2, ky = jax.random.split(key, 5)

    A1 = jax.random.normal(k1, (cod.n, x1.n), dtype=jnp.float64)
    A2 = jax.random.normal(k2, (cod.n, x2.n), dtype=jnp.float64)

    op1 = DenseArrayLinOp(dom=x1, cod=cod, A=A1)
    op2 = DenseArrayLinOp(dom=x2, cod=cod, A=A2)
    A = SumToSingleLinOp(dom=dom, cod=cod, ops=(op1, op2))

    x = (jax.random.normal(kx1, (x1.n,), dtype=jnp.float64),
         jax.random.normal(kx2, (x2.n,), dtype=jnp.float64))
    yv = jax.random.normal(ky, (cod.n,), dtype=jnp.float64)

    lhs = cod.inner(A.apply(x), yv)
    rhs = dom.inner(x, A.rapply(yv))
    assert np.allclose(np.asarray(lhs), np.asarray(rhs))


def test_sum_to_single_is_jittable(jax_ctx_no_checks):
    jax, jnp = _jax_import()

    x1 = DenseVectorSpace(ctx=jax_ctx_no_checks, shape=(5,), n=5)
    x2 = DenseVectorSpace(ctx=jax_ctx_no_checks, shape=(3,), n=3)
    y = DenseVectorSpace(ctx=jax_ctx_no_checks, shape=(7,), n=7)

    dom = ProductSpace(spaces=(x1, x2))
    cod = y

    key = jax.random.PRNGKey(5)
    k1, k2, kx1, kx2 = jax.random.split(key, 4)

    A1 = jax.random.normal(k1, (cod.n, x1.n), dtype=jnp.float64)
    A2 = jax.random.normal(k2, (cod.n, x2.n), dtype=jnp.float64)

    op1 = DenseArrayLinOp(dom=x1, cod=cod, A=A1)
    op2 = DenseArrayLinOp(dom=x2, cod=cod, A=A2)
    A = SumToSingleLinOp(dom=dom, cod=cod, ops=(op1, op2))

    x = (jax.random.normal(kx1, (x1.n,), dtype=jnp.float64),
         jax.random.normal(kx2, (x2.n,), dtype=jnp.float64))

    out_eager = A.apply(x)

    f = jax.jit(lambda op_, x_: op_.apply(x_))
    out_jit = f(A, x)
    assert np.allclose(np.asarray(out_jit), np.asarray(out_eager))


# ---------------------------------------------------------------------
# StackedLinOp tests:  X -> (Y1×Y2)
# ---------------------------------------------------------------------
def test_stacked_apply_rapply_correctness(spaces):
    jax, jnp = _jax_import()
    _, _, y1, y2, x, _ = spaces

    dom = x
    cod = ProductSpace(spaces=(y1, y2))

    key = jax.random.PRNGKey(6)
    k1, k2, kv, ky1, ky2 = jax.random.split(key, 5)

    A1 = jax.random.normal(k1, (y1.n, x.n), dtype=jnp.float64)
    A2 = jax.random.normal(k2, (y2.n, x.n), dtype=jnp.float64)

    op1 = DenseArrayLinOp(dom=dom, cod=y1, A=A1)
    op2 = DenseArrayLinOp(dom=dom, cod=y2, A=A2)
    A = StackedLinOp(dom=dom, cod=cod, ops=(op1, op2))

    v = jax.random.normal(kv, (x.n,), dtype=jnp.float64)
    out = A.apply(v)

    assert out[0].shape == (y1.n,)
    assert out[1].shape == (y2.n,)
    assert np.allclose(np.asarray(out[0]), np.asarray(A1 @ v))
    assert np.allclose(np.asarray(out[1]), np.asarray(A2 @ v))

    w = (jax.random.normal(ky1, (y1.n,), dtype=jnp.float64),
         jax.random.normal(ky2, (y2.n,), dtype=jnp.float64))
    vt = A.rapply(w)

    assert vt.shape == (x.n,)
    assert np.allclose(np.asarray(vt), np.asarray(A1.T.conj() @ w[0] + A2.T.conj() @ w[1]))


def test_stacked_adjoint_property(spaces):
    jax, jnp = _jax_import()
    _, _, y1, y2, x, _ = spaces

    dom = x
    cod = ProductSpace(spaces=(y1, y2))

    key = jax.random.PRNGKey(7)
    k1, k2, kv, ky1, ky2 = jax.random.split(key, 5)

    A1 = jax.random.normal(k1, (y1.n, x.n), dtype=jnp.float64)
    A2 = jax.random.normal(k2, (y2.n, x.n), dtype=jnp.float64)

    op1 = DenseArrayLinOp(dom=dom, cod=y1, A=A1)
    op2 = DenseArrayLinOp(dom=dom, cod=y2, A=A2)
    A = StackedLinOp(dom=dom, cod=cod, ops=(op1, op2))

    v = jax.random.normal(kv, (x.n,), dtype=jnp.float64)
    w = (jax.random.normal(ky1, (y1.n,), dtype=jnp.float64),
         jax.random.normal(ky2, (y2.n,), dtype=jnp.float64))

    lhs = cod.inner(A.apply(v), w)
    rhs = dom.inner(v, A.rapply(w))
    assert np.allclose(np.asarray(lhs), np.asarray(rhs))


def test_stacked_is_jittable(jax_ctx_no_checks):
    jax, jnp = _jax_import()

    x = DenseVectorSpace(ctx=jax_ctx_no_checks, shape=(6,), n=6)
    y1 = DenseVectorSpace(ctx=jax_ctx_no_checks, shape=(4,), n=4)
    y2 = DenseVectorSpace(ctx=jax_ctx_no_checks, shape=(2,), n=2)

    dom = x
    cod = ProductSpace(spaces=(y1, y2))

    key = jax.random.PRNGKey(8)
    k1, k2, kv, ky1, ky2 = jax.random.split(key, 5)

    A1 = jax.random.normal(k1, (y1.n, x.n), dtype=jnp.float64)
    A2 = jax.random.normal(k2, (y2.n, x.n), dtype=jnp.float64)

    op1 = DenseArrayLinOp(dom=dom, cod=y1, A=A1)
    op2 = DenseArrayLinOp(dom=dom, cod=y2, A=A2)
    A = StackedLinOp(dom=dom, cod=cod, ops=(op1, op2))

    v = jax.random.normal(kv, (x.n,), dtype=jnp.float64)
    out_eager = A.apply(v)

    f = jax.jit(lambda op_, x_: op_.apply(x_))
    out_jit = f(A, v)

    assert np.allclose(np.asarray(out_jit[0]), np.asarray(out_eager[0]))
    assert np.allclose(np.asarray(out_jit[1]), np.asarray(out_eager[1]))


# ---------------------------------------------------------------------
# Construction mismatch tests (backend / type)
# ---------------------------------------------------------------------
def test_dense_linop_rejects_numpy_vs_jax_backend(jax_ctx):
    jax, jnp = _jax_import()

    # JAX space
    x = DenseVectorSpace(ctx=jax_ctx, shape=(5,), n=5)

    # NumPy space (different backend type)
    from qotlib.core.backend.numpy import NumpyOps
    np_ctx = BackendContext(
        ops=NumpyOps(),
        dtype=np.float64,
        allow_sparse=True,
        enable_checks=True,
    )
    y = DenseVectorSpace(ctx=np_ctx, shape=(4,), n=4)

    A = jax.random.normal(jax.random.PRNGKey(0), (y.n, x.n), dtype=jnp.float64)

    with pytest.raises(ValueError, match="Domain and codomain backends are not compatible"):
        _ = DenseArrayLinOp(dom=x, cod=y, A=A)
