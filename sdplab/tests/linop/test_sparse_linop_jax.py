# qotlib/tests/test_sparse_linop_jax.py
from __future__ import annotations

import numpy as np
import pytest

from qotlib.core.linop._sparse import SparseArrayLinOp
from qotlib.core.backend import BackendContext
from qotlib.core.space import DenseVectorSpace


def _jax_import():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _jax_ops_import():
    # Adjust this import path if your project structure differs.
    from qotlib.core.backend.jax import JaxOps  # noqa: WPS433

    return JaxOps


def _jsparse_import():
    jax, _ = _jax_import()
    jsparse = pytest.importorskip("jax.experimental.sparse")
    return jax, jsparse


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
def dom_cod(jax_ctx):
    dom_n = 6
    cod_n = 4
    dom = DenseVectorSpace(ctx=jax_ctx, shape=(dom_n,), n=dom_n)
    cod = DenseVectorSpace(ctx=jax_ctx, shape=(cod_n,), n=cod_n)
    return dom, cod


def _make_sparse_from_dense(A_dense):
    """
    Convert a dense (m,n) JAX array to a JAX sparse array.
    Prefer BCOO since it's widely supported and PyTree-friendly.
    """
    _, jsparse = _jsparse_import()
    return jsparse.BCOO.fromdense(A_dense)


# -------------------------
# Construction tests
# -------------------------


def test_sparse_linop_allows_different_jaxops_instances(jax_ctx):
    jax, jnp = _jax_import()
    JaxOps = _jax_ops_import()
    jsparse = pytest.importorskip("jax.experimental.sparse")

    from qotlib.core.linop._sparse import SparseArrayLinOp

    x = DenseVectorSpace(ctx=jax_ctx, shape=(5,), n=5)

    other_ctx = BackendContext(
        ops=JaxOps(),  # different instance, SAME type -> should be OK
        dtype=np.float64,
        allow_sparse=True,
        enable_checks=True,
    )
    y = DenseVectorSpace(ctx=other_ctx, shape=(4,), n=4)

    key = jax.random.PRNGKey(0)
    A_dense = jax.random.normal(key, (y.n, x.n), dtype=jnp.float64)
    # Make it somewhat sparse then convert to JAX sparse
    mask = (jax.random.uniform(key, (y.n, x.n)) < 0.5)
    A_dense = A_dense * mask
    A = jsparse.BCOO.fromdense(A_dense)

    _ = SparseArrayLinOp(dom=x, cod=y, A=A)  # should NOT raise


def test_sparse_linop_rejects_numpy_vs_jax_backend(jax_ctx):
    jax, jnp = _jax_import()
    jsparse = pytest.importorskip("jax.experimental.sparse")

    from qotlib.core.backend.numpy import NumpyOps
    from qotlib.core.linop._sparse import SparseArrayLinOp

    x = DenseVectorSpace(ctx=jax_ctx, shape=(5,), n=5)

    np_ctx = BackendContext(
        ops=NumpyOps(),
        dtype=np.float64,
        allow_sparse=True,
        enable_checks=True,
    )
    y = DenseVectorSpace(ctx=np_ctx, shape=(4,), n=4)

    key = jax.random.PRNGKey(1)
    A_dense = jax.random.normal(key, (y.n, x.n), dtype=jnp.float64)
    A = jsparse.BCOO.fromdense(A_dense)

    with pytest.raises(ValueError, match="backends are not compatible"):
        _ = SparseArrayLinOp(dom=x, cod=y, A=A)


def test_construct_rejects_non_sparse(dom_cod):
    """Reject non-sparse operator arrays under JAX backend."""
    jax, jnp = _jax_import()

    dom, cod = dom_cod
    key = jax.random.PRNGKey(1)
    A_dense = jax.random.normal(key, (cod.n, dom.n), dtype=jnp.float64)  # dense, not sparse

    with pytest.raises(TypeError, match="Expected sparse"):
        SparseArrayLinOp(dom=dom, cod=cod, A=A_dense)


def test_construct_rejects_wrong_A_shape(dom_cod):
    """Reject sparse operator tensors with invalid shape."""
    jax, jnp = _jax_import()

    dom, cod = dom_cod
    key = jax.random.PRNGKey(2)
    A_dense = jax.random.normal(key, (cod.n, dom.n + 1), dtype=jnp.float64)
    A = _make_sparse_from_dense(A_dense)

    with pytest.raises(TypeError, match=r"Expected A\.shape == cod\.shape \+ dom\.shape"):
        SparseArrayLinOp(dom=dom, cod=cod, A=A)


# -------------------------
# apply / rapply correctness + jittability
# -------------------------

def test_apply_matches_dense_and_is_jittable(dom_cod):
    """Check apply matches dense matmul and works under JIT."""
    jax, jnp = _jax_import()

    dom, cod = dom_cod
    key = jax.random.PRNGKey(3)
    kA, kx = jax.random.split(key, 2)

    # Make a genuinely sparse-ish dense matrix then sparsify
    A_dense = jax.random.normal(kA, (cod.n, dom.n), dtype=jnp.float64)
    mask = (jax.random.uniform(kA, (cod.n, dom.n)) < 0.5)
    A_dense = A_dense * mask

    A = _make_sparse_from_dense(A_dense)
    op = SparseArrayLinOp(dom=dom, cod=cod, A=A)

    x = jax.random.normal(kx, (dom.n,), dtype=jnp.float64)

    y = op.apply(x)
    y_expected = A_dense @ x

    assert y.shape == (cod.n,)
    assert np.allclose(np.asarray(y), np.asarray(y_expected))

    y_jit = jax.jit(lambda op_, x_: op_.apply(x_))(op, x)
    assert np.allclose(np.asarray(y_jit), np.asarray(y_expected))


def test_rapply_matches_adjoint_and_is_jittable(dom_cod):
    """Check rapply matches adjoint and works under JIT."""
    jax, jnp = _jax_import()

    dom, cod = dom_cod
    key = jax.random.PRNGKey(4)
    kA, ky = jax.random.split(key, 2)

    A_dense = jax.random.normal(kA, (cod.n, dom.n), dtype=jnp.float64)
    mask = (jax.random.uniform(kA, (cod.n, dom.n)) < 0.6)
    A_dense = A_dense * mask

    A = _make_sparse_from_dense(A_dense)
    op = SparseArrayLinOp(dom=dom, cod=cod, A=A)

    y = jax.random.normal(ky, (cod.n,), dtype=jnp.float64)

    x = op.rapply(y)
    x_expected = A_dense.T.conj() @ y

    assert x.shape == (dom.n,)
    assert np.allclose(np.asarray(x), np.asarray(x_expected))

    x_jit = jax.jit(lambda op_, y_: op_.rapply(y_))(op, y)
    assert np.allclose(np.asarray(x_jit), np.asarray(x_expected))


def test_call_delegates_to_apply_and_is_jittable(dom_cod):
    """Ensure __call__ delegates to apply and is JIT-friendly."""
    jax, jnp = _jax_import()

    dom, cod = dom_cod
    key = jax.random.PRNGKey(5)
    kA, kx = jax.random.split(key, 2)

    A_dense = jax.random.normal(kA, (cod.n, dom.n), dtype=jnp.float64)
    A_dense = A_dense * (jax.random.uniform(kA, (cod.n, dom.n)) < 0.5)

    A = _make_sparse_from_dense(A_dense)
    op = SparseArrayLinOp(dom=dom, cod=cod, A=A)

    x = jax.random.normal(kx, (dom.n,), dtype=jnp.float64)

    assert np.allclose(np.asarray(op(x)), np.asarray(op.apply(x)))

    y_expected = A_dense @ x
    y_jit = jax.jit(lambda op_, x_: op_(x_))(op, x)
    assert np.allclose(np.asarray(y_jit), np.asarray(y_expected))


def test_rapply_uses_conjugate_transpose_for_complex_and_is_jittable():
    """Ensure complex rapply uses conjugate transpose and is JIT-friendly."""
    jax, jnp = _jax_import()
    JaxOps = _jax_ops_import()

    ctx = BackendContext(
        ops=JaxOps(),
        dtype=np.complex128,
        allow_sparse=True,
        enable_checks=True,
    )
    dom_n = 6
    cod_n = 4
    dom = DenseVectorSpace(ctx=ctx, shape=(dom_n,), n=dom_n)
    cod = DenseVectorSpace(ctx=ctx, shape=(cod_n,), n=cod_n)

    key = jax.random.PRNGKey(6)
    kA, ky = jax.random.split(key, 2)

    A_re = jax.random.normal(kA, (cod_n, dom_n), dtype=jnp.float64)
    A_im = jax.random.normal(kA, (cod_n, dom_n), dtype=jnp.float64)
    A_dense = (A_re + 1j * A_im) * (jax.random.uniform(kA, (cod_n, dom_n)) < 0.6)

    A = _make_sparse_from_dense(A_dense)
    op = SparseArrayLinOp(dom=dom, cod=cod, A=A)

    y_re = jax.random.normal(ky, (cod_n,), dtype=jnp.float64)
    y_im = jax.random.normal(ky, (cod_n,), dtype=jnp.float64)
    y = y_re + 1j * y_im

    x = op.rapply(y)
    x_expected = A_dense.T.conj() @ y
    assert np.allclose(np.asarray(x), np.asarray(x_expected))

    x_jit = jax.jit(lambda op_, y_: op_.rapply(y_))(op, y)
    assert np.allclose(np.asarray(x_jit), np.asarray(x_expected))


def test_adjoint_property_complex_and_is_jittable():
    """Verify adjoint property holds in eager and JIT for complex data."""
    jax, jnp = _jax_import()
    JaxOps = _jax_ops_import()

    ctx = BackendContext(
        ops=JaxOps(),
        dtype=np.complex128,
        allow_sparse=True,
        enable_checks=True,
    )
    dom_n = 6
    cod_n = 4
    dom = DenseVectorSpace(ctx=ctx, shape=(dom_n,), n=dom_n)
    cod = DenseVectorSpace(ctx=ctx, shape=(cod_n,), n=cod_n)

    key = jax.random.PRNGKey(7)
    kA, kx, ky = jax.random.split(key, 3)

    A_dense = (jax.random.normal(kA, (cod_n, dom_n), dtype=jnp.float64)
               + 1j * jax.random.normal(kA, (cod_n, dom_n), dtype=jnp.float64))
    A_dense = A_dense * (jax.random.uniform(kA, (cod_n, dom_n)) < 0.5)

    A = _make_sparse_from_dense(A_dense)
    op = SparseArrayLinOp(dom=dom, cod=cod, A=A)

    x = (jax.random.normal(kx, (dom_n,), dtype=jnp.float64)
         + 1j * jax.random.normal(kx, (dom_n,), dtype=jnp.float64))
    y = (jax.random.normal(ky, (cod_n,), dtype=jnp.float64)
         + 1j * jax.random.normal(ky, (cod_n,), dtype=jnp.float64))

    lhs = cod.inner(op.apply(x), y)      # <Ax, y>
    rhs = dom.inner(x, op.rapply(y))     # <x, A^* y>
    assert np.allclose(np.asarray(lhs), np.asarray(rhs))

    def _lhs_rhs(op_, x_, y_):
        return cod.inner(op_.apply(x_), y_), dom.inner(x_, op_.rapply(y_))

    lhs_jit, rhs_jit = jax.jit(_lhs_rhs)(op, x, y)
    assert np.allclose(np.asarray(lhs_jit), np.asarray(rhs_jit))


# -------------------------
# Checks under enable_checks=True
# -------------------------

def test_domain_check_enabled(dom_cod):
    """Ensure domain membership checks are enforced."""
    jax, jnp = _jax_import()

    dom, cod = dom_cod
    key = jax.random.PRNGKey(8)
    kA, kx = jax.random.split(key, 2)

    A_dense = jax.random.normal(kA, (cod.n, dom.n), dtype=jnp.float64)
    A = _make_sparse_from_dense(A_dense)
    op = SparseArrayLinOp(dom=dom, cod=cod, A=A)

    x_bad = jax.random.normal(kx, (dom.n + 1,), dtype=jnp.float64)
    with pytest.raises(TypeError, match="Expected shape"):
        op.apply(x_bad)


def test_codomain_check_enabled(dom_cod):
    """Ensure codomain membership checks are enforced."""
    jax, jnp = _jax_import()

    dom, cod = dom_cod
    key = jax.random.PRNGKey(9)
    kA, ky = jax.random.split(key, 2)

    A_dense = jax.random.normal(kA, (cod.n, dom.n), dtype=jnp.float64)
    A = _make_sparse_from_dense(A_dense)
    op = SparseArrayLinOp(dom=dom, cod=cod, A=A)

    y_bad = jax.random.normal(ky, (cod.n + 2,), dtype=jnp.float64)
    with pytest.raises(TypeError, match="Expected shape"):
        op.rapply(y_bad)
