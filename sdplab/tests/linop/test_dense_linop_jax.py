from __future__ import annotations

import numpy as np
import pytest

from qotlib.core.linop._dense import DenseArrayLinOp
from qotlib.core.backend import BackendContext
from qotlib.core.space import DenseVectorSpace


def _jax_import():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    # Ensure consistent semantics with NumPy tests (float64/complex128 supported)
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _jax_ops_import():
    """
    Keep imports explicit and fail loudly if your path differs.
    Adjust the import below if your package layout is different.
    """
    from qotlib.core.backend.jax import JaxOps  # noqa: WPS433 (explicit import for tests)

    return JaxOps


@pytest.fixture()
def jax_ctx():
    _, _ = _jax_import()
    JaxOps = _jax_ops_import()

    ops = JaxOps()
    return BackendContext(
        ops=ops,
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


# -------------------------
# Construction tests
# -------------------------

def test_dense_linop_allows_different_jaxops_instances(jax_ctx):
    jax, jnp = _jax_import()
    JaxOps = _jax_ops_import()

    x = DenseVectorSpace(ctx=jax_ctx, shape=(5,), n=5)

    other_ctx = BackendContext(
        ops=JaxOps(),  # different instance, SAME type -> should be OK now
        dtype=np.float64,
        allow_sparse=True,
        enable_checks=True,
    )
    y = DenseVectorSpace(ctx=other_ctx, shape=(4,), n=4)

    A = jax.random.normal(jax.random.PRNGKey(0), (y.n, x.n), dtype=jnp.float64)

    _ = DenseArrayLinOp(dom=x, cod=y, A=A)  # should NOT raise


def test_dense_linop_rejects_numpy_vs_jax_backend(jax_ctx):
    jax, jnp = _jax_import()

    x = DenseVectorSpace(ctx=jax_ctx, shape=(5,), n=5)

    from qotlib.core.backend.numpy import NumpyOps

    np_ctx = BackendContext(
        ops=NumpyOps(),  # different type from JaxOps
        dtype=np.float64,
        allow_sparse=True,
        enable_checks=True,
    )
    y = DenseVectorSpace(ctx=np_ctx, shape=(4,), n=4)

    A = jax.random.normal(jax.random.PRNGKey(1), (y.n, x.n), dtype=jnp.float64)

    with pytest.raises(ValueError, match="backends are not compatible"):
        _ = DenseArrayLinOp(dom=x, cod=y, A=A)


def test_construct_rejects_non_dense(dom_cod):
    """Reject non-dense operator arrays under JAX backend."""
    # Pass a NumPy array instead of jax.Array; Jax backend should reject.
    dom, cod = dom_cod
    A = np.zeros((cod.n, dom.n), dtype=np.float64)

    with pytest.raises(TypeError):
        DenseArrayLinOp(dom=dom, cod=cod, A=A)  # type: ignore[arg-type]


def test_construct_rejects_wrong_A_shape(dom_cod):
    """Reject operator tensors with invalid shape."""
    jax, jnp = _jax_import()

    dom, cod = dom_cod
    key = jax.random.PRNGKey(1)

    A = jax.random.normal(key, (cod.n, dom.n + 1), dtype=jnp.float64)
    with pytest.raises(TypeError, match=r"Expected A\.shape == cod\.shape \+ dom\.shape"):
        DenseArrayLinOp(dom=dom, cod=cod, A=A)


# -------------------------
# apply / rapply correctness + jittability
# -------------------------

def test_apply_matches_matrix_form_and_is_jittable(dom_cod):
    """Check apply matches matmul and works under JIT."""
    jax, jnp = _jax_import()

    dom, cod = dom_cod
    key = jax.random.PRNGKey(2)
    kA, kx = jax.random.split(key, 2)

    A = jax.random.normal(kA, (cod.n, dom.n), dtype=jnp.float64)
    x = jax.random.normal(kx, (dom.n,), dtype=jnp.float64)

    op = DenseArrayLinOp(dom=dom, cod=cod, A=A)

    y = op.apply(x)
    y_expected = A @ x
    assert y.shape == (cod.n,)
    assert np.allclose(np.asarray(y), np.asarray(y_expected))

    f = jax.jit(lambda op_, x_: op_.apply(x_))
    y_jit = f(op, x)
    assert np.allclose(np.asarray(y_jit), np.asarray(y_expected))


def test_rapply_matches_adjoint_and_is_jittable(dom_cod):
    """Check rapply matches adjoint and works under JIT."""
    jax, jnp = _jax_import()

    dom, cod = dom_cod
    key = jax.random.PRNGKey(3)
    kA, ky = jax.random.split(key, 2)

    A = jax.random.normal(kA, (cod.n, dom.n), dtype=jnp.float64)
    y = jax.random.normal(ky, (cod.n,), dtype=jnp.float64)

    op = DenseArrayLinOp(dom=dom, cod=cod, A=A)

    x = op.rapply(y)
    x_expected = A.T.conj() @ y
    assert x.shape == (dom.n,)
    assert np.allclose(np.asarray(x), np.asarray(x_expected))

    f = jax.jit(lambda op_, y_: op_.rapply(y_))
    x_jit = f(op, y)
    assert np.allclose(np.asarray(x_jit), np.asarray(x_expected))


def test_call_delegates_to_apply_and_is_jittable(dom_cod):
    """Ensure __call__ delegates to apply and is JIT-friendly."""
    jax, jnp = _jax_import()

    dom, cod = dom_cod
    key = jax.random.PRNGKey(4)
    kA, kx = jax.random.split(key, 2)

    A = jax.random.normal(kA, (cod.n, dom.n), dtype=jnp.float64)
    x = jax.random.normal(kx, (dom.n,), dtype=jnp.float64)

    op = DenseArrayLinOp(dom=dom, cod=cod, A=A)

    assert np.allclose(np.asarray(op(x)), np.asarray(op.apply(x)))

    f = jax.jit(lambda op_, x_: op_(x_))
    y_jit = f(op, x)
    y_expected = A @ x
    assert np.allclose(np.asarray(y_jit), np.asarray(y_expected))


def test_rapply_uses_conjugate_transpose_for_complex_and_is_jittable():
    """Ensure complex rapply uses conjugate transpose and is JIT-friendly."""
    jax, jnp = _jax_import()
    JaxOps = _jax_ops_import()

    ops = JaxOps()
    ctx = BackendContext(
        ops=ops,
        dtype=np.complex128,
        allow_sparse=True,
        enable_checks=True,
    )

    dom_n = 6
    cod_n = 4
    dom = DenseVectorSpace(ctx=ctx, shape=(dom_n,), n=dom_n)
    cod = DenseVectorSpace(ctx=ctx, shape=(cod_n,), n=cod_n)

    key = jax.random.PRNGKey(5)
    kA, ky = jax.random.split(key, 2)

    A = jax.random.normal(kA, (cod_n, dom_n), dtype=jnp.float64) + 1j * jax.random.normal(
        kA, (cod_n, dom_n), dtype=jnp.float64
    )
    y = jax.random.normal(ky, (cod_n,), dtype=jnp.float64) + 1j * jax.random.normal(
        ky, (cod_n,), dtype=jnp.float64
    )

    op = DenseArrayLinOp(dom=dom, cod=cod, A=A)

    x = op.rapply(y)
    x_expected = A.T.conj() @ y
    assert np.allclose(np.asarray(x), np.asarray(x_expected))

    f = jax.jit(lambda op_, y_: op_.rapply(y_))
    x_jit = f(op, y)
    assert np.allclose(np.asarray(x_jit), np.asarray(x_expected))


def test_adjoint_property_complex_and_is_jittable():
    """Verify adjoint property holds in eager and JIT for complex data."""
    jax, jnp = _jax_import()
    JaxOps = _jax_ops_import()

    ops = JaxOps()
    ctx = BackendContext(
        ops=ops,
        dtype=np.complex128,
        allow_sparse=True,
        enable_checks=True,
    )

    dom_n = 6
    cod_n = 4
    dom = DenseVectorSpace(ctx=ctx, shape=(dom_n,), n=dom_n)
    cod = DenseVectorSpace(ctx=ctx, shape=(cod_n,), n=cod_n)

    key = jax.random.PRNGKey(6)
    kA, kx, ky = jax.random.split(key, 3)

    A = jax.random.normal(kA, (cod_n, dom_n), dtype=jnp.float64) + 1j * jax.random.normal(
        kA, (cod_n, dom_n), dtype=jnp.float64
    )
    x = jax.random.normal(kx, (dom_n,), dtype=jnp.float64) + 1j * jax.random.normal(
        kx, (dom_n,), dtype=jnp.float64
    )
    y = jax.random.normal(ky, (cod_n,), dtype=jnp.float64) + 1j * jax.random.normal(
        ky, (cod_n,), dtype=jnp.float64
    )

    op = DenseArrayLinOp(dom=dom, cod=cod, A=A)

    # Non-jit check
    lhs = cod.inner(op.apply(x), y)      # <Ax, y>
    rhs = dom.inner(x, op.rapply(y))     # <x, A^* y>
    assert np.allclose(np.asarray(lhs), np.asarray(rhs))

    # Jit check (compute both sides inside jit)
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
    key = jax.random.PRNGKey(7)
    kA, kx = jax.random.split(key, 2)

    A = jax.random.normal(kA, (cod.n, dom.n), dtype=jnp.float64)
    op = DenseArrayLinOp(dom=dom, cod=cod, A=A)

    x_bad = jax.random.normal(kx, (dom.n + 1,), dtype=jnp.float64)
    with pytest.raises(TypeError, match="Expected shape"):
        op.apply(x_bad)


def test_codomain_check_enabled(dom_cod):
    """Ensure codomain membership checks are enforced."""
    jax, jnp = _jax_import()

    dom, cod = dom_cod
    key = jax.random.PRNGKey(8)
    kA, ky = jax.random.split(key, 2)

    A = jax.random.normal(kA, (cod.n, dom.n), dtype=jnp.float64)
    op = DenseArrayLinOp(dom=dom, cod=cod, A=A)

    y_bad = jax.random.normal(ky, (cod.n + 2,), dtype=jnp.float64)
    with pytest.raises(TypeError, match="Expected shape"):
        op.rapply(y_bad)
