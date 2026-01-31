from __future__ import annotations

import numpy as np
import pytest

from qotlib.core.linop._dense import DenseArrayLinOp
from qotlib.core.backend import BackendContext
from qotlib.core.backend.numpy import NumpyOps
from qotlib.core.space import DenseVectorSpace


@pytest.fixture()
def np_ctx() -> BackendContext:
    ops = NumpyOps()
    return BackendContext(
        ops=ops,
        dtype=np.float64,
        allow_sparse=True,
        enable_checks=True,
    )


@pytest.fixture()
def dom_cod(np_ctx):
    # Space requires explicit shape; DenseVectorSpace also stores n.
    dom_n = 6
    cod_n = 4
    dom = DenseVectorSpace(ctx=np_ctx, shape=(dom_n,), n=dom_n)
    cod = DenseVectorSpace(ctx=np_ctx, shape=(cod_n,), n=cod_n)
    return dom, cod


# -------------------------
# Construction tests
# -------------------------

def test_dense_linop_rejects_different_backend_types(np_ctx):
    # New semantics: compatibility is by ops TYPE, not instance identity.
    # So we must test *different types*.
    from qotlib.core.backend.numpy import NumpyOps
    from qotlib.core.backend import BackendContext
    from qotlib.core.space import DenseVectorSpace
    from qotlib.core.linop._dense import DenseArrayLinOp

    class DummyNumpyOps(NumpyOps):
        pass

    x = DenseVectorSpace(ctx=np_ctx, shape=(5,), n=5)

    other_ctx = BackendContext(
        ops=DummyNumpyOps(),  # different type
        dtype=np.float64,
        allow_sparse=True,
        enable_checks=True,
    )
    y = DenseVectorSpace(ctx=other_ctx, shape=(4,), n=4)

    A = np.random.default_rng(0).normal(size=(y.n, x.n))

    with pytest.raises(ValueError, match="backends are not compatible"):
        _ = DenseArrayLinOp(dom=x, cod=y, A=A)



def test_construct_rejects_non_dense(dom_cod):
    """Reject non-dense operator arrays."""
    dom, cod = dom_cod
    A = [[0.0] * dom.n for _ in range(cod.n)]  # Python list, not ndarray
    with pytest.raises(TypeError):
        DenseArrayLinOp(dom=dom, cod=cod, A=A)  # type: ignore[arg-type]


def test_construct_rejects_wrong_A_shape(dom_cod):
    """Reject operator tensors with invalid shape."""
    dom, cod = dom_cod
    A = np.zeros((cod.n, dom.n + 1), dtype=np.float64)
    with pytest.raises(TypeError, match=r"Expected A\.shape == cod\.shape \+ dom\.shape"):
        DenseArrayLinOp(dom=dom, cod=cod, A=A)


# -------------------------
# apply / rapply correctness
# -------------------------

def test_apply_matches_matrix_form(dom_cod):
    """Check apply matches explicit matrix multiplication."""
    dom, cod = dom_cod
    rng = np.random.default_rng(0)

    A = rng.normal(size=(cod.n, dom.n))
    op = DenseArrayLinOp(dom=dom, cod=cod, A=A)

    x = rng.normal(size=(dom.n,))
    y = op.apply(x)

    assert y.shape == (cod.n,)
    assert np.allclose(y, A @ x)


def test_rapply_matches_adjoint_real(dom_cod):
    """Check rapply matches adjoint for real matrices."""
    dom, cod = dom_cod
    rng = np.random.default_rng(1)

    A = rng.normal(size=(cod.n, dom.n))
    op = DenseArrayLinOp(dom=dom, cod=cod, A=A)

    y = rng.normal(size=(cod.n,))
    x = op.rapply(y)

    assert x.shape == (dom.n,)
    assert np.allclose(x, A.T.conj() @ y)


def test_rapply_uses_conjugate_transpose_for_complex():
    """Ensure rapply uses conjugate transpose for complex matrices."""
    ops = NumpyOps()
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

    rng = np.random.default_rng(2)
    A = rng.normal(size=(cod_n, dom_n)) + 1j * rng.normal(size=(cod_n, dom_n))
    op = DenseArrayLinOp(dom=dom, cod=cod, A=A)

    y = rng.normal(size=(cod_n,)) + 1j * rng.normal(size=(cod_n,))
    x = op.rapply(y)

    assert np.allclose(x, A.T.conj() @ y)


def test_adjoint_property_complex():
    """Verify adjoint property <Ax,y> = <x,A*y> for complex case."""
    ops = NumpyOps()
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

    rng = np.random.default_rng(3)
    A = rng.normal(size=(cod_n, dom_n)) + 1j * rng.normal(size=(cod_n, dom_n))
    op = DenseArrayLinOp(dom=dom, cod=cod, A=A)

    x = rng.normal(size=(dom_n,)) + 1j * rng.normal(size=(dom_n,))
    y = rng.normal(size=(cod_n,)) + 1j * rng.normal(size=(cod_n,))

    lhs = cod.inner(op.apply(x), y)   # <Ax, y>
    rhs = dom.inner(x, op.rapply(y))  # <x, A^* y>
    assert np.allclose(lhs, rhs)


# -------------------------
# API behavior + checks
# -------------------------

def test_call_delegates_to_apply(dom_cod):
    """Ensure __call__ delegates to apply."""
    dom, cod = dom_cod
    rng = np.random.default_rng(4)

    A = rng.normal(size=(cod.n, dom.n))
    op = DenseArrayLinOp(dom=dom, cod=cod, A=A)

    x = rng.normal(size=(dom.n,))
    assert np.allclose(op(x), op.apply(x))


def test_domain_check_enabled(dom_cod):
    """Ensure domain membership checks are enforced."""
    dom, cod = dom_cod
    rng = np.random.default_rng(5)

    A = rng.normal(size=(cod.n, dom.n))
    op = DenseArrayLinOp(dom=dom, cod=cod, A=A)

    x_bad = rng.normal(size=(dom.n + 1,))
    with pytest.raises(TypeError, match="Expected shape"):
        op.apply(x_bad)


def test_codomain_check_enabled(dom_cod):
    """Ensure codomain membership checks are enforced."""
    dom, cod = dom_cod
    rng = np.random.default_rng(6)

    A = rng.normal(size=(cod.n, dom.n))
    op = DenseArrayLinOp(dom=dom, cod=cod, A=A)

    y_bad = rng.normal(size=(cod.n + 2,))
    with pytest.raises(TypeError, match="Expected shape"):
        op.rapply(y_bad)
