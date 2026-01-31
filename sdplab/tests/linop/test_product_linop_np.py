from __future__ import annotations

import numpy as np
import pytest

from qotlib.core.backend import BackendContext
from qotlib.core.backend.numpy import NumpyOps
from qotlib.core.linop._dense import DenseArrayLinOp
from qotlib.core.space import DenseVectorSpace, ProductSpace

from qotlib.core.linop import BlockDiagonalLinOp, SumToSingleLinOp, StackedLinOp


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
def spaces(np_ctx):
    # domain blocks
    x1 = DenseVectorSpace(ctx=np_ctx, shape=(5,), n=5)
    x2 = DenseVectorSpace(ctx=np_ctx, shape=(3,), n=3)

    # codomain blocks
    y1 = DenseVectorSpace(ctx=np_ctx, shape=(4,), n=4)
    y2 = DenseVectorSpace(ctx=np_ctx, shape=(2,), n=2)

    # single spaces
    x = DenseVectorSpace(ctx=np_ctx, shape=(6,), n=6)
    y = DenseVectorSpace(ctx=np_ctx, shape=(7,), n=7)

    return x1, x2, y1, y2, x, y


# ---------------------------------------------------------------------
# BlockDiagonalLinOp tests:  (X1×X2) -> (Y1×Y2)
# ---------------------------------------------------------------------
def test_blockdiag_apply_rapply_correctness(np_ctx, spaces):
    x1, x2, y1, y2, _, _ = spaces
    rng = np.random.default_rng(0)

    dom = ProductSpace(spaces=(x1, x2))
    cod = ProductSpace(spaces=(y1, y2))

    A1 = rng.normal(size=(y1.n, x1.n))
    A2 = rng.normal(size=(y2.n, x2.n))

    op1 = DenseArrayLinOp(dom=x1, cod=y1, A=A1)
    op2 = DenseArrayLinOp(dom=x2, cod=y2, A=A2)

    A = BlockDiagonalLinOp(dom=dom, cod=cod, ops=(op1, op2))

    x = (rng.normal(size=(x1.n,)), rng.normal(size=(x2.n,)))
    y = A.apply(x)
    assert y[0].shape == (y1.n,)
    assert y[1].shape == (y2.n,)
    assert np.allclose(y[0], A1 @ x[0])
    assert np.allclose(y[1], A2 @ x[1])

    z = (rng.normal(size=(y1.n,)), rng.normal(size=(y2.n,)))
    xt = A.rapply(z)
    assert xt[0].shape == (x1.n,)
    assert xt[1].shape == (x2.n,)
    assert np.allclose(xt[0], A1.T.conj() @ z[0])
    assert np.allclose(xt[1], A2.T.conj() @ z[1])


def test_blockdiag_adjoint_property(np_ctx, spaces):
    x1, x2, y1, y2, _, _ = spaces
    rng = np.random.default_rng(1)

    dom = ProductSpace(spaces=(x1, x2))
    cod = ProductSpace(spaces=(y1, y2))

    A1 = rng.normal(size=(y1.n, x1.n))
    A2 = rng.normal(size=(y2.n, x2.n))
    op1 = DenseArrayLinOp(dom=x1, cod=y1, A=A1)
    op2 = DenseArrayLinOp(dom=x2, cod=y2, A=A2)
    A = BlockDiagonalLinOp(dom=dom, cod=cod, ops=(op1, op2))

    x = (rng.normal(size=(x1.n,)), rng.normal(size=(x2.n,)))
    y = (rng.normal(size=(y1.n,)), rng.normal(size=(y2.n,)))

    lhs = cod.inner(A.apply(x), y)      # <Ax, y>
    rhs = dom.inner(x, A.rapply(y))     # <x, A^*y>
    assert np.allclose(lhs, rhs)


def test_blockdiag_construct_requires_product_spaces(np_ctx, spaces):
    x1, x2, y1, y2, _, _ = spaces
    rng = np.random.default_rng(2)

    # wrong dom/cod types (not ProductSpace)
    A1 = rng.normal(size=(y1.n, x1.n))
    op1 = DenseArrayLinOp(dom=x1, cod=y1, A=A1)

    with pytest.raises(TypeError, match="expects dom and cod to be ProductSpace"):
        BlockDiagonalLinOp(dom=x1, cod=y1, ops=(op1,))  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# SumToSingleLinOp tests:  (X1×X2) -> Y
# ---------------------------------------------------------------------
def test_sum_to_single_apply_rapply_correctness(np_ctx, spaces):
    x1, x2, _, _, _, y = spaces
    rng = np.random.default_rng(3)

    dom = ProductSpace(spaces=(x1, x2))
    cod = y

    A1 = rng.normal(size=(cod.n, x1.n))
    A2 = rng.normal(size=(cod.n, x2.n))
    op1 = DenseArrayLinOp(dom=x1, cod=cod, A=A1)
    op2 = DenseArrayLinOp(dom=x2, cod=cod, A=A2)

    A = SumToSingleLinOp(dom=dom, cod=cod, ops=(op1, op2))

    x = (rng.normal(size=(x1.n,)), rng.normal(size=(x2.n,)))
    out = A.apply(x)
    assert out.shape == (cod.n,)
    assert np.allclose(out, A1 @ x[0] + A2 @ x[1])

    u = rng.normal(size=(cod.n,))
    xt = A.rapply(u)
    assert xt[0].shape == (x1.n,)
    assert xt[1].shape == (x2.n,)
    assert np.allclose(xt[0], A1.T.conj() @ u)
    assert np.allclose(xt[1], A2.T.conj() @ u)


def test_sum_to_single_adjoint_property(np_ctx, spaces):
    x1, x2, _, _, _, y = spaces
    rng = np.random.default_rng(4)

    dom = ProductSpace(spaces=(x1, x2))
    cod = y

    A1 = rng.normal(size=(cod.n, x1.n))
    A2 = rng.normal(size=(cod.n, x2.n))
    op1 = DenseArrayLinOp(dom=x1, cod=cod, A=A1)
    op2 = DenseArrayLinOp(dom=x2, cod=cod, A=A2)
    A = SumToSingleLinOp(dom=dom, cod=cod, ops=(op1, op2))

    x = (rng.normal(size=(x1.n,)), rng.normal(size=(x2.n,)))
    yv = rng.normal(size=(cod.n,))

    lhs = cod.inner(A.apply(x), yv)
    rhs = dom.inner(x, A.rapply(yv))
    assert np.allclose(lhs, rhs)


def test_sum_to_single_construct_requires_product_domain(np_ctx, spaces):
    x1, _, _, _, _, y = spaces
    rng = np.random.default_rng(5)

    A1 = rng.normal(size=(y.n, x1.n))
    op1 = DenseArrayLinOp(dom=x1, cod=y, A=A1)

    with pytest.raises(TypeError, match="expects dom to be ProductSpace"):
        SumToSingleLinOp(dom=x1, cod=y, ops=(op1,))  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# StackedLinOp tests:  X -> (Y1×Y2)
# ---------------------------------------------------------------------
def test_stacked_apply_rapply_correctness(np_ctx, spaces):
    _, _, y1, y2, x, _ = spaces
    rng = np.random.default_rng(6)

    dom = x
    cod = ProductSpace(spaces=(y1, y2))

    A1 = rng.normal(size=(y1.n, x.n))
    A2 = rng.normal(size=(y2.n, x.n))
    op1 = DenseArrayLinOp(dom=dom, cod=y1, A=A1)
    op2 = DenseArrayLinOp(dom=dom, cod=y2, A=A2)

    A = StackedLinOp(dom=dom, cod=cod, ops=(op1, op2))

    v = rng.normal(size=(x.n,))
    out = A.apply(v)
    assert out[0].shape == (y1.n,)
    assert out[1].shape == (y2.n,)
    assert np.allclose(out[0], A1 @ v)
    assert np.allclose(out[1], A2 @ v)

    w = (rng.normal(size=(y1.n,)), rng.normal(size=(y2.n,)))
    vt = A.rapply(w)
    assert vt.shape == (x.n,)
    assert np.allclose(vt, A1.T.conj() @ w[0] + A2.T.conj() @ w[1])


def test_stacked_adjoint_property(np_ctx, spaces):
    _, _, y1, y2, x, _ = spaces
    rng = np.random.default_rng(7)

    dom = x
    cod = ProductSpace(spaces=(y1, y2))

    A1 = rng.normal(size=(y1.n, x.n))
    A2 = rng.normal(size=(y2.n, x.n))
    op1 = DenseArrayLinOp(dom=dom, cod=y1, A=A1)
    op2 = DenseArrayLinOp(dom=dom, cod=y2, A=A2)
    A = StackedLinOp(dom=dom, cod=cod, ops=(op1, op2))

    v = rng.normal(size=(x.n,))
    w = (rng.normal(size=(y1.n,)), rng.normal(size=(y2.n,)))

    lhs = cod.inner(A.apply(v), w)
    rhs = dom.inner(v, A.rapply(w))
    assert np.allclose(lhs, rhs)


def test_stacked_construct_requires_product_codomain(np_ctx, spaces):
    _, _, y1, _, x, _ = spaces
    rng = np.random.default_rng(8)

    A1 = rng.normal(size=(y1.n, x.n))
    op1 = DenseArrayLinOp(dom=x, cod=y1, A=A1)

    with pytest.raises(TypeError, match="expects cod to be ProductSpace"):
        StackedLinOp(dom=x, cod=y1, ops=(op1,))  # type: ignore[arg-type]
