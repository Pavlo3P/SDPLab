import numpy as np
import pytest

from qotlib.core.backend import BackendContext, NumpyOps
from qotlib.core.linop import DenseArrayLinOp, SparseArrayLinOp
from qotlib.core.space import DenseVectorSpace


def _make_spaces():
    ops = NumpyOps()
    ctx = BackendContext(ops=ops, dtype=np.float64)
    dom = DenseVectorSpace(ctx=ctx, shape=(2,), n=2)
    cod = DenseVectorSpace(ctx=ctx, shape=(3,), n=3)
    return ops, dom, cod


def test_dense_array_linop_apply_and_rapply():
    """Check dense linear operator apply/rapply and pytree round-trip."""
    ops, dom, cod = _make_spaces()
    A = ops.asarray([[1.0, 2.0], [0.0, -1.0], [3.0, 4.0]])
    linop = DenseArrayLinOp(dom=dom, cod=cod, A=A)

    x = ops.asarray([1.0, -1.0])
    y = linop.apply(x)
    assert np.allclose(y, A @ x)

    x_back = linop.rapply(y)
    assert np.allclose(x_back, A.T.conj() @ y)

    children, aux = linop.tree_flatten()
    rebuilt = DenseArrayLinOp.tree_unflatten(aux, children)
    assert np.allclose(rebuilt.apply(x), y)


def test_dense_array_linop_shape_validation():
    """Ensure dense linop rejects tensors with invalid shapes."""
    ops, dom, cod = _make_spaces()
    with pytest.raises(TypeError):
        DenseArrayLinOp(dom=dom, cod=cod, A=ops.asarray([[1.0, 2.0]]))


def test_sparse_array_linop_apply_and_rapply():
    """Check sparse linear operator apply/rapply and pytree round-trip."""
    ops, dom, cod = _make_spaces()
    sp = pytest.importorskip("scipy").sparse

    dense_A = np.array([[1.0, 2.0], [0.0, -1.0], [3.0, 4.0]])
    sparse_A = sp.csr_matrix(dense_A)

    linop = SparseArrayLinOp(dom=dom, cod=cod, A=sparse_A)

    x = ops.asarray([1.0, -1.0])
    y = linop.apply(x)
    assert np.allclose(y, dense_A @ x)

    x_back = linop.rapply(y)
    assert np.allclose(x_back, dense_A.T.conj() @ y)

    children, aux = linop.tree_flatten()
    rebuilt = SparseArrayLinOp.tree_unflatten(aux, children)
    assert np.allclose(rebuilt.apply(x), y)
