from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from math import prod

from ._base import LinOp
from ..types import DenseArray, SparseArray
from ..backend import jax_pytree_class


@jax_pytree_class
@dataclass(slots=True)
class SparseArrayLinOp(LinOp):
    """
    Sparse linear operator implementing the tensor map A : dom -> cod where
    conceptually A has shape cod.shape + dom.shape, but stored as a 2D sparse matrix:

        A2.shape == (prod(cod.shape), prod(dom.shape))

    apply:  y = A ⋅ x  (contract over dom axes)
    rapply: x = A^* ⋅ y  (contract over cod axes)
    """

    A: SparseArray

    def __post_init__(self) -> None:
        self._check_backends()
        ctx = self.dom.ctx

        A = ctx.assert_sparse(self.A)
        object.__setattr__(self, "A", A)

        # Shape validation
        expected = tuple(self.cod.shape) + tuple(self.dom.shape)
        if tuple(A.shape) != expected:
            raise TypeError(f"Expected A.shape == cod.shape + dom.shape == {expected}, got {A.shape}")

    def apply(self, x: DenseArray) -> DenseArray:
        """
        Forward action: y = A ⋅ x with y in cod.shape.

        x must have shape dom.shape (dense).
        """
        ctx = self.dom.ctx
        self.assert_domain(x)

        m = prod(self.cod.shape)
        n = prod(self.dom.shape)

        A = self.A.reshape((m, n))
        x1 = x.reshape((n,))
        y1 = ctx.ops.sparse_matmul(A, x1)   # (m,)
        y = self.cod.unflatten(y1)
        return y

    def rapply(self, y: DenseArray) -> DenseArray:
        """
        Adjoint action: x = A^* ⋅ y with x in dom.shape.

        y must have shape cod.shape (dense).
        """
        ctx = self.dom.ctx
        self.assert_codomain(y)

        m = prod(self.cod.shape)
        n = prod(self.dom.shape)

        y1 = y.reshape((m,))

        AT = self.A.reshape((m, n)).T
        x1 = ctx.ops.sparse_matmul(AT, y1.conj()).conj()

        x = self.dom.unflatten(x1)
        return x

    def tree_flatten(self):
        aux = (self.dom, self.cod)
        children = (self.A,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        dom, cod = aux
        A = children[0]
        return cls(dom, cod, A)
