from __future__ import annotations

from dataclasses import dataclass
from math import prod

from ._base import LinOp
from ..types import DenseArray
from ..backend import jax_pytree_class


@jax_pytree_class
@dataclass(slots=True)
class DenseArrayLinOp(LinOp):
    """
    Dense linear operator defined by an array A with shape:

        A.shape == cod.shape + dom.shape

    apply:  y = A ⋅ x  (contract over dom axes)
    rapply: x = A^* ⋅ y  (contract over cod axes)
    """

    A: DenseArray

    def __post_init__(self) -> None:
        self._check_backends()
        A = self.dom.ctx.assert_dense(self.A)
        object.__setattr__(self, "A", A)

        expected = tuple(self.cod.shape) + tuple(self.dom.shape)
        if tuple(A.shape) != expected:
            raise TypeError(f"Expected A.shape == cod.shape + dom.shape == {expected}, got {A.shape}")

    def apply(self, x: DenseArray) -> DenseArray:
        """
        Forward action: y = A ⋅ x with y in cod.shape.

        No membership checks here; validate once outside loops.
        """
        self.assert_domain(x)

        m = prod(self.cod.shape)
        n = prod(self.dom.shape)

        A = self.A.reshape((m, n))
        x1 = x.reshape((n,))
        y1 = A @ x1
        y  = self.cod.unflatten(y1)
        return y

    def rapply(self, y: DenseArray) -> DenseArray:
        """
        Adjoint action: x = A^* ⋅ y with x in dom.shape.

        For complex A, uses conjugate-transpose of the 2D reshaped matrix.
        """
        self.assert_codomain(y)

        m = prod(self.cod.shape)
        n = prod(self.dom.shape)

        A2 = self.A.reshape((m, n))
        y1 = y.reshape((m,))
        x1 = A2.T.conj() @ y1
        x  = self.dom.unflatten(x1)
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
