from __future__ import annotations

from dataclasses import dataclass, field

from qotlib.core import DenseArray, DenseHermitianMatrixSpace, jax_pytree_class
from qotlib.sdp import SDPDual, SDPPrimal
from qotlib.sdp.variables.primal import Array
from ._block_space import BlockMatrixSpace

@jax_pytree_class
@dataclass
class QOTDual(SDPDual[DenseArray]):
    def __post_init__(self):
        if not isinstance(self.space, BlockMatrixSpace):
            raise TypeError(f"`space` must be a `BlockMatrixSpace`, got {type(self.space)}")
        super(QOTDual, self).__post_init__()

    def tree_flatten(self):
        children = (self.y,)
        aux = (self.space,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (space,) = aux
        (y,) = children
        return cls(space, y)

    @property
    def d(self):
        return self.space.shape[1]

    @property
    def N(self):
        return self.space.shape[0]

    @property
    def dim(self) -> int:
        return self.d ** self.N

    def __getitem__(self, k: int) -> DenseArray:
        return self.y[k, :, :]

    def __repr__(self):
        return f"QOTDual(d={self.d}, N={self.N})"

@jax_pytree_class
@dataclass
class QOTPrimal(SDPPrimal[DenseArray]):
    d: int = field(kw_only=True)
    N: int = field(kw_only=True)

    def _new_like(self, new_val: DenseArray) -> "QOTPrimal":
        return QOTPrimal(self.space, new_val, d=self.d, N=self.N)

    def __post_init__(self):
        if not isinstance(self.space, DenseHermitianMatrixSpace):
            raise TypeError(f"`space` must be a `DenseHermitianMatrixSpace`, got {type(self.space)}")

        expected_shape = (self.d ** self.N, self.d ** self.N)
        if self.space.shape != expected_shape:
            raise ValueError(f"Space shape and expected shape do not match: {self.space.shape} != {expected_shape} (d ** N, d ** N).")
        super(QOTPrimal, self).__post_init__()

    @property
    def dim(self) -> int:
        return self.d ** self.N

    def eigh(self, k: int = None) -> tuple[Array, Array]:
        return self.space.eigh(self.X)

    def tree_flatten(self):
        children = (self.X, )
        aux = (self.d, self.N, self.space)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        d, N, space = aux
        X = children[0]
        return cls(space, X, d=d, N=N)

    def __repr__(self):
        return f"QOTPrimal(d={self.d}, N={self.N})"
