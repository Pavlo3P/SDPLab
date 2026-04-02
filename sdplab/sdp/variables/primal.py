from __future__ import annotations

from typing import Tuple
from spacecore import Space, jax_pytree_class, Context, ArrayLike, DenseArray
from ._base import SDPVar

@jax_pytree_class
class SDPPrimal(SDPVar):
    def __init__(
        self,
        space: Space,
        X: ArrayLike,
        ctx: Context | str | None = None,
    ):
        super(SDPPrimal, self).__init__(space, ctx)
        self.space.check_member(X)
        self.X = self.space.ctx.asarray(X)

    @property
    def val(self) -> ArrayLike:
        return self.X

    def _new_like(self, new_val: ArrayLike) -> SDPPrimal:
        return SDPPrimal(self.space, new_val)

    def eigh(self, k: int | None = None) -> Tuple[DenseArray, ArrayLike]:
        return self.space.eigh(self.X, k)

    def tree_flatten(self):
        return (self.X,), (self.space,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (X,) = children
        (space,) = aux
        return cls(space, X)
