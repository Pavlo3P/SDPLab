from __future__ import annotations

from typing import Any, Generic, TypeVar
from dataclasses import dataclass

from qotlib.core import ArrayLike, Space, jax_pytree_class
from ._base import SDPVar, Array

@jax_pytree_class
@dataclass
class SDPPrimal(SDPVar[Array]):
    space: Space
    X: Array

    @property
    def val(self) -> Array:
        return self.X

    def _new_like(self, new_val: Array) -> "SDPPrimal[Array]":
        return SDPPrimal(self.space, new_val)

    def eigh(self, k: int = None) -> tuple[Array, Array]:
        return self.space.eigh(self.X, k)

    def tree_flatten(self):
        return (self.X,), (self.space,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (X,) = children
        (space,) = aux
        return cls(space, X)
