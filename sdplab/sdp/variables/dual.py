from __future__ import annotations

from dataclasses import dataclass

from qotlib.core import Space, jax_pytree_class
from ._base import SDPVar, Array


@jax_pytree_class
@dataclass
class SDPDual(SDPVar[Array]):
    space: Space
    y: Array

    @property
    def val(self) -> Array:
        return self.y

    def _new_like(self, new_val: Array) -> "SDPDual[Array]":
        return SDPDual(self.space, new_val)

    def tree_flatten(self):
        return (self.y,), (self.space,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (y,) = children
        (space,) = aux
        return cls(space, y)
