from __future__ import annotations

from typing import Any, Generic, TypeVar, Tuple
from abc import ABC, abstractmethod

from ...core import ArrayLike, Space, BackendContext, BackendOps

Array = TypeVar("Array", bound=ArrayLike)

class SDPVar(ABC, Generic[Array]):
    space: Space

    @property
    def ops(self) -> BackendOps:
        return self.space.ctx.ops

    @property
    def ctx(self) -> BackendContext:
        return self.space.ctx

    @property
    @abstractmethod
    def val(self) -> Array:
        ...

    def __post_init__(self):
        self.space.check_member(self.val)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.val.shape

    @abstractmethod
    def _new_like(self, new_val: Array) -> "SDPVar":
        ...

    def __add__(self, other: Any):
        if type(other) is not type(self):
            return NotImplemented
        y = self.space.add(self.val, other.val)
        return self._new_like(y)

    def __radd__(self, other: Any):
        # For sum(..., start=0) patterns; treat 0 as neutral if desired.
        if other == 0:
            return self
        return NotImplemented

    def __sub__(self, other: Any):
        if type(other) is not type(self):
            return NotImplemented
        y = self.space.add(self.val, self.space.scale(-1, other.val))
        return self._new_like(y)

    def __neg__(self):
        return self._new_like(self.space.scale(-1, self.val))

    def __mul__(self, a: Any):
        y = self.space.scale(a, self.val)
        return self._new_like(y)

    def __rmul__(self, a: Any):
        y = self.space.scale(a, self.val)
        return self._new_like(y)

    def __truediv__(self, a: Any):
        y = self.space.scale(1 / a, self.val)
        return self._new_like(y)

    def axpy(self, a: Any, x: "SDPVar"):
        if type(x) is not type(self):
            raise TypeError(f"axpy expects {type(self).__name__}, got {type(x).__name__}")
        y = self.space.axpy(a, x.val, self.val)
        return self._new_like(y)

    def inner(self, other: "SDPVar"):
        if type(other) is not type(self):
            raise TypeError(f"inner expects {type(self).__name__}, got {type(other).__name__}")
        return self.space.inner(self.val, other.val)

    def norm(self):
        return self.space.norm(self.val)
