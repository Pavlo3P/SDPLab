from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

from ._dtype import DType


@runtime_checkable
class ArrayLike(Protocol):
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def dtype(self) -> DType: ...

    def conj(self) -> "ArrayLike": ...
    @property
    def T(self) -> "ArrayLike": ...
