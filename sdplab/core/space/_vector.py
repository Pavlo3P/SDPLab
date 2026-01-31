from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple

from ._base import Space
from ..types import DenseArray


@dataclass
class DenseVectorSpace(Space):
    """
    Dense vector space R^n or C^n.

    Elements:
      - backend-native dense arrays (NumPy or JAX)
      - canonical shape is (n,)

    Geometry:
      - Euclidean / ℓ2 inner product
            ⟨x, y⟩ = vdot(x, y).
    """

    shape: Tuple[int, ...] = field(init=False)
    n: int

    def __post_init__(self):
        if self.n <= 0:
            raise ValueError("n must be positive.")
        self.shape = (self.n,)

    def _check_member(self, x: Any) -> None:
        X = self.ctx.assert_dense(x)
        if tuple(getattr(X, "shape", ())) != self.shape:
            raise TypeError(f"Expected shape {self.shape}, got {getattr(X, 'shape', None)}")

    def zeros(self) -> DenseArray:
        return self.ctx.ops.zeros(self.shape, dtype=self.ctx.dtype)

    def add(self, x: Any, y: Any) -> DenseArray:
        self.check_member(x)
        self.check_member(y)
        return x + y

    def scale(self, a: Any, x: Any) -> DenseArray:
        self.check_member(x)
        return a * x

    def inner(self, x: Any, y: Any) -> Any:
        self.check_member(x)
        self.check_member(y)
        ops = self.ctx.ops
        return ops.vdot(x, y)

    def eigh(self, x: Any, k: int = None) -> Any:
        raise TypeError(
            f"{type(self).__name__}.eigh is not defined for vector spaces."
        )

    def flatten(self, x: Any) -> DenseArray:
        self.check_member(x)
        return x

    def unflatten(self, v: DenseArray) -> DenseArray:
        V = self.ctx.assert_dense(v)
        return self.ctx.ops.reshape(V, self.shape)
