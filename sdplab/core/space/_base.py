from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple

from ..backend import BackendContext
from ..types import DenseArray


@dataclass
class Space(ABC):
    """
    Abstract Space.

    A Space owns the *geometry* (inner product, norm) and the basic linear
    structure (add/scale/axpy) for its elements.

    Solvers should use only this API.
    """

    ctx: BackendContext
    shape: Tuple[int, ...]  # Canonical dense shape of an element in this space

    @abstractmethod
    def _check_member(self, x: Any) -> None:
        """
        Raise if `x` is not a valid element of this space.

        Typical checks:
          - x.space is self (if your elements carry a .space)
          - backend family consistency (via ctx)
          - representation is supported
          - shape/structure constraints (Hermitian, block sizes, etc.)
        """

    def check_member(self, x: Any) -> None:
        if self.ctx.enable_checks:
            self._check_member(x)

    @abstractmethod
    def zeros(self) -> Any:
        """Return the additive identity in the requested representation."""

    @abstractmethod
    def add(self, x: Any, y: Any) -> Any:
        """Return x + y."""

    @abstractmethod
    def scale(self, a: Any, x: Any) -> Any:
        """Return a * x."""

    def axpy(self, a: Any, x: Any, y: Any) -> Any:
        """Return a*x + y."""
        return self.add(self.scale(a, x), y)

    @abstractmethod
    def inner(self, x: Any, y: Any) -> Any:
        """
        Inner product ⟨x, y⟩ for elements of this space.
        """

    def norm(self, x: Any) -> Any:
        """Induced norm ||x|| = sqrt(real(⟨x,x⟩)). Override if you can do better."""
        v = self.ctx.ops.real(self.inner(x, x))
        return self.ctx.ops.sqrt(v)

    @abstractmethod
    def eigh(self, x: Any, k: int = None) -> Any:
        """Eigendecomposition of x (if applicable).)"""

    @abstractmethod
    def flatten(self, x: Any) -> DenseArray:
        """
        Return a dense 1D coordinate vector (backend-native dense array).

        If a representation forbids materialization, raise a policy/capability error.
        """

    @abstractmethod
    def unflatten(self, v: DenseArray) -> Any:
        """Inverse of flatten; returns an element in the requested representation."""
        raise NotImplementedError
