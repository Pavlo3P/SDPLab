from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from ..space import Space

Domain = TypeVar('Domain', bound=Space)
Codomain = TypeVar('Codomain', bound=Space)

@dataclass(slots=True)
class LinOp(ABC, Generic[Domain, Codomain]):
    """
    Minimal linear operator (morphism) between two spaces.

    This class is intentionally small:
      - no matrix semantics
      - no arithmetic
      - no storage assumptions

    Its sole purpose is to represent a linear map
        A : dom -> cod
    with access to both forward and adjoint actions.
    """

    dom: Domain
    cod: Codomain

    # ------------------------------------------------------------------
    # Core linear actions
    # ------------------------------------------------------------------

    def _check_backends(self):
        if type(self.dom.ctx.ops) is not type(self.cod.ctx.ops):
            raise ValueError('Domain and codomain backends are not compatible.')

    @abstractmethod
    def apply(self, x: Any) -> Any:
        """
        Forward application: y = A x

        Contract:
          - x is an element of self.dom
          - return value is an element of self.cod
        """
        raise NotImplementedError

    @abstractmethod
    def rapply(self, y: Any) -> Any:
        """
        Adjoint application: x = A^* y

        Contract:
          - y is an element of self.cod
          - return value is an element of self.dom
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def __call__(self, x: Any) -> Any:
        return self.apply(x)

    # ------------------------------------------------------------------
    # Optional safety checks (can be disabled for performance)
    # ------------------------------------------------------------------

    def assert_domain(self, x: Any) -> None:
        self.dom.check_member(x)

    def assert_codomain(self, y: Any) -> None:
        self.cod.check_member(y)
