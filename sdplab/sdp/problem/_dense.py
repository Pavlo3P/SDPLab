from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

from qotlib.core import DenseArray, LinOp, jax_pytree_class
from ..variables import SDPPrimal, SDPDual
from ._base import SDPProblem


@dataclass
class SDPDenseProblem(SDPProblem):
    tau: Optional[float] | None = None

    def primal_objective(self, primal: SDPPrimal[DenseArray]) -> float:
        return self.A.dom.ctx.ops.real(self.A.dom.inner(self.C, primal.X))

    def dual_constr_eig_decomp(self, dual: SDPDual, k: int = None) -> tuple[DenseArray, DenseArray]:
        """Eigendecomposition of A.T @ dual - C."""
        lhs = self.A.rapply(dual.y) - self.C
        return self.A.dom.eigh(lhs)
