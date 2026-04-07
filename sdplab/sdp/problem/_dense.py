from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

from spacecore import Context, DenseArray
from spacecore.linop import LinOp
from ..variables import SDPPrimal, SDPDual
from ._base import SDPProblem


@dataclass(init=False)
class SDPDenseProblem(SDPProblem):
    tau: Optional[float] | None = None
    def __init__(self,
                 C: DenseArray,
                 A: LinOp,
                 b: DenseArray,
                 tau: float | None = None,
                 ctx: Context | str | None = None,
                 ):
        super(SDPDenseProblem, self).__init__(C, A, b, ctx)
        self.tau = float(tau) if tau is not None else None

    def primal_objective(self, primal: SDPPrimal) -> float:
        return self.A.dom.ctx.ops.real(self.A.dom.inner(self.C, primal.X))

    def dual_constr_eig_decomp(self, dual: SDPDual, k: int = None) -> tuple[DenseArray, DenseArray]:
        """Eigendecomposition of A.T @ dual - C."""
        lhs = self.A.rapply(dual.y) - self.C
        return self.A.dom.eigh(lhs)

    def _convert(self, new_ctx: Context) -> SDPDenseProblem:
        new_C = new_ctx.asarray(self.C)
        new_b = new_ctx.asarray(self.b)
        return SDPDenseProblem(new_C, self.A, new_b, self.tau, new_ctx)