from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from spacecore import (
    ArrayLike, DenseArray, BackendOps, Context, Space
)
from spacecore.linop import LinOp
from spacecore._contextual import ContextBound, ctx_manager

from ...sdp.variables import SDPPrimal, SDPDual


@dataclass(init=False)
class SDPProblem(ContextBound):
    def __init__(self,
                 C: ArrayLike,
                 A: LinOp,
                 b: ArrayLike,
                 ctx: Context | str | None = None,
                 ):
        ctx = ctx_manager.resolve_context_priority(ctx, A)
        super(SDPProblem, self).__init__(ctx)

        self.A = A.convert(ctx)
        self.A.dom.check_member(C)
        self.A.cod.check_member(b)
        self.C = C
        self.b = b

    @property
    def dom(self) -> Space:
        return self.A.dom

    @property
    def cod(self) -> Space:
        return self.A.cod

    @abstractmethod
    def primal_objective(self, primal: SDPPrimal) -> float:
        raise NotImplementedError()

    def dual_objective(self, dual: SDPDual) -> float:
        return self.ops.real(self.cod.inner(self.b, dual.y))

    def A_apply(self, primal: SDPPrimal) -> SDPDual:
        return self.dual_from_array(self.A.apply(primal.X))

    def AT_apply(self, dual: SDPDual) -> SDPPrimal:
        return self.primal_from_array(self.A.rapply(dual.y))

    def dual_from_array(self, array: ArrayLike) -> SDPDual:
        return SDPDual(self.cod, array)

    def primal_from_array(self, array: ArrayLike) -> SDPPrimal:
        return SDPPrimal(self.dom, array)

    def primal_from_eigendecomp(self, eigvals: DenseArray, eigvecs: DenseArray) -> SDPPrimal:
        raise NotImplementedError()

    @abstractmethod
    def dual_constr_eig_decomp(self, dual: SDPDual, k: int = None) -> tuple[DenseArray, DenseArray]:
        """Eigendecomposition of A.T @ dual - C."""
        raise NotImplementedError()
