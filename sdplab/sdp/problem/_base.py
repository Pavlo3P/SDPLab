from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ...core import (
    ArrayLike, DenseArray, LowRankMatrix,
    LinOp, BackendOps, BackendContext, Space
)
from ...sdp.variables import SDPPrimal, SDPDual


@dataclass
class SDPProblem(ABC):
    C: ArrayLike
    A: LinOp
    b: ArrayLike

    def __post_init__(self):
        if self.C.dtype != self.A.dom.ctx.dtype:
            raise TypeError(f"Cost matrix dtype {self.C.dtype} does not match primal space context dtype (i.e. A.dom.ctx.dtype) {self.A.dom.ctx.dtype}.")
        if self.C.shape != self.A.dom.shape:
            raise ValueError(f"Cost matrix shape {self.C.shape} does not match primal domain shape (i.e. A.dom.shape) {self.A.dom.shape}.")

        if self.b.dtype != self.cod.ctx.dtype:
            raise TypeError(f"Constraint RHS dtype {self.b.dtype} does not match dual space context dtype (i.e. A.cod.ctx.dtype) {self.A.cod.ctx.dtype}.")
        if self.b.shape != self.cod.shape:
            raise ValueError(f"Constraint RHS shape {self.b.shape} does not match dual domain shape (i.e. A.cod.shape) {self.A.cod.shape}.")

    @property
    def dom(self) -> Space:
        return self.A.dom

    @property
    def cod(self) -> Space:
        return self.A.cod

    @property
    def ctx(self) -> BackendContext:
        return self.A.dom.ctx

    @property
    def ops(self) -> BackendOps:
        return self.A.dom.ctx.ops

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
        max_rank = eigvals.shape[0]
        X = LowRankMatrix(ctx=self.ctx, max_rank=max_rank, eigvals=eigvals, eigvecs=eigvecs)
        return SDPPrimal(self.dom, X)

    @abstractmethod
    def dual_constr_eig_decomp(self, dual: SDPDual, k: int = None) -> tuple[DenseArray, DenseArray]:
        """Eigendecomposition of A.T @ dual - C."""
        raise NotImplementedError()
