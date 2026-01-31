from dataclasses import dataclass, field

from qotlib.core import (
    DenseArray, LowRankMatrix, BackendContext, DenseHermitianMatrixSpace
)
from qotlib.sdp import SDPProblem, SDPDenseProblem, SDPDual, SDPPrimal
from ._variables import QOTDual, QOTPrimal
from ._constraint_op import QOTConstraintOp


@dataclass(init=False)
class QOTProblem(SDPProblem):
    d: int
    N: int

    def __init__(self, ctx: BackendContext, C: DenseArray, marginals: DenseArray, *, d: int, N: int, atol: float = 0., rtol: float = 0., enforce_hermitian: bool = True):
        dom = DenseHermitianMatrixSpace(ctx, d ** N, atol=atol, rtol=rtol, enforce_hermitian=enforce_hermitian)
        A = QOTConstraintOp(dom, d=d, N=N)
        super(QOTProblem, self).__init__(C, A, marginals)
        self.d = d
        self.N = N

    def primal_objective(self, primal: QOTPrimal) -> float:
        return self.A.dom.ctx.ops.real(self.A.dom.inner(self.C, primal.X))

    def dual_from_array(self, y: DenseArray) -> QOTDual:
        return QOTDual(self.A.cod, y)

    def primal_from_array(self, X: DenseArray) -> QOTPrimal:
        return QOTPrimal(self.A.dom, X, d=self.d, N=self.N)

    def primal_from_eigendecomp(self, eigvals: DenseArray, eigvecs: DenseArray) -> QOTPrimal:
        max_rank = eigvals.shape[0]
        X = LowRankMatrix(ctx=self.A.dom.ctx, max_rank=max_rank, eigvals=eigvals, eigvecs=eigvecs).to_dense()
        return QOTPrimal(self.A.dom, X, d=self.d, N=self.N)

    def dual_constr_eig_decomp(self, dual: QOTDual, k: int = None) -> tuple[DenseArray, DenseArray]:
        """Eigendecomposition of A.T @ dual - C."""
        lhs = self.A.rapply(dual.y) - self.C
        return self.A.dom.eigh(lhs)
