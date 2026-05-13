from __future__ import annotations

from typing import Any

from spacecore import DenseArray, HermitianSpace, jax_pytree_class, Context, LinOp

from ._linalg import _compute_ptraces, kron_sum, make_perm
from ._block_space import BlockMatrixSpace


@jax_pytree_class
class QOTConstraintOp(LinOp[HermitianSpace, BlockMatrixSpace]):
    def __init__(self,
                 *,
                 d: int,
                 N: int,
                 atol: float = 0.0,
                 rtol: float = 0.0,
                 enforce_herm: bool = True,
                 ctx: Context | str | None = None
                 ):
        if d <= 0 or type(d) is not int:
            raise ValueError("d must be positive integer.")
        if N <= 0 or type(N) is not int:
            raise ValueError("N must be positive integer.")

        atol = float(atol)
        rtol = float(rtol)
        enforce_herm = bool(enforce_herm)

        dom = HermitianSpace(d ** N, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=ctx)
        cod = BlockMatrixSpace(d=d, N=N, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=ctx)
        super(QOTConstraintOp, self).__init__(dom, cod, ctx)

        self.d = d
        self.N = N
        self.perms = tuple(make_perm(i, self.N) for i in range(self.N))

    def apply(self, X: DenseArray) -> DenseArray:
        self.dom.check_member(X)
        return _compute_ptraces(self.dom.ctx, X, d=self.d, N=self.N, perms=self.perms)

    def rapply(self, y: DenseArray) -> Any:
        self.cod.check_member(y)
        return kron_sum(self.cod.ctx, y)

    def tree_flatten(self):
        aux = (
            self.d,
            self.N,
            self.dom.atol,
            self.dom.rtol,
            self.dom.enforce_herm,
            self.cod.ctx,
        )
        return (), aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        d, N, atol, rtol, enforce_herm, ctx = aux
        return cls(d=d, N=N, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=ctx)

    def _convert(self, new_ctx: Context) -> QOTConstraintOp:
        return QOTConstraintOp(d=self.d, N=self.N, atol=self.dom.atol, rtol=self.dom.rtol, enforce_herm=self.dom.enforce_herm, ctx=new_ctx)
