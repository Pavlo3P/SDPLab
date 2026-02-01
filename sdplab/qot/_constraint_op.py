from dataclasses import dataclass, field
from typing import Tuple, Any

from ..core import LinOp, DenseArray, DenseHermitianMatrixSpace, jax_pytree_class
from ..linalg import _compute_ptraces, kron_sum, make_perm
from ._block_space import BlockMatrixSpace


@jax_pytree_class
@dataclass(slots=True)
class QOTConstraintOp(LinOp[DenseHermitianMatrixSpace, BlockMatrixSpace]):
    d: int = field(kw_only=True)
    N: int = field(kw_only=True)
    perms: Tuple[Tuple[int, ...], ...] = field(init=False)
    cod: BlockMatrixSpace = field(init=False)

    def __post_init__(self) -> None:
        dom = self.dom
        d = self.d
        N = self.N
        self.cod = BlockMatrixSpace(dom.ctx, d=d, N=N, atol=dom.atol, rtol=dom.rtol, enforce_hermitian=dom.enforce_hermitian)
        self.perms = tuple(make_perm(i, self.N) for i in range(self.N))

    def apply(self, X: DenseArray) -> DenseArray:
        self.dom.check_member(X)
        return _compute_ptraces(self.dom.ctx, X, d=self.d, N=self.N, perms=self.perms)

    def rapply(self, y: DenseArray) -> Any:
        self.cod.check_member(y)
        return kron_sum(self.cod.ctx, y)

    def tree_flatten(self):
        return (), (self.d, self.N, self.dom)

    @classmethod
    def tree_unflatten(cls, aux, children):
        d, N, dom = aux
        return cls(dom, d=d, N=N)
