from dataclasses import dataclass
from typing import Any, Tuple

from ..types import DenseArray
from ..backend import BackendContext, jax_pytree_class


@jax_pytree_class
@dataclass
class LowRankMatrix:
    """
    Low-rank Hermitian matrix representation.

    Represents n-dimensional Hermitian matrix  X = V diag(s) V^\dagger
    whose maximum rank is r.
    """
    ctx: BackendContext
    max_rank: int  # r
    eigvals: DenseArray  # Shape (r,)
    eigvecs: DenseArray  # Shape (n, r)

    def __post_init__(self) -> None:
        if not isinstance(self.max_rank, int) or self.max_rank <= 0:
            raise ValueError("max_rank must be a positive integer.")
        self.check_shapes()
        self.ctx.assert_dense(self.eigvals)
        self.ctx.assert_dense(self.eigvecs)

    def check_shapes(self) -> None:
        r = int(self.max_rank)
        if tuple(self.eigvals.shape) != (r,):
            raise TypeError(f"eigvals must have shape ({r},), got {self.eigvals.shape}")
        if len(self.eigvecs.shape) != 2 or self.eigvecs.shape[1] != r:
            raise TypeError(f"eigvecs must have shape (n, {r}), got {self.eigvecs.shape}")

    @property
    def r(self) -> int:
        return self.max_rank

    @property
    def dim(self) -> int:
        return int(self.eigvecs.shape[0])

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.dim, self.dim)

    @property
    def T(self) -> "LowRankMatrix":
        return LowRankMatrix(
            ctx=self.ctx,
            max_rank=self.max_rank,
            eigvals=self.eigvals,
            eigvecs=self.eigvecs.conj(),
        )

    def conj(self) -> "LowRankMatrix":
        """
        Elementwise complex conjugation of the matrix:
            conj(X) = conj(V) diag(conj(s)) conj(V)^H
        """
        return LowRankMatrix(
            ctx=self.ctx,
            max_rank=self.max_rank,
            eigvals=self.eigvals.conj(),
            eigvecs=self.eigvecs.conj(),
        )

    def to_dense(self) -> DenseArray:
        X = (self.eigvecs * self.eigvals) @ self.eigvecs.T.conj()
        return X

    def matvec(self, x: DenseArray) -> DenseArray:
        u = self.eigvecs.T.conj() @ x
        u = self.eigvals * u
        u = self.eigvecs @ u
        return u

    def inner(self, other: "LowRankMatrix") -> Any:
        ops = self.ctx.ops

        Vx = self.eigvecs  # (rx, n)
        sx = self.eigvals  # (rx,)
        Vy = other.eigvecs  # (ry, n)
        sy = other.eigvals  # (ry,)

        # Gram of row vectors: G_{i,j} = <v_i^x, v_j^y> = v_i^x (v_j^y)^H
        G = Vx.T.conj() @ Vy  # (rx, ry)

        # |G|^2 without forming abs(): elementwise G * conj(G)
        abs2 = G * G.conj()  # (rx, ry)

        # weights: conj(sx_i) * sy_j  (broadcast to (rx, ry))
        W = sx.conj()[:, None] * sy[None, :]

        # scalar result
        return ops.sum(W * abs2)

    def l2_norm(self) -> float:
        ops = self.ctx.ops
        return ops.sqrt(ops.vdot(self.eigvals, self.eigvals))

    def trace(self):
        return self.eigvals.sum()

    def tree_flatten(self):
        children = (self.eigvals, self.eigvecs)
        aux = (self.ctx, self.max_rank)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        ctx, max_rank = aux
        (eigvals, eigvecs) = children
        return cls(ctx, max_rank, eigvals, eigvecs)
