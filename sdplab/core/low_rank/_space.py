from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from ..types import DenseArray
from ..space import Space
from ._matrix import LowRankMatrix


@dataclass
class LowRankHermitianMatrixSpace(Space):
    """
    Space of n×n Hermitian matrices represented in low-rank eigen form:

        X = V diag(s) V^H

    where:
      - max_rank = r
      - eigvals s has shape (r,)
      - eigvecs V has shape (r, n) (rows are eigenvectors)

    Linear structure:
      - scale and inner are supported natively in low-rank form.
      - add is supported only via a policy:
          * "materialize": return a dense (n,n) array (NOT LowRankMatrix)
          * "disallow": raise
        because LowRank + LowRank generally increases rank.

    NOTE: This Space's *canonical dense shape* is still (n, n) per Space.shape,
    even though the representation is LowRankMatrix.
    """

    max_rank: int
    n: int

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive.")
        if self.max_rank <= 0:
            raise ValueError("max_rank must be positive.")

    def _check_member(self, x: Any) -> None:
        if not isinstance(x, LowRankMatrix):
            raise TypeError(f"Expected LowRankMatrix, got {type(x)}")
        if x.dim != self.n:
            raise TypeError(f"Expected n={self.n}, got n={x.dim}")
        if x.ctx is not self.ctx:
            raise TypeError(f"Expected ctx={self.ctx}, got ctx={x.ctx}")
        x.check_shapes()
        self.check_dtypes(x)

    def check_dtypes(self, x: LowRankMatrix) -> None:
        self.ctx.assert_dense(x.eigvals)
        self.ctx.assert_dense(x.eigvecs)

        eigval_dtype = x.eigvals.dtype
        eigvec_dtype = x.eigvecs.dtype
        eigvec_real_dtype = x.eigvecs.real.dtype

        if eigvec_dtype != self.ctx.dtype:
            raise TypeError(f"Expected dtype={self.ctx.dtype}, got dtype={eigvec_dtype}.")
        if eigval_dtype != eigvec_real_dtype:
            raise TypeError(f"Incompatible eigval and eigvec dtype. Expected eigval.dtype=eigvec.real.dtype={eigvec_real_dtype}, got eigval.dtype={eigval_dtype}.")

    def zeros(self) -> LowRankMatrix:
        ops = self.ctx.ops
        v0 = ops.zeros((self.n, self.max_rank), dtype=self.ctx.dtype)
        s0 = ops.zeros((self.max_rank,), dtype=v0.real.dtype)
        return LowRankMatrix(ctx=self.ctx, max_rank=self.max_rank, eigvals=s0, eigvecs=v0)

    def add(self, x: Any, y: Any) -> DenseArray:
        self.check_member(x)
        self.check_member(y)

        # materialize -> dense (n, n)
        return x.to_dense() + y.to_dense()

    def scale(self, a: Any, x: Any) -> LowRankMatrix:
        z = LowRankMatrix(ctx=self.ctx, max_rank=x.max_rank, eigvals=a * x.eigvals, eigvecs=x.eigvecs)
        self.check_member(z)  # Will raise if `a` is complex
        return z

    def inner(self, x: Any, y: Any) -> Any:
        self.check_member(x)
        self.check_member(y)
        return x.inner(y)  # Uses LowRankMatrix.inner (no ctx needed)

    def eigh(self, x: Any, k: int = None) -> Tuple[DenseArray, DenseArray]:
        self.check_member(x)
        return x.eigvals, x.eigvecs

    def flatten(self, x: Any) -> DenseArray:
        """
        Return a dense coordinate vector. Policy: materialize dense and ravel.
        This is expensive (O(n^2)) but sometimes necessary for generic solvers.
        """
        self.check_member(x)
        X = x.to_dense()
        return self.ctx.ops.ravel(X)

    def unflatten(self, v: DenseArray) -> Any:
        """
        Inverse of flatten. Policy: materialize dense matrix then compress via eigh,
        keeping max_rank eigenpairs.

        This requires an eigendecomposition and is therefore expensive; it is intended
        for boundary use, not optimization loops.
        """
        vv = self.ctx.assert_dense(v)
        X = self.ctx.ops.reshape(vv, (self.n, self.n))

        # Boundary-only: compute eigenpairs and truncate to max_rank.
        w, U = self.ctx.ops.eigh(X)

        # Keep the largest-magnitude eigenvalues by default.
        # If you prefer "largest positive" or "largest algebraic", adjust here.
        ops = self.ctx.ops
        idx = ops.argsort(ops.abs(w))[::-1]  # descending by |w|
        idx = idx[: self.max_rank]

        U_r = U[:, idx]       # (n, r) columns
        w_r = w[idx].astype(U_r.real.dtype)  # (r,)

        # Convert to our convention: eigvecs (r, n) as rows.
        V_r = U_r    # (r, n)

        return LowRankMatrix(ctx=self.ctx, max_rank=self.max_rank, eigvals=w_r, eigvecs=V_r)
