from typing import Any

from spacecore import VectorSpace, Context, DenseArray


class BlockMatrixSpace(VectorSpace):
    def __init__(
        self,
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
        shape = (N, d, d)
        super(BlockMatrixSpace, self).__init__(shape, ctx)

        self.atol = float(atol)
        self.rtol = float(rtol)
        self.enforce_herm = bool(enforce_herm)

    def is_hermitian(self, X: DenseArray) -> bool:
        ops = self.ctx.ops
        Xh = ops.conj(ops.transpose(X, (0, 2, 1)))
        diff = X - Xh

        adiff = ops.abs(diff)
        aX = ops.abs(X)

        max_adiff = adiff.max()
        max_aX = aX.max()

        def _as_float(v):
            try:
                return float(v)
            except TypeError:
                return float(v.item())

        max_adiff_f = _as_float(max_adiff)
        max_aX_f = _as_float(max_aX)

        thresh = float(self.atol) + float(self.rtol) * max_aX_f
        return max_adiff_f <= thresh

    def _check_member(self, x: Any) -> None:
        self.ctx.assert_dense(x)

        if tuple(x.shape) != self.shape:
            raise ValueError(f"Expected shape {self.shape}, got {x.shape}")

        if x.dtype != self.ctx.dtype:
            raise TypeError(f"Expected dtype {self.ctx.dtype}, got {x.dtype}")

        if self.enforce_herm and not self.is_hermitian(x):
            raise TypeError("Block is not Hermitian (within the specified tolerances).")

    def flatten(self, x: Any) -> DenseArray:
        """
        Return a dense 1D coordinate vector (backend-native dense array).

        If a representation forbids materialization, raise a policy/capability error.
        """
        self.check_member(x)
        return self.ctx.ops.reshape(x, -1)

    def symmetrize(self, X: DenseArray) -> DenseArray:
        ops = self.ctx.ops
        return (X + ops.conj(ops.transpose(X, (0, 2, 1)))) * 0.5

    def unflatten(self, v: DenseArray) -> Any:
        """Inverse of flatten; returns an element in the requested representation."""
        X = self.ctx.ops.reshape(v, self.shape)
        X = self.ctx.asarray(X)
        X = self.symmetrize(X)
        self.check_member(X)
        return X
