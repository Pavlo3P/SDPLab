from typing import Any
from dataclasses import dataclass

from qotlib.core import Space, BackendContext, DenseArray


class BlockMatrixSpace(Space):
    def __init__(
        self,
        ctx: BackendContext,
        *,
        d: int,
        N: int,
        atol: float = 0.0,
        rtol: float = 0.0,
        enforce_hermitian: bool = True,
    ):
        if d <= 0:
            raise ValueError("d must be positive.")
        if N <= 0:
            raise ValueError("N must be positive.")

        self.atol = float(atol)
        self.rtol = float(rtol)
        self.enforce_hermitian = bool(enforce_hermitian)
        shape = (N, d, d)
        super(BlockMatrixSpace, self).__init__(ctx, shape)

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

        if self.enforce_hermitian and not self.is_hermitian(x):
            raise TypeError("Block is not Hermitian (within the specified tolerances).")

    def zeros(self) -> Any:
        return self.ctx.ops.zeros(self.shape, dtype=self.ctx.dtype)

    def add(self, x: Any, y: Any) -> Any:
        self.check_member(x)
        self.check_member(y)
        return x + y

    def scale(self, a: Any, x: Any) -> Any:
        self.check_member(x)
        return a * x

    # ---------------------------------------------------------------------
    # Geometry
    # ---------------------------------------------------------------------

    def inner(self, x: Any, y: Any) -> Any:
        """
        Inner product ⟨x, y⟩ for elements of this space.

        Must support cross-representation pairing for all representation pairs
        you want the space to accept; unsupported pairs should raise a clear
        capability error.
        """

        self.check_member(x)
        self.check_member(y)
        return self.ctx.ops.vdot(x, y)

    def norm(self, x: Any) -> Any:
        """Induced norm ||x|| = sqrt(real(⟨x,x⟩)). Override if you can do better."""
        self.check_member(x)
        v = self.ctx.ops.real(self.inner(x, x))
        # Backend-agnostic: assume scalar supports **0.5; if complex, spaces may
        # want ctx.ops.real(v) here.
        return v ** 0.5

    def eigh(self, x: Any, k: int = None) -> Any:
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # Coordinate interface (optional but often useful)
    # ---------------------------------------------------------------------

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
