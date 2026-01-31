from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from ._base import Space
from ..backend import BackendContext
from ..types import DenseArray


@dataclass(init=False)
class DenseHermitianMatrixSpace(Space):
    """
    Space of dense n×n Hermitian matrices.

    - Elements are backend-native dense arrays, shape (n, n).
    - Membership enforces Hermitian structure up to tolerances.
    - Inner product is Frobenius / Hilbert–Schmidt:
        ⟨X, Y⟩ = vdot(vec(X), vec(Y)),
      where vdot conjugates the first argument (backend-defined).
    """

    atol: float = 0.0
    rtol: float = 0.0
    enforce_hermitian: bool = True

    def __init__(self, ctx: BackendContext, n: int, atol: float = 0., rtol: float = 0., enforce_hermitian: bool = True):
        if n <= 0:
            raise ValueError("n must be positive.")

        shape = (n, n)
        super(DenseHermitianMatrixSpace, self).__init__(ctx, shape)

        self.atol = atol
        self.rtol = rtol
        self.enforce_hermitian = enforce_hermitian

    @property
    def n(self) -> int:
        return self.shape[0]

    def _check_member(self, x: Any) -> None:
        X = self.ctx.assert_dense(x)

        if tuple(X.shape) != self.shape:
            raise TypeError(f"Expected shape {self.shape}, got {X.shape}")

        if not self.is_hermitian(X):
            raise TypeError("Matrix is not Hermitian (within the specified tolerances).")

    def is_hermitian(self, X: DenseArray) -> bool:
        ops = self.ctx.ops
        Xh = ops.conj(X).T
        diff = X - Xh

        # Validation is typically done outside jit, so it is OK to reduce via
        # backend's .max when available (NumPy/JAX arrays have it).
        adiff = ops.abs(diff)
        aX = ops.abs(X)

        # max(abs(diff)) <= atol + rtol*max(abs(X))
        max_adiff = adiff.max()
        max_aX = aX.max()

        # Convert 0-d arrays to Python scalars if needed
        def _as_float(v):
            try:
                return float(v)
            except TypeError:
                return float(v.item())

        max_adiff_f = _as_float(max_adiff)
        max_aX_f = _as_float(max_aX)

        thresh = float(self.atol) + float(self.rtol) * max_aX_f
        return max_adiff_f <= thresh

    def symmetrize(self, X: DenseArray) -> DenseArray:
        """Project onto the Hermitian cone: (X + X^H)/2."""
        ops = self.ctx.ops
        return (X + ops.conj(X).T) * 0.5

    def zeros(self) -> DenseArray:
        return self.ctx.ops.zeros(self.shape, dtype=self.ctx.dtype)

    def add(self, X: DenseArray, Y: DenseArray) -> DenseArray:
        self.check_member(X)
        self.check_member(Y)
        return X + Y

    def scale(self, a: Any, X: DenseArray) -> DenseArray:
        Z = a * X
        self.check_member(Z)  # Will raise if `a` is complex
        return Z

    def inner(self, X: DenseArray, Y: DenseArray) -> Any:
        self.check_member(X)
        self.check_member(Y)
        ops = self.ctx.ops
        return ops.vdot(X, Y)

    def eigh(self, X: DenseArray, k: int = None) -> Tuple[DenseArray, DenseArray]:
        self.check_member(X)
        return self.ctx.ops.eigh(X)

    def flatten(self, X: DenseArray) -> DenseArray:
        self.check_member(X)
        return self.ctx.ops.ravel(X)

    def unflatten(self, v: DenseArray) -> DenseArray:
        vv = self.ctx.assert_dense(v)
        X = self.ctx.ops.reshape(vv, self.shape)
        return self.symmetrize(X)

    def psd_proj(self, X: DenseArray) -> DenseArray:
        self.check_member(X)
        ops = self.ctx.ops
        evals, evecs = ops.eigh(X)
        evals = ops.maximum(evals, 0.)
        return self.dense_from_eig_decomp(evals, evecs)

    def dense_from_eig_decomp(self, evals: DenseArray, evecs: DenseArray) -> DenseArray:
        self.ctx.assert_dense(evals)
        self.ctx.assert_dense(evecs)
        X = (evecs * evals) @ evecs.T.conj()
        self.check_member(X)
        return X
