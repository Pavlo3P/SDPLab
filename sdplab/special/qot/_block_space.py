r"""Block-matrix vector space for QOT marginal constraints.

For the QOT constraint operator :math:`\mathcal{A}`, the codomain is

.. math::

    \operatorname{cod}(\mathcal{A}) = \operatorname{Herm}(d)^N.

An element of :math:`\operatorname{cod}(\mathcal{A})` is an ordered tuple
:math:`\gamma = (\gamma_0, \ldots, \gamma_{N-1})` of one-body marginal
matrices, with each :math:`\gamma_k \in \operatorname{Herm}(d)`. In array
form this space is represented by shape ``(N, d, d)``.
"""

from typing import Any

from spacecore import VectorSpace, Context, DenseArray


class BlockMatrixSpace(VectorSpace):
    r"""Space :math:`\operatorname{Herm}(d)^N` of Hermitian matrix blocks.

    Elements have shape ``(N, d, d)`` and are interpreted as tuples
    :math:`\gamma = (\gamma_0, \ldots, \gamma_{N-1})` with
    :math:`\gamma_k \in \operatorname{Herm}(d)`. For QOT, this is the space
    :math:`\operatorname{cod}(\mathcal{A})` containing prescribed one-body
    marginals and dual block variables.

    The vector-space operations are blockwise operations on this tuple, and
    Hermitian membership is checked independently on each block.
    """

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
        r"""Create :math:`\operatorname{Herm}(d)^N` for QOT one-body marginals."""
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
        r"""Return True if every block :math:`X_k` is Hermitian within tolerance."""
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
        r"""Validate that ``x`` represents an element of :math:`\operatorname{Herm}(d)^N`."""
        self.ctx.assert_dense(x)

        if tuple(x.shape) != self.shape:
            raise ValueError(f"Expected shape {self.shape}, got {x.shape}")

        if x.dtype != self.ctx.dtype:
            raise TypeError(f"Expected dtype {self.ctx.dtype}, got {x.dtype}")

        if self.enforce_herm and not self.is_hermitian(x):
            raise TypeError("Block is not Hermitian (within the specified tolerances).")

    def flatten(self, x: Any) -> DenseArray:
        r"""Return the coordinate vector obtained by stacking all block entries.

        The inverse operation reshapes this vector back to ``(N, d, d)`` and
        symmetrizes each block into :math:`\operatorname{Herm}(d)`.
        """
        self.check_member(x)
        return self.ctx.ops.reshape(x, -1)

    def symmetrize(self, X: DenseArray) -> DenseArray:
        r"""Return the blockwise Hermitian symmetrization of ``X``."""
        ops = self.ctx.ops
        return (X + ops.conj(ops.transpose(X, (0, 2, 1)))) * 0.5

    def unflatten(self, v: DenseArray) -> Any:
        r"""Convert a coordinate vector back to an element of :math:`\operatorname{Herm}(d)^N`."""
        X = self.ctx.ops.reshape(v, self.shape)
        X = self.ctx.asarray(X)
        X = self.symmetrize(X)
        self.check_member(X)
        return X
