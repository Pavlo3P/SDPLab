"""Stacked tuples of quadratic phase-space operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from spacecore import Context, ContextBound, DenseArray, jax_pytree_class, resolve_context_priority

from ._quadratic_operator import QuadraticOperator
from ._repr import array_summary, backend_label, html_repr, plain_repr
from ._spaces import GaussianPhaseSpace, _check_no_complex_input, _is_real_dtype


@jax_pytree_class
@dataclass(init=False)
class QuadraticOperatorTuple(ContextBound):
    """Represent several quadratic operators with stacked coefficients.

    Use this container for dual variables in
    :class:`QOTGaussianBarycenterProblem`. All operators share one
    :class:`GaussianPhaseSpace`; their scalar, linear, and quadratic
    coefficients are stacked along the leading axis.

    Parameters
    ----------
    space : GaussianPhaseSpace
        Common phase space for all stored operators.
    constants : array-like
        Scalar coefficients with shape ``(N,)``.
    linears : array-like
        Linear coefficients with shape ``(N, space.dim)``.
    quadratics : array-like
        Symmetric quadratic coefficients with shape
        ``(N, space.dim, space.dim)``.
    ctx : Context, str, or None, optional
        Backend context. If ``None``, inferred from ``space``.

    Attributes
    ----------
    space : GaussianPhaseSpace
        Common phase space converted to this object's context.
    constants : array-like
        Stacked scalar coefficients.
    linears : array-like
        Stacked linear coefficient vectors.
    quadratics : array-like
        Stacked quadratic coefficient matrices.

    Raises
    ------
    TypeError
        If any coefficient stack is complex-valued or not real-valued after
        conversion.
    ValueError
        If stack shapes are inconsistent or any quadratic block is not
        symmetric.

    Examples
    --------
    >>> import numpy as np
    >>> from spacecore import Context, NumpyOps
    >>> from sdplab.special.qot.barycenter import GaussianPhaseSpace
    >>> from sdplab.special.qot.barycenter import QuadraticOperatorTuple
    >>> ctx = Context(NumpyOps(), dtype=np.float64)
    >>> space = GaussianPhaseSpace(1, ctx=ctx)
    >>> ops = QuadraticOperatorTuple(space, [0.0, 1.0], np.zeros((2, 2)),
    ...                              np.stack([np.eye(2), 2.0 * np.eye(2)]),
    ...                              ctx=ctx)
    >>> len(ops)
    2
    """

    def __init__(
        self,
        space: GaussianPhaseSpace,
        constants: DenseArray,
        linears: DenseArray,
        quadratics: DenseArray,
        *,
        ctx: Context | str | None = None,
    ) -> None:
        ctx = resolve_context_priority(ctx, space)
        super(QuadraticOperatorTuple, self).__init__(ctx)
        self.space = space.convert(self.ctx)

        if self.ctx.enable_checks:
            _check_no_complex_input(constants, "constants")
            _check_no_complex_input(linears, "linears")
            _check_no_complex_input(quadratics, "quadratics")

        constants = self.ctx.asarray(constants)
        linears = self.ctx.asarray(linears)
        quadratics = self.ctx.asarray(quadratics)

        if len(constants.shape) != 1:
            raise ValueError(f"constants must have shape (N,), got {tuple(constants.shape)}.")
        N = int(constants.shape[0])
        if tuple(linears.shape) != (N, self.space.dim):
            raise ValueError(
                f"linears must have shape {(N, self.space.dim)}, got {tuple(linears.shape)}."
            )
        if tuple(quadratics.shape) != (N, self.space.dim, self.space.dim):
            raise ValueError(
                "quadratics must have shape "
                f"{(N, self.space.dim, self.space.dim)}, got {tuple(quadratics.shape)}."
            )
        if self.ctx.enable_checks:
            if not _is_real_dtype(constants.dtype):
                raise TypeError("constants must have a real dtype.")
            if not _is_real_dtype(linears.dtype):
                raise TypeError("linears must have a real dtype.")
            if not _is_real_dtype(quadratics.dtype):
                raise TypeError("quadratics must have a real dtype.")

            q_np = np.asarray(quadratics, dtype=float)
            q_t = np.swapaxes(q_np, -1, -2)
            if not np.allclose(q_np, q_t, atol=self.space.atol, rtol=self.space.rtol):
                raise ValueError("every quadratic block must be symmetric.")

        self.constants = constants
        self.linears = linears
        self.quadratics = quadratics

    @classmethod
    def from_operators(
        cls,
        operators: Sequence[QuadraticOperator],
        *,
        ctx: Context | str | None = None,
    ) -> "QuadraticOperatorTuple":
        """Build a stacked tuple from individual operators.

        Parameters
        ----------
        operators : sequence of QuadraticOperator
            Non-empty sequence of operators on compatible phase spaces.
        ctx : Context, str, or None, optional
            Backend context for the returned tuple. If ``None``, inferred from
            the operators.

        Returns
        -------
        QuadraticOperatorTuple
            Stacked representation of ``operators``.

        Raises
        ------
        ValueError
            If ``operators`` is empty or contains incompatible phase spaces.
        """
        if len(operators) == 0:
            raise ValueError("operators must be non-empty.")
        resolved_ctx = resolve_context_priority(ctx, *operators)
        space = operators[0].space.convert(resolved_ctx)
        converted = []
        for op in operators:
            if not space.is_compatible(op.space):
                raise ValueError("all operators must live on compatible phase spaces.")
            converted.append(op.convert(resolved_ctx))
        ops = space.ops
        return cls(
            space,
            ops.stack([op.constant for op in converted]),
            ops.stack([op.linear for op in converted]),
            ops.stack([op.quadratic for op in converted]),
            ctx=resolved_ctx,
        )

    def __len__(self) -> int:
        """Return the number of stored operators."""
        return int(self.constants.shape[0])

    def _repr_rows(self) -> tuple[tuple[str, object], ...]:
        """Return rows shared by plain and HTML reprs."""
        return (
            ("operators", len(self)),
            ("space", f"m={self.space.m}, dim={self.space.dim}"),
            ("constants", array_summary("constants", self.constants, ops=self.ops)),
            ("linears", array_summary("linears", self.linears, ops=self.ops)),
            ("quadratics", array_summary("quadratics", self.quadratics, ops=self.ops)),
            ("backend", backend_label(self)),
        )

    def __repr__(self) -> str:
        """Return a readable representation."""
        return plain_repr(type(self).__name__, self._repr_rows())

    def _repr_html_(self) -> str:
        """Return a notebook-friendly representation."""
        return html_repr(type(self).__name__, self._repr_rows())

    def __getitem__(self, index):
        """Return one operator for an integer index, or a sliced tuple."""
        if isinstance(index, slice):
            return QuadraticOperatorTuple(
                self.space,
                self.constants[index],
                self.linears[index],
                self.quadratics[index],
                ctx=self.ctx,
            )
        return QuadraticOperator(
            self.space,
            self.constants[index],
            self.linears[index],
            self.quadratics[index],
            ctx=self.ctx,
        )

    def stacked_params(self) -> tuple[DenseArray, DenseArray, DenseArray]:
        """Return the stacked coefficient arrays.

        Returns
        -------
        constants : array-like
            Stacked scalar coefficients.
        linears : array-like
            Stacked linear coefficient vectors.
        quadratics : array-like
            Stacked quadratic coefficient matrices.
        """
        return self.constants, self.linears, self.quadratics

    def with_params(
        self,
        constants: DenseArray,
        linears: DenseArray,
        quadratics: DenseArray,
    ) -> "QuadraticOperatorTuple":
        """Return a tuple with replacement coefficients.

        Parameters
        ----------
        constants : array-like
            Replacement scalar coefficients with shape ``(N,)``.
        linears : array-like
            Replacement linear coefficients with shape ``(N, space.dim)``.
        quadratics : array-like
            Replacement quadratic coefficients with shape
            ``(N, space.dim, space.dim)``.

        Returns
        -------
        QuadraticOperatorTuple
            New tuple on the same phase space.
        """
        return QuadraticOperatorTuple(self.space, constants, linears, quadratics, ctx=self.ctx)

    def sum_weighted(self, weights: DenseArray) -> QuadraticOperator:
        r"""Return a weighted sum as one operator.

        Parameters
        ----------
        weights : array-like
            Real weights with shape ``(len(self),)``.

        Returns
        -------
        QuadraticOperator
            Operator :math:`\sum_s w_s A_s` on the common phase space.

        Raises
        ------
        TypeError
            If ``weights`` is complex-valued or not real-valued after
            conversion.
        ValueError
            If ``weights`` has an invalid shape.
        """
        _check_no_complex_input(weights, "weights")
        weights = self.ctx.asarray(weights)
        if tuple(weights.shape) != (len(self),):
            raise ValueError(f"weights must have shape {(len(self),)}, got {tuple(weights.shape)}.")
        if not _is_real_dtype(weights.dtype):
            raise TypeError("weights must have a real dtype.")
        constant = self.ops.einsum("s,s->", weights, self.constants)
        linear = self.ops.einsum("s,sd->d", weights, self.linears)
        quadratic = self.ops.einsum("s,sij->ij", weights, self.quadratics)
        return QuadraticOperator(self.space, constant, linear, quadratic, ctx=self.ctx)

    def tree_flatten(self):
        """Return children and auxiliary data for JAX PyTree flattening."""
        return (self.constants, self.linears, self.quadratics), (self.space,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild a tuple from JAX PyTree data."""
        constants, linears, quadratics = children
        (space,) = aux
        return cls(space, constants, linears, quadratics)

    def _convert(self, new_ctx: Context) -> "QuadraticOperatorTuple":
        """Convert to ``new_ctx``.

        Converts the common phase space and all stacked coefficient arrays
        through the :class:`QuadraticOperatorTuple` construction path.
        """
        return QuadraticOperatorTuple(
            self.space,
            self.constants,
            self.linears,
            self.quadratics,
            ctx=new_ctx,
        )
