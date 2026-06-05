"""Quadratic phase-space operators."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from numbers import Number
from typing import Any

from spacecore import Context, ContextBound, DenseArray, jax_pytree_class, resolve_context_priority

from ._repr import array_summary, backend_label, html_repr, plain_repr, safe_float
from ._spaces import (
    GaussianPhaseSpace,
    _check_no_complex_input,
    _check_shape,
    _is_real_dtype,
)


def _scalar_array(ctx: Context, value: Any, name: str, *, check: bool = True) -> DenseArray:
    """Convert and validate a real scalar array."""
    if check:
        _check_no_complex_input(value, name)
    arr = ctx.asarray(value)
    if tuple(arr.shape) != ():
        if int(prod(arr.shape)) != 1:
            raise ValueError(f"{name} must be a scalar.")
        arr = ctx.ops.reshape(arr, ())
    if check and not _is_real_dtype(arr.dtype):
        raise TypeError(f"{name} must have a real dtype.")
    return arr


def _is_scalar_like(value: Any) -> bool:
    """Return whether ``value`` can be interpreted as a scalar multiplier."""
    if isinstance(value, Number):
        return True
    return hasattr(value, "shape") and tuple(value.shape) == ()


@jax_pytree_class
@dataclass(init=False)
class QuadraticOperator(ContextBound):
    r"""Represent a quadratic phase-space operator.

    Store the coefficients of

    .. math::

        A = c + a^T R + \frac{1}{2} R^T G R,

    where ``R`` is the coordinate vector of a :class:`GaussianPhaseSpace`.
    Coefficients are real-valued and ``quadratic`` is required to be
    symmetric when validation is enabled.

    Parameters
    ----------
    space : GaussianPhaseSpace
        Phase space on which the operator acts.
    constant : float or array-like, optional
        Scalar coefficient :math:`c`. Default is 0.
    linear : array-like or None, optional
        Linear coefficient :math:`a` with shape ``(space.dim,)``. If ``None``,
        the zero vector is used.
    quadratic : array-like or None, optional
        Symmetric quadratic coefficient :math:`G` with shape
        ``(space.dim, space.dim)``. If ``None``, the zero matrix is used.
    ctx : Context, str, or None, optional
        Backend context. If ``None``, inferred from ``space``.

    Attributes
    ----------
    space : GaussianPhaseSpace
        Phase space converted to this object's context.
    constant : array-like
        Scalar constant coefficient.
    linear : array-like
        Linear coefficient vector.
    quadratic : array-like
        Symmetric quadratic coefficient matrix.

    Raises
    ------
    TypeError
        If supplied coefficients are complex-valued or not real-valued after
        conversion.
    ValueError
        If coefficient shapes are invalid or ``quadratic`` is not symmetric.

    Examples
    --------
    >>> import numpy as np
    >>> from spacecore import Context, NumpyOps
    >>> from sdplab.special.qot.barycenter import GaussianPhaseSpace, QuadraticOperator
    >>> ctx = Context(NumpyOps(), dtype=np.float64)
    >>> space = GaussianPhaseSpace(1, ctx=ctx)
    >>> op = QuadraticOperator(space, 1.0, [0.5, 0.0], np.eye(2), ctx=ctx)
    >>> np.allclose(op.as_tuple()[0], 1.0)
    True
    """

    def __init__(
        self,
        space: GaussianPhaseSpace,
        constant: float | DenseArray = 0.0,
        linear: DenseArray | None = None,
        quadratic: DenseArray | None = None,
        *,
        ctx: Context | str | None = None,
    ) -> None:
        ctx = resolve_context_priority(ctx, space)
        super(QuadraticOperator, self).__init__(ctx)
        self.space = space.convert(self.ctx)

        if linear is None:
            linear = self.space.zero_mean()
        if quadratic is None:
            quadratic = self.ops.zeros((self.space.dim, self.space.dim))

        linear_arr = self.ctx.asarray(linear)
        quadratic_arr = self.ctx.asarray(quadratic)
        _check_shape(linear_arr, (self.space.dim,), "linear")
        _check_shape(quadratic_arr, (self.space.dim, self.space.dim), "quadratic")

        if self.ctx.enable_checks:
            self.space.check_mean(linear)
            self.space.check_symmetric_matrix(quadratic, name="quadratic")
            if not _is_real_dtype(linear_arr.dtype):
                raise TypeError("linear must have a real dtype.")
            if not _is_real_dtype(quadratic_arr.dtype):
                raise TypeError("quadratic must have a real dtype.")

        self.constant = _scalar_array(
            self.ctx,
            constant,
            "constant",
            check=self.ctx.enable_checks,
        )
        self.linear = linear_arr
        self.quadratic = quadratic_arr

    def expectation(self, state: "GaussianState") -> DenseArray:
        r"""Evaluate this operator in a Gaussian state.

        Parameters
        ----------
        state : GaussianState
            Gaussian state on a compatible phase space.

        Returns
        -------
        array-like
            Scalar expectation :math:`\operatorname{Tr}(\rho A)`.
        """
        return state.expect_quadratic(self)

    def as_tuple(self) -> tuple[DenseArray, DenseArray, DenseArray]:
        """Return the three coefficient arrays.

        Returns
        -------
        constant : array-like
            Scalar constant coefficient.
        linear : array-like
            Linear coefficient vector.
        quadratic : array-like
            Quadratic coefficient matrix.
        """
        return self.constant, self.linear, self.quadratic

    def _repr_rows(self) -> tuple[tuple[str, object], ...]:
        """Return rows shared by plain and HTML reprs."""
        return (
            ("space", f"m={self.space.m}, dim={self.space.dim}"),
            ("constant", safe_float(self.constant)),
            ("linear", array_summary("linear", self.linear, ops=self.ops)),
            ("quadratic", array_summary("quadratic", self.quadratic, ops=self.ops)),
            ("backend", backend_label(self)),
        )

    def __repr__(self) -> str:
        """Return a readable representation."""
        return plain_repr(type(self).__name__, self._repr_rows())

    def _repr_html_(self) -> str:
        """Return a notebook-friendly representation."""
        return html_repr(type(self).__name__, self._repr_rows())

    def _check_compatible(self, other: "QuadraticOperator") -> None:
        """Validate that ``other`` is a compatible quadratic operator."""
        if not isinstance(other, QuadraticOperator):
            raise TypeError(f"expected QuadraticOperator, got {type(other)!r}.")
        if not self.space.is_compatible(other.space):
            raise ValueError("operators live on incompatible phase spaces.")

    def _new_like(
        self,
        constant: DenseArray,
        linear: DenseArray,
        quadratic: DenseArray,
    ) -> "QuadraticOperator":
        """Build a new operator on this operator's phase space."""
        return QuadraticOperator(self.space, constant, linear, quadratic, ctx=self.ctx)

    def __add__(self, other: Any):
        if not isinstance(other, QuadraticOperator):
            return NotImplemented
        self._check_compatible(other)
        other = other.convert(self.ctx)
        return self._new_like(
            self.constant + other.constant,
            self.linear + other.linear,
            self.quadratic + other.quadratic,
        )

    def __radd__(self, other: Any):
        if other == 0:
            return self
        return NotImplemented

    def __sub__(self, other: Any):
        if not isinstance(other, QuadraticOperator):
            return NotImplemented
        self._check_compatible(other)
        other = other.convert(self.ctx)
        return self._new_like(
            self.constant - other.constant,
            self.linear - other.linear,
            self.quadratic - other.quadratic,
        )

    def __neg__(self):
        return self._new_like(-self.constant, -self.linear, -self.quadratic)

    def __mul__(self, scalar: Any):
        if not _is_scalar_like(scalar):
            return NotImplemented
        return self._new_like(
            self.constant * scalar,
            self.linear * scalar,
            self.quadratic * scalar,
        )

    def __rmul__(self, scalar: Any):
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Any):
        if not _is_scalar_like(scalar):
            return NotImplemented
        return self._new_like(
            self.constant / scalar,
            self.linear / scalar,
            self.quadratic / scalar,
        )

    def tree_flatten(self):
        """Return children and auxiliary data for JAX PyTree flattening."""
        return (self.constant, self.linear, self.quadratic), (self.space,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild a quadratic operator from JAX PyTree data."""
        constant, linear, quadratic = children
        (space,) = aux
        return cls(space, constant, linear, quadratic)

    def _convert(self, new_ctx: Context) -> "QuadraticOperator":
        """Convert to ``new_ctx``.

        Converts the phase space and coefficient arrays through the
        :class:`QuadraticOperator` construction path under ``new_ctx``.
        """
        return QuadraticOperator(
            self.space,
            self.constant,
            self.linear,
            self.quadratic,
            ctx=new_ctx,
        )
