"""Moment-level Gaussian states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spacecore import Context, ContextBound, DenseArray, jax_pytree_class, resolve_context_priority

from ._repr import array_summary, backend_label, html_repr, plain_repr, safe_float
from ._spaces import GaussianPhaseSpace, _as_float, _check_no_complex_input, _is_real_dtype


@jax_pytree_class
@dataclass(init=False)
class GaussianState(ContextBound):
    r"""Represent a Gaussian trace-class operator by moments.

    Store the mean, covariance, and trace normalization of a Gaussian operator
    on a :class:`GaussianPhaseSpace`. The pair ``mean`` and ``cov`` describes
    the normalized Gaussian shape, while ``normalization`` stores
    :math:`\operatorname{Tr}(\rho)`.

    Parameters
    ----------
    space : GaussianPhaseSpace
        Phase space on which the Gaussian operator lives.
    mean : array-like
        First moment vector with shape ``(space.dim,)``.
    cov : array-like
        Symmetric covariance matrix with shape ``(space.dim, space.dim)``.
        Checked against the uncertainty principle of ``space`` when context
        checks are enabled.
    normalization : float or array-like, optional
        Positive scalar trace normalization. Use 1 for density operators.
        Default is 1.
    ctx : Context, str, or None, optional
        Backend context. If ``None``, inferred from ``space``.

    Attributes
    ----------
    space : GaussianPhaseSpace
        Phase space converted to this object's context.
    mean : array-like
        Stored mean vector.
    cov : array-like
        Stored covariance matrix.
    normalization : array-like
        Scalar trace normalization.

    Raises
    ------
    TypeError
        If ``mean``, ``cov``, or ``normalization`` is complex-valued.
    ValueError
        If shapes are invalid, ``cov`` violates the uncertainty principle, or
        ``normalization`` is not positive and finite.

    Examples
    --------
    >>> import numpy as np
    >>> from spacecore import Context, NumpyOps
    >>> from sdplab.special.qot.barycenter import GaussianPhaseSpace, GaussianState
    >>> ctx = Context(NumpyOps(), dtype=np.float64)
    >>> space = GaussianPhaseSpace(1, ctx=ctx)
    >>> state = GaussianState(space, [0.0, 0.0], np.eye(2), ctx=ctx)
    >>> state.dim
    2
    >>> np.allclose(state.second_moment(), np.eye(2))
    True
    """

    def __init__(
        self,
        space: GaussianPhaseSpace,
        mean: DenseArray,
        cov: DenseArray,
        normalization: float | DenseArray = 1.0,
        *,
        ctx: Context | str | None = None,
    ) -> None:
        ctx = resolve_context_priority(ctx, space)
        super(GaussianState, self).__init__(ctx)
        self.space = space.convert(self.ctx)

        mean_arr = self.ctx.asarray(mean)
        cov_arr = self.ctx.asarray(cov)
        norm = self.ctx.asarray(normalization)
        if tuple(mean_arr.shape) != (self.space.dim,):
            raise ValueError(f"mean must have shape {(self.space.dim,)}, got {tuple(mean_arr.shape)}.")
        if tuple(cov_arr.shape) != (self.space.dim, self.space.dim):
            raise ValueError(
                f"covariance must have shape {(self.space.dim, self.space.dim)}, "
                f"got {tuple(cov_arr.shape)}."
            )
        if tuple(norm.shape) != ():
            if int(np.prod(norm.shape)) != 1:
                raise ValueError("normalization must be a scalar.")
            norm = self.ops.reshape(norm, ())

        if self.ctx.enable_checks:
            self.space.check_mean(mean)
            self.space.check_uncertainty(cov)
            _check_no_complex_input(normalization, "normalization")
            if not _is_real_dtype(norm.dtype):
                raise TypeError("normalization must have a real dtype.")
            norm_f = _as_float(norm)
            if not np.isfinite(norm_f) or norm_f <= 0.0:
                raise ValueError("normalization must be positive and finite.")

        self.mean = mean_arr
        self.cov = cov_arr
        self.normalization = norm

    @property
    def dim(self) -> int:
        """Return the phase-space dimension."""
        return self.space.dim

    def second_moment(self) -> DenseArray:
        r"""Return the normalized second moment.

        Returns
        -------
        array-like
            Matrix :math:`\Gamma + d d^T`, where ``d`` is :attr:`mean` and
            :math:`\Gamma` is :attr:`cov`.
        """
        return self.cov + self.ops.einsum("i,j->ij", self.mean, self.mean)

    def weighted_mean(self) -> DenseArray:
        r"""Return the trace-weighted first moment.

        Returns
        -------
        array-like
            Vector :math:`\operatorname{Tr}(\rho R)`.
        """
        return self.normalization * self.mean

    def weighted_second_moment(self) -> DenseArray:
        r"""Return the trace-weighted second moment.

        Returns
        -------
        array-like
            Matrix :math:`\operatorname{Tr}(\rho R R^T)` in the symmetric
            moment convention.
        """
        return self.normalization * self.second_moment()

    def expect_quadratic(self, op: "QuadraticOperator") -> DenseArray:
        r"""Evaluate the expectation of a quadratic operator.

        Parameters
        ----------
        op : QuadraticOperator
            Quadratic operator on a compatible phase space.

        Returns
        -------
        array-like
            Scalar value :math:`\operatorname{Tr}(\rho A)`.

        Raises
        ------
        ValueError
            If ``op`` lives on an incompatible phase space.
        """
        if not self.space.is_compatible(op.space):
            raise ValueError("state and operator live on incompatible phase spaces.")
        op = op.convert(self.ctx)
        second = self.second_moment()
        value = (
            op.constant
            + self.ops.einsum("i,i->", op.linear, self.mean)
            + 0.5 * self.ops.einsum("ij,ji->", op.quadratic, second)
        )
        return self.normalization * value

    def moment_vector(self) -> tuple[DenseArray, DenseArray]:
        """Return the normalized first and second moments.

        Returns
        -------
        mean : array-like
            Stored mean vector.
        second : array-like
            Normalized second moment matrix.
        """
        return self.mean, self.second_moment()

    def _repr_rows(self) -> tuple[tuple[str, object], ...]:
        """Return rows shared by plain and HTML reprs."""
        cov_trace = None
        try:
            cov_trace = safe_float(self.ops.trace(self.cov))
        except Exception:
            cov_trace = None
        cov_text = array_summary("cov", self.cov, ops=self.ops)
        if cov_trace is not None:
            cov_text = f"{cov_text}, trace={cov_trace}"
        return (
            ("space", f"m={self.space.m}, dim={self.space.dim}"),
            ("normalization", safe_float(self.normalization)),
            ("mean", array_summary("mean", self.mean, ops=self.ops)),
            ("covariance", cov_text),
            ("backend", backend_label(self)),
        )

    def __repr__(self) -> str:
        """Return a readable representation."""
        return plain_repr(type(self).__name__, self._repr_rows())

    def _repr_html_(self) -> str:
        """Return a notebook-friendly representation."""
        return html_repr(type(self).__name__, self._repr_rows())

    def tree_flatten(self):
        """Return children and auxiliary data for JAX PyTree flattening."""
        return (self.mean, self.cov, self.normalization), (self.space,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild a Gaussian state from JAX PyTree data."""
        mean, cov, normalization = children
        (space,) = aux
        return cls(space, mean, cov, normalization)

    def _convert(self, new_ctx: Context) -> "GaussianState":
        """Convert to ``new_ctx``.

        Converts the phase space, mean, covariance, and normalization through
        the :class:`Context` construction path used by :class:`GaussianState`.
        """
        return GaussianState(
            self.space,
            self.mean,
            self.cov,
            self.normalization,
            ctx=new_ctx,
        )
