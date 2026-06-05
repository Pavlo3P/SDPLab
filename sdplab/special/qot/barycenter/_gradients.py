"""Stacked dual gradients for Gaussian QOT barycenters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spacecore import Context, ContextBound, DenseArray, jax_pytree_class

from ._repr import array_summary, backend_label, html_repr, plain_repr

if TYPE_CHECKING:
    from ._problem import QOTGaussianBarycenterProblem
from ._gaussian_state import GaussianState
from ._quadratic_operator_tuple import QuadraticOperatorTuple


@jax_pytree_class
@dataclass(init=False)
class QOTGaussianBarycenterGradients(ContextBound):
    """Store stacked ascent gradients for Gaussian barycenter dual variables.

    The arrays follow the same ordering as the dual tuples ``U`` and ``V``:
    constants, linears, and quadratics for ``U`` first, then constants,
    linears, and quadratics for ``V``.

    Parameters
    ----------
    grad_U_constants : array-like
        Gradient for ``U`` constants with shape ``(N,)``.
    grad_U_linears : array-like
        Gradient for ``U`` linears with shape ``(N, space0.dim)``.
    grad_U_quadratics : array-like
        Gradient for ``U`` quadratics with shape
        ``(N, space0.dim, space0.dim)``.
    grad_V_constants : array-like
        Gradient for ``V`` constants with shape ``(N,)``.
    grad_V_linears : array-like
        Gradient for ``V`` linears with shape ``(N, space.dim)``.
    grad_V_quadratics : array-like
        Gradient for ``V`` quadratics with shape
        ``(N, space.dim, space.dim)``.
    ctx : Context, str, or None, optional
        Backend context for the stored arrays.

    Attributes
    ----------
    grad_U_constants, grad_U_linears, grad_U_quadratics : array-like
        Stacked gradients for the barycenter-space dual tuple.
    grad_V_constants, grad_V_linears, grad_V_quadratics : array-like
        Stacked gradients for the input-space dual tuple.

    Examples
    --------
    >>> import numpy as np
    >>> from spacecore import Context, NumpyOps
    >>> from sdplab.special.qot.barycenter._gradients import QOTGaussianBarycenterGradients
    >>> ctx = Context(NumpyOps(), dtype=np.float64)
    >>> grad = QOTGaussianBarycenterGradients(
    ...     np.zeros(1), np.zeros((1, 2)), np.zeros((1, 2, 2)),
    ...     np.zeros(1), np.zeros((1, 2)), np.zeros((1, 2, 2)), ctx=ctx)
    >>> len(grad)
    6
    """

    def __init__(
        self,
        grad_U_constants: DenseArray,
        grad_U_linears: DenseArray,
        grad_U_quadratics: DenseArray,
        grad_V_constants: DenseArray,
        grad_V_linears: DenseArray,
        grad_V_quadratics: DenseArray,
        *,
        ctx: Context | str | None = None,
    ) -> None:
        super(QOTGaussianBarycenterGradients, self).__init__(ctx)
        self.grad_U_constants = self.ctx.asarray(grad_U_constants)
        self.grad_U_linears = self.ctx.asarray(grad_U_linears)
        self.grad_U_quadratics = self.ctx.asarray(grad_U_quadratics)
        self.grad_V_constants = self.ctx.asarray(grad_V_constants)
        self.grad_V_linears = self.ctx.asarray(grad_V_linears)
        self.grad_V_quadratics = self.ctx.asarray(grad_V_quadratics)

    def as_tuple(self):
        """Return gradients in the documented order.

        Returns
        -------
        tuple of array-like
            Six arrays ordered as ``grad_U_constants``, ``grad_U_linears``,
            ``grad_U_quadratics``, ``grad_V_constants``, ``grad_V_linears``,
            and ``grad_V_quadratics``.
        """
        return (
            self.grad_U_constants,
            self.grad_U_linears,
            self.grad_U_quadratics,
            self.grad_V_constants,
            self.grad_V_linears,
            self.grad_V_quadratics,
        )

    def __iter__(self):
        """Iterate over gradients in the documented order."""
        return iter(self.as_tuple())

    def __len__(self) -> int:
        """Return the number of stacked gradient arrays."""
        return 6

    def __getitem__(self, index):
        """Return one gradient array by tuple order."""
        return self.as_tuple()[index]

    def _repr_rows(self) -> tuple[tuple[str, object], ...]:
        """Return rows shared by plain and HTML reprs."""
        return (
            ("grad_U_constants", array_summary("grad_U_constants", self.grad_U_constants, ops=self.ops)),
            ("grad_U_linears", array_summary("grad_U_linears", self.grad_U_linears, ops=self.ops)),
            ("grad_U_quadratics", array_summary("grad_U_quadratics", self.grad_U_quadratics, ops=self.ops)),
            ("grad_V_constants", array_summary("grad_V_constants", self.grad_V_constants, ops=self.ops)),
            ("grad_V_linears", array_summary("grad_V_linears", self.grad_V_linears, ops=self.ops)),
            ("grad_V_quadratics", array_summary("grad_V_quadratics", self.grad_V_quadratics, ops=self.ops)),
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
        return self.as_tuple(), (self.ctx,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild gradients from JAX PyTree data."""
        (ctx,) = aux
        return cls(*children, ctx=ctx)

    def _convert(self, new_ctx: Context) -> "QOTGaussianBarycenterGradients":
        """Convert to ``new_ctx``.

        Converts all six stacked gradient arrays through the
        :class:`QOTGaussianBarycenterGradients` construction path.
        """
        return QOTGaussianBarycenterGradients(*self.as_tuple(), ctx=new_ctx)


def _stack_state_arrays(problem: "QOTGaussianBarycenterProblem", states: tuple[GaussianState, ...]):
    """Return stacked normalizations, means, and covariances."""
    ops = problem.ops
    converted = [state.convert(problem.ctx) for state in states]
    return (
        ops.stack([state.normalization for state in converted]),
        ops.stack([state.mean for state in converted]),
        ops.stack([state.cov for state in converted]),
    )


def _weighted_second(ops, normalizations: DenseArray, means: DenseArray, covs: DenseArray):
    """Return stacked normalized weights times second moments."""
    seconds = covs + means[:, :, None] * means[:, None, :]
    return normalizations[:, None, None] * seconds


def _moment_difference_params(
    ops,
    pos_norms: DenseArray,
    pos_means: DenseArray,
    pos_covs: DenseArray,
    neg_norms: DenseArray,
    neg_means: DenseArray,
    neg_covs: DenseArray,
):
    """Convert signed Gaussian moment differences into quadratic coefficients."""
    grad_constants = pos_norms - neg_norms
    grad_linears = pos_norms[:, None] * pos_means - neg_norms[:, None] * neg_means
    grad_quadratics = 0.5 * (
        _weighted_second(ops, pos_norms, pos_means, pos_covs)
        - _weighted_second(ops, neg_norms, neg_means, neg_covs)
    )
    return grad_constants, grad_linears, grad_quadratics


def dual_gradients(
    problem: "QOTGaussianBarycenterProblem",
    U_tuple: QuadraticOperatorTuple,
    V_tuple: QuadraticOperatorTuple,
) -> QOTGaussianBarycenterGradients:
    r"""Return the ascent gradient of a Gaussian barycenter dual objective.

    The convention is the true ascent gradient:

    .. math::

        \nabla_{V_s} = \sigma_s - \operatorname{Tr}_{0}(\rho_s),
        \qquad
        \nabla_{U_s} = \eta - \operatorname{Tr}_{1}(\rho_s).

    Parameters
    ----------
    problem : QOTGaussianBarycenterProblem
        Gaussian barycenter problem defining the objective.
    U_tuple : QuadraticOperatorTuple
        Barycenter-space dual variables.
    V_tuple : QuadraticOperatorTuple
        Input-space dual variables.

    Returns
    -------
    QOTGaussianBarycenterGradients
        Stacked ascent gradients for ``U_tuple`` and ``V_tuple``.

    Raises
    ------
    ValueError
        If either dual tuple has the wrong length or lives on an incompatible
        phase space.
    """
    U_tuple, V_tuple = problem._validate_dual_tuples(U_tuple, V_tuple)
    couplings = problem.dual_state_couplings(U_tuple, V_tuple)
    eta = problem.dual_barycenter_state(U_tuple).convert(problem.ctx)

    joint_norms, joint_means, joint_covs = _stack_state_arrays(problem, couplings)
    d0 = problem.space0.dim

    left_norms = joint_norms
    left_means = joint_means[:, :d0]
    left_covs = joint_covs[:, :d0, :d0]
    right_norms = joint_norms
    right_means = joint_means[:, d0:]
    right_covs = joint_covs[:, d0:, d0:]

    N = len(U_tuple)
    ops = problem.ops
    eta_norms = ops.broadcast_to(eta.normalization, (N,))
    eta_means = ops.broadcast_to(eta.mean, (N, problem.space0.dim))
    eta_covs = ops.broadcast_to(eta.cov, (N, problem.space0.dim, problem.space0.dim))

    grad_U = _moment_difference_params(
        ops,
        eta_norms,
        eta_means,
        eta_covs,
        left_norms,
        left_means,
        left_covs,
    )
    grad_V = _moment_difference_params(
        ops,
        problem._sigma_normalizations,
        problem._sigma_means,
        problem._sigma_covs,
        right_norms,
        right_means,
        right_covs,
    )

    return QOTGaussianBarycenterGradients(
        grad_U[0],
        grad_U[1],
        grad_U[2],
        grad_V[0],
        grad_V[1],
        grad_V[2],
        ctx=problem.ctx,
    )
