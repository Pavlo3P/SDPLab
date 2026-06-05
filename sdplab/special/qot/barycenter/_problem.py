"""Gaussian barycenter problem for entropic quantum optimal transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from spacecore import Context, ContextBound, DenseArray, jax_pytree_class, resolve_context_priority

from ._gaussian_calculus import (
    gaussian_trace_entropy,
    quadratic_gibbs_log_partition_from_params,
    quadratic_gibbs_log_partitions_from_params,
    quadratic_gibbs_states_from_params,
    _quadratic_gibbs_state_from_params,
)
from ._gaussian_state import GaussianState
from ._gradients import QOTGaussianBarycenterGradients, dual_gradients
from ._quadratic_operator import QuadraticOperator
from ._quadratic_operator_tuple import QuadraticOperatorTuple
from ._repr import array_summary, backend_label, html_repr, plain_repr
from ._spaces import GaussianPhaseSpace, _check_no_complex_input, _is_real_dtype


@jax_pytree_class
@dataclass(init=False)
class QOTGaussianBarycenterProblem(ContextBound):
    r"""Represent an entropic Gaussian QOT barycenter problem.

    Store the spaces, input Gaussian states, quadratic cost, weights, and
    regularization parameters needed to evaluate the Gaussian QOT primal and
    dual formulas. Dual variables are represented by
    :class:`QuadraticOperatorTuple` objects.

    Parameters
    ----------
    space0 : GaussianPhaseSpace
        Phase space for the barycenter state.
    space : GaussianPhaseSpace
        Phase space for each input state. Must use the same ``hbar`` as
        ``space0``.
    sigma : sequence of GaussianState
        Non-empty input states on ``space``.
    cost : QuadraticOperator
        Quadratic cost on the product phase space with ``space0.m + space.m``
        modes.
    alpha : array-like
        Nonnegative input weights with shape ``(len(sigma),)`` and positive
        sum.
    epsilon : float
        Positive entropic regularization weight for couplings.
    tau : float
        Positive entropic regularization weight for the barycenter state.
    use_log_partition : bool, optional
        If ``True``, evaluate dual normalizing terms with log-partitions and
        return normalized Gibbs states from dual state accessors. Default is
        ``False``.
    ctx : Context, str, or None, optional
        Backend context. If ``None``, inferred from supplied objects.

    Attributes
    ----------
    space0 : GaussianPhaseSpace
        Barycenter phase space converted to this object's context.
    space : GaussianPhaseSpace
        Input phase space converted to this object's context.
    joint_space : GaussianPhaseSpace
        Product phase space for couplings.
    sigma : tuple of GaussianState
        Input states converted to this object's context.
    cost : QuadraticOperator
        Product-space quadratic cost.
    alpha : array-like
        Stored weight vector.
    epsilon : float
        Coupling entropy regularization.
    tau : float
        Barycenter entropy regularization.
    use_log_partition : bool
        Whether dual objectives use log-partition arithmetic.
    N : int
        Number of input states.

    Raises
    ------
    ValueError
        If inputs are empty, spaces are incompatible, ``alpha`` has invalid
        shape or values, or regularization parameters are not positive.
    TypeError
        If ``alpha`` is complex-valued or not real-valued after conversion.

    Notes
    -----
    The entropy convention is :math:`x(\log x - 1)`. Its scalar conjugate is
    :math:`\exp(t)`, so the dual Gibbs exponents do not include an extra
    ``-1`` constant.

    Examples
    --------
    >>> import numpy as np
    >>> from spacecore import Context, NumpyOps
    >>> from sdplab.special.qot.barycenter import (
    ...     GaussianPhaseSpace, GaussianState, QuadraticOperator,
    ...     QOTGaussianBarycenterProblem)
    >>> ctx = Context(NumpyOps(), dtype=np.float64)
    >>> space0 = GaussianPhaseSpace(1, ctx=ctx)
    >>> space = GaussianPhaseSpace(1, ctx=ctx)
    >>> joint = GaussianPhaseSpace(2, ctx=ctx)
    >>> sigma = [GaussianState(space, [0.0, 0.0], np.eye(2), ctx=ctx)]
    >>> cost = QuadraticOperator(joint, 0.0, np.zeros(4), np.eye(4), ctx=ctx)
    >>> problem = QOTGaussianBarycenterProblem(
    ...     space0, space, sigma, cost, [1.0], epsilon=1.0, tau=1.0, ctx=ctx)
    >>> problem.N
    1
    """

    def __init__(
        self,
        space0: GaussianPhaseSpace,
        space: GaussianPhaseSpace,
        sigma: Sequence[GaussianState],
        cost: QuadraticOperator,
        alpha: DenseArray,
        epsilon: float,
        tau: float,
        *,
        use_log_partition: bool = False,
        ctx: Context | str | None = None,
    ) -> None:
        if len(sigma) == 0:
            raise ValueError("sigma must contain at least one Gaussian state.")
        ctx = resolve_context_priority(ctx, space0, space, cost, *sigma)
        super(QOTGaussianBarycenterProblem, self).__init__(ctx)

        self.space0 = space0.convert(self.ctx)
        self.space = space.convert(self.ctx)
        hbar_scale = max(1.0, abs(self.space0.hbar), abs(self.space.hbar))
        hbar_tol = max(self.space0.atol, self.space.atol) + max(self.space0.rtol, self.space.rtol) * hbar_scale
        if abs(self.space0.hbar - self.space.hbar) > hbar_tol:
            raise ValueError("space0 and space must use the same hbar.")
        self.joint_space = GaussianPhaseSpace(
            self.space0.m + self.space.m,
            hbar=self.space0.hbar,
            atol=max(self.space0.atol, self.space.atol),
            rtol=max(self.space0.rtol, self.space.rtol),
            ctx=self.ctx,
        )

        self.sigma = tuple(state.convert(self.ctx) for state in sigma)
        for state in self.sigma:
            if not self.space.is_compatible(state.space):
                raise ValueError("every sigma state must live on the input space.")

        if not self.joint_space.is_compatible(cost.space):
            raise ValueError("cost must live on the product phase space.")
        self.cost = cost.convert(self.ctx)

        if self.ctx.enable_checks:
            _check_no_complex_input(alpha, "alpha")
        alpha = self.ctx.asarray(alpha)
        if tuple(alpha.shape) != (len(self.sigma),):
            raise ValueError(f"alpha must have shape {(len(self.sigma),)}, got {tuple(alpha.shape)}.")
        if self.ctx.enable_checks:
            if not _is_real_dtype(alpha.dtype):
                raise TypeError("alpha must have a real dtype.")
            alpha_np = np.asarray(alpha, dtype=float)
            if not np.all(np.isfinite(alpha_np)):
                raise ValueError("alpha must contain only finite values.")
            if np.any(alpha_np < 0.0):
                raise ValueError("alpha weights must be nonnegative.")
            if float(alpha_np.sum()) <= 0.0:
                raise ValueError("at least one alpha weight must be positive.")

        epsilon = float(epsilon)
        tau = float(tau)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be positive and finite.")
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("tau must be positive and finite.")

        self.alpha = alpha
        self.epsilon = epsilon
        self.tau = tau
        self.use_log_partition = bool(use_log_partition)
        self.N = len(self.sigma)

        self._sigma_normalizations = self.ops.stack([state.normalization for state in self.sigma])
        self._sigma_means = self.ops.stack([state.mean for state in self.sigma])
        self._sigma_covs = self.ops.stack([state.cov for state in self.sigma])

    def _repr_rows(self) -> tuple[tuple[str, object], ...]:
        """Return rows shared by plain and HTML reprs."""
        return (
            ("inputs", self.N),
            ("barycenter space", f"m={self.space0.m}, dim={self.space0.dim}"),
            ("input space", f"m={self.space.m}, dim={self.space.dim}"),
            ("joint space", f"m={self.joint_space.m}, dim={self.joint_space.dim}"),
            (
                "regularization",
                f"epsilon={self.epsilon:g}, tau={self.tau:g}, "
                f"use_log_partition={self.use_log_partition}",
            ),
            ("alpha", array_summary("alpha", self.alpha, ops=self.ops, stat="sum")),
            ("cost", f"QuadraticOperator(dim={self.cost.space.dim})"),
            ("backend", backend_label(self)),
        )

    def __repr__(self) -> str:
        """Return a readable representation."""
        return plain_repr(type(self).__name__, self._repr_rows())

    def _repr_html_(self) -> str:
        """Return a notebook-friendly representation."""
        return html_repr(type(self).__name__, self._repr_rows())

    def _validate_dual_tuples(
        self,
        U_tuple: QuadraticOperatorTuple,
        V_tuple: QuadraticOperatorTuple,
    ) -> tuple[QuadraticOperatorTuple, QuadraticOperatorTuple]:
        """Validate and convert dual tuples to the problem context."""
        if len(U_tuple) != self.N or len(V_tuple) != self.N:
            raise ValueError("U_tuple and V_tuple must have one operator per sigma state.")
        if not self.space0.is_compatible(U_tuple.space):
            raise ValueError("U_tuple must live on the barycenter space.")
        if not self.space.is_compatible(V_tuple.space):
            raise ValueError("V_tuple must live on the input space.")
        return U_tuple.convert(self.ctx), V_tuple.convert(self.ctx)

    def _validate_v_tuple(self, V_tuple: QuadraticOperatorTuple) -> QuadraticOperatorTuple:
        """Validate and convert the input-space dual tuple."""
        if len(V_tuple) != self.N:
            raise ValueError("V_tuple must have one operator per sigma state.")
        if not self.space.is_compatible(V_tuple.space):
            raise ValueError("V_tuple must live on the input space.")
        return V_tuple.convert(self.ctx)

    def _joint_exponent_params(
        self,
        U_tuple: QuadraticOperatorTuple,
        V_tuple: QuadraticOperatorTuple,
    ) -> tuple[DenseArray, DenseArray, DenseArray]:
        """Return stacked parameters of coupling exponents."""
        U_tuple, V_tuple = self._validate_dual_tuples(U_tuple, V_tuple)
        N = self.N
        d0 = self.space0.dim
        d1 = self.space.dim
        zeros_01 = self.ops.zeros((N, d0, d1))
        zeros_10 = self.ops.zeros((N, d1, d0))
        block_top = self.ops.concatenate([U_tuple.quadratics, zeros_01], axis=2)
        block_bottom = self.ops.concatenate([zeros_10, V_tuple.quadratics], axis=2)
        block_quad = self.ops.concatenate([block_top, block_bottom], axis=1)
        block_linear = self.ops.concatenate([U_tuple.linears, V_tuple.linears], axis=1)

        constants = (
            U_tuple.constants
            + V_tuple.constants
            - self.alpha * self.cost.constant
        ) / self.epsilon
        linears = (
            block_linear
            - self.alpha[:, None] * self.cost.linear[None, :]
        ) / self.epsilon
        quadratics = (
            block_quad
            - self.alpha[:, None, None] * self.cost.quadratic[None, :, :]
        ) / self.epsilon
        return constants, linears, quadratics

    def _barycenter_exponent_params(
        self,
        U_tuple: QuadraticOperatorTuple,
    ) -> tuple[DenseArray, DenseArray, DenseArray]:
        """Return parameters of the barycenter Gibbs exponent."""
        if len(U_tuple) != self.N:
            raise ValueError("U_tuple must have one operator per sigma state.")
        if not self.space0.is_compatible(U_tuple.space):
            raise ValueError("U_tuple must live on the barycenter space.")
        U_tuple = U_tuple.convert(self.ctx)
        constants = -self.ops.sum(U_tuple.constants, axis=0) / self.tau
        linears = -self.ops.sum(U_tuple.linears, axis=0) / self.tau
        quadratics = -self.ops.sum(U_tuple.quadratics, axis=0) / self.tau
        return constants, linears, quadratics

    def _v_sigma_expectations(self, V_tuple: QuadraticOperatorTuple) -> DenseArray:
        """Return stacked ``Tr[V_s sigma_s]`` values."""
        V_tuple = self._validate_v_tuple(V_tuple)
        weighted_means = self._sigma_normalizations[:, None] * self._sigma_means
        second = self._sigma_covs + self._sigma_means[:, :, None] * self._sigma_means[:, None, :]
        weighted_seconds = self._sigma_normalizations[:, None, None] * second
        return (
            V_tuple.constants * self._sigma_normalizations
            + self.ops.einsum("sd,sd->s", V_tuple.linears, weighted_means)
            + 0.5 * self.ops.einsum("sij,sji->s", V_tuple.quadratics, weighted_seconds)
        )

    def primal_objective(
        self,
        rho: GaussianState,
        couplings: Sequence[GaussianState],
    ) -> DenseArray:
        """Evaluate the Gaussian primal objective.

        Parameters
        ----------
        rho : GaussianState
            Candidate barycenter state on ``space0``.
        couplings : sequence of GaussianState
            Coupling states on ``joint_space``, one per input state.

        Returns
        -------
        array-like
            Scalar primal objective value.

        Raises
        ------
        ValueError
            If ``rho`` or any coupling lives on an incompatible phase space, or
            if the number of couplings is not ``N``.
        """
        rho = rho.convert(self.ctx)
        if not self.space0.is_compatible(rho.space):
            raise ValueError("rho must live on the barycenter space.")
        if len(couplings) != self.N:
            raise ValueError("couplings must have one joint state per sigma state.")

        cost_terms = []
        entropy_terms = []
        for coupling in couplings:
            coupling = coupling.convert(self.ctx)
            if not self.joint_space.is_compatible(coupling.space):
                raise ValueError("each coupling must live on the joint product space.")
            cost_terms.append(coupling.expect_quadratic(self.cost))
            entropy_terms.append(gaussian_trace_entropy(coupling))

        cost_vals = self.ops.stack(cost_terms)
        entropy_vals = self.ops.stack(entropy_terms)
        return (
            self.ops.einsum("s,s->", self.alpha, cost_vals)
            + self.epsilon * self.ops.sum(entropy_vals, axis=0)
            + self.tau * self.ctx.asarray(gaussian_trace_entropy(rho))
        )

    def dual_state_couplings(
        self,
        U_tuple: QuadraticOperatorTuple,
        V_tuple: QuadraticOperatorTuple,
    ) -> tuple[GaussianState, ...]:
        r"""Return dual coupling states.

        Parameters
        ----------
        U_tuple : QuadraticOperatorTuple
            Barycenter-space dual variables.
        V_tuple : QuadraticOperatorTuple
            Input-space dual variables.

        Returns
        -------
        tuple of GaussianState
            Coupling states proportional to
            :math:`\exp((U_s + V_s - \alpha_s C) / \epsilon)`.
        """
        constants, linears, quadratics = self._joint_exponent_params(U_tuple, V_tuple)
        return quadratic_gibbs_states_from_params(
            self.joint_space,
            constants,
            linears,
            quadratics,
            normalize=self.use_log_partition,
        )

    def dual_barycenter_state(self, U_tuple: QuadraticOperatorTuple) -> GaussianState:
        r"""Return the dual barycenter state.

        Parameters
        ----------
        U_tuple : QuadraticOperatorTuple
            Barycenter-space dual variables.

        Returns
        -------
        GaussianState
            State proportional to :math:`\exp(-\sum_s U_s / \tau)`.
        """
        constants, linears, quadratics = self._barycenter_exponent_params(U_tuple)
        return _quadratic_gibbs_state_from_params(
            self.space0,
            constants,
            linears,
            quadratics,
            normalize=self.use_log_partition,
        )

    def hamiltonian_margins(
        self,
        U_tuple: QuadraticOperatorTuple,
        V_tuple: QuadraticOperatorTuple,
    ) -> tuple[DenseArray, DenseArray]:
        r"""Return positive-definiteness margins of Gibbs Hamiltonians.

        Parameters
        ----------
        U_tuple : QuadraticOperatorTuple
            Barycenter-space dual variables.
        V_tuple : QuadraticOperatorTuple
            Input-space dual variables.

        Returns
        -------
        coupling_margin : array-like
            Minimum eigenvalue of the coupling Gibbs Hamiltonians.
        eta_margin : array-like
            Minimum eigenvalue of the barycenter Gibbs Hamiltonian.

        Notes
        -----
        For an exponent :math:`A = c + a^T R + R^T G R / 2`, trace-class
        Gaussian calculus requires :math:`H = -G` to be positive definite.
        Values near or below zero predict a confining-Hamiltonian
        :class:`ValueError`.
        """
        _, _, coupling_quadratics = self._joint_exponent_params(U_tuple, V_tuple)
        _, _, eta_quadratic = self._barycenter_exponent_params(U_tuple)
        coupling_margin = self.ops.min(self.ops.eigvalsh(-coupling_quadratics))
        eta_margin = self.ops.min(self.ops.eigvalsh(-eta_quadratic))
        return coupling_margin, eta_margin

    def dual_objective(
        self,
        U_tuple: QuadraticOperatorTuple,
        V_tuple: QuadraticOperatorTuple,
    ) -> DenseArray:
        """Evaluate the entropic Gaussian QOT dual objective.

        Parameters
        ----------
        U_tuple : QuadraticOperatorTuple
            Barycenter-space dual variables.
        V_tuple : QuadraticOperatorTuple
            Input-space dual variables.

        Returns
        -------
        array-like
            Scalar dual objective value.

        Raises
        ------
        ValueError
            If either dual tuple has the wrong length or incompatible phase
            space.
        """
        U_tuple, V_tuple = self._validate_dual_tuples(U_tuple, V_tuple)
        linear_term = self.ops.sum(self._v_sigma_expectations(V_tuple), axis=0)
        if self.use_log_partition:
            coupling_constants, coupling_linears, coupling_quadratics = self._joint_exponent_params(
                U_tuple,
                V_tuple,
            )
            coupling_log_z = quadratic_gibbs_log_partitions_from_params(
                self.joint_space,
                coupling_constants,
                coupling_linears,
                coupling_quadratics,
            )
            eta_constants, eta_linears, eta_quadratics = self._barycenter_exponent_params(U_tuple)
            eta_log_z = quadratic_gibbs_log_partition_from_params(
                self.space0,
                eta_constants,
                eta_linears,
                eta_quadratics,
            )
            return (
                linear_term
                - self.epsilon * self.ops.sum(coupling_log_z, axis=0)
                - self.tau * eta_log_z
            )

        couplings = self.dual_state_couplings(U_tuple, V_tuple)
        eta = self.dual_barycenter_state(U_tuple)
        coupling_traces = self.ops.stack([state.normalization for state in couplings])
        return (
            linear_term
            - self.epsilon * self.ops.sum(coupling_traces, axis=0)
            - self.tau * eta.normalization
        )

    def gradients(
        self,
        U_tuple: QuadraticOperatorTuple,
        V_tuple: QuadraticOperatorTuple,
    ) -> QOTGaussianBarycenterGradients:
        """Return stacked ascent gradients of the dual objective.

        Parameters
        ----------
        U_tuple : QuadraticOperatorTuple
            Barycenter-space dual variables.
        V_tuple : QuadraticOperatorTuple
            Input-space dual variables.

        Returns
        -------
        QOTGaussianBarycenterGradients
            Gradients in the coefficient order of ``U_tuple`` and ``V_tuple``.
        """
        return dual_gradients(self, U_tuple, V_tuple)

    def tree_flatten(self):
        """Return children and auxiliary data for JAX PyTree flattening."""
        children = (self.alpha,)
        aux = (
            self.space0,
            self.space,
            self.sigma,
            self.cost,
            self.epsilon,
            self.tau,
            self.use_log_partition,
            self.ctx,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild a problem from JAX PyTree data."""
        (alpha,) = children
        space0, space, sigma, cost, epsilon, tau, use_log_partition, ctx = aux
        return cls(
            space0,
            space,
            sigma,
            cost,
            alpha,
            epsilon,
            tau,
            use_log_partition=use_log_partition,
            ctx=ctx,
        )

    def _convert(self, new_ctx: Context) -> "QOTGaussianBarycenterProblem":
        """Convert to ``new_ctx``.

        Converts the spaces, states, cost, and weight vector through the
        :class:`QOTGaussianBarycenterProblem` construction path. Scalar
        regularization parameters and ``use_log_partition`` are preserved.
        """
        return QOTGaussianBarycenterProblem(
            self.space0,
            self.space,
            self.sigma,
            self.cost,
            self.alpha,
            self.epsilon,
            self.tau,
            use_log_partition=self.use_log_partition,
            ctx=new_ctx,
        )
