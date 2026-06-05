"""Gaussian calculus backend for quadratic moment-level QOT."""

from __future__ import annotations

import math

from spacecore import DenseArray

from ._gaussian_state import GaussianState
from ._quadratic_operator import QuadraticOperator
from ._spaces import GaussianPhaseSpace


def _array_namespace(space: GaussianPhaseSpace):
    """Return the array namespace attached to ``space``'s backend."""
    if hasattr(space.ops, "jnp"):
        return space.ops.jnp
    if hasattr(space.ops, "np"):
        return space.ops.np
    raise TypeError(f"unsupported backend family {space.ops.family!r}.")


def _complex_dtype(space: GaussianPhaseSpace):
    """Return a complex dtype matching the backend real precision."""
    xp = _array_namespace(space)
    dtype_name = str(space.dtype)
    if "64" in dtype_name and hasattr(xp, "complex128"):
        return xp.complex128
    return xp.complex64


def _as_complex(space: GaussianPhaseSpace, x: DenseArray):
    """Convert ``x`` to a complex backend array without using the real context dtype."""
    xp = _array_namespace(space)
    return xp.asarray(x, dtype=_complex_dtype(space))


def _check_real_spectrum(space: GaussianPhaseSpace, eigvals: DenseArray, name: str) -> None:
    """Validate a numerically real spectrum in eager checked contexts."""
    if not space.ctx.enable_checks:
        return
    ops = space.ops
    imag_scale = ops.max(ops.abs(ops.imag(eigvals)))
    value_scale = ops.maximum(1.0, ops.max(ops.abs(eigvals)))
    if float(imag_scale) > 1e-7 * float(value_scale):
        raise ValueError(f"{name} has a non-negligible imaginary spectrum.")


def _paired_abs_spectrum(space: GaussianPhaseSpace, eigvals: DenseArray) -> DenseArray:
    """Return one value from each +/- symplectic eigenvalue pair."""
    xp = _array_namespace(space)
    abs_vals = xp.sort(xp.abs(xp.real(eigvals)))
    indices = xp.arange(1, 2 * space.m, 2)
    return xp.take(abs_vals, indices)


def _symplectic_eigvals_from_generator(space: GaussianPhaseSpace, H: DenseArray) -> DenseArray:
    """Return positive symplectic eigenvalues of a quadratic generator."""
    xp = _array_namespace(space)
    omega = _as_complex(space, space.omega)
    H = _as_complex(space, H)
    eigvals = xp.linalg.eigvals(1j * omega @ H)
    _check_real_spectrum(space, eigvals, "quadratic generator")
    return _paired_abs_spectrum(space, eigvals)


def _symplectic_eigvals_covariance(space: GaussianPhaseSpace, cov: DenseArray) -> DenseArray:
    """Return covariance symplectic eigenvalues."""
    xp = _array_namespace(space)
    omega = _as_complex(space, space.omega)
    gamma = _as_complex(space, cov)
    eigvals = xp.linalg.eigvals(1j * omega @ gamma)
    _check_real_spectrum(space, eigvals, "covariance")
    return _paired_abs_spectrum(space, eigvals)


def _matrix_coth(space: GaussianPhaseSpace, A: DenseArray) -> DenseArray:
    """Evaluate the matrix hyperbolic cotangent by backend eigendecomposition."""
    xp = _array_namespace(space)
    eigvals, eigvecs = xp.linalg.eig(A)
    coth_vals = 1.0 / xp.tanh(eigvals)
    inv_eigvecs = xp.linalg.solve(eigvecs, xp.eye(eigvecs.shape[0], dtype=eigvecs.dtype))
    return eigvecs @ xp.diag(coth_vals) @ inv_eigvecs


def _thermal_covariance(space: GaussianPhaseSpace, H: DenseArray) -> DenseArray:
    r"""Return covariance of ``exp(-1/2 R^T H R)``."""
    xp = _array_namespace(space)
    omega = _as_complex(space, space.omega)
    H = _as_complex(space, H)
    hbar = space.hbar
    if hbar == 0.0:
        cov = xp.linalg.solve(H, xp.eye(H.shape[0], dtype=H.dtype))
        cov = 0.5 * (cov + xp.swapaxes(cov, -1, -2))
        return space.ctx.asarray(xp.real(cov))
    K = 0.5j * hbar * omega @ H
    cov_complex = 0.5 * hbar * _matrix_coth(space, K) @ (1j * omega)
    if space.ctx.enable_checks:
        imag_scale = space.ops.max(space.ops.abs(space.ops.imag(cov_complex)))
        value_scale = space.ops.maximum(1.0, space.ops.max(space.ops.abs(cov_complex)))
        if float(imag_scale) > 1e-7 * float(value_scale):
            raise ValueError("thermal covariance has a non-negligible imaginary part.")
    cov = xp.real(cov_complex)
    cov = 0.5 * (cov + xp.swapaxes(cov, -1, -2))
    return space.ctx.asarray(cov)


def _check_confining(space: GaussianPhaseSpace, H: DenseArray) -> None:
    """Validate positive definiteness in eager checked contexts."""
    if not space.ctx.enable_checks:
        return
    eigvals = space.ops.eigvalsh(H)
    min_eig = float(space.ops.min(eigvals))
    threshold = max(space.atol, 1e-12)
    if min_eig <= threshold:
        raise ValueError(
            "quadratic Gibbs state requires a strictly negative definite "
            "quadratic coefficient for exp(A). Equivalently, the Hamiltonian "
            f"H=-G must be positive definite; min eig(H)={min_eig:.6g}, "
            f"required > {threshold:.6g}. Try increasing the cost ridge, "
            "reducing the ascent step, or starting from smaller dual "
            "quadratic coefficients."
        )


def _quadratic_partition_log(space: GaussianPhaseSpace, H: DenseArray) -> DenseArray:
    r"""Return ``log Tr exp(-1/2 R^T H R)``."""
    xp = _array_namespace(space)
    if space.hbar == 0.0:
        sign, logdet = xp.linalg.slogdet(H)
        if space.ctx.enable_checks and float(sign) <= 0.0:
            raise ValueError("classical quadratic partition requires positive determinant.")
        return 0.5 * H.shape[0] * xp.log(2.0 * math.pi) - 0.5 * logdet
    nu = _symplectic_eigvals_from_generator(space, H)
    return -xp.sum(xp.log(2.0 * xp.sinh(0.5 * space.hbar * nu)))


def _quadratic_gibbs_state_from_params(
    space: GaussianPhaseSpace,
    constant: DenseArray,
    linear: DenseArray,
    quadratic: DenseArray,
    *,
    normalize: bool = False,
) -> GaussianState:
    """Return the Gaussian operator proportional to ``exp(A)`` from parameters."""
    mean, cov, log_z = _quadratic_gibbs_moments_log_partition_from_params(
        space,
        constant,
        linear,
        quadratic,
    )
    normalization = space.ops.ones(()) if normalize else space.ops.exp(log_z)
    return GaussianState(space, mean, cov, normalization=normalization)


def _quadratic_gibbs_moments_log_partition_from_params(
    space: GaussianPhaseSpace,
    constant: DenseArray,
    linear: DenseArray,
    quadratic: DenseArray,
) -> tuple[DenseArray, DenseArray, DenseArray]:
    """Return mean, covariance, and ``log Tr exp(A)`` from quadratic parameters."""
    c = space.ctx.asarray(constant)
    a = space.ctx.asarray(linear)
    G = space.ctx.asarray(quadratic)
    G = 0.5 * (G + space.ops.swapaxes(G, -1, -2))
    H = -G
    _check_confining(space, H)

    mean = space.ops.solve(H, a)
    shifted_constant = c + 0.5 * space.ops.einsum("i,i->", a, mean)
    cov = _thermal_covariance(space, H)
    log_z = shifted_constant + _quadratic_partition_log(space, H)
    return mean, cov, log_z


def quadratic_gibbs_state(op: QuadraticOperator) -> GaussianState:
    r"""Return the Gaussian operator :math:`\exp(A)`.

    Convert a confining quadratic exponent into its moment-level Gaussian
    representation, including the trace normalization.

    Parameters
    ----------
    op : QuadraticOperator
        Quadratic exponent :math:`A`. Its quadratic coefficient must be
        strictly negative definite in the Hamiltonian sense.

    Returns
    -------
    GaussianState
        Trace-class Gaussian operator with moments and normalization matching
        :math:`\exp(A)`.

    Raises
    ------
    ValueError
        If the exponent is not confining when checks are enabled.

    Notes
    -----
    This module uses the primal scalar regularization
    :math:`x(\log x - 1)`, whose conjugate is :math:`\exp(t)`. No extra
    ``-1`` constant is inserted here.

    Examples
    --------
    >>> import numpy as np
    >>> from spacecore import Context, NumpyOps
    >>> from sdplab.special.qot.barycenter import GaussianPhaseSpace, QuadraticOperator
    >>> from sdplab.special.qot.barycenter._gaussian_calculus import quadratic_gibbs_state
    >>> ctx = Context(NumpyOps(), dtype=np.float64)
    >>> space = GaussianPhaseSpace(1, ctx=ctx)
    >>> op = QuadraticOperator(space, 0.0, np.zeros(2), -np.eye(2), ctx=ctx)
    >>> state = quadratic_gibbs_state(op)
    >>> np.allclose(state.mean, np.zeros(2))
    True
    """
    return _quadratic_gibbs_state_from_params(
        op.space,
        op.constant,
        op.linear,
        op.quadratic,
    )


def quadratic_gibbs_states_from_params(
    space: GaussianPhaseSpace,
    constants: DenseArray,
    linears: DenseArray,
    quadratics: DenseArray,
    *,
    normalize: bool = False,
) -> tuple[GaussianState, ...]:
    """Compute Gaussian Gibbs states from stacked parameters.

    Each leading-axis slice defines one quadratic exponent on the same phase
    space.

    Parameters
    ----------
    space : GaussianPhaseSpace
        Phase space for every returned state.
    constants : array-like
        Scalar exponent constants with shape ``(N,)``.
    linears : array-like
        Linear exponent coefficients with shape ``(N, space.dim)``.
    quadratics : array-like
        Quadratic exponent coefficients with shape
        ``(N, space.dim, space.dim)``.
    normalize : bool, optional
        If ``True``, return normalized states with trace one. Default is
        ``False``.

    Returns
    -------
    tuple of GaussianState
        Gibbs states, one per leading-axis parameter block.
    """
    states = []
    for s in range(int(constants.shape[0])):
        states.append(
            _quadratic_gibbs_state_from_params(
                space,
                constants[s],
                linears[s],
                quadratics[s],
                normalize=normalize,
            )
        )
    return tuple(states)


def quadratic_gibbs_log_partition_from_params(
    space: GaussianPhaseSpace,
    constant: DenseArray,
    linear: DenseArray,
    quadratic: DenseArray,
) -> DenseArray:
    r"""Return the log-partition of a quadratic Gibbs exponent.

    The constant, linear, and quadratic arrays parameterize one exponent on
    ``space``.

    Parameters
    ----------
    space : GaussianPhaseSpace
        Phase space for the exponent.
    constant : array-like
        Scalar exponent constant.
    linear : array-like
        Linear exponent coefficient with shape ``(space.dim,)``.
    quadratic : array-like
        Quadratic exponent coefficient with shape ``(space.dim, space.dim)``.

    Returns
    -------
    array-like
        Scalar value :math:`\log \operatorname{Tr}(\exp(A))`.
    """
    _, _, log_z = _quadratic_gibbs_moments_log_partition_from_params(
        space,
        constant,
        linear,
        quadratic,
    )
    return log_z


def quadratic_gibbs_log_partitions_from_params(
    space: GaussianPhaseSpace,
    constants: DenseArray,
    linears: DenseArray,
    quadratics: DenseArray,
) -> DenseArray:
    r"""Return log-partitions for stacked quadratic exponents.

    Each leading-axis slice is evaluated independently and returned in the
    same order.

    Parameters
    ----------
    space : GaussianPhaseSpace
        Phase space for every exponent.
    constants : array-like
        Scalar exponent constants with shape ``(N,)``.
    linears : array-like
        Linear exponent coefficients with shape ``(N, space.dim)``.
    quadratics : array-like
        Quadratic exponent coefficients with shape
        ``(N, space.dim, space.dim)``.

    Returns
    -------
    array-like
        Vector with entries :math:`\log \operatorname{Tr}(\exp(A_s))`.
    """
    return space.ops.stack(
        [
            quadratic_gibbs_log_partition_from_params(
                space,
                constants[s],
                linears[s],
                quadratics[s],
            )
            for s in range(int(constants.shape[0]))
        ]
    )


def partial_trace_gaussian_joint_to_left(
    joint_state: GaussianState,
    left_space: GaussianPhaseSpace,
    right_space: GaussianPhaseSpace,
) -> GaussianState:
    """Trace out the right subsystem of a joint Gaussian state.

    The returned state keeps the left block of first and second moments.

    Parameters
    ----------
    joint_state : GaussianState
        Joint state on the product of ``left_space`` and ``right_space``.
    left_space : GaussianPhaseSpace
        Phase space retained in the returned state.
    right_space : GaussianPhaseSpace
        Phase space traced out.

    Returns
    -------
    GaussianState
        Left marginal with the same normalization as ``joint_state``.

    Raises
    ------
    ValueError
        If ``joint_state`` is not on the product space.
    """
    if joint_state.space.m != left_space.m + right_space.m:
        raise ValueError("joint state space is not the product of left and right spaces.")
    left_space = left_space.convert(joint_state.ctx)
    d0 = left_space.dim
    return GaussianState(
        left_space,
        joint_state.mean[:d0],
        joint_state.cov[:d0, :d0],
        normalization=joint_state.normalization,
        ctx=joint_state.ctx,
    )


def partial_trace_gaussian_joint_to_right(
    joint_state: GaussianState,
    left_space: GaussianPhaseSpace,
    right_space: GaussianPhaseSpace,
) -> GaussianState:
    """Trace out the left subsystem of a joint Gaussian state.

    The returned state keeps the right block of first and second moments.

    Parameters
    ----------
    joint_state : GaussianState
        Joint state on the product of ``left_space`` and ``right_space``.
    left_space : GaussianPhaseSpace
        Phase space traced out.
    right_space : GaussianPhaseSpace
        Phase space retained in the returned state.

    Returns
    -------
    GaussianState
        Right marginal with the same normalization as ``joint_state``.

    Raises
    ------
    ValueError
        If ``joint_state`` is not on the product space.
    """
    if joint_state.space.m != left_space.m + right_space.m:
        raise ValueError("joint state space is not the product of left and right spaces.")
    right_space = right_space.convert(joint_state.ctx)
    d0 = left_space.dim
    return GaussianState(
        right_space,
        joint_state.mean[d0:],
        joint_state.cov[d0:, d0:],
        normalization=joint_state.normalization,
        ctx=joint_state.ctx,
    )


def gaussian_trace_entropy(state: GaussianState) -> DenseArray:
    r"""Return the Gaussian trace entropy functional.

    Evaluate the entropy contribution used by the primal objective for a
    possibly unnormalized Gaussian operator.

    Parameters
    ----------
    state : GaussianState
        Possibly unnormalized Gaussian operator.

    Returns
    -------
    array-like
        Scalar value :math:`\operatorname{Tr}(X(\log X - 1))`.

    Notes
    -----
    For ``state.space.hbar == 0``, the implementation uses the classical
    differential-entropy convention for Gaussian densities.
    """
    xp = _array_namespace(state.space)
    if state.space.hbar == 0.0:
        sign, logdet = xp.linalg.slogdet(state.cov)
        if state.space.ctx.enable_checks and float(sign) <= 0.0:
            raise ValueError("classical Gaussian entropy requires positive determinant.")
        z = state.normalization
        entropy = 0.5 * (state.space.dim * (1.0 + xp.log(2.0 * math.pi)) + logdet)
        return z * xp.log(z) - z * entropy - z
    z = state.normalization
    nu = _symplectic_eigvals_covariance(state.space, state.cov)
    nbar = xp.maximum(nu / state.space.hbar - 0.5, 0.0)
    n_safe = xp.maximum(nbar, 1e-30)
    entropy_terms = (nbar + 1.0) * xp.log(nbar + 1.0) - nbar * xp.log(n_safe)
    entropy = xp.sum(entropy_terms)
    return z * xp.log(z) - z * entropy - z
