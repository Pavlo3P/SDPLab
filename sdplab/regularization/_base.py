r"""Base class for spectral regularizers for SDPs.

The base SDP supplies a cost matrix
:math:`C \in \operatorname{dom}(\mathcal{A})`, a linear operator
:math:`\mathcal{A} : \operatorname{dom}(\mathcal{A}) \to
\operatorname{cod}(\mathcal{A})`, and constraint RHS
:math:`b \in \operatorname{cod}(\mathcal{A})`.

Most regularizers in this package are separable: they act on a primal matrix
only through a sum over scalar functions of its eigenvalues.
For

.. math::

    X = V \operatorname{diag}(\lambda) V^\dagger
    \in \operatorname{dom}(\mathcal{A}),

and regularization strength :math:`\varepsilon > 0`, the lifted penalty is

.. math::

    R_\varepsilon(X)
    = \varepsilon \operatorname{Tr}[\varphi(X)]
    = \varepsilon \sum_i \varphi(\lambda_i).

The Legendre transform of :math:`\varphi` is denoted by :math:`\psi`:

.. math::

    \psi(s) = \sup_t \{s t - \varphi(t)\}.

The dual regularization term evaluates :math:`\psi` spectrally at the scaled
matrix

.. math::

    \frac{\mathcal{A}^\dagger y - C}{\varepsilon},
    \qquad y \in \operatorname{cod}(\mathcal{A}),

and returns, for separable regularizers,

.. math::

    \varepsilon \operatorname{Tr}\left[
        \psi\left(\frac{\mathcal{A}^\dagger y - C}{\varepsilon}\right)
    \right].

To implement a new regularizer, define the four scalar operations:

    phi:
        The scalar penalty :math:`\varphi` applied to primal eigenvalues.
    phi_star:
        The Legendre transform :math:`\psi` applied to scaled dual-slack
        eigenvalues.
    phi_star_prime:
        The derivative :math:`\psi'` used to recover primal eigenvalues from
        scaled dual-slack eigenvalues.
    log_phi_star_prime:
        The same derivative in log form for numerically stable normalization.

Coupled regularizers, such as the log-trace-exponential entropy variant, may
override the spectral aggregation and derivative methods when no meaningful
elementwise :math:`\psi` exists.
"""
from __future__ import annotations

from typing import Any, Tuple
from abc import abstractmethod
from dataclasses import dataclass

from spacecore import (
    DenseArray,
    ArrayLike,
    Context,
    jax_pytree_class,
    ContextBound,
    EuclideanJordanAlgebraSpace,
    resolve_context_priority,
)


def _validate_positive_scalar(
    val: DenseArray | float,
    ctx: Context,
) -> float:
    """Convert ``val`` to ``ctx`` and validate that it is a positive real scalar."""
    value = ctx.asarray(val)

    if tuple(value.shape) != ():
        raise ValueError(
            f"Expected a scalar value, got shape {value.shape}."
        )

    dtype = ctx.ops.get_dtype(value)
    if ctx.ops.is_complex_dtype(dtype):
        raise TypeError(
            "Expected a real scalar value."
        )

    value = float(value)
    if not bool(value > 0):
        raise ValueError(
            f"Expected a strictly positive value, got {value}."
        )

    return value


@jax_pytree_class
@dataclass(init=False)
class Regularizer(ContextBound):
    r"""Base class for scalar spectral regularizers.

    Subclasses define:

        phi(t): scalar convex penalty :math:`\varphi(t)`,
        phi_star(s): Legendre transform :math:`\psi(s)`,
        phi_star_prime(s): derivative :math:`\psi'(s)`,
        log_phi_star_prime(s): logarithm :math:`\log(\psi'(s))`.

    The base class lifts these one-dimensional formulas to matrices through
    the spectral calculus. If
    :math:`X \in \operatorname{dom}(\mathcal{A})` has eigenvalues
    :math:`\lambda_i(X)`, then

    .. math::

        \operatorname{Tr}[\varphi(X)]
        = \sum_i \varphi(\lambda_i(X)).

    If :math:`\mathcal{A}^\dagger y - C` has eigenvalues :math:`s_i`, then
    the transformed dual regularization term is

    .. math::

        \varepsilon \operatorname{Tr}\left[
            \psi\left(\frac{\mathcal{A}^\dagger y - C}{\varepsilon}\right)
        \right]
        = \varepsilon \sum_i \psi(s_i / \varepsilon).

    This abstraction keeps the SDP code independent of the concrete choice of
    :math:`\varphi`, for example entropy or a quadratic penalty.
    """

    def __init__(self,
                 val: DenseArray | float,
                 space: EuclideanJordanAlgebraSpace | None = None,
                 ctx: Context | str | None = None):
        r"""Create a regularizer with strength :math:`\varepsilon = \texttt{val}`.

        Larger :math:`\varepsilon` makes the spectral penalty more influential.
        Smaller :math:`\varepsilon` makes the model closer to the original
        unregularized SDP. ``space`` may be omitted for scalar formula use; it
        is required before evaluating matrix-level methods.
        """
        ctx = resolve_context_priority(ctx, space, val)
        super(Regularizer, self).__init__(ctx)

        if space is not None and not space.is_euclidean:
            raise NotImplementedError(
                "Regularization currently supports only Euclidean matrix spaces."
            )

        self.val = _validate_positive_scalar(val, ctx)
        self.space = space.convert(ctx) if space is not None else None

    @abstractmethod
    def phi(self, x: DenseArray) -> DenseArray:
        r"""Evaluate the scalar convex penalty :math:`\varphi` elementwise.

        The input represents eigenvalues :math:`\lambda_i(X)` of a primal
        matrix :math:`X \in \operatorname{dom}(\mathcal{A})`.
        """

    @abstractmethod
    def phi_star(self, x: DenseArray) -> DenseArray:
        r"""Evaluate the Legendre transform :math:`\psi` elementwise.

        The input represents scaled eigenvalues
        :math:`s_i / \varepsilon` of
        :math:`\mathcal{A}^\dagger y - C`.
        """

    @abstractmethod
    def phi_star_prime(self, x: DenseArray) -> DenseArray:
        r"""Evaluate :math:`\psi'(x)` elementwise.

        This derivative converts scaled dual-slack eigenvalues
        :math:`s_i / \varepsilon` into primal eigenvalues
        :math:`\lambda_i(X)` during primal recovery.
        """

    @abstractmethod
    def log_phi_star_prime(self, x: DenseArray) -> DenseArray:
        r"""Evaluate :math:`\log(\psi'(x))` elementwise.

        Use this form when :math:`\psi'(s_i / \varepsilon)` may overflow or
        underflow before normalization.
        """

    def _phi(self, eigvals: DenseArray) -> DenseArray:
        r"""Return :math:`\operatorname{Tr}[\varphi(X)]` for primal eigenvalues.

        Args:
            eigvals: Eigenvalues :math:`\lambda_i(X)` of a matrix :math:`X`.

        Returns:
            The spectral trace
            :math:`\operatorname{Tr}[\varphi(X)] = \sum_i \varphi(\lambda_i(X))`.
        """
        phi_eigvals = self.phi(eigvals)
        return self.ops.sum(phi_eigvals)

    def _phi_star(self, eigvals: DenseArray) -> DenseArray:
        r"""Return :math:`\operatorname{Tr}[\varphi^*(X)]` for primal eigenvalues.

        Args:
            eigvals: Eigenvalues :math:`\lambda_i(X)` of a matrix :math:`X`.

        Returns:
            The spectral trace
            :math:`\operatorname{Tr}[\varphi^*(X)] = \sum_i \varphi^*(\lambda_i(X))`.
        """
        phi_star_eigvals = self.phi_star(eigvals)
        return self.ops.sum(phi_star_eigvals)

    def __call__(self, X: ArrayLike) -> DenseArray:
        r"""Evaluate :math:`\varepsilon \operatorname{Tr}[\varphi(X)]`."""
        if self.space is None:
            raise ValueError("Matrix evaluation requires a regularizer space.")
        eigvals = self.ops.real(self.space.spectrum(X))
        return self.val * self._phi(eigvals)

    def legendre(self, X: ArrayLike) -> DenseArray:
        r"""Evaluate the matrix conjugate term at ``X``."""
        if self.space is None:
            raise ValueError("Matrix evaluation requires a regularizer space.")
        eigvals = self.ops.real(self.space.spectrum(X))
        return self.val * self._phi_star(eigvals / self.val)

    def _convert(self, new_ctx: Context) -> Regularizer:
        """Return this regularizer represented in ``new_ctx``."""
        space = self.space.convert(new_ctx) if self.space is not None else None
        return type(self)(self.val, space, ctx=new_ctx)

    def phi_star_prime_matrix(self, X: ArrayLike, normalized: bool = True) -> ArrayLike:
        r"""
        Return (\varphi^*)'(X / \varepsilon) matrix.
        """
        if self.space is None:
            raise ValueError("Matrix evaluation requires a regularizer space.")
        eigvals, eigvecs = self.space.spectral_decompose(X)
        eigvals = eigvals / self.val
        if normalized:
            eigvals = self._robust_normalization(eigvals)
        else:
            eigvals = self.phi_star_prime(eigvals)
        return self.space.from_spectrum(eigvals, eigvecs)

    def legendre_and_grad(self, X: ArrayLike, normalized: bool = False) -> Tuple[DenseArray, ArrayLike]:
        if self.space is None:
            raise ValueError("Matrix evaluation requires a regularizer space.")
        eigvals, eigvecs = self.space.spectral_decompose(X)
        eigvals = self.ops.real(eigvals) / self.val
        legendre = self.val * self._phi_star(eigvals)
        if normalized:
            phi_star_prime_eigvals = self._robust_normalization(eigvals)
        else:
            phi_star_prime_eigvals = self.phi_star_prime(eigvals)
        phi_star_prime_X = self.space.from_spectrum(phi_star_prime_eigvals, eigvecs)

        return legendre, phi_star_prime_X

    def _robust_normalization(self, eigvals: DenseArray) -> DenseArray:
        r"""Normalize :math:`\log(\psi'(s_i / \varepsilon))` with log-sum-exp.

        Here ``eigvals`` stores the scaled slack eigenvalues
        :math:`s_i / \varepsilon` from :math:`\mathcal{A}^\dagger y - C`.
        """
        log_phi_sp = self.log_phi_star_prime(eigvals)
        lse = self.ops.logsumexp(log_phi_sp)
        normalized = self.ops.exp(log_phi_sp - lse)
        return normalized

    def tree_flatten(self):
        """Return children and auxiliary data for JAX PyTree flattening."""
        return (self.val,), (self.space, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild a regularizer from JAX PyTree data."""
        (reg,) = children
        (space, ctx) = aux
        obj = cls.__new__(cls)
        obj._ctx = ctx
        obj.space = space
        obj.val = reg
        return obj


@jax_pytree_class
@dataclass(init=False)
class SDPRegularized(ContextBound):
    r"""Semidefinite program equipped with a spectral regularizer.

    The regularized primal objective is

    .. math::

        \operatorname{Tr}[C X] + \varepsilon\operatorname{Tr}[\varphi(X)]

    for separable regularizers. The dual-side regularizer is delegated to
    ``reg.legendre`` so coupled variants such as ``EntropyRegLog`` use their
    own matrix conjugate.
    """

    def __init__(self, sdp: Any, reg: Regularizer, ctx: Context | str | None = None):
        ctx = resolve_context_priority(ctx, sdp, reg)
        super(SDPRegularized, self).__init__(ctx)
        self.sdp = sdp.convert(ctx) if getattr(sdp, "ctx", None) != ctx else sdp
        self.reg = self._bind_regularizer(reg.convert(ctx))

    def _bind_regularizer(self, reg: Regularizer) -> Regularizer:
        space = self.sdp.dom
        if reg.space is None:
            return type(reg)(reg.val, space, ctx=self.ctx)
        if reg.space != space:
            raise ValueError("Regularizer space must match the SDP primal domain.")
        return reg

    def _cost_matrix(self):
        cost = self.sdp.C
        return cost.to_dense() if hasattr(cost, "to_dense") else cost

    def _dual_slack(self, dual: Any):
        return self.sdp.A.rapply(dual.y) - self._cost_matrix()

    def primal_objective_reg(self, primal: Any) -> DenseArray:
        r"""Return the regularized primal objective."""
        linear = self.sdp.C.inner(primal.X) if hasattr(self.sdp.C, "inner") else self.sdp.primal_objective(primal)
        return self.ops.real(linear) + self.reg(primal.X)

    def dual_objective_reg(self, dual: Any) -> DenseArray:
        r"""Return the smooth regularized dual objective."""
        return self.sdp.dual_objective(dual) - self.reg.legendre(self._dual_slack(dual))

    def primal_from_dual(self, dual: Any, normalized: bool = True, k: int | None = None) -> Any:
        r"""Recover a primal variable from a dual iterate."""
        if k is not None:
            raise NotImplementedError("Truncated primal recovery is not available for this SDP implementation.")
        X = self.reg.phi_star_prime_matrix(self._dual_slack(dual), normalized=normalized)
        return self.sdp.primal_from_array(X)

    def _convert(self, new_ctx: Context) -> SDPRegularized:
        return type(self)(self.sdp, self.reg, ctx=new_ctx)

    def tree_flatten(self):
        """Return children and auxiliary data for JAX PyTree flattening."""
        return (self.reg,), (self.sdp, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild a regularized SDP from JAX PyTree data."""
        (reg,) = children
        (sdp, ctx) = aux
        return cls(sdp, reg, ctx=ctx)
