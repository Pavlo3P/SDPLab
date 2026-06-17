r"""Base class for separable spectral regularizers for SDPs.

The base SDP supplies a cost matrix
:math:`C \in \operatorname{dom}(\mathcal{A})`, a linear operator
:math:`\mathcal{A} : \operatorname{dom}(\mathcal{A}) \to
\operatorname{cod}(\mathcal{A})`, and constraint RHS
:math:`b \in \operatorname{cod}(\mathcal{A})`.

A spectral regularizer acts on a primal matrix only through its eigenvalues.
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

and returns

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
"""
from __future__ import annotations

from math import isfinite
from typing import Tuple
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
    if not isfinite(value) or value <= 0.0:
        raise ValueError(
            f"Expected a finite strictly positive value, got {value}."
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
                 space: EuclideanJordanAlgebraSpace,
                 ctx: Context | str | None = None):
        r"""Create a regularizer with strength :math:`\varepsilon = \texttt{val}`.

        Larger :math:`\varepsilon` makes the spectral penalty more influential.
        Smaller :math:`\varepsilon` makes the model closer to the original
        unregularized SDP.
        """
        ctx = resolve_context_priority(ctx, space, val)
        super(Regularizer, self).__init__(ctx)

        if not space.is_euclidean:
            raise NotImplementedError(
                "Regularization currently supports only Euclidean matrix spaces."
            )

        self.val = _validate_positive_scalar(val, ctx)
        self.space = space.convert(ctx)

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
        eigvals = self.ops.real(self.space.spectrum(X))
        return self.val * self._phi(eigvals)

    def legendre(self, X: ArrayLike) -> DenseArray:
        r"""Evaluate :math:`\varepsilon \operatorname{Tr}[\varphi^*(X / \varepsilon)]`."""
        eigvals = self.ops.real(self.space.spectrum(X))
        return self.val * self._phi_star(eigvals / self.val)

    def _convert(self, new_ctx: Context) -> Regularizer:
        """Return this regularizer represented in ``new_ctx``."""
        return type(self)(self.val, self.space.convert(new_ctx), ctx=new_ctx)

    def phi_star_prime_matrix(self, X: ArrayLike, normalized: bool = True) -> ArrayLike:
        r"""
        Return (\varphi^*)'(X / \varepsilon) matrix.
        """
        eigvals, eigvecs = self.space.spectral_decompose(X)
        eigvals = eigvals / self.val
        if normalized:
            eigvals = self._robust_normalization(eigvals)
        else:
            eigvals = self.phi_star_prime(eigvals)
        return self.space.from_spectrum(eigvals, eigvecs)

    def legendre_and_grad(self, X: ArrayLike, normalized: bool = False) -> Tuple[DenseArray, ArrayLike]:
        eigvals, eigvecs = self.space.spectral_decompose(X)
        eigvals = self.ops.real(eigvals) / self.val
        legendre =  self.val * self._phi_star(eigvals)
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
