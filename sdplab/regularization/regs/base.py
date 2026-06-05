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

from abc import abstractmethod
from dataclasses import dataclass

from spacecore import DenseArray, Context, jax_pytree_class, ContextBound

from ...sdp import SDPProblem, SDPPrimal, SDPDual

@jax_pytree_class
@dataclass(init=False)
class AbstractRegularizer(ContextBound):
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

    def __init__(self, val: DenseArray | float, ctx: Context | str | None = None):
        r"""Create a regularizer with strength :math:`\varepsilon = \texttt{val}`.

        Larger :math:`\varepsilon` makes the spectral penalty more influential.
        Smaller :math:`\varepsilon` makes the model closer to the original
        unregularized SDP.
        """
        super(AbstractRegularizer, self).__init__(ctx)
        self.val = self.ctx.asarray(val)

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

    def _phi(self, primal_eigvals: DenseArray) -> DenseArray:
        r"""Return :math:`\operatorname{Tr}[\varphi(X)]` for primal eigenvalues.

        Args:
            primal_eigvals: Eigenvalues :math:`\lambda_i(X)` of a primal
                matrix :math:`X \in \operatorname{dom}(\mathcal{A})`.

        Returns:
            The spectral trace
            :math:`\operatorname{Tr}[\varphi(X)] = \sum_i \varphi(\lambda_i(X))`.
        """
        phi_eigvals = self.phi(primal_eigvals)
        return self.ops.sum(phi_eigvals)

    def _phi_star(self, constr_eigvals: DenseArray) -> DenseArray:
        r"""Return the spectral trace of :math:`\psi` on scaled dual slack.

        Args:
            constr_eigvals: Eigenvalues :math:`s_i / \varepsilon` of
                :math:`(\mathcal{A}^\dagger y - C) / \varepsilon`.

        Returns:
            The trace
            :math:`\operatorname{Tr}[\psi(S)] = \sum_i \psi(s_i / \varepsilon)`,
            where
            :math:`S = (\mathcal{A}^\dagger y - C) / \varepsilon`.
        """
        phi_star_eigvals = self.phi_star(constr_eigvals)
        return self.ops.sum(phi_star_eigvals)

    def __call__(self, primal: SDPPrimal, k: int = None) -> DenseArray:
        r"""Evaluate :math:`R_\varepsilon(X) = \varepsilon \operatorname{Tr}[\varphi(X)]`."""
        primal_eigvals, _ = primal.eigh(k)
        return self.val * self._phi(primal_eigvals)

    def legendre(self, sdp: SDPProblem, dual: SDPDual, k: int = None) -> DenseArray:
        r"""Evaluate the dual regularization trace for a dual variable.

        For :math:`y \in \operatorname{cod}(\mathcal{A})`, this returns

        .. math::

            \varepsilon \operatorname{Tr}\left[
                \psi\left(\frac{\mathcal{A}^\dagger y - C}{\varepsilon}\right)
            \right],

        where :math:`s_i` are the eigenvalues of
        :math:`\mathcal{A}^\dagger y - C`.
        """
        constr_eigvals, _ = sdp.dual_constr_eig_decomp(dual, k)
        constr_eigvals = self.ops.real(constr_eigvals / self.val)
        return self.val * self._phi_star(constr_eigvals)

    def _convert(self, new_ctx: Context) -> AbstractRegularizer:
        """Return this regularizer represented in ``new_ctx``."""
        return type(self)(self.val, ctx=new_ctx)

    def tree_flatten(self):
        """Return children and auxiliary data for JAX PyTree flattening."""
        return (self.val,), (self.ctx,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild a regularizer from JAX PyTree data."""
        (reg,) = children
        (ctx,) = aux
        obj = cls.__new__(cls)
        obj._ctx = ctx
        obj.val = reg
        return obj
