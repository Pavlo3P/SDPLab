r"""Entropy spectral regularizers for regularized SDPs.

For a primal matrix :math:`X \in \operatorname{dom}(\mathcal{A})`, entropy
regularization contributes

.. math::

    R_\varepsilon(X)
    = \varepsilon \operatorname{Tr}[\varphi(X)],
    \qquad
    \varphi(t) = t(\log t - 1).

For the separable entropy regularizer, its Legendre transform :math:`\psi` is
evaluated spectrally on the scaled dual slack

.. math::

    \frac{\mathcal{A}^\dagger y - C}{\varepsilon},
    \qquad y \in \operatorname{cod}(\mathcal{A}).
"""

from spacecore import DenseArray, jax_pytree_class
from ._base import Regularizer


@jax_pytree_class
class EntropyReg(Regularizer):
    r"""Negative von Neumann entropy spectral regularizer.

    The scalar convex function is

    .. math::

        \varphi(t) =
        \begin{cases}
        t(\log t - 1), & t > 0,\\
        0, & t = 0,\\
        +\infty, & t < 0.
        \end{cases}

    Applied spectrally, it gives

    .. math::

        R_\varepsilon(X)
        = \varepsilon \operatorname{Tr}[\varphi(X)]
        = \varepsilon \sum_i \lambda_i(X)(\log \lambda_i(X) - 1).

    The Legendre transform is

    .. math::

        \psi(s) = \exp(s).

    If :math:`s_i` are the eigenvalues of
    :math:`\mathcal{A}^\dagger y - C`, then ``primal_from_dual`` uses

    .. math::

        \lambda_i(X) = \psi'(s_i / \varepsilon)
        = \exp(s_i / \varepsilon),

    optionally normalized by ``SDPRegularized.primal_from_dual``.

    In plain language: entropy regularization turns scaled dual slack
    eigenvalues into exponential/Gibbs weights.
    """

    def phi(self, x: DenseArray) -> DenseArray:
        r"""Return the extended-value entropy penalty elementwise.

        The value is :math:`x(\log x - 1)` for positive entries, ``0`` at
        zero, and ``+inf`` for negative entries. The implementation never
        evaluates ``log`` on nonpositive entries.
        """
        ops = self.ctx.ops
        safe_x = ops.where(x > 0., x, 1.)
        positive_values = safe_x * (ops.log(safe_x) - 1.)
        return ops.where(x > 0., positive_values, ops.where(x == 0., 0., ops.inf))

    def phi_star(self, x: DenseArray) -> DenseArray:
        r"""Return :math:`\psi(x) = \exp(x)` elementwise."""
        ops = self.ctx.ops
        return ops.exp(x)

    def log_phi_star_prime(self, x: DenseArray) -> DenseArray:
        r"""Return :math:`\log(\psi'(x))` elementwise.

        For entropy regularization, :math:`\psi'(x) = \exp(x)`, hence
        :math:`\log(\psi'(x)) = x`.
        """
        return x

    def phi_star_prime(self, x: DenseArray) -> DenseArray:
        r"""Return :math:`\psi'(x) = \exp(x)` elementwise."""
        ops = self.ctx.ops
        return ops.exp(x)


@jax_pytree_class
class EntropyRegLog(EntropyReg):
    r"""Trace-normalized entropy variant with coupled logarithmic dual term.

    This regularizer represents the coupled conjugate

    .. math::

        F^*(X) =
        \varepsilon \log \operatorname{Tr}\exp(X / \varepsilon),

    not an elementwise scalar :math:`\psi` summed over eigenvalues. If
    :math:`s_i` are eigenvalues of :math:`X`, then
    ``legendre(X)`` returns
    :math:`\varepsilon\log\left(\sum_i \exp(s_i / \varepsilon)\right)`.

    Its derivative is the normalized exponential spectrum, so matrix gradients
    are Gibbs states with trace one. This is useful for trace-normalized
    problems with :math:`\operatorname{Tr}[X] = 1`, such as density-matrix
    SDPs.
    """

    def phi_star(self, x: DenseArray) -> DenseArray:
        r"""Reject the misleading elementwise entropy conjugate.

        ``EntropyRegLog`` represents
        :math:`\varepsilon\log\operatorname{Tr}\exp(X/\varepsilon)`, whose
        conjugate term is coupled across all eigenvalues. Use ``legendre`` or
        ``_phi_star`` for matrix/eigenvalue-vector evaluations.
        """
        raise NotImplementedError(
            "EntropyRegLog has no elementwise phi_star; it represents the "
            "coupled function eps * log Tr exp(X / eps). Use legendre() or "
            "_phi_star() on the full spectrum instead."
        )

    def _phi_star(self, constr_eigvals: DenseArray) -> DenseArray:
        r"""Return the logarithmic spectral Legendre term.

        Args:
            constr_eigvals: Eigenvalues :math:`s_i / \varepsilon` of the scaled
                dual slack
                :math:`(\mathcal{A}^\dagger y - C) / \varepsilon`.

        Returns:
            The value
            :math:`\log\left(\sum_i \exp(s_i / \varepsilon)\right)`.
        """
        ops = self.ctx.ops
        return ops.logsumexp(constr_eigvals)

    def log_phi_star_prime(self, x: DenseArray) -> DenseArray:
        r"""Return log normalized exponential weights for the full spectrum."""
        ops = self.ctx.ops
        return x - ops.logsumexp(x)

    def phi_star_prime(self, x: DenseArray) -> DenseArray:
        r"""Return normalized exponential weights for the full spectrum."""
        ops = self.ctx.ops
        return ops.exp(self.log_phi_star_prime(x))
