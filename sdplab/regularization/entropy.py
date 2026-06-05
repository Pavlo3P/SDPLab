r"""Entropy separable spectral regularizers for regularized SDPs.

For a primal matrix :math:`X \in \operatorname{dom}(\mathcal{A})`, entropy
regularization contributes

.. math::

    R_\varepsilon(X)
    = \varepsilon \operatorname{Tr}[\varphi(X)],
    \qquad
    \varphi(t) = t(\log t - 1).

Its Legendre transform :math:`\psi` is evaluated spectrally on the scaled dual
slack

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

        \varphi(t) = t(\log t - 1), \qquad t > 0.

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
        r"""Return :math:`\varphi(x) = x(\log x - 1)` elementwise.

        Nonpositive entries are assigned value zero by convention in this
        implementation.
        """
        ops = self.ctx.ops
        return ops.where(x > 0, x * (ops.log(x) - 1.), 0.)

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
    r"""Trace-normalized entropy variant with logarithmic dual term.

    The usual entropy dual regularization term is

    .. math::

        \varepsilon \operatorname{Tr}\left[
            \psi\left(\frac{\mathcal{A}^\dagger y - C}{\varepsilon}\right)
        \right].

    This variant replaces the spectral trace by its logarithm:

    .. math::

        \varepsilon \log \operatorname{Tr}\left[
            \psi\left(\frac{\mathcal{A}^\dagger y - C}{\varepsilon}\right)
        \right]
        =
        \varepsilon \log\left(\sum_i \exp(s_i / \varepsilon)\right),

    where :math:`s_i` are eigenvalues of
    :math:`\mathcal{A}^\dagger y - C`.

    This is useful for trace-normalized problems with :math:`\operatorname{Tr}[X] = 1`,
    such as density-matrix SDPs.
    """

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
