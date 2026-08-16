r"""Quadratic separable spectral regularizer for regularized SDPs.

For a primal matrix :math:`X \in \operatorname{dom}(\mathcal{A})`, the
regularizer contributes

.. math::

    R_\varepsilon(X)
    = \varepsilon \operatorname{Tr}[\varphi(X)],
    \qquad
    \varphi(t) = \frac{t^2}{2} + \iota_{[0,\infty)}(t).

Its Legendre transform is applied spectrally to the scaled dual slack

.. math::

    \frac{\mathcal{A}^\dagger y - C}{\varepsilon},
    \qquad y \in \operatorname{cod}(\mathcal{A}).
"""

from spacecore import DenseArray, jax_pytree_class
from ._base import NEG_EIG_TOL, Regularizer


@jax_pytree_class
class QuadraticReg(Regularizer):
    r"""Quadratic spectral regularizer on nonnegative spectra.

    The scalar convex function is

    .. math::

        \varphi(t) = \frac{t^2}{2} + \iota_{[0,\infty)}(t).

    The indicator term is zero for :math:`t \ge 0` and :math:`+\infty` for
    :math:`t < 0`. Since the primal constraint is :math:`X \succeq 0`, this
    defines the separable spectral penalty

    .. math::

        R_\varepsilon(X)
        = \varepsilon \operatorname{Tr}[\varphi(X)]
        = \varepsilon \sum_i \frac{\lambda_i(X)^2}{2}.

    The Legendre transform of :math:`\varphi` restricted to
    nonnegative primal eigenvalues is

    .. math::

        \psi(s) = \frac{\max\{s, 0\}^2}{2}.

    If :math:`s_i` are the eigenvalues of
    :math:`\mathcal{A}^\dagger y - C`, then ``primal_from_dual`` uses

    .. math::

        \lambda_i(X) = \psi'(s_i / \varepsilon)
        = \max\{s_i / \varepsilon, 0\}.

    In plain language: the quadratic regularizer clips negative scaled slack
    eigenvalues to zero and keeps positive scaled slack eigenvalues linearly.
    """

    def phi(self, x: DenseArray) -> DenseArray:
        r"""Return :math:`x^2 / 2 + \iota_{[0,\infty)}(x)` elementwise.

        Round-off-negative eigenvalues (down to ``-NEG_EIG_TOL``) evaluate at
        the limit :math:`\varphi(0)=0` rather than out of domain.
        """
        ops = self.ctx.ops
        safe = ops.maximum(x, 0.)
        return ops.where(x >= -NEG_EIG_TOL, safe ** 2 / 2, float("inf"))

    def phi_star(self, x: DenseArray) -> DenseArray:
        r"""Return :math:`\psi(x) = \max\{x, 0\}^2 / 2` elementwise."""
        ops = self.ctx.ops
        return ops.maximum(x, 0.) ** 2 / 2

    def phi_star_prime(self, x: DenseArray) -> DenseArray:
        r"""Return :math:`\psi'(x) = \max\{x, 0\}` elementwise."""
        ops = self.ctx.ops
        return ops.maximum(x, 0.)

    def log_phi_star_prime(self, x: DenseArray) -> DenseArray:
        r"""Return :math:`\log(\psi'(x))` elementwise.

        For the quadratic regularizer this is :math:`\log(\max\{x, 0\})`,
        with :math:`-\infty` on nonpositive entries.
        """
        ops = self.ctx.ops
        safe = ops.where(x > 0., x, 1.)
        return ops.where(x > 0., ops.log(safe), float("-inf"))
