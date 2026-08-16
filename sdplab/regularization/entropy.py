from spacecore import jax_pytree_class, DenseArray

from ._base import NEG_EIG_TOL, Regularizer


@jax_pytree_class
class EntropyReg(Regularizer):
    r"""Entropy regularizer, :math:`\varphi(t) = t(\log t - 1)`.

    Its conjugate is the exponential :math:`\psi(s) = e^{s}`. The *free*
    smoothed dual term (``legendre(..., normalized=False)``) is the separable
    :math:`\varepsilon \sum_i e^{s_i/\varepsilon}`, which is **unbounded**
    away from dual feasibility and overflows at small :math:`\varepsilon`. Set
    ``normalized=True`` (on the regularizer methods, or on
    :class:`~sdplab.regularization.BoundDualFunctional`) to select the
    fixed-trace form: the globally bounded log-partition
    :math:`\varepsilon \log \operatorname{Tr}\exp(X/\varepsilon)` with the
    unit-trace Gibbs-state gradient. Because :math:`\log\psi'(x) = x` is affine
    here, that fixed-trace form is the *exact* log-partition dual — the target
    for the entropy dual at small :math:`\varepsilon`.
    """

    def phi(self, x: DenseArray) -> DenseArray:
        r"""Return :math:`x(\log x - 1)` on :math:`x\ge0`, else :math:`+\infty`.

        Round-off-negative eigenvalues (down to ``-NEG_EIG_TOL``) evaluate at
        the limit :math:`\varphi(0)=0` rather than out of domain.
        """
        ops = self.ops
        safe_x = ops.where(x > 0., x, 1.)
        positive_values = safe_x * (ops.log(safe_x) - 1.)
        return ops.where(
            x > 0.,
            positive_values,
            ops.where(x >= -NEG_EIG_TOL, 0., float("inf")),
        )

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
