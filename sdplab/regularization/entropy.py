from spacecore import jax_pytree_class, DenseArray

from ._base import Regularizer


@jax_pytree_class
class EntropyReg(Regularizer):
    def phi(self, x: DenseArray) -> DenseArray:
        ops = self.ops
        safe_x = ops.where(x > 0., x, 1.)
        positive_values = safe_x * (ops.log(safe_x) - 1.)
        return ops.where(x > 0., positive_values, ops.where(x == 0., 0., float("inf")))

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
