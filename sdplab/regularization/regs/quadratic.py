from ...core import DenseArray, jax_pytree_class
from .base import AbstractRegularizer


@jax_pytree_class
class QuadraticReg(AbstractRegularizer):
    def phi(self, x: DenseArray) -> DenseArray:
        return x ** 2 / 2

    def phi_star(self, x: DenseArray) -> DenseArray:
        ops = self.ctx.ops
        return ops.maximum(x, 0.) ** 2 / 2

    def phi_star_prime(self, x: DenseArray) -> DenseArray:
        ops = self.ctx.ops
        return ops.maximum(x, 0.)

    def log_phi_star_prime(self, x: DenseArray) -> DenseArray:
        ops = self.ctx.ops
        return ops.where(x > 0., ops.log(x), -ops.inf)
