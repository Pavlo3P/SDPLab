from spacecore import DenseArray, jax_pytree_class
from .base import AbstractRegularizer


@jax_pytree_class
class EntropyReg(AbstractRegularizer):
    def phi(self, x: DenseArray) -> DenseArray:
        ops = self.ctx.ops
        return ops.where(x > 0, x * (ops.log(x) - 1.), 0.)

    def phi_star(self, x: DenseArray) -> DenseArray:
        ops = self.ctx.ops
        return ops.exp(x)

    def log_phi_star_prime(self, x: DenseArray) -> DenseArray:
        return x

    def phi_star_prime(self, x: DenseArray) -> DenseArray:
        ops = self.ctx.ops
        return ops.exp(x)


@jax_pytree_class
class EntropyRegLog(EntropyReg):
    def _phi_star(self, constr_eigvals: DenseArray) -> DenseArray:
        """
        Core conjugate Tr[φ*((A.T @ dual - C) / eps)].
        :param constr_eigvals: Eigenvalues of (A.T @ dual - C) / eps λ_1, ..., λ_n.
        :return Regularization conjugate value Σ_{i = 1}^n φ*(λ_i).
        """
        ops = self.ctx.ops
        return ops.logsumexp(constr_eigvals)
