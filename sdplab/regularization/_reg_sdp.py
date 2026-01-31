from dataclasses import dataclass

from qotlib.core import DenseArray, jax_pytree_class
from qotlib.sdp import SDPProblem, SDPPrimal, SDPDual
from qotlib.regularization import AbstractRegularizer


@jax_pytree_class
@dataclass
class SDPRegularized:
    sdp: SDPProblem
    reg: AbstractRegularizer

    def primal_objective_reg(self, primal: SDPPrimal) -> DenseArray:
        return self.sdp.primal_objective(primal) + self.reg(primal)

    def dual_objective_reg(self, dual: SDPDual) -> DenseArray:
        return self.sdp.dual_objective(dual) - self.reg.legendre(self.sdp, dual)

    def primal_from_dual(self, dual: SDPDual, normalized: bool = True, k: int = None) -> SDPPrimal:
        eigvals, eigvecs = self.sdp.dual_constr_eig_decomp(dual, k)
        eigvals = eigvals / self.reg.val
        if normalized:
            eigvals = self._robust_normalization(eigvals)
        else:
            eigvals = self.reg.phi_star_prime(eigvals)
        return self.sdp.primal_from_eigendecomp(eigvals, eigvecs)

    def _robust_normalization(self, eigvals: DenseArray) -> DenseArray:
        log_phi_sp = self.reg.log_phi_star_prime(eigvals)
        lse = self.sdp.A.dom.ctx.ops.logsumexp(log_phi_sp)
        normalized = self.sdp.A.dom.ctx.ops.exp(log_phi_sp - lse)
        return normalized

    def tree_flatten(self):
        return (self.reg,), (self.sdp,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (reg,) = children
        (sdp,) = aux
        return cls(sdp, reg)
