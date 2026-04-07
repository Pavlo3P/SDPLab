from abc import abstractmethod
from dataclasses import dataclass

from spacecore import DenseArray, Context, jax_pytree_class
from spacecore._contextual import ContextBound

from ...sdp import SDPProblem, SDPPrimal, SDPDual

@jax_pytree_class
@dataclass(init=False)
class AbstractRegularizer(ContextBound):
    def __init__(self, val: DenseArray, ctx: Context | str | None = None):
        super(AbstractRegularizer, self).__init__(ctx)
        self.val = self.ctx.asarray(val)

    @abstractmethod
    def phi(self, x: DenseArray) -> DenseArray:
        """Regularization function φ(x)."""

    @abstractmethod
    def phi_star(self, x: DenseArray) -> DenseArray:
        """Legendre transform of the regularization function φ*(x)."""

    @abstractmethod
    def phi_star_prime(self, x: DenseArray) -> DenseArray:
        """Legendre transform derivative of the regularization function φ*(x), i.e. (φ*(x))'."""

    @abstractmethod
    def log_phi_star_prime(self, x: DenseArray) -> DenseArray:
        """Natural logarithm of Legendre transform derivative of the regularization function φ*(x), i.e. log((φ*(x))')."""

    def _phi(self, primal_eigvals: DenseArray) -> DenseArray:
        """
        Core convex function Tr[φ(X)].
        :param primal_eigvals: Eigenvalues of the primal variable λ_1, ..., λ_n.
        :return: Regularization value Σ_{i = 1}^n φ(λ_i).
        """
        phi_eigvals = self.phi(primal_eigvals)
        return phi_eigvals.sum()

    def _phi_star(self, constr_eigvals: DenseArray) -> DenseArray:
        """
        Core conjugate Tr[φ*((A.T @ dual - C) / eps)].
        :param constr_eigvals: Eigenvalues of (A.T @ dual - C) / eps λ_1, ..., λ_n.
        :return Regularization conjugate value Σ_{i = 1}^n φ*(λ_i).
        """
        phi_star_eigvals = self.phi_star(constr_eigvals)
        return phi_star_eigvals.sum()

    def __call__(self, primal: SDPPrimal, k: int = None) -> DenseArray:
        # R(X) = reg · Tr[φ(primal)]
        primal_eigvals, _ = primal.eigh(k)
        return self.val * self._phi(primal_eigvals)

    def legendre(self, sdp: SDPProblem, dual: SDPDual, k: int = None) -> DenseArray:
        # R*(dual) = reg · Tr[φ*( (A.T @ dual - C) / reg )]
        # y = A.T @ dual - C
        constr_eigvals, _ = sdp.dual_constr_eig_decomp(dual, k)
        constr_eigvals = constr_eigvals / self.val
        return self.val * self._phi_star(constr_eigvals)

    def tree_flatten(self):
        return (self.val,), (self.ctx,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (reg,) = children
        (ctx,) = aux
        obj = cls.__new__(cls)
        obj._ctx = ctx
        obj.val = reg
        return obj
