r"""Wrappers that combine an SDP with a separable spectral regularizer.

A base SDP supplies a cost matrix :math:`C \in \operatorname{dom}(\mathcal{A})`,
a linear operator
:math:`\mathcal{A} : \operatorname{dom}(\mathcal{A}) \to \operatorname{cod}(\mathcal{A})`,
and a constraint RHS :math:`b \in \operatorname{cod}(\mathcal{A})`. It defines

.. math::

    \min_{X \in \operatorname{dom}(\mathcal{A})}\ &\operatorname{Tr}[C X] \\
    \text{s.t.}\quad &\mathcal{A}X = b, \\
                 &X \succeq 0.

A spectral regularizer supplies a scalar convex function :math:`\varphi` and
strength :math:`\varepsilon > 0`. Together they define the regularized primal
objective

.. math::

    P_\varepsilon(X)
    = \operatorname{Tr}[C X] + \varepsilon \operatorname{Tr}[\varphi(X)].

If :math:`X = V\operatorname{diag}(\lambda)V^\dagger`, then :math:`\varphi` is
applied spectrally:

.. math::

    \operatorname{Tr}[\varphi(X)] = \sum_i \varphi(\lambda_i).

The corresponding smooth dual objective is

.. math::

    D_\varepsilon(y)
    = \operatorname{Tr}[b\ y]
      - \varepsilon \operatorname{Tr}\left[
          \psi\left(\frac{\mathcal{A}^\dagger y - C}{\varepsilon}\right)
        \right],
    \qquad y \in \operatorname{cod}(\mathcal{A}),

where :math:`\psi` is the Legendre transform of :math:`\varphi`.

In plain language: optimize in the dual space, then reconstruct a primal matrix
from the eigendecomposition of :math:`\mathcal{A}^\dagger y - C`.
"""

from dataclasses import dataclass

from spacecore import DenseArray, jax_pytree_class
from ..sdp import SDPProblem, SDPPrimal, SDPDual
from ..regularization import AbstractRegularizer


@jax_pytree_class
@dataclass
class SDPRegularized:
    r"""Semidefinite program equipped with a separable spectral regularizer.

    The base SDP supplies a cost matrix
    :math:`C \in \operatorname{dom}(\mathcal{A})`, a linear operator
    :math:`\mathcal{A} : \operatorname{dom}(\mathcal{A}) \to \operatorname{cod}(\mathcal{A})`,
    and constraint RHS :math:`b \in \operatorname{cod}(\mathcal{A})`.
    The regularizer supplies a scalar convex function :math:`\varphi` and
    strength :math:`\varepsilon > 0`. Together they define the regularized SDP
    problem

    .. math::

        \min_{X \in \operatorname{dom}(\mathcal{A})}\ &P_\varepsilon(X)
               = \operatorname{Tr}[C X] + \varepsilon \operatorname{Tr}[\varphi(X)] \\
        \text{s.t.}\quad &\mathcal{A}X = b, \\
                     &X \succeq 0.

    The corresponding dual unconstrained problem is

    .. math::

        \max_{y \in \operatorname{cod}(\mathcal{A})} D_\varepsilon(y) =
        \operatorname{Tr}[b\ y]
        - \varepsilon \operatorname{Tr}\left[
            \psi\left(\frac{\mathcal{A}^\dagger y - C}{\varepsilon}\right)
          \right],

    where :math:`\psi` is the Legendre transform of :math:`\varphi`. The map
    ``primal_from_dual`` uses the first-order relation
    :math:`\lambda_i(X) = \psi'(s_i / \varepsilon)` in the eigenbasis of
    :math:`\mathcal{A}^\dagger y - C`, where :math:`s_i` are its eigenvalues.

    In plain language: solve in the dual space, then turn the dual solution
    back into a primal matrix by converting slack eigenvalues into primal
    eigenvalues.
    """

    sdp: SDPProblem
    reg: AbstractRegularizer

    def primal_objective_reg(self, primal: SDPPrimal) -> DenseArray:
        r"""Return the regularized primal objective.

        For :math:`X \in \operatorname{dom}(\mathcal{A})`, this computes

        .. math::

            P_\varepsilon(X)
            = \operatorname{Tr}[C X]
              + \varepsilon \operatorname{Tr}[\varphi(X)].

        If :math:`X = V\operatorname{diag}(\lambda)V^\dagger`, then

        .. math::

            \operatorname{Tr}[\varphi(X)] = \sum_i \varphi(\lambda_i).
        """
        return self.sdp.primal_objective(primal) + self.reg(primal)

    def dual_objective_reg(self, dual: SDPDual) -> DenseArray:
        r"""Return the smooth regularized dual objective.

        For :math:`y \in \operatorname{cod}(\mathcal{A})`, this computes

        .. math::

            D_\varepsilon(y) =
            \operatorname{Tr}[b\ y]
            - \varepsilon \operatorname{Tr}\left[
                \psi\left(\frac{\mathcal{A}^\dagger y - C}{\varepsilon}\right)
              \right],

        where :math:`\psi` is the Legendre transform of :math:`\varphi`.
        """
        return self.sdp.dual_objective(dual) - self.reg.legendre(self.sdp, dual)

    def primal_from_dual(self, dual: SDPDual, normalized: bool = True, k: int = None) -> SDPPrimal:
        r"""Recover a primal variable from a dual iterate.

        Let

        .. math::

            \mathcal{A}^\dagger y - C = V \operatorname{diag}(s) V^\dagger.

        The unnormalized recovery sets

        .. math::

            \lambda_i(X) = \psi'(s_i / \varepsilon),

        and returns

        .. math::

            X = V \operatorname{diag}(\lambda(X)) V^\dagger.

        With ``normalized=True``, the eigenvalues are normalized by log-sum-exp
        so that :math:`\sum_i \lambda_i(X) = 1`. This is useful when
        :math:`X` is intended to be a trace-one density matrix.

        Args:
            dual: Dual variable :math:`y \in \operatorname{cod}(\mathcal{A})`
                used to form :math:`\mathcal{A}^\dagger y - C`.
            normalized: If True, normalize the recovered eigenvalues with a
                log-sum-exp transform. If False, return the raw derivative
                :math:`\psi'(s_i / \varepsilon)`.
            k: Optional eigensolver truncation parameter forwarded to the SDP.
                It specifies how many leading eigenpairs are used in the
                reconstruction. If ``None``, use the full eigendecomposition.

        Returns:
            A primal variable :math:`X \in \operatorname{dom}(\mathcal{A})`
            reconstructed from the eigendecomposition of
            :math:`\mathcal{A}^\dagger y - C`.

        Notes:
            This method does not solve the dual problem by itself. It assumes
            ``dual`` is already a meaningful dual candidate, for example the
            output of a dual optimizer.
        """
        eigvals, eigvecs = self.sdp.dual_constr_eig_decomp(dual, k)
        eigvals = eigvals / self.reg.val
        if normalized:
            eigvals = self._robust_normalization(eigvals)
        else:
            eigvals = self.reg.phi_star_prime(eigvals)
        return self.sdp.primal_from_eigendecomp(eigvals, eigvecs)

    def _robust_normalization(self, eigvals: DenseArray) -> DenseArray:
        r"""Normalize :math:`\log(\psi'(s_i / \varepsilon))` with log-sum-exp.

        Here ``eigvals`` stores the scaled slack eigenvalues
        :math:`s_i / \varepsilon` from :math:`\mathcal{A}^\dagger y - C`.
        """
        log_phi_sp = self.reg.log_phi_star_prime(eigvals)
        lse = self.sdp.A.dom.ctx.ops.logsumexp(log_phi_sp)
        normalized = self.sdp.A.dom.ctx.ops.exp(log_phi_sp - lse)
        return normalized

    def tree_flatten(self):
        """Return children and auxiliary data for JAX PyTree flattening."""
        return (self.reg,), (self.sdp,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild a regularized SDP from JAX PyTree data."""
        (reg,) = children
        (sdp,) = aux
        return cls(sdp, reg)
