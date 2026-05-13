r"""Dense SDP problem implementation.

The dense problem represents matrix SDPs of the form

.. math::

    \min_{X \in \operatorname{dom}(\mathcal{A})}\ &\operatorname{Re}\operatorname{Tr}[C X] \\
    \text{s.t.}\quad &\mathcal{A}X = b, \\
                 &X \succeq 0, \\
                 &\operatorname{Tr}[X] = \tau
                  \quad \text{if } \tau \text{ is provided}.

The domain :math:`\operatorname{dom}(\mathcal{A})` is expected to be a dense
symmetric or Hermitian matrix space. Thus,
:math:`C \in \operatorname{dom}(\mathcal{A})` is a symmetric or Hermitian cost
matrix and :math:`X \in \operatorname{dom}(\mathcal{A})` is the primal matrix
variable. The operator

.. math::

    \mathcal{A} : \operatorname{dom}(\mathcal{A})
    \to \operatorname{cod}(\mathcal{A})

maps matrices into a finite-dimensional constraint space. For a dense operator
stored as matrices :math:`A_0, \ldots, A_{m-1}`, the constraint
:math:`\mathcal{A}X = b` usually means
:math:`\operatorname{Tr}[A_i X] = b_i` for each :math:`i`.

Example interpretation:

    If :math:`X` is a density matrix, :math:`X \succeq 0` says it is physically
    valid and :math:`\operatorname{Tr}[X] = 1` says it has unit mass.
    Constraint rows can encode measured observables, marginal constraints, or
    any other linear equations.
"""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

from spacecore import Context, DenseArray
from spacecore.linop import LinOp
from ..variables import SDPPrimal, SDPDual
from ._base import SDPProblem


@dataclass(init=False)
class SDPDenseProblem(SDPProblem):
    r"""Dense semidefinite program with optional trace constraint.

    .. math::

        \min_{X \in \operatorname{dom}(\mathcal{A})}\ &\operatorname{Re}\operatorname{Tr}[C X] \\
        \text{s.t.}\quad &\mathcal{A}X = b, \\
                     &X \succeq 0, \\
                     &\operatorname{Tr}[X] = \tau
                      \quad \text{if } \tau \text{ is provided}.

    The domain :math:`\operatorname{dom}(\mathcal{A})` is expected to be a
    dense symmetric or Hermitian matrix space. Thus,
    :math:`C \in \operatorname{dom}(\mathcal{A})` is a symmetric or Hermitian (dense)
    cost matrix and :math:`X \in \operatorname{dom}(\mathcal{A})` is the primal
    matrix variable. The operator
    :math:`\mathcal{A} : \operatorname{dom}(\mathcal{A}) \to
    \operatorname{cod}(\mathcal{A})` maps matrices into a finite-dimensional
    constraint space.

    The positive-semidefinite constraint :math:`X \succeq 0` is not stored as a
    linear equality. It is part of the interpretation of the matrix domain and
    is enforced by solvers that operate on this problem class. The optional
    scalar :math:`\tau` adds the affine constraint
    :math:`\operatorname{Tr}[X] = \tau`; this is common when :math:`X` is a
    density matrix and :math:`\tau = 1`.

    The class does not store an explicit cone object. The cone is implied by
    the dense matrix domain and by solvers such as the CVXPY backend.
    """

    tau: Optional[float] | None = None
    def __init__(self,
                 C: DenseArray,
                 A: LinOp,
                 b: DenseArray,
                 tau: float | None = None,
                 ctx: Context | str | None = None,
                 ):
        r"""Create dense SDP data :math:`(C, \mathcal{A}, b)`.

        Args:
            C: Dense symmetric or Hermitian cost matrix in
                :math:`\operatorname{dom}(\mathcal{A})`.
            A: Linear constraint operator
                :math:`\mathcal{A} : \operatorname{dom}(\mathcal{A}) \to
                \operatorname{cod}(\mathcal{A})`.
            b: Dense right-hand side in :math:`\operatorname{cod}(\mathcal{A})`.
            tau: Optional trace value. If set, feasible matrices satisfy
                :math:`\operatorname{Tr}[X] = \tau`.
            ctx: Optional backend context.
        """
        super(SDPDenseProblem, self).__init__(C, A, b, ctx)
        self.tau = float(tau) if tau is not None else None

    def primal_objective(self, primal: SDPPrimal) -> float:
        r"""Return :math:`\operatorname{Re}\operatorname{Tr}[C X]` for the dense primal matrix :math:`X`.

        For real symmetric matrices this is the usual trace/Frobenius
        objective. For complex Hermitian matrices this is the real part of the
        Hermitian trace pairing, so the objective value is real.
        """
        return self.A.dom.ctx.ops.real(self.A.dom.inner(self.C, primal.X))

    def dual_constr_eig_decomp(self, dual: SDPDual, k: int = None) -> tuple[DenseArray, DenseArray]:
        r"""Return the eigendecomposition of :math:`\mathcal{A}^\dagger y - C`.

        The eigenvalues tell algorithms how the dual slack behaves in the
        semidefinite order. Regularized solvers also use this basis to build a
        primal matrix from a dual iterate.

        Args:
            dual: Dual variable used to form the slack expression.
            k: Number of first eigenpairs to return. If ``None``, return the
                full eigendecomposition.
        """
        lhs = self.A.rapply(dual.y) - self.C
        return self.A.dom.eigh(lhs)

    def _convert(self, new_ctx: Context) -> SDPDenseProblem:
        """Return an equivalent dense problem in ``new_ctx``."""
        new_C = new_ctx.asarray(self.C)
        new_b = new_ctx.asarray(self.b)
        return SDPDenseProblem(new_C, self.A, new_b, self.tau, new_ctx)
