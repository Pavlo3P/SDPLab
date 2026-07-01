r"""Quantum optimal transport spaces, operators, examples, and solvers.

QOT constraints use the partial-trace map
:math:`\mathcal{A}\Gamma = (\operatorname{Tr}^k[\Gamma])_k`
from ``Herm(d^N)`` to ``Herm(d)^N`` and its Kronecker-sum adjoint
:math:`\mathcal{A}^\dagger`. The coupling is denoted by :math:`\Gamma`; its
one-body marginals are denoted by :math:`\gamma_k`. Here
:math:`\operatorname{Tr}^k` means the partial trace that keeps subsystem
:math:`k` and traces out all other subsystems.
"""

from ._constraint_op import QOTConstraintOp
from ._cvxpy import solve_qot_dual

__all__ = [
    "QOTConstraintOp",

    "solve_qot_dual",
]
