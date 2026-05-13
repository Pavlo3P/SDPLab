r"""SDP problem classes for storing ``(C, A, b)`` data.

Every problem class represents a version of

.. math::

    \min_X \quad \operatorname{Tr}[C X]
    \quad \text{s.t.} \quad \mathcal{A}X = b,\quad X \succeq 0.

The problem object is intentionally just data plus the basic mathematical
operations needed by solvers:

    ``primal_objective(X)`` computes :math:`\operatorname{Tr}[C X]`.
    ``A_apply(X)`` computes :math:`\mathcal{A}X`.
    ``AT_apply(y)`` computes :math:`\mathcal{A}^\dagger y`.
    ``dual_constr_eig_decomp(y)`` diagonalizes
    :math:`\mathcal{A}^\dagger y - C`.

This separation keeps modeling and solving separate: construct the problem
once, then pass it to different solvers or regularizers.
"""

from ._base import SDPProblem
from ._dense import SDPDenseProblem

__all__ = [
    "SDPProblem",
    "SDPDenseProblem",
]
