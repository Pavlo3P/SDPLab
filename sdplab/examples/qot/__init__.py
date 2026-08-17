r"""Random quantum optimal-transport instances: a coupling
:math:`\Gamma \in \operatorname{Herm}(d^N)` of minimal cost with prescribed
one-body marginals :math:`\operatorname{Tr}^k[\Gamma] = \gamma_k`.

The codomain is a stack of Hermitian blocks, so the dual is a tuple of matrices
rather than a vector of scalars. :func:`generate_random_qot` derives the
marginals from a reference coupling, so the returned ``(problem, state)``
carries a feasible point.
"""

from ...special.qot import QOTConstraintOp
from ._random import generate_random_qot

__all__ = [
    "QOTConstraintOp",
    "generate_random_qot",
]
