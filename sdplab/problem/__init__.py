r"""Conic problem data ``(C, A, b)`` over Euclidean Jordan algebra domains.

The problem class represents

.. math::

    \min_X \quad \langle C, X\rangle
    \quad \text{s.t.} \quad \mathcal{A}X = b,\quad X \succeq 0,

where the domain of :math:`\mathcal{A}` may be any Euclidean Jordan algebra
space — Hermitian matrices, elementwise (orthant) spaces, or trees of such
leaves. The cost :math:`C` is a :class:`Cost` — in general an *operator*, not
just a stored matrix: dense/sparse Hermitian costs also act on vectors
(``matvec``), and :class:`ElementCost` covers plain elements of any domain.
The problem object is intentionally just data plus the basic mathematical
operations every solver needs:

    ``primal_objective(X)`` computes :math:`\langle C, X\rangle`.
    ``dual_objective(y)`` computes :math:`\langle b, y\rangle`.
    ``dual_slack(y)`` computes :math:`\mathcal{A}^\dagger y - C`.
    ``feasibility_gap(X)`` computes :math:`\mathcal{A}X - b`.

This separation keeps modeling and solving separate: construct the problem
once, then pass it to different solvers or regularizers.
"""

from ._base import SDPProblem, as_member
from ._constraint import (
    ConstraintOp,
    MatrixFreeConstraintOp,
    WrappedConstraintOp,
    DenseConstraintOp,
    SparseConstraintOp,
)
from ._cost import (
    Cost,
    ElementCost,
    HermitianCost,
    DenseHermitianCost,
    SparseHermitianCost,
)

__all__ = [
    "SDPProblem",
    "as_member",
    "ConstraintOp",
    "MatrixFreeConstraintOp",
    "WrappedConstraintOp",
    "DenseConstraintOp",
    "SparseConstraintOp",
    "Cost",
    "ElementCost",
    "HermitianCost",
    "DenseHermitianCost",
    "SparseHermitianCost",
]
