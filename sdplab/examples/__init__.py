r"""Ready-made SDP instances for testing and benchmarking.

Each builder returns a :class:`~sdplab.problem.SDPProblem`; they differ in the
structure of :math:`\mathcal{A}`.

``generate_max_cut``
    Diagonal extraction, :math:`\operatorname{Herm}(n) \to \mathbb{R}^n`. Real,
    and **gauge-degenerate**: :math:`\mathcal{A}^\dagger \mathbf{1} = \mathbb{1}`,
    so the dual is fixed only modulo :math:`\operatorname{span}\{\mathbf 1\}`.
``generate_random_qot``
    One-body partial traces,
    :math:`\operatorname{Herm}(d^N) \to \operatorname{Herm}(d)^N`. Complex,
    stacked-block codomain. Returns ``(problem, state)``, ``state`` feasible.
``generate_qubit_tomography``
    :math:`\operatorname{Tr}[M_i X] = b_i` plus a unit-trace row. Zero cost, so
    pure feasibility.

All three domains are :class:`~spacecore.HermitianSpace`, but the interfaces
are written against the general Euclidean Jordan algebra contract -- orthant
spaces, block products, trees mixing cone types. Prefer space-level operations
(``spectrum``, ``flatten``, ``axpy``) over anything assuming a dense block.
"""

from ._max_cut import (
    MaxCutOperator,
    generate_erdos_renyi_graph_laplacian,
    generate_max_cut,
)
from .q_tomography import generate_qubit_tomography
from .qot import QOTConstraintOp, generate_random_qot

__all__ = [
    # problem generators
    "generate_max_cut",
    "generate_qubit_tomography",
    "generate_random_qot",
    # constraint operators, for hand-assembled instances
    "MaxCutOperator",
    "QOTConstraintOp",
    # supporting utilities
    "generate_erdos_renyi_graph_laplacian",
]
