r"""Dispatch layer for CVXPY-backed SDP solvers.

CVXPY is used here as a reference convex optimization backend. The dispatcher
chooses the concrete CVXPY implementation based on the problem class. Each
supported problem is interpreted through the SDP data
:math:`C \in \operatorname{dom}(\mathcal{A})`,
:math:`\mathcal{A} : \operatorname{dom}(\mathcal{A}) \to \operatorname{cod}(\mathcal{A})`,
and :math:`b \in \operatorname{cod}(\mathcal{A})`.
"""

import cvxpy as cp
from typing import Tuple

from ...sdp import SDPDenseProblem, SDPProblem, SDPPrimal, SDPDual

from ._sdp import solve_sdp_primal


def run_cvxpy_solver(
    problem: SDPProblem,
    solver: str = 'MOSEK',
    verbose: bool = False,
    return_problem: bool = False,
    *args, **kwargs
) -> Tuple[SDPPrimal, SDPDual] | Tuple[SDPPrimal, SDPDual, cp.Problem]:
    r"""Solve a supported SDP problem through CVXPY.

    Mathematically, this sends

    .. math::

        \min_{X \in \operatorname{dom}(\mathcal{A})}\quad\operatorname{Tr}[C X]
        \quad \text{s.t.} \quad
            \mathcal{A}X = b,\\
            X \succeq 0.

    to CVXPY, with any problem-specific extra constraints such as
    :math:`\operatorname{Tr}[X] = \tau` added by the concrete solver.

    Args:
        problem: SDP problem instance to solve.
        solver: CVXPY solver name.
        verbose: Whether CVXPY should print solver output.
        *args: Additional positional arguments passed to the concrete solver.
        **kwargs: Additional keyword arguments passed to the concrete solver.

    Returns:
        A pair ``(primal, dual)`` with solution variables in the original
        problem context.
    """
    if isinstance(problem, SDPDenseProblem):
        return solve_sdp_primal(problem, solver, verbose, return_problem, *args, **kwargs)
    else:
        raise ValueError('Unknown problem type: {}'.format(type(problem)))
