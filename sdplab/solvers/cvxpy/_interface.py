from qotlib.sdp import SDPDenseProblem, SDPProblem, SDPPrimal, SDPDual
from qotlib.qot import QOTProblem

from ._sdp import solve_sdp_primal
from ._qot import solve_qot_dual


def run_cvxpy_solver(
    problem: SDPProblem,
    solver: str = 'MOSEK',
    verbose: bool = False,
    *args, **kwargs
) -> tuple[SDPPrimal, SDPDual]:
    if isinstance(problem, SDPDenseProblem):
        return solve_sdp_primal(problem, solver, verbose, *args, **kwargs)
    elif isinstance(problem, QOTProblem):
        return solve_qot_dual(problem, solver, verbose, *args, **kwargs)
    else:
        raise ValueError('Unknown problem type: {}'.format(type(problem)))
