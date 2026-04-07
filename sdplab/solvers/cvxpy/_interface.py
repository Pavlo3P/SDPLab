from ...sdp import SDPDenseProblem, SDPProblem, SDPPrimal, SDPDual

from ._sdp import solve_sdp_primal


def run_cvxpy_solver(
    problem: SDPProblem,
    solver: str = 'MOSEK',
    verbose: bool = False,
    *args, **kwargs
) -> tuple[SDPPrimal, SDPDual]:
    if isinstance(problem, SDPDenseProblem):
        return solve_sdp_primal(problem, solver, verbose, *args, **kwargs)
    else:
        raise ValueError('Unknown problem type: {}'.format(type(problem)))
