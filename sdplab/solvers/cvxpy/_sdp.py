from __future__ import annotations

from typing import Tuple
import numpy as np
import cvxpy as cp

from spacecore import Context, NumpyOps
from ...sdp import SDPDenseProblem, SDPPrimal, SDPDual


def solve_sdp_primal(
    sdp: SDPDenseProblem,
    solver: str = 'MOSEK',
    verbose: bool = False,
    *args, **kwargs
) -> Tuple[SDPPrimal, SDPDual]:
    """
    Solve the dense SDP

        minimize   ⟨C, X⟩
        s.t.       ⟨A_k, X⟩ = b_k,  k=1..m
                   X ⪰ 0

    where data may be real (symmetric) or complex (Hermitian).

    Returns (SDPPrimal, SDPDual).
    """

    # 1) dense backend data
    np_ctx = Context(ops=NumpyOps(), dtype=sdp.ctx.dtype)
    sdp_np = sdp.convert(np_ctx)
    n = sdp_np.dom.shape[0]
    m = sdp_np.cod.shape[0]

    C = sdp_np.C
    b = sdp_np.b
    A = sdp_np.A.A
    tau = sdp_np.tau

    # 2) detect complex data
    is_complex = np.iscomplexobj(C) or np.iscomplexobj(A)

    if is_complex:
        X = cp.Variable((n, n), hermitian=True)
        eq_constraints = [cp.real(cp.trace(A[i] @ X)) == b[i] for i in range(m)]
        constraints = list(eq_constraints)
        if tau is not None:
            constraints.append(cp.real(cp.trace(X)) == tau)
        constraints.append(X >> 0)
        objective = cp.Minimize(cp.real(cp.trace(C @ X)))
    else:
        X = cp.Variable((n, n), symmetric=True)
        eq_constraints = [cp.trace(A[i] @ X) == b[i] for i in range(m)]
        constraints = list(eq_constraints)
        if tau is not None:
            constraints.append(cp.trace(X) == tau)
        constraints.append(X >> 0)
        objective = cp.Minimize(cp.trace(C @ X))

    # 6) solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose, *args, **kwargs)

    if X.value is None:
        raise ValueError(f'{solver} solver did not return a solution.')

    # 7) pack results
    X_val = sdp.ctx.asarray(X.value)

    y_val = [con.dual_value for con in eq_constraints]
    y_val = sdp.ctx.asarray(y_val)

    primal = sdp.primal_from_array(X_val)
    dual   = sdp.dual_from_array(y_val)
    return primal, dual
