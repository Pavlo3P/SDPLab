from __future__ import annotations

from typing import Tuple
import numpy as np
import cvxpy as cp

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

    where data may be real (symmetric) or complex Hermitian.

    Returns (SDPPrimal, SDPDual) using your dense wrappers.
    """

    # 1) dense backend data
    C_np = np.asarray(sdp.C)
    A_np = np.asarray(sdp.A)
    b_np = np.asarray(sdp.b)
    tau = sdp.tau
    m, n, _ = A_np.shape

    # 2) detect complex data
    is_complex = np.iscomplexobj(C_np) or np.iscomplexobj(A_np) or np.iscomplexobj(b_np)

    # 3) variable
    if is_complex:
        # Complex Hermitian variable
        X = cp.Variable((n, n), hermitian=True)
    else:
        # Real symmetric variable
        X = cp.Variable((n, n), symmetric=True)

    # 4) equality constraints  ⟨A_k, X⟩ = b_k
    #    ⟨A, X⟩ = trace(A @ X) (for Hermitian A,X this equals trace(A^H X) and is real;
    #    if data are complex, CVXPY splits equality into real/imag automatically).
    eq_constraints = [cp.trace(A_np[i] @ X) == b_np[i] for i in range(m)]
    constraints = list(eq_constraints)
    if tau is not None:
        constraints.append(cp.trace(X) == tau)
    constraints.append(X >> 0)

    # 5) objective: minimize ⟨C, X⟩
    objective = cp.Minimize(cp.real(cp.trace(C_np @ X)))

    # 6) solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose, *args, **kwargs)

    if X.value is None:
        raise ValueError(f'{solver} solver did not return a solution.')

    # 7) pack results
    X_val = sdp.A.dom.ctx.ops.asarray(X.value, dtype=sdp.A.dom.ctx.dtype)

    y_val = [con.dual_value for con in eq_constraints]
    y_val = sdp.A.cod.ctx.ops.asarray(y_val, dtype=sdp.A.cod.ctx.dtype)

    primal = SDPPrimal(sdp.A.dom, X_val)
    dual   = SDPDual(sdp.A.cod, y_val)
    return primal, dual
