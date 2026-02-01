from __future__ import annotations

import numpy as np
import cvxpy as cp

from ...qot import QOTPrimal, QOTDual, QOTProblem


def solve_qot_dual(
    problem: QOTProblem,
    solver: str = 'MOSEK',
    verbose: bool = False,
    *args, **kwargs
) -> tuple[QOTPrimal, QOTDual]:
    """
    Solve the QOT dual SDP:

        maximize   sum_k trace(U_k @ ρ_k)
        subject to K := sum_k  I⊗...⊗U_k⊗...⊗I   ⪯   C
                   U_k Hermitian   (real case: symmetric)

    where ρ_k are the single-site partial traces and C is the cost matrix
    of size D×D with D = d**N.

    Returns:
        (primal, dual) where
          - primal is QOTPrimal (dense) — the dual multiplier of C ⪰ K
          - dual   is QOTDual with blocks U_k ∈ ℂ^{d×d}
    """


    C_np = np.asarray(problem.C)
    marginals = np.asarray(problem.b)

    d = problem.d
    N = problem.N
    is_complex = np.iscomplexobj(C_np) or np.iscomplexobj(marginals)

    if is_complex:
        U = [cp.Variable((d, d), hermitian=True) for _ in range(N)]
    else:
        U = [cp.Variable((d, d), symmetric=True) for _ in range(N)]

    obj = cp.Maximize(cp.real(cp.sum([cp.trace(Uk @ Rk) for Uk, Rk in zip(U, marginals)])))

    # ----------- build K = sum_k I⊗...⊗U_k⊗...⊗I -----------
    eye = cp.Constant(np.eye(d, d, dtype=problem.A.cod.ctx.dtype))
    def kron_chain(k: int) -> cp.Expression:
        expr = None
        for idx in range(N):
            term = U[idx] if idx == k else eye
            expr = term if expr is None else cp.kron(expr, term)
        return expr

    K = sum(kron_chain(k) for k in range(N))
    constraints = [C_np - K >> 0]

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver, verbose=verbose, *args, **kwargs)

    if U[0].value is None:
        raise ValueError(f'{solver} solver did not return a solution.')

    gamma_val = problem.A.dom.ctx.ops.asarray(constraints[0].dual_value, dtype=problem.A.dom.ctx.dtype)
    if np.iscomplexobj(gamma_val):
        gamma_val *= 2.
    primal = QOTPrimal(problem.A.dom, gamma_val, d=d, N=N)

    u_vals = problem.A.cod.ctx.ops.stack(
        [problem.A.cod.ctx.ops.asarray(Uk.value, dtype=problem.A.cod.ctx.dtype) for Uk in U],
        axis=0
    )
    dual = QOTDual(problem.A.cod, u_vals)

    return primal, dual
