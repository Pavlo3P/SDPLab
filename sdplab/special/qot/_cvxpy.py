from __future__ import annotations

import numpy as np
import cvxpy as cp

from spacecore import Context, NumpyOps
from sdplab.sdp import SDPDenseProblem, SDPPrimal, SDPDual
from ._block_space import BlockMatrixSpace
from ._constraint_op import QOTConstraintOp

def solve_qot_dual(
    qot: SDPDenseProblem,
    solver: str = 'MOSEK',
    verbose: bool = False,
    *args, **kwargs
) -> tuple[SDPPrimal, SDPDual]:
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

    if not (isinstance(qot.A, QOTConstraintOp) or isinstance(qot.cod, BlockMatrixSpace)):
        raise TypeError("Input problem is not Quantum Optimal Transport.")

    np_ctx = Context(ops=NumpyOps(), dtype=qot.ctx.dtype)
    problem = qot.convert(np_ctx)

    C = cp.Constant(qot.C)
    marginals = qot.b

    d = problem.A.d
    N = problem.A.N
    is_complex = np.iscomplexobj(C) or np.iscomplexobj(marginals)

    if is_complex:
        U = [cp.Variable((d, d), hermitian=True) for _ in range(N)]
        obj = cp.Maximize(cp.real(cp.sum([cp.trace(Uk @ cp.Constant(Rk)) for Uk, Rk in zip(U, marginals)])))
    else:
        U = [cp.Variable((d, d), symmetric=True) for _ in range(N)]
        obj = cp.Maximize(cp.sum([cp.trace(Uk @ cp.Constant(Rk)) for Uk, Rk in zip(U, marginals)]))

    # ----------- build K = sum_k I⊗...⊗U_k⊗...⊗I -----------
    eye = cp.Constant(np.eye(d, d, dtype=qot.ctx.dtype))
    def kron_chain(k: int) -> cp.Expression:
        expr = None
        for idx in range(N):
            term = U[idx] if idx == k else eye
            expr = term if expr is None else cp.kron(expr, term)
        return expr

    K = sum(kron_chain(k) for k in range(N))
    constraints = [C - K >> 0]

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver, verbose=verbose, *args, **kwargs)

    if U[0].value is None:
        raise ValueError(f'{solver} solver did not return a solution.')

    gamma_val = problem.A.dom.ctx.ops.asarray(constraints[0].dual_value, dtype=problem.A.dom.ctx.dtype)
    if np.iscomplexobj(gamma_val):
        gamma_val *= 2.
    primal = qot.primal_from_array(gamma_val)

    u_vals = problem.A.cod.ctx.ops.stack(
        [problem.A.cod.ctx.ops.asarray(Uk.value, dtype=problem.A.cod.ctx.dtype) for Uk in U],
        axis=0
    )
    dual = qot.dual_from_array(u_vals)

    return primal, dual
