r"""CVXPY solver for the QOT dual semidefinite program."""

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
    r"""Solve the QOT dual SDP.

    For the QOT constraint operator :math:`\mathcal{A}`, the primal coupling
    satisfies
    :math:`\Gamma \in \operatorname{dom}(\mathcal{A}) = \operatorname{Herm}(d^N)`
    and the marginal data satisfy
    :math:`\gamma = (\gamma_0, \ldots, \gamma_{N-1})
    \in \operatorname{cod}(\mathcal{A}) = \operatorname{Herm}(d)^N`.
    The dual problem is

    .. math::

        \max_{U \in \operatorname{cod}(\mathcal{A})}\quad
        \sum_k \operatorname{Tr}[U_k \gamma_k]
        \quad \text{s.t.}\quad
        \mathcal{A}^\dagger U \preceq C.

    Here :math:`C \in \operatorname{dom}(\mathcal{A})` is the cost matrix and
    :math:`U = (U_0, \ldots, U_{N-1})` is the block dual variable. The adjoint
    constraint is

    .. math::

        \mathcal{A}^\dagger U
        =
        U_0 \oplus \cdots \oplus U_{N-1}
        =
        \sum_k I \otimes \cdots \otimes U_k \otimes \cdots \otimes I
        \preceq C,

    equivalently :math:`C - \mathcal{A}^\dagger U \succeq 0`.

    Returns:
        A pair ``(primal, dual)``. The primal value stores the coupling
        :math:`\Gamma`, represented as the positive-semidefinite multiplier
        for :math:`C - \mathcal{A}^\dagger U \succeq 0`. The dual value stores
        the optimized blocks :math:`U_k` in
        :math:`\operatorname{cod}(\mathcal{A})`.
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
        r"""Return the Kronecker embedding of the ``k``-th dual block."""
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

    Gamma_val = problem.A.dom.ctx.ops.asarray(constraints[0].dual_value, dtype=problem.A.dom.ctx.dtype)
    if np.iscomplexobj(Gamma_val):
        Gamma_val *= 2.
    primal = qot.primal_from_array(Gamma_val)

    u_vals = problem.A.cod.ctx.ops.stack(
        [problem.A.cod.ctx.ops.asarray(Uk.value, dtype=problem.A.cod.ctx.dtype) for Uk in U],
        axis=0
    )
    dual = qot.dual_from_array(u_vals)

    return primal, dual
