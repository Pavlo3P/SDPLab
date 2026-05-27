r"""CVXPY implementation of dense SDP primal solves.

This module solves dense instances of the SDP model

.. math::

    \min_{X \in \operatorname{dom}(\mathcal{A})}\quad\operatorname{Re}\operatorname{Tr}[C X]
    \quad \text{s.t.} \quad
        \mathcal{A}X = b,\quad
        X \succeq 0,

with an optional trace constraint :math:`\operatorname{Tr}[X] = \tau`.
"""

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
    return_problem: bool = False,
    *args, **kwargs
) -> Tuple[SDPPrimal, SDPDual] | Tuple[SDPPrimal, SDPDual, cp.Problem]:
    r"""Solve a dense SDP primal problem.

    The solved problem is

    .. math::

        \min_{X \in \operatorname{dom}(\mathcal{A})}\quad\operatorname{Re}\operatorname{Tr}[C X]
        \quad \text{s.t.} \quad
            \mathcal{A}X = b,\\
            X \succeq 0.

    In the common dense representation, :math:`\mathcal{A}` is stored through
    matrices :math:`A_1, \ldots, A_m`, and the equality constraint means
    :math:`\operatorname{Tr}[A_k X] = b_k` for :math:`k = 1, \ldots, m`.
    Here :math:`C, X \in \operatorname{dom}(\mathcal{A})`, while
    :math:`b \in \operatorname{cod}(\mathcal{A})`.

    If ``sdp.tau`` is set, the additional affine constraint
    :math:`\operatorname{Tr}[X] = \tau` is included. The returned dual variable
    contains the CVXPY multipliers for :math:`\mathcal{A}X = b`.

    Args:
        sdp: Dense SDP data ``(C, A, b)`` plus optional trace value.
        solver: CVXPY solver name, for example ``"MOSEK"``.
        verbose: Whether CVXPY should print solver progress.
        *args: Extra positional arguments passed to ``Problem.solve``.
        **kwargs: Extra keyword arguments passed to ``Problem.solve``.

    Returns:
        ``(primal, dual)`` where ``primal.X`` is the optimized matrix and
        ``dual.y`` stores equality-constraint multipliers in
        :math:`\operatorname{cod}(\mathcal{A})`.
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

    if return_problem:
        return primal, dual, prob

    return primal, dual
