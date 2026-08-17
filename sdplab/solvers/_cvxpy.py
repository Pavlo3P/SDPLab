r"""CVXPY reference backend for conic problems over a Hermitian domain.

The single public entry point is :func:`run_cvxpy_solver`.

Encoding
--------
The backend asks the problem's :class:`~sdplab.problem.ConstraintOp` and
:class:`~sdplab.problem.Cost` for their cvxpy forms and assembles the standard
SDP directly, one scalar equality per constraint:

.. math::

    \min_{X \succeq 0}\ \operatorname{Re}\operatorname{Tr}[C\,X]
    \quad\text{s.t.}\quad
    \operatorname{Re}\operatorname{Tr}[A_i\,X] = b_i,\quad i = 0, \ldots, m-1,

where ``[A_0, ..., A_{m-1}] = A.to_cvxpy()`` are the per-constraint matrices,
``b = A.rhs_to_cvxpy(problem.b)`` the matching real right-hand side, and
``C = cost.to_cvxpy()`` the Hermitian cost matrix. The ``trace`` convention is
what cvxpy consumes natively -- a sparse ``A_i`` is accepted directly in
``cp.trace(A_i @ X)``.

The primal variable is a Hermitian (complex) or symmetric (real) cvxpy matrix
with ``X >> 0``. The domain must be a single :class:`~spacecore.HermitianSpace`;
structured (tree/stacked) domains are not handled by this backend.

Dual convention
---------------
cvxpy returns the negative of the standard-form equality dual, so the solver
passes ``-lambda`` (the stacked per-constraint multipliers) to
``A.dual_from_cvxpy``, which reassembles it into a ``cod`` element (a plain
vector for scalar constraints, Hermitian marginal blocks for QOT).
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import cvxpy as cp

from spacecore import Context, HermitianSpace, NumpyOps

from ..problem import SDPProblem, as_member


def _hermitian_variable(space: HermitianSpace) -> Tuple[cp.Variable, Any]:
    """Return ``(X, extract)`` for a Hermitian/symmetric cone variable ``X >> 0``."""
    n = space.n
    if space.field == "complex":
        X = cp.Variable((n, n), hermitian=True)
    else:
        X = cp.Variable((n, n), symmetric=True)

    def extract() -> np.ndarray:
        value = np.asarray(X.value)
        return (value + value.conj().T) / 2

    return X, extract


def run_cvxpy_solver(
    sdp: SDPProblem,
    solver: str = "MOSEK",
    verbose: bool = False,
    return_problem: bool = False,
    *args,
    **kwargs,
) -> Tuple[Any, Any] | Tuple[Any, Any, cp.Problem]:
    r"""Solve a Hermitian-domain SDP through CVXPY in per-constraint form.

    Args:
        sdp: Problem data ``(C, A, b)`` with a Hermitian domain. ``A`` is a
            :class:`~sdplab.problem.ConstraintOp` (plain ``LinOp`` inputs are
            wrapped by :class:`~sdplab.problem.SDPProblem`).
        solver: CVXPY solver name, e.g. ``"MOSEK"`` or ``"CLARABEL"``.
        verbose: Whether CVXPY prints solver progress.
        return_problem: Also return the ``cvxpy.Problem``.
        *args: Extra positional arguments passed to ``Problem.solve``.
        **kwargs: Extra keyword arguments passed to ``Problem.solve``.

    Returns:
        ``(X, y)`` in the original problem context: the optimized primal element
        of ``dom`` and the equality dual reassembled into ``cod``. With
        ``return_problem`` the ``cvxpy.Problem`` is appended.

    Raises:
        TypeError: If ``sdp`` is not an :class:`~sdplab.problem.SDPProblem`.
        NotImplementedError: If the domain is not a single Hermitian space.
        ValueError: If the solver does not return an optimal solution.
    """
    if not isinstance(sdp, SDPProblem):
        raise TypeError(
            f"run_cvxpy_solver expects an SDPProblem, got {type(sdp).__name__}."
        )

    # numpy backend data
    np_ctx = Context(ops=NumpyOps(), dtype=sdp.ctx.dtype)
    sdp_np = sdp.convert(np_ctx)
    dom = sdp_np.dom

    if not isinstance(dom, HermitianSpace):
        raise NotImplementedError(
            "The cvxpy backend supports only a single Hermitian domain; got "
            f"{type(dom).__name__}."
        )

    A_list = sdp_np.A.to_cvxpy()
    b_vec = np.asarray(sdp_np.A.rhs_to_cvxpy(sdp_np.b)).reshape(-1)
    C_mat = sdp_np.C.to_cvxpy()

    # 1) cone variable and per-constraint equalities
    X, extract = _hermitian_variable(dom)
    equalities = []
    for A_i, b_i in zip(A_list, b_vec):
        lhs = cp.trace(cp.Constant(A_i) @ X)
        equalities.append((cp.real(lhs) if lhs.is_complex() else lhs) == float(b_i))

    pairing = cp.trace(cp.Constant(C_mat) @ X)
    objective = cp.Minimize(cp.real(pairing) if pairing.is_complex() else pairing)

    # 2) solve
    prob = cp.Problem(objective, [X >> 0, *equalities])
    prob.solve(solver=solver, verbose=verbose, *args, **kwargs)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise ValueError(f"{solver} solver did not return a solution ({prob.status}).")

    # 3) pack results in the original context
    X_np = extract()

    lam = np.array([np.real(eq.dual_value) for eq in equalities], dtype=float)
    # cvxpy's equality dual is the negation of the standard-form dual.
    y_np = sdp_np.A.dual_from_cvxpy(-lam)

    X_val = as_member(sdp.dom, X_np, sdp.ctx)
    y_val = as_member(sdp.cod, y_np, sdp.ctx)

    if return_problem:
        return X_val, y_val, prob

    return X_val, y_val
