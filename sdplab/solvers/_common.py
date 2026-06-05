"""Shared helpers for regularized SDP dual solvers."""

from __future__ import annotations

from spacecore import DenseArray

from ..regularization import SDPRegularized


def dual_objective_array(problem: SDPRegularized, y: DenseArray) -> DenseArray:
    r"""Evaluate the regularized dual objective using only backend arrays."""
    sdp = problem.sdp
    reg = problem.reg

    slack = sdp.A.rapply(y) - sdp.C
    eigvals, _ = sdp.dom.spectral_decompose(slack)
    eigvals = reg.ops.real(eigvals / reg.val)

    linear = sdp.ops.real(sdp.cod.inner(sdp.b, y))
    return linear - reg.val * reg._phi_star(eigvals)


def problem_summary(problem: SDPRegularized) -> str:
    """Return a compact problem summary for solver logs."""
    try:
        reg_val = float(problem.reg.val)
    except TypeError:
        reg_val = problem.reg.val
    return (
        f"{type(problem).__name__} with "
        f"{type(problem.reg).__name__}(eps={reg_val})"
    )


__all__ = ["dual_objective_array", "problem_summary"]
