from __future__ import annotations

from spacecore import DenseArray

from ...sdp import SDPDenseProblem, SDPPrimal, SDPDual


def pdhg_primal_update(
    problem: SDPDenseProblem,
    primal_prev: DenseArray,
    dual_new: DenseArray,
    tau: float,
    sq_reg: float = 0.0,
) -> DenseArray:
    r"""Return the primal PDHG update as a raw backend array."""
    x_new = primal_prev - tau * (problem.C + problem.A.rapply(dual_new))
    x_new = x_new / (1.0 + tau * sq_reg)
    return problem.dom.psd_proj(x_new)


def pdhg_dual_update(
    problem: SDPDenseProblem,
    dual_prev: DenseArray,
    primal_bar: DenseArray,
    sigma: float,
) -> DenseArray:
    r"""Return the dual ascent PDHG update as a raw backend array."""
    return dual_prev + sigma * (problem.A.apply(primal_bar) - problem.b)
