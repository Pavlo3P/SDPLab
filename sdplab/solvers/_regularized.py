"""Backend dispatcher for smooth regularized SDP dual solvers."""

from __future__ import annotations

import logging
import math
import time

from spacecore import JaxOps, NumpyOps

try:
    from spacecore import TorchOps
except ImportError:  # pragma: no cover - depends on optional torch install
    TorchOps = ()

from ..regularization import SDPRegularized
from ..sdp import SDPDual
from ._common import problem_summary
from ._runner import OptimizeResult, _ensure_info_logging
from ._torch import solve_torch
from .jax import solve_optax


logger = logging.getLogger(__name__)


def run_regularized_solver(
    problem: SDPRegularized,
    init_dual: SDPDual | None = None,
    *,
    max_iter: int = 1000,
    tol: float = 1e-6,
    method: str | None = None,
    opt=None,
    learning_rate: float = 1e-2,
    verbose: int = 1,
    ascii_only: bool = False,
    color: bool | None = None,
    **kwargs,
) -> OptimizeResult:
    """Optimize a regularized SDP dual with the matching backend driver."""
    if init_dual is None:
        init_dual = problem.sdp.dual_from_array(problem.sdp.cod.zeros())

    ops = problem.sdp.ctx.ops
    if isinstance(ops, JaxOps):
        if opt is None:
            import optax

            opt = optax.adam(learning_rate)
        return solve_optax(
            problem,
            init_dual,
            opt=opt,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            ascii_only=ascii_only,
            color=color,
            **kwargs,
        )

    elif isinstance(ops, TorchOps):
        return solve_torch(
            problem,
            init_dual,
            opt=opt,
            max_iter=max_iter,
            tol=tol,
            learning_rate=learning_rate,
            verbose=verbose,
            ascii_only=ascii_only,
            color=color,
            **kwargs,
        )

    elif isinstance(ops, NumpyOps):
        return solve_scipy(
            problem,
            init_dual=init_dual,
            method=method,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            ascii_only=ascii_only,
            color=color,
            **kwargs,
        )

    raise ValueError(f"Unsupported regularized SDP backend: {type(ops).__name__}")


def solve_scipy(
    problem: SDPRegularized,
    *,
    init_dual: SDPDual | None = None,
    method: str | None = "L-BFGS-B",
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: int = 1,
    ascii_only: bool = False,
    color: bool | None = None,
    **kwargs,
) -> OptimizeResult:
    """Solve a regularized SDP dual through ``scipy.optimize.minimize``.

    SciPy owns its optimization loop, so progress reporting and per-iteration
    diagnostics are limited compared to ``solve_optax`` and ``solve_torch``.
    """
    if not isinstance(problem, SDPRegularized):
        raise TypeError("solve_scipy expects SDPRegularized.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    import numpy as np
    from scipy import optimize

    if init_dual is None:
        init_dual = problem.sdp.dual_from_array(problem.sdp.cod.zeros())

    x0 = np.asarray(init_dual.y).reshape(-1)
    shape = tuple(init_dual.y.shape)
    objective_values: list[float] = []

    def unpack(x):
        return problem.sdp.dual_from_array(np.asarray(x).reshape(shape))

    def objective(x):
        value = problem.dual_objective_reg(unpack(x))
        dual_obj = float(np.real(np.asarray(value)))
        objective_values.append(dual_obj)
        return -dual_obj

    options = dict(kwargs.pop("options", {}))
    options.setdefault("maxiter", max_iter)

    if int(verbose) >= 1:
        _ensure_info_logging(logger)
        logger.info("=" * 80)
        logger.info("Solver: scipy%s", f", method={method}" if method else "")
        logger.info("Problem: %s", problem_summary(problem))
        logger.info("Tolerance: scipy tol < %g", tol)
        logger.info("Max iterations: %d", max_iter)
        logger.info("=" * 80)

    start = time.perf_counter()
    scipy_result = optimize.minimize(
        objective,
        x0,
        method=method,
        tol=tol,
        options=options,
        **kwargs,
    )
    elapsed = time.perf_counter() - start

    final_grad_norm = _scipy_grad_norm(getattr(scipy_result, "jac", None))
    final_loss = -float(
        getattr(
            scipy_result,
            "fun",
            -objective_values[-1] if objective_values else math.nan,
        )
    )
    num_iters = int(getattr(scipy_result, "nit", len(objective_values)))

    result = OptimizeResult(
        dual=unpack(scipy_result.x),
        converged=bool(scipy_result.success),
        num_iters=num_iters,
        final_loss=final_loss,
        final_grad_norm=final_grad_norm,
        elapsed_seconds=elapsed,
        loss_history=objective_values,
        grad_norm_history=None,
        step_times=None,
    )
    result.scipy_result = scipy_result

    if int(verbose) >= 1:
        logger.info(result.summary())
        logger.info("=" * 80)

    return result


def _scipy_grad_norm(jac) -> float:
    if jac is None:
        return math.nan

    import numpy as np

    return float(np.linalg.norm(np.asarray(jac).reshape(-1)))


__all__ = ["run_regularized_solver", "solve_scipy"]
