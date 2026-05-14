"""Backend dispatcher for smooth regularized SDP dual optimization."""

from __future__ import annotations

from time import time as Time

from spacecore import JaxOps, NumpyOps

from sdplab.regularization import SDPRegularized
from sdplab.sdp import SDPDual
from sdplab.solvers import ConvergenceInfo, run_optax_solver


def run_regularized_solver(
    problem: SDPRegularized,
    init_dual: SDPDual | None = None,
    *,
    max_iter: int = 1000,
    tol: float = 1e-6,
    method: str | None = None,
    opt=None,
    learning_rate: float = 1e-2,
    verbose: bool = False,
    **kwargs,
) -> ConvergenceInfo:
    """Optimize a regularized SDP dual with the solver matching its backend.

    NumPy-backed problems are sent to ``scipy.optimize.minimize``. JAX-backed
    problems are sent to the existing Optax loop.
    """
    if init_dual is None:
        init_dual = problem.sdp.dual_from_array(problem.sdp.cod.zeros())

    ops = problem.sdp.ctx.ops
    if isinstance(ops, JaxOps):
        if opt is None:
            import optax

            opt = optax.adam(learning_rate)
        return run_optax_solver(
            problem,
            init_dual,
            opt=opt,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            **kwargs,
        )

    if isinstance(ops, NumpyOps):
        return _run_scipy_solver(
            problem,
            init_dual,
            max_iter=max_iter,
            tol=tol,
            method=method,
            **kwargs,
        )

    raise ValueError(f"Unsupported regularized SDP backend: {type(ops).__name__}")


def _run_scipy_solver(
    problem: SDPRegularized,
    init_dual: SDPDual,
    *,
    max_iter: int,
    tol: float,
    method: str | None,
    **kwargs,
) -> ConvergenceInfo:
    """Run SciPy's scalar optimizer on the NumPy dual objective."""
    import numpy as np
    from scipy import optimize

    x0 = np.asarray(init_dual.y).reshape(-1)
    shape = tuple(init_dual.y.shape)
    dual_obj = []

    def unpack(x):
        return problem.sdp.dual_from_array(np.asarray(x).reshape(shape))

    def objective(x):
        value = problem.dual_objective_reg(unpack(x))
        value = float(np.real(np.asarray(value)))
        dual_obj.append(value)
        return -value

    options = dict(kwargs.pop("options", {}))
    options.setdefault("maxiter", max_iter)

    start = Time()
    result = optimize.minimize(
        objective,
        x0,
        method=method,
        tol=tol,
        options=options,
        **kwargs,
    )
    end = Time()

    final_dual = unpack(result.x)
    return ConvergenceInfo(
        dual=final_dual,
        dual_obj=np.asarray(dual_obj, dtype=float),
        tol_reached=bool(result.success),
        time=end - start,
    )
