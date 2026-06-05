r"""Pure-JAX first-order solver for regularized SDP duals.

This module is the no-Optax JAX counterpart to ``solve_optax``.  It uses
only JAX primitives for value/gradient evaluation and loop compilation.  The
update rule is caller-configurable; by default it performs fixed-step gradient
descent on the negative regularized dual objective.
"""

from __future__ import annotations

from time import time as Time
from typing import Callable

from spacecore import Context, DenseArray

from ...regularization import SDPRegularized
from ...sdp import SDPDual
from .._common import dual_objective_array
from .._runner import OptimizeResult
from ._optax import (
    _array_from_params,
    _params_from_array,
)


JaxUpdate = Callable[[object, object, DenseArray, DenseArray], object]


def gradient_descent_update(params, grad, value, learning_rate):
    """Default pure-JAX update: ``params - learning_rate * grad``."""
    import jax

    return jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g,
        params,
        grad,
    )


def _validate_jaxlib_inputs(
    problem: SDPRegularized,
    max_iter: int,
    tol: float,
    log_every: int,
) -> None:
    """Validate public ``run_jaxlib_solver`` inputs."""
    if not isinstance(problem, SDPRegularized):
        raise TypeError("run_jaxlib_solver expects SDPRegularized.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")
    if getattr(problem.sdp.ops, "jax", None) is None:
        raise ValueError("run_jaxlib_solver requires a JAX-backed SDP.")


def _problem_for_jit(problem: SDPRegularized) -> SDPRegularized:
    """Return an equivalent problem with runtime membership checks disabled."""
    ctx = Context(
        problem.sdp.ops,
        dtype=problem.sdp.ctx.dtype,
        enable_checks=False,
    )
    reg = type(problem.reg)(ctx.asarray(problem.reg.val), ctx=ctx)
    return SDPRegularized(problem.sdp.convert(ctx), reg)


def _tree_l2_norm(tree):
    """Return Euclidean norm of a real-valued JAX PyTree."""
    import jax
    import jax.numpy as jnp

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError("gradient tree has no leaves.")

    return jnp.sqrt(sum(jnp.real(jnp.vdot(leaf, leaf)) for leaf in leaves))


def _run_jaxlib_loop(
    problem: SDPRegularized,
    init_params,
    update: JaxUpdate,
    learning_rate: float,
    max_iter: int,
    tol: float,
):
    """Run the compiled pure-JAX loop."""
    import jax

    ops = problem.sdp.ops

    def loss_fun(params):
        y = _array_from_params(params)
        return -ops.real(dual_objective_array(problem, y))

    value_and_grad = jax.value_and_grad(loss_fun)

    carry0 = (
        init_params,
        ops.asarray(ops.inf),
        ops.zeros((max_iter,)),
        ops.zeros((max_iter,)),
        ops.asarray(0),
    )

    def cond_fun(carry):
        _, grad_norm, _, _, it = carry
        return (it < max_iter) & (grad_norm >= tol)

    def body_fun(carry):
        params, _, obj_log, grad_log, it = carry

        value, grad = value_and_grad(params)
        dual_obj = -value
        grad_norm = _tree_l2_norm(grad)
        params = update(params, grad, value, learning_rate)

        obj_log = ops.index_set(obj_log, it, dual_obj)
        grad_log = ops.index_set(grad_log, it, grad_norm)

        return params, grad_norm, obj_log, grad_log, it + 1

    params, grad_norm, obj_log, grad_log, n_iters = ops.while_loop(
        cond_fun,
        body_fun,
        carry0,
    )
    return _array_from_params(params), n_iters, grad_norm, obj_log, grad_log


def run_jaxlib_solver(
    sdp: SDPRegularized,
    init_dual: SDPDual | None = None,
    update: JaxUpdate | None = None,
    learning_rate: float = 1e-2,
    max_iter: int = 100000,
    tol: float = 1e-6,
    verbose: bool = False,
    log_every: int = 50,
) -> OptimizeResult:
    r"""Run a pure-JAX optimizer loop on the regularized SDP dual objective.

    Args:
        sdp: Regularized SDP whose dual objective is maximized.
        init_dual: Optional initial dual variable. If omitted, the zero element
            of the SDP codomain is used.
        update: Optional JAX-compatible update callable with signature
            ``update(params, grad, value, learning_rate) -> params``. If
            omitted, fixed-step gradient descent is used.
        learning_rate: Scalar passed to ``update``.
        max_iter: Maximum number of iterations.
        tol: Stop once the gradient norm is below this tolerance.
        verbose: If true, print diagnostics after the compiled loop returns.
        log_every: Diagnostic print interval when ``verbose`` is enabled.

    Returns:
        ``OptimizeResult`` with final dual variable, dual-objective history,
        gradient norm history, convergence status, and elapsed time.
    """
    _validate_jaxlib_inputs(
        problem=sdp,
        max_iter=max_iter,
        tol=tol,
        log_every=log_every,
    )
    if update is None:
        update = gradient_descent_update
    if init_dual is None:
        init_dual = SDPDual(sdp.sdp.cod, sdp.sdp.cod.zeros(), ctx=sdp.sdp.ctx)

    loop_problem = _problem_for_jit(sdp)
    init_y = loop_problem.sdp.ctx.asarray(init_dual.y)
    init_params = _params_from_array(loop_problem, init_y)

    def loop():
        return _run_jaxlib_loop(
            problem=loop_problem,
            init_params=init_params,
            update=update,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
        )

    jax = loop_problem.sdp.ops.jax
    compiled = jax.jit(loop).lower().compile()

    start = Time()
    final_y, n_iters, grad_norm, obj_hist, grad_hist = compiled()
    elapsed = Time() - start

    n_iters = int(n_iters)
    obj_hist = obj_hist[:n_iters]
    grad_hist = grad_hist[:n_iters]

    if verbose:
        for it in range(0, n_iters, log_every):
            print(
                f"[iter {it}] "
                f"dual_obj={float(obj_hist[it]):.6e} "
                f"grad_norm={float(grad_hist[it]):.6e}"
            )

    loss_history = [float(obj_hist[it]) for it in range(n_iters)]
    grad_norm_history = [float(grad_hist[it]) for it in range(n_iters)]
    final_loss = loss_history[-1] if loss_history else float("nan")
    return OptimizeResult(
        dual=SDPDual(sdp.sdp.cod, final_y, ctx=sdp.sdp.ctx),
        converged=float(grad_norm) < tol,
        num_iters=n_iters,
        final_loss=final_loss,
        final_grad_norm=float(grad_norm),
        elapsed_seconds=elapsed,
        loss_history=loss_history,
        grad_norm_history=grad_norm_history,
        step_times=None,
    )


__all__ = [
    "JaxUpdate",
    "gradient_descent_update",
    "run_jaxlib_solver",
]
