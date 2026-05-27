r"""PyTorch-based first-order solver for regularized SDP duals.

This is the eager Torch analogue of :func:`run_optax_solver`.  It maximizes
the smooth regularized dual objective by minimizing its negative with a
``torch.optim`` optimizer.  The optimization variable is a raw Torch tensor;
``SDPDual`` is used only at the public boundary.
"""

from __future__ import annotations

from time import time as Time
from typing import Any

from spacecore import Context, DenseArray

from ..regularization import SDPRegularized
from ..sdp import SDPDual
from ._info import ConvergenceInfo
from .jax._optax import _dual_objective_array


def _validate_torch_inputs(
    problem: SDPRegularized,
    max_iter: int,
    tol: float,
    log_every: int,
) -> None:
    """Validate public ``run_torch_solver`` inputs."""
    if not isinstance(problem, SDPRegularized):
        raise TypeError("run_torch_solver expects SDPRegularized.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")
    if getattr(problem.sdp.ops, "torch", None) is None:
        raise ValueError("run_torch_solver requires a Torch-backed SDP.")


def _problem_for_torch_loop(problem: SDPRegularized) -> SDPRegularized:
    """Return an equivalent problem with runtime membership checks disabled."""
    ctx = Context(
        problem.sdp.ops,
        dtype=problem.sdp.ctx.dtype,
        enable_checks=False,
    )
    reg = type(problem.reg)(ctx.asarray(problem.reg.val), ctx=ctx)
    return SDPRegularized(problem.sdp.convert(ctx), reg)


def _make_torch_optimizer(
    torch,
    opt,
    params: list,
    learning_rate: float | None,
    opt_kwargs: dict[str, Any],
):
    """Build a Torch optimizer from a class or factory."""
    if opt is None:
        opt = torch.optim.Adam

    if isinstance(opt, torch.optim.Optimizer):
        raise TypeError(
            "Pass a Torch optimizer class or factory, not an already-created "
            "optimizer instance."
        )

    kwargs = dict(opt_kwargs)
    if learning_rate is not None and "lr" not in kwargs:
        kwargs["lr"] = learning_rate
    return opt(params, **kwargs)


def _grad_norm(param, torch) -> DenseArray:
    """Return the Euclidean norm of the current parameter gradient."""
    if param.grad is None:
        return torch.zeros((), dtype=torch.real(param).dtype, device=param.device)
    return torch.sqrt(torch.sum(torch.abs(param.grad.detach()) ** 2))


def _run_torch_loop(
    problem: SDPRegularized,
    init_y: DenseArray,
    opt,
    learning_rate: float | None,
    max_iter: int,
    tol: float,
    opt_kwargs: dict[str, Any],
):
    """Run the eager Torch optimization loop."""
    torch = problem.sdp.ops.torch

    param = torch.nn.Parameter(init_y.detach().clone())
    optimizer = _make_torch_optimizer(
        torch=torch,
        opt=opt,
        params=[param],
        learning_rate=learning_rate,
        opt_kwargs=opt_kwargs,
    )

    dual_obj = problem.sdp.ops.zeros((max_iter,))
    grad_hist = problem.sdp.ops.zeros((max_iter,))
    grad_norm = problem.sdp.ops.asarray(problem.sdp.ops.inf)
    n_iters = 0

    def loss_fun():
        return -problem.sdp.ops.real(_dual_objective_array(problem, param))

    is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)

    for it in range(max_iter):
        if float(grad_norm) < tol:
            break

        if is_lbfgs:
            last_loss = None
            last_grad_norm = None

            def closure():
                nonlocal last_loss, last_grad_norm
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fun()
                loss.backward()
                last_loss = loss.detach()
                last_grad_norm = _grad_norm(param, torch)
                return loss

            optimizer.step(closure)
            loss = last_loss
            grad_norm = last_grad_norm
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fun()
            loss.backward()
            grad_norm = _grad_norm(param, torch)
            optimizer.step()

        dual_obj[it] = -loss.detach()
        grad_hist[it] = grad_norm.detach()
        n_iters = it + 1

    return param.detach().clone(), n_iters, grad_norm.detach(), dual_obj, grad_hist


def run_torch_solver(
    sdp: SDPRegularized,
    init_dual: SDPDual | None = None,
    opt=None,
    max_iter: int = 100000,
    tol: float = 1e-6,
    learning_rate: float | None = 1e-2,
    verbose: bool = False,
    log_every: int = 50,
    **opt_kwargs,
) -> ConvergenceInfo:
    r"""Run a PyTorch optimizer on the regularized SDP dual objective.

    Args:
        sdp: Regularized SDP whose dual objective is maximized.
        init_dual: Optional initial dual variable. If omitted, the zero element
            of the SDP codomain is used.
        opt: Torch optimizer class or factory. Examples:
            ``torch.optim.Adam``, ``torch.optim.SGD``, ``torch.optim.LBFGS``.
            Defaults to ``torch.optim.Adam``.
        max_iter: Maximum number of optimizer iterations.
        tol: Stop once the gradient norm is below this tolerance.
        learning_rate: Default ``lr`` passed to the optimizer when ``lr`` is
            not present in ``opt_kwargs``. Set to ``None`` to use the optimizer
            default.
        verbose: If true, print logged diagnostics after the loop returns.
        log_every: Diagnostic print interval when ``verbose`` is enabled.
        **opt_kwargs: Additional keyword arguments forwarded to the optimizer.

    Returns:
        ``ConvergenceInfo`` with the final dual variable, dual-objective
        history, gradient norm history, tolerance flag, and elapsed time.
    """
    _validate_torch_inputs(
        problem=sdp,
        max_iter=max_iter,
        tol=tol,
        log_every=log_every,
    )

    if init_dual is None:
        init_dual = SDPDual(sdp.sdp.cod, sdp.sdp.cod.zeros(), ctx=sdp.sdp.ctx)

    loop_problem = _problem_for_torch_loop(sdp)
    init_y = loop_problem.sdp.ctx.asarray(init_dual.y)

    start = Time()
    final_y, n_iters, grad_norm, dual_obj, grad_hist = _run_torch_loop(
        problem=loop_problem,
        init_y=init_y,
        opt=opt,
        learning_rate=learning_rate,
        max_iter=max_iter,
        tol=tol,
        opt_kwargs=opt_kwargs,
    )
    elapsed = Time() - start

    dual_obj = dual_obj[:n_iters]
    grad_hist = grad_hist[:n_iters]

    if verbose:
        for it in range(0, n_iters, log_every):
            print(
                f"[iter {it}] "
                f"dual_obj={float(dual_obj[it]):.6e} "
                f"grad_norm={float(grad_hist[it]):.6e}"
            )

    return ConvergenceInfo(
        dual=SDPDual(sdp.sdp.cod, final_y, ctx=sdp.sdp.ctx),
        dual_obj=dual_obj,
        grad_norm=grad_hist,
        tol_reached=float(grad_norm) < tol,
        time=elapsed,
    )


__all__ = ["run_torch_solver"]
