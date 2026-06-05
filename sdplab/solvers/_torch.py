r"""PyTorch-based first-order solver for regularized SDP duals.

This is the eager Torch analogue of :func:`solve_optax`.  It maximizes
the smooth regularized dual objective by minimizing its negative with a
``torch.optim`` optimizer.  The optimization variable is a raw Torch tensor;
``SDPDual`` is used only at the public boundary.
"""

from __future__ import annotations

from typing import Any

from spacecore import Context, DenseArray

from ..regularization import SDPRegularized
from ..sdp import SDPDual
from ._common import dual_objective_array, problem_summary
from ._runner import OptimizeResult, run_solver


def _validate_torch_inputs(
    problem: SDPRegularized,
    max_iter: int,
    tol: float,
    log_every: int,
) -> None:
    """Validate public ``solve_torch`` inputs."""
    if not isinstance(problem, SDPRegularized):
        raise TypeError("solve_torch expects SDPRegularized.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")
    if getattr(problem.sdp.ops, "torch", None) is None:
        raise ValueError("solve_torch requires a Torch-backed SDP.")


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


def solve_torch(
    sdp: SDPRegularized,
    init_dual: SDPDual | None = None,
    opt=None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    learning_rate: float | None = 1e-2,
    verbose: int = 1,
    log_every: int = 50,
    ascii_only: bool = False,
    color: bool | None = None,
    **opt_kwargs,
) -> OptimizeResult:
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
        verbose: Verbosity level. ``0`` is silent, ``1`` prints header/footer,
            ``2`` prints periodic diagnostics, ``3`` prints every iteration,
            and ``4`` uses boxed verbose diagnostics.
        log_every: Diagnostic print interval when ``verbose >= 2``.
        ascii_only: Use ASCII-only verbose output when true.
        color: Whether to use ANSI color in fancy output. ``None`` auto-detects.
        **opt_kwargs: Additional keyword arguments forwarded to the optimizer.

    Returns:
        ``OptimizeResult`` with the final dual variable, loss and gradient
        histories, convergence status, and timing information.
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

    torch = loop_problem.sdp.ops.torch
    param = torch.nn.Parameter(init_y.detach().clone())
    optimizer = _make_torch_optimizer(
        torch=torch,
        opt=opt,
        params=[param],
        learning_rate=learning_rate,
        opt_kwargs=opt_kwargs,
    )

    def loss_fun():
        return -loop_problem.sdp.ops.real(dual_objective_array(loop_problem, param))

    is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
    lbfgs_loss = [None]
    lbfgs_grad_norm = [None]

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fun()
        loss.backward()
        lbfgs_loss[0] = loss.detach()
        lbfgs_grad_norm[0] = _grad_norm(param, torch)
        return loss

    def step_fn(state):
        del state
        if is_lbfgs:
            optimizer.step(closure)
            with torch.no_grad():
                accepted_loss = loss_fun().detach()
            return None, -accepted_loss, lbfgs_grad_norm[0].detach()

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fun()
        loss.backward()
        grad_norm = _grad_norm(param, torch)
        optimizer.step()
        return None, -loss.detach(), grad_norm.detach()

    def finalize_fn(state):
        del state
        return SDPDual(sdp.sdp.cod, param.detach().clone(), ctx=sdp.sdp.ctx)

    return run_solver(
        init_state=None,
        step_fn=step_fn,
        finalize_fn=finalize_fn,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        log_every=log_every,
        solver_name=f"torch, optimizer={optimizer.__class__.__name__}, backend=torch",
        problem_summary=problem_summary(sdp),
        initial_dual_norm=float(loop_problem.sdp.cod.norm(init_y)),
        ascii_only=ascii_only,
        color=color,
    )


__all__ = ["solve_torch"]
