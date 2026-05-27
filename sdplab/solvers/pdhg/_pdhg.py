r"""Backend-neutral PDHG driver for dense semidefinite programs.

The public function validates library-level inputs and wraps final arrays back
into SDP variables.  The optimization loop itself works only with backend
arrays and uses ``BackendOps`` control-flow primitives, which keeps the same
loop usable by eager and JIT-capable backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import time as Time
from typing import TypeAlias

from spacecore import Context, DenseArray

from ...regularization import QuadraticReg, SDPRegularized
from ...sdp import SDPDenseProblem, SDPDual, SDPPrimal
from .._info import ConvergenceInfo
from ._steps import (
    pdhg_dual_update,
    pdhg_primal_update,
)


SDPOrQuadraticReg: TypeAlias = SDPDenseProblem | SDPRegularized


@dataclass(frozen=True)
class _PDHGProblemData:
    """Problem data normalized before entering the array-only loop."""

    sdp: SDPDenseProblem
    sq_reg: DenseArray
    sq_reg_float: float
    quadratic_problem: SDPRegularized | None


@dataclass(frozen=True)
class _PDHGLoopResult:
    """Raw fixed-size output returned by the backend loop."""

    primal: DenseArray
    dual: DenseArray
    n_iters: DenseArray
    residual: DenseArray
    primal_obj: DenseArray
    dual_obj: DenseArray
    residuals: DenseArray


def _base_sdp(problem: SDPOrQuadraticReg) -> SDPDenseProblem:
    """Return the dense SDP stored directly or inside ``SDPRegularized``."""
    sdp = problem.sdp if isinstance(problem, SDPRegularized) else problem
    if not isinstance(sdp, SDPDenseProblem):
        raise TypeError("PDHG currently supports only SDPDenseProblem data.")
    return sdp


def _quadratic_strength(
    problem: SDPOrQuadraticReg,
    sq_reg: float | None,
) -> DenseArray:
    """Return the quadratic strength as a scalar in the problem context."""
    sdp = _base_sdp(problem)

    if isinstance(problem, SDPRegularized):
        if not isinstance(problem.reg, QuadraticReg):
            raise TypeError("PDHG supports SDPRegularized only with QuadraticReg.")
        if sq_reg is not None:
            raise ValueError("sq_reg must be None when problem is SDPRegularized.")
        return problem.reg.val

    return sdp.ctx.asarray(0.0 if sq_reg is None else sq_reg)


def _regularization_strength_float(sdp: SDPDenseProblem, sq_reg: DenseArray) -> float:
    """Return ``sq_reg`` as a Python float for wrapper-side validation."""
    ops = sdp.ops
    imag = float(ops.abs(ops.imag(sq_reg)))
    if imag > 0.0:
        raise ValueError("sq_reg must be real.")
    return float(ops.real(sq_reg))


def _as_quadratic_problem(
    problem: SDPOrQuadraticReg,
    sq_reg: DenseArray,
    sq_reg_float: float,
) -> SDPRegularized | None:
    """Return an ``SDPRegularized`` view for nonzero quadratic strength."""
    if sq_reg_float <= 0.0:
        return None

    if isinstance(problem, SDPRegularized):
        return problem

    sdp = _base_sdp(problem)
    return SDPRegularized(sdp, QuadraticReg(sq_reg, ctx=sdp.ctx))


def _prepare_problem_data(
    problem: SDPOrQuadraticReg,
    sq_reg: float | None,
) -> _PDHGProblemData:
    """Normalize plain and regularized inputs before launching the loop."""
    sdp = _base_sdp(problem)
    sq_reg_arr = _quadratic_strength(problem, sq_reg)
    sq_reg_float = _regularization_strength_float(sdp, sq_reg_arr)

    if sq_reg_float < 0.0:
        raise ValueError("sq_reg must be nonnegative.")

    return _PDHGProblemData(
        sdp=sdp,
        sq_reg=sq_reg_arr,
        sq_reg_float=sq_reg_float,
        quadratic_problem=_as_quadratic_problem(problem, sq_reg_arr, sq_reg_float),
    )


def _quadratic_value(
    problem: SDPRegularized,
    primal_x: DenseArray,
) -> DenseArray:
    """Evaluate ``QuadraticReg`` through SDP-domain spectral calculus."""
    eigvals, _ = problem.sdp.dom.eigh(primal_x)
    return problem.reg.val * problem.reg._phi(eigvals)


def _primal_objective(
    data: _PDHGProblemData,
    primal_x: DenseArray,
) -> DenseArray:
    r"""Return the logged primal objective.

    For ``sq_reg > 0`` this computes
    :math:`\langle C,X\rangle + \mu \operatorname{Tr}[\varphi(X)]` with
    ``QuadraticReg``'s spectral calculus.
    """
    linear = data.sdp.ops.real(data.sdp.dom.inner(data.sdp.C, primal_x))
    if data.quadratic_problem is None:
        return linear
    return linear + _quadratic_value(data.quadratic_problem, primal_x)


def _dual_objective(
    data: _PDHGProblemData,
    dual_pdhg_y: DenseArray,
) -> DenseArray:
    r"""Return the regularized dual objective in package sign convention."""
    sdp = data.sdp
    reg = data.quadratic_problem.reg
    y = -dual_pdhg_y
    slack = sdp.A.rapply(y) - sdp.C
    slack_eigvals, _ = sdp.dom.eigh(slack)
    slack_eigvals = reg.ops.real(slack_eigvals / reg.val)
    obj = sdp.ops.real(sdp.cod.inner(sdp.b, y))
    return obj - reg.val * reg._phi_star(slack_eigvals)


def pdhg_residual(
    sdp: SDPDenseProblem,
    primal_new: SDPPrimal,
    primal_prev: SDPPrimal,
    dual_new: SDPDual,
    dual_prev: SDPDual,
) -> DenseArray:
    r"""Return a PDHG residual diagnostic for typed SDP variables."""
    return _pdhg_residual_array(
        sdp=sdp,
        primal_new=primal_new.X,
        primal_prev=primal_prev.X,
        dual_new=dual_new.y,
        dual_prev=dual_prev.y,
    )


def _pdhg_residual_array(
    sdp: SDPDenseProblem,
    primal_new: DenseArray,
    primal_prev: DenseArray,
    dual_new: DenseArray,
    dual_prev: DenseArray,
) -> DenseArray:
    r"""Return the PDHG residual using only backend arrays."""
    feasibility = sdp.A.apply(primal_new) - sdp.b
    dx = primal_new - primal_prev
    dy = dual_new - dual_prev
    return sdp.ops.sqrt(
        sdp.cod.norm(feasibility) ** 2
        + sdp.dom.norm(dx) ** 2
        + sdp.cod.norm(dy) ** 2
    )


def pdhg_iteration(
    problem: SDPDenseProblem,
    primal_prev: DenseArray,
    dual_prev: DenseArray,
    primal_bar: DenseArray,
    tau: float,
    sigma: float,
    theta: float = 1.0,
    sq_reg: float = 0.0,
) -> tuple[DenseArray, DenseArray, DenseArray]:
    r"""Run one PDHG step on raw backend arrays."""
    dual_new = pdhg_dual_update(
        problem=problem,
        dual_prev=dual_prev,
        primal_bar=primal_bar,
        sigma=sigma,
    )

    primal_new = pdhg_primal_update(
        problem=problem,
        primal_prev=primal_prev,
        dual_new=dual_new,
        tau=tau,
        sq_reg=sq_reg,
    )

    primal_bar_new = primal_new + theta * (primal_new - primal_prev)
    return primal_new, dual_new, primal_bar_new


def _run_pdhg_loop(
    data: _PDHGProblemData,
    init_primal: DenseArray,
    init_dual: DenseArray,
    tau: float,
    sigma: float,
    theta: float,
    max_iter: int,
    tol: float,
) -> tuple[DenseArray, DenseArray, DenseArray, DenseArray, DenseArray, DenseArray, DenseArray]:
    r"""Run the array-only backend loop.

    Histories are allocated at ``max_iter`` length and trimmed by the wrapper.
    The stopping condition and conditional dual-objective logging use
    ``BackendOps`` primitives instead of Python control flow.
    """
    sdp = data.sdp
    ops = sdp.ops
    has_dual_objective = data.quadratic_problem is not None

    carry0 = (
        init_primal,
        init_dual,
        init_primal,
        ops.zeros((max_iter,)),
        ops.zeros((max_iter,)),
        ops.zeros((max_iter,)),
        ops.asarray(0),
        ops.asarray(ops.inf),
    )

    def cond_fun(carry):
        *_, it, residual = carry
        return (it < max_iter) & (residual >= tol)

    def body_fun(carry):
        (
            primal,
            dual,
            primal_bar,
            primal_obj_log,
            dual_obj_log,
            residual_log,
            it,
            residual_prev,
        ) = carry

        primal_prev = primal
        dual_prev = dual
        primal, dual, primal_bar = pdhg_iteration(
            problem=sdp,
            primal_prev=primal_prev,
            dual_prev=dual_prev,
            primal_bar=primal_bar,
            tau=tau,
            sigma=sigma,
            theta=theta,
            sq_reg=data.sq_reg,
        )

        it_next = it + 1
        check_residual = (it_next % 64 == 0) | (it_next == max_iter)

        primal_obj = _primal_objective(data, primal)
        residual = ops.cond(
            check_residual,
            lambda args: _pdhg_residual_array(
                sdp=sdp,
                primal_new=args[0],
                primal_prev=args[1],
                dual_new=args[2],
                dual_prev=args[3],
            ),
            lambda args: args[4],
            (primal, primal_prev, dual, dual_prev, residual_prev),
        )
        dual_obj = (
            _dual_objective(data, dual)
            if has_dual_objective
            else ops.asarray(0.0)
        )

        primal_obj_log = ops.index_set(primal_obj_log, it, primal_obj)
        residual_log = ops.index_set(residual_log, it, residual)
        dual_obj_log = ops.cond(
            has_dual_objective,
            lambda log: ops.index_set(log, it, dual_obj),
            lambda log: log,
            dual_obj_log,
        )

        return (
            primal,
            dual,
            primal_bar,
            primal_obj_log,
            dual_obj_log,
            residual_log,
            it_next,
            residual,
        )

    (
        primal,
        dual,
        _,
        primal_obj,
        dual_obj,
        residuals,
        n_iters,
        residual,
    ) = ops.while_loop(cond_fun, body_fun, carry0)

    return primal, dual, n_iters, residual, primal_obj, dual_obj, residuals


def _validate_pdhg_inputs(
    problem: SDPOrQuadraticReg,
    init_primal: SDPPrimal | None,
    init_dual: SDPDual | None,
    tau: float,
    sigma: float,
    theta: float,
    max_iter: int,
    tol: float,
    log_every: int,
) -> None:
    """Validate public ``run_pdhg_solver`` inputs."""
    _base_sdp(problem)

    if tau is None:
        raise ValueError("tau must be provided.")
    if sigma is None:
        raise ValueError("sigma must be provided.")
    if max_iter is None:
        raise ValueError("max_iter must be provided.")
    if tol is None:
        raise ValueError("tol must be provided.")
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if theta < 0 or theta > 1:
        raise ValueError("theta must lie in [0, 1].")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")


def _loop_data_for_execution(data: _PDHGProblemData, jit: bool | None) -> _PDHGProblemData:
    """Return loop data, disabling membership checks only for JIT tracing."""
    if not jit:
        return data

    no_check_ctx = Context(
        data.sdp.ops,
        dtype=data.sdp.ctx.dtype,
        enable_checks=False,
    )
    loop_sdp = data.sdp.convert(no_check_ctx)
    loop_sq_reg = no_check_ctx.asarray(data.sq_reg)
    loop_quadratic_problem = (
        SDPRegularized(loop_sdp, QuadraticReg(loop_sq_reg, ctx=no_check_ctx))
        if data.quadratic_problem is not None
        else None
    )
    return _PDHGProblemData(
        sdp=loop_sdp,
        sq_reg=loop_sq_reg,
        sq_reg_float=data.sq_reg_float,
        quadratic_problem=loop_quadratic_problem,
    )


def _execute_loop(
    data: _PDHGProblemData,
    init_primal: SDPPrimal | None,
    init_dual: SDPDual | None,
    tau: float,
    sigma: float,
    theta: float,
    max_iter: int,
    tol: float,
    jit: bool | None,
) -> _PDHGLoopResult:
    """Run the array-only loop eagerly or through backend JIT."""
    loop_data = _loop_data_for_execution(data, jit)
    init_primal_x = (
        loop_data.sdp.dom.zeros()
        if init_primal is None
        else loop_data.sdp.ctx.asarray(init_primal.X)
    )
    init_dual_y = (
        loop_data.sdp.cod.zeros()
        if init_dual is None
        else loop_data.sdp.ctx.asarray(init_dual.y)
    )

    def loop():
        return _run_pdhg_loop(
            data=loop_data,
            init_primal=init_primal_x,
            init_dual=init_dual_y,
            tau=tau,
            sigma=sigma,
            theta=theta,
            max_iter=max_iter,
            tol=tol,
        )

    if not jit:
        return _PDHGLoopResult(*loop())

    jax = getattr(loop_data.sdp.ops, "jax", None)
    if jax is None:
        raise ValueError("jit=True requires a backend exposing a JAX jit.")

    return _PDHGLoopResult(*jax.jit(loop).lower().compile()())


def _tidy_pdhg_output(
    data: _PDHGProblemData,
    raw: _PDHGLoopResult,
    tol: float,
    elapsed: float,
    verbose: bool,
    log_every: int,
) -> ConvergenceInfo:
    """Slice fixed-size loop histories and build ``ConvergenceInfo``."""
    n_iters = int(raw.n_iters)
    primal_obj = raw.primal_obj[:n_iters]
    residuals = raw.residuals[:n_iters]
    dual_obj = (
        raw.dual_obj[:n_iters]
        if data.quadratic_problem is not None
        else None
    )

    if verbose:
        for it in range(0, n_iters, log_every):
            print(
                f"[iter {it}] "
                f"primal_obj={float(primal_obj[it]):.6e} "
                f"resid={float(residuals[it]):.6e}"
            )

    return ConvergenceInfo(
        primal=SDPPrimal(data.sdp.dom, raw.primal, ctx=data.sdp.ctx),
        dual=SDPDual(data.sdp.cod, raw.dual, ctx=data.sdp.ctx),
        primal_obj=primal_obj,
        dual_obj=dual_obj,
        grad_norm=residuals,
        tol_reached=float(raw.residual) < tol,
        time=elapsed,
    )


def run_pdhg_solver(
    problem: SDPOrQuadraticReg,
    init_primal: SDPPrimal | None = None,
    init_dual: SDPDual | None = None,
    tau: float | None = None,
    sigma: float | None = None,
    max_iter: int | None = None,
    tol: float | None = None,
    theta: float = 1.0,
    sq_reg: float | None = None,
    jit: bool | None = None,
    verbose: bool = False,
    log_every: int = 50,
) -> ConvergenceInfo:
    r"""Run PDHG on a dense SDP with optional zero-centered quadratic regularization.

    The wrapper validates inputs, prepares the scalar quadratic strength,
    optionally JIT-compiles the backend loop, and trims fixed-size histories
    returned by that loop.
    """
    _validate_pdhg_inputs(
        problem=problem,
        init_primal=init_primal,
        init_dual=init_dual,
        tau=tau,
        sigma=sigma,
        theta=theta,
        max_iter=max_iter,
        tol=tol,
        log_every=log_every,
    )

    data = _prepare_problem_data(problem=problem, sq_reg=sq_reg)

    start = Time()
    raw = _execute_loop(
        data=data,
        init_primal=init_primal,
        init_dual=init_dual,
        tau=tau,
        sigma=sigma,
        theta=theta,
        max_iter=max_iter,
        tol=tol,
        jit=jit,
    )
    elapsed = Time() - start

    return _tidy_pdhg_output(
        data=data,
        raw=raw,
        tol=tol,
        elapsed=elapsed,
        verbose=verbose,
        log_every=log_every,
    )
