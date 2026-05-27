r"""Optax-based first-order solver for regularized SDP duals.

The solver maximizes the smooth regularized dual objective

.. math::

    D_\varepsilon(y)
    =
    \langle b,y\rangle
    - \varepsilon\operatorname{Tr}\left[
        \psi\left((\mathcal A^\dagger y-C)/\varepsilon\right)
      \right].

Optax minimizes functions, so the compiled loop minimizes ``-D_eps``.  The loop
works directly on backend arrays and PyTrees; SDP variable wrappers are used
only at the public boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import time as Time

from spacecore import Context, DenseArray, jax_pytree_class

from ...regularization import SDPRegularized
from ...sdp import SDPDual
from .._info import ConvergenceInfo


@jax_pytree_class
@dataclass(frozen=True)
class DualReIm:
    r"""Real-valued PyTree representation of a complex dual array."""

    re: DenseArray
    im: DenseArray

    @classmethod
    def from_array(cls, array: DenseArray, ops) -> "DualReIm":
        """Split ``array`` into real and imaginary PyTree leaves."""
        return cls(ops.real(array), ops.imag(array))

    def get_array(self) -> DenseArray:
        """Recombine real and imaginary leaves into a complex array."""
        return self.re + 1j * self.im

    def tree_flatten(self):
        """Return PyTree children and auxiliary data for JAX."""
        return (self.re, self.im), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild the split representation from PyTree leaves."""
        re, im = children
        return cls(re, im)


def _is_complex_array(sdp: SDPRegularized, array: DenseArray) -> bool:
    """Return whether ``array`` has a complex dtype in the problem backend."""
    dtype = sdp.sdp.ops.get_dtype(array)
    return getattr(dtype, "kind", None) == "c" or "complex" in str(dtype)


def _params_from_array(sdp: SDPRegularized, array: DenseArray):
    """Return optimizer parameters for ``array``."""
    if _is_complex_array(sdp, array):
        return DualReIm.from_array(array, sdp.sdp.ops)
    return array


def _array_from_params(params):
    """Return a dual array from optimizer parameters."""
    if isinstance(params, DualReIm):
        return params.get_array()
    return params


def _dual_objective_array(problem: SDPRegularized, y: DenseArray) -> DenseArray:
    r"""Evaluate the regularized dual objective using only backend arrays."""
    sdp = problem.sdp
    reg = problem.reg

    slack = sdp.A.rapply(y) - sdp.C
    eigvals, _ = sdp.dom.eigh(slack)
    eigvals = reg.ops.real(eigvals / reg.val)

    linear = sdp.ops.real(sdp.cod.inner(sdp.b, y))
    return linear - reg.val * reg._phi_star(eigvals)


def _validate_optax_inputs(
    problem: SDPRegularized,
    opt,
    max_iter: int,
    tol: float,
    log_every: int,
) -> None:
    """Validate public ``run_optax_solver`` inputs."""
    if not isinstance(problem, SDPRegularized):
        raise TypeError("run_optax_solver expects SDPRegularized.")
    if opt is None:
        raise ValueError("opt must be provided.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")
    if getattr(problem.sdp.ops, "jax", None) is None:
        raise ValueError("run_optax_solver requires a JAX-backed SDP.")


def _problem_for_jit(problem: SDPRegularized) -> SDPRegularized:
    """Return an equivalent problem with runtime membership checks disabled."""
    ctx = Context(
        problem.sdp.ops,
        dtype=problem.sdp.ctx.dtype,
        enable_checks=False,
    )
    reg = type(problem.reg)(ctx.asarray(problem.reg.val), ctx=ctx)
    return SDPRegularized(problem.sdp.convert(ctx), reg)


def _run_optax_loop(
    problem: SDPRegularized,
    init_params,
    opt,
    max_iter: int,
    tol: float,
):
    """Run the compiled Optax loop and return raw arrays/histories."""
    import jax
    import optax as optax_lib

    ops = problem.sdp.ops

    def loss_fun(params):
        y = _array_from_params(params)
        return -ops.real(_dual_objective_array(problem, y))

    value_and_grad = jax.value_and_grad(loss_fun)
    state0 = opt.init(init_params)

    carry0 = (
        init_params,
        state0,
        ops.asarray(ops.inf),
        ops.zeros((max_iter,)),
        ops.zeros((max_iter,)),
        ops.asarray(0),
    )

    def cond_fun(carry):
        _, _, grad_norm, _, _, it = carry
        return (it < max_iter) & (grad_norm >= tol)

    def body_fun(carry):
        params, state, _, obj_log, grad_log, it = carry

        loss, grad = value_and_grad(params)
        grad_norm = optax_lib.global_norm(grad)
        updates, state = opt.update(
            grad,
            state,
            params,
            value=loss,
            grad=grad,
            value_fn=loss_fun,
        )
        params = optax_lib.apply_updates(params, updates)

        obj_log = ops.index_set(obj_log, it, -loss)
        grad_log = ops.index_set(grad_log, it, grad_norm)

        return params, state, grad_norm, obj_log, grad_log, it + 1

    params, _, grad_norm, obj_log, grad_log, n_iters = ops.while_loop(
        cond_fun,
        body_fun,
        carry0,
    )
    return _array_from_params(params), n_iters, grad_norm, obj_log, grad_log


def run_optax_solver(
    sdp: SDPRegularized,
    init_dual: SDPDual | None = None,
    opt=None,
    max_iter: int = 100000,
    tol: float = 1e-6,
    verbose: bool = False,
    log_every: int = 50,
) -> ConvergenceInfo:
    r"""Run a JAX/Optax optimizer on the regularized SDP dual objective.

    Args:
        sdp: Regularized SDP whose dual objective is maximized.
        init_dual: Optional initial dual variable. If omitted, the zero element
            of the SDP codomain is used.
        opt: Optax gradient transformation.
        max_iter: Maximum number of optimizer iterations.
        tol: Stop once the gradient norm is below this tolerance.
        verbose: If true, print logged diagnostics after the compiled loop
            returns.
        log_every: Diagnostic print interval when ``verbose`` is enabled.

    Returns:
        ``ConvergenceInfo`` with the final dual variable, dual-objective
        history, gradient norm history, tolerance flag, and elapsed time.
    """
    _validate_optax_inputs(
        problem=sdp,
        opt=opt,
        max_iter=max_iter,
        tol=tol,
        log_every=log_every,
    )

    if init_dual is None:
        init_dual = sdp.sdp.dual_from_array(sdp.sdp.cod.zeros())

    loop_problem = _problem_for_jit(sdp)
    init_y = loop_problem.sdp.ctx.asarray(init_dual.y)
    init_params = _params_from_array(loop_problem, init_y)

    def loop():
        return _run_optax_loop(
            problem=loop_problem,
            init_params=init_params,
            opt=opt,
            max_iter=max_iter,
            tol=tol,
        )

    jax = loop_problem.sdp.ops.jax
    compiled = jax.jit(loop).lower().compile()

    start = Time()
    final_y, n_iters, grad_norm, dual_obj, grad_hist = compiled()
    elapsed = Time() - start

    n_iters = int(n_iters)
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
        dual=sdp.sdp.dual_from_array(final_y),
        dual_obj=dual_obj,
        grad_norm=grad_hist,
        tol_reached=float(grad_norm) < tol,
        time=elapsed,
    )


__all__ = ["run_optax_solver", ]
