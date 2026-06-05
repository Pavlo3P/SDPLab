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

from spacecore import Context, DenseArray, jax_pytree_class

from ...regularization import SDPRegularized
from ...sdp import SDPDual
from .._common import dual_objective_array, problem_summary
from .._runner import OptimizeResult, run_solver


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


def _validate_optax_inputs(
    problem: SDPRegularized,
    opt,
    max_iter: int,
    tol: float,
    log_every: int,
) -> None:
    """Validate public ``solve_optax`` inputs."""
    if not isinstance(problem, SDPRegularized):
        raise TypeError("solve_optax expects SDPRegularized.")
    if opt is None:
        raise ValueError("opt must be provided.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")
    if getattr(problem.sdp.ops, "jax", None) is None:
        raise ValueError("solve_optax requires a JAX-backed SDP.")


def _problem_for_jit(problem: SDPRegularized) -> SDPRegularized:
    """Return an equivalent problem with runtime membership checks disabled."""
    ctx = Context(
        problem.sdp.ops,
        dtype=problem.sdp.ctx.dtype,
        enable_checks=False,
    )
    reg = type(problem.reg)(ctx.asarray(problem.reg.val), ctx=ctx)
    return SDPRegularized(problem.sdp.convert(ctx), reg)


def solve_optax(
    sdp: SDPRegularized,
    init_dual: SDPDual | None = None,
    opt=None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: int = 1,
    log_every: int = 50,
    ascii_only: bool = False,
    color: bool | None = None,
) -> OptimizeResult:
    r"""Run a JAX/Optax optimizer on the regularized SDP dual objective.

    Args:
        sdp: Regularized SDP whose dual objective is maximized.
        init_dual: Optional initial dual variable. If omitted, the zero element
            of the SDP codomain is used.
        opt: Optax gradient transformation.
        max_iter: Maximum number of optimizer iterations.
        tol: Stop once the gradient norm is below this tolerance.
        verbose: Verbosity level. ``0`` is silent, ``1`` prints header/footer,
            ``2`` prints periodic diagnostics, ``3`` prints every iteration,
            and ``4`` uses boxed verbose diagnostics.
        log_every: Diagnostic print interval when ``verbose >= 2``.
        ascii_only: Use ASCII-only verbose output when true.
        color: Whether to use ANSI color in fancy output. ``None`` auto-detects.

    Returns:
        ``OptimizeResult`` with the final dual variable, loss and gradient
        histories, convergence status, and timing information.
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

    import jax
    import optax as optax_lib

    ops = loop_problem.sdp.ops

    def loss_fun(params):
        y = _array_from_params(params)
        return -ops.real(dual_objective_array(loop_problem, y))

    value_and_grad = jax.value_and_grad(loss_fun)
    init_opt_state = opt.init(init_params)

    @jax.jit
    def step_fn(state):
        params, opt_state = state
        loss, grad = value_and_grad(params)
        dual_obj = -loss
        grad_norm = optax_lib.global_norm(grad)
        updates, opt_state = opt.update(
            grad,
            opt_state,
            params,
            value=loss,
            grad=grad,
            value_fn=loss_fun,
        )
        params = optax_lib.apply_updates(params, updates)
        return (params, opt_state), dual_obj, grad_norm

    def finalize_fn(state):
        params, _ = state
        return sdp.sdp.dual_from_array(_array_from_params(params))

    return run_solver(
        init_state=(init_params, init_opt_state),
        step_fn=step_fn,
        finalize_fn=finalize_fn,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        log_every=log_every,
        solver_name="optax, backend=jax",
        problem_summary=problem_summary(sdp),
        initial_dual_norm=float(loop_problem.sdp.cod.norm(init_y)),
        ascii_only=ascii_only,
        color=color,
    )


__all__ = ["solve_optax"]
