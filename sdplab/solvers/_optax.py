from functools import partial
from dataclasses import dataclass, field
from time import time as Time

from qotlib.core import DenseArray
from qotlib.regularization import SDPRegularized
from qotlib.sdp import SDPDual
from ._info import ConvergenceInfo


@dataclass
class DualReIm:
    re: DenseArray = field(init=False)
    im: DenseArray = field(init=False)

    def __init__(self, array: DenseArray):
        self.re = array.real
        self.im = array.imag

    def tree_flatten(self):
        children = (self.re, self.im)
        return children, None

    def get_array(self) -> DenseArray:
        return self.re + 1j * self.im

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)
        obj.re = children[0]
        obj.im = children[1]
        return obj


def run_optax_solver(
        sdp: SDPRegularized,
        init_dual: SDPDual,
        opt: "optax.GradientTransformation",
        max_iter: int,
        tol: float,
        verbose: bool = False,
        log_every: int = 50,
):
    import jax
    import jax.numpy as jnp
    from jax import tree_util
    import optax
    import optax.tree_utils as otu

    try:
        tree_util.register_pytree_node_class(DualReIm)
    except Exception as e:
        if "Duplicate custom PyTreeDef type registration" in str(e):
            pass
        else:
            raise e

    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
    def _run_optax_solver(
            sdp: SDPRegularized,
            init_dual: DualReIm,
            opt: optax.GradientTransformation,
            max_iter: int = 100000,
            tol: float = 1e-6,
            verbose: bool = False,
            log_every: int = 50,
    ):
        """
        Runs optax solver on the Regularized QOT dual.
        Adds counters for iteration, logs the objective value each step,
        and prints progress every `log_every` iters if `verbose=True`.

        Returns a dict with:
          - 'params': final dual variables
          - 'state': final Optax state (contains 'count' & 'grad')
          - 'loss_history': array of objective values per iteration
        """

        # 1) Define objective
        def fun(d: DualReIm) -> jnp.ndarray:
            arr = d.get_array()
            dual = sdp.sdp.dual_from_array(arr)
            return -sdp.dual_objective_reg(dual)

        # 3) Setup optimized
        value_and_grad_fun = jax.value_and_grad(fun)
        state0 = opt.init(init_dual)

        # 4) Pre-allocate loss log
        loss_log0 = jnp.zeros((max_iter,), dtype=jnp.float64)
        grad_log0 = jnp.zeros((max_iter,), dtype=jnp.float64)

        # 5) Loop body
        def step(carry):
            params, state, grad_norm, loss_log, grad_log, iter_num = carry

            # compute value & gradient
            value, grad = value_and_grad_fun(params)
            grad_norm = otu.tree_l2_norm(grad)

            # Optimizer update
            updates, state = opt.update(
                grad, state, params,
                value=value, grad=grad, value_fn=fun
            )
            params = optax.apply_updates(params, updates)

            # Log objective value
            loss_log = loss_log.at[iter_num].set(-value)
            grad_log = grad_log.at[iter_num].set(grad_norm)

            # optional debug print
            cond = verbose & (iter_num % log_every == 0)

            def _print(args):
                i, v = args
                jax.debug.print("[iter {i}] value={v:.6e} grad norm={g:.6e}", i=i, v=-v, g=grad_norm)

            _ = jax.lax.cond(cond, _print, lambda x: None,
                             operand=(iter_num, value))

            return params, state, grad_norm, loss_log, grad_log, iter_num + 1

        # 6) Loop condition
        def cont(carry):
            _, _, grad_norm, _, _, it = carry
            # it = otu.tree_get(state, 'count')
            # return (it == 0) | ((it < max_iter) & (err >= tol))
            return (it < 2) | ((it < max_iter) & (grad_norm >= tol))

        # 7) Run loop
        init_carry = (init_dual, state0, 0., loss_log0, grad_log0, 0)
        final_params, final_state, _, loss_log, grad_log, n_iters = jax.lax.while_loop(
            cont, step, init_carry
        )

        # 8) Slice to actual iterations
        # n_iters = otu.tree_get(final_state, 'count')
        tol_reached = jnp.where(n_iters < max_iter, True, False)
        return {
            'params': final_params,
            'n_iters': n_iters,
            'state': final_state[0],
            'loss_history': loss_log,
            'grad_history': grad_log,
            "tol_reached": tol_reached
        }

    def to_compile(_sdp, _init_d):
        return _run_optax_solver(_sdp, _init_d, opt=opt, max_iter=max_iter, tol=tol, verbose=verbose, log_every=log_every)
    arr = init_dual.tree_flatten()[0][0]
    init_d = DualReIm(arr)
    compiled = jax.jit(to_compile).lower(sdp, init_d).compile()

    start = Time()
    log = compiled(sdp, init_d)
    end = Time()

    final_arr = log['params'].get_array()
    dual = sdp.sdp.dual_from_array(final_arr)

    return ConvergenceInfo(
        dual=dual,
        dual_obj=log['loss_history'][:log['n_iters']],
        grad_norm=log['grad_history'][:log['n_iters']],
        tol_reached=bool(log['tol_reached']),
        time=end-start
    )
