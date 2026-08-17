# Copyright 2026 Pavlo Pelikh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dispatcher for smooth regularized dual solves via ``spacecore.optimize``.

No optimization loop lives here. Iteration, line search, convergence testing,
history recording, and timing are owned by :func:`spacecore.minimize_scipy` and
:func:`spacecore.minimize_optax`; this module only fixes the maximization sign
and translates results into the package-level :class:`OptimizeResult`.

The entry point takes a :class:`BoundDualFunctional` -- ε and the normalization
travel with the functional rather than being passed alongside it, so there is
no way to solve at one ε while recovering the primal at another. Build one with
``RegularizedSDPDualFunctional.bind(eps, normalized=...)``.
"""

from __future__ import annotations

import logging

from spacecore import minimize_optax, minimize_scipy

from ..regularization import BoundDualFunctional
from ._common import OptimizeResult, loop_functional, problem_summary

logger = logging.getLogger(__name__)


def run_regularized_solver(
    problem: BoundDualFunctional,
    init_dual=None,
    *,
    method: str | None = None,
    opt=None,
    learning_rate: float = 1e-2,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: int = 1,
    **kwargs,
) -> OptimizeResult:
    """Optimize a regularized dual objective with the matching backend driver.

    Args:
        problem: The :class:`BoundDualFunctional` to maximize; its ``eps_val``
            and ``normalized`` select the strength and the primal recovery.
        init_dual: Initial dual iterate, a plain ``cod`` element. Defaults to
            zeros.
        method: ``"scipy"`` or ``"optax"``. ``None`` selects ``"optax"`` on a
            JAX backend and ``"scipy"`` otherwise.
        opt: Optax optimizer for the ``optax`` route (default
            ``optax.adam(learning_rate)``).
        **kwargs: Forwarded to the underlying driver
            (:func:`spacecore.minimize_scipy` /
            :func:`spacecore.minimize_optax`).

    Returns:
        An :class:`OptimizeResult` whose ``final_loss``/``loss_history``
        report the maximized dual value :math:`D_\\varepsilon` and whose
        ``raw`` field carries the untranslated backend result.
    """
    if not isinstance(problem, BoundDualFunctional):
        raise TypeError(
            "run_regularized_solver expects a BoundDualFunctional; got "
            f"{type(problem).__name__}. Call .bind(eps, normalized=...) on a "
            "RegularizedSDPDualFunctional first."
        )

    if method is None:
        family = getattr(problem.ops, "family", None)
        method = "optax" if family == "jax" else "scipy"

    if init_dual is None:
        init_dual = problem.domain.zeros()

    if method == "scipy":
        return _solve_scipy(
            problem, init_dual,
            max_iter=max_iter, tol=tol, verbose=verbose, **kwargs,
        )
    if method == "optax":
        return _solve_optax(
            problem, init_dual,
            opt=opt, learning_rate=learning_rate,
            max_iter=max_iter, tol=tol, verbose=verbose, **kwargs,
        )

    raise ValueError(f"Unknown method {method!r}; use 'scipy' or 'optax'.")


def _loop_bound(problem: BoundDualFunctional) -> BoundDualFunctional:
    """Return ``problem`` with runtime membership checks disabled.

    :func:`loop_functional` converts the eps-per-call base, so the bound view
    is rebuilt around it carrying the same ``eps_val`` and ``normalized``.
    """
    return loop_functional(problem.base).bind(
        problem.eps_val, normalized=problem.normalized
    )


def _solve_scipy(
    problem: BoundDualFunctional,
    init_dual,
    *,
    max_iter: int,
    tol: float,
    verbose: int,
    scipy_method: str = "L-BFGS-B",
    **kwargs,
) -> OptimizeResult:
    """Maximize the dual with :func:`spacecore.minimize_scipy` (L-BFGS-B default).

    :func:`spacecore.minimize_scipy` requires a real inner-product domain --
    it flattens through ``F.domain.flatten`` and hands SciPy real coordinate
    vectors -- so a complex codomain is rejected up front rather than left to
    fail inside SciPy.
    """
    import time

    import numpy as np

    cod = problem.domain
    if getattr(cod, "field", "real") != "real":
        raise NotImplementedError(
            "method='scipy' needs a real codomain; spacecore.minimize_scipy "
            f"cannot flatten {type(cod).__name__} (field="
            f"{getattr(cod, 'field', None)!r}) into real coordinates. Use "
            "method='optax' on a JAX backend."
        )

    # Negate through the spacecore functional algebra: SciPy minimizes -D_eps.
    F = -_loop_bound(problem)

    if int(verbose) >= 1:
        logger.info(
            "solver: scipy/%s on %s",
            scipy_method, problem_summary(problem.base, problem.eps_val),
        )

    options = dict(kwargs.pop("options", {}))
    options.setdefault("maxiter", int(max_iter))
    start = time.perf_counter()
    result = minimize_scipy(
        F, init_dual, method=scipy_method, tol=tol, options=options, **kwargs
    )
    elapsed = time.perf_counter() - start

    jac = getattr(result, "jac", None)
    grad_norm = (
        float(np.linalg.norm(np.asarray(jac).reshape(-1)))
        if jac is not None
        else float("nan")
    )
    wrapped = OptimizeResult(
        dual=result.x_element,
        converged=bool(result.success),
        num_iters=int(getattr(result, "nit", -1)),
        final_loss=-float(result.fun),
        final_grad_norm=grad_norm,
        elapsed_seconds=elapsed,
        raw=result,
    )
    if int(verbose) >= 1:
        logger.info(wrapped.summary())
    return wrapped


def _solve_optax(
    problem: BoundDualFunctional,
    init_dual,
    *,
    opt,
    learning_rate: float,
    max_iter: int,
    tol: float,
    verbose: int,
    **kwargs,
) -> OptimizeResult:
    """Maximize the dual with spacecore's compiled optax loop.

    The whole loop runs inside ``jax.jit(lax.while_loop)`` with a fused
    ``value_and_grad`` per iteration, tolerance-based stopping, history
    sampling, and compile/execution timing -- none of that lives in sdplab.
    """
    if opt is None:
        import optax

        opt = optax.adam(learning_rate)

    # Negate through the spacecore functional algebra: optax minimizes -D_eps.
    F = -_loop_bound(problem)
    result = minimize_optax(
        F,
        init_dual,
        opt,
        max_iter=max_iter,
        tol=tol,
        verbose=min(int(verbose), 2),
        **kwargs,
    )

    history = result.history or {}
    loss_history = (
        [-v for v in history["value"]] if "value" in history else None
    )
    grad_history = (
        list(history["grad_norm"]) if "grad_norm" in history else None
    )
    return OptimizeResult(
        dual=result.x_element,
        converged=bool(result.success),
        num_iters=int(result.num_iters),
        final_loss=-float(result.final_value),
        final_grad_norm=float(result.final_grad_norm),
        elapsed_seconds=float(result.execution_seconds),
        loss_history=loss_history,
        grad_norm_history=grad_history,
        raw=result,
        extra={"compile_seconds": result.compile_seconds},
    )


__all__ = ["run_regularized_solver"]
