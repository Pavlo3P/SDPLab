"""Solver entry points and convergence records."""

from ._regularized import run_regularized_solver, solve_scipy
from ._runner import OptimizeResult, run_solver
from .jax import run_jaxlib_solver, solve_optax
from ._torch import solve_torch
from .cvxpy import run_cvxpy_solver

__all__ = [
    "run_jaxlib_solver",
    "run_cvxpy_solver",
    "run_regularized_solver",
    "solve_optax",
    "solve_torch",
    "solve_scipy",
    "run_solver",
    "OptimizeResult",
]
