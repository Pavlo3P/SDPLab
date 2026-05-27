"""Solver entry points and convergence records."""

from ._info import ConvergenceInfo
from .jax import run_jaxlib_solver, run_optax_solver
from ._torch import run_torch_solver
from .cvxpy import run_cvxpy_solver

__all__ = [
    "run_jaxlib_solver",
    "run_cvxpy_solver",
    "run_optax_solver",
    "run_torch_solver",
    "ConvergenceInfo",
]
