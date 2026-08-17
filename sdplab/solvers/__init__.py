"""Solver entry points and convergence records.

First-order regularized dual solves are delegated to ``spacecore.optimize``
(:func:`spacecore.minimize_scipy` / :func:`spacecore.minimize_optax`) through
:func:`run_regularized_solver`.
"""

from ._regularized import run_regularized_solver
from ._cvxpy import run_cvxpy_solver

__all__ = [
    "run_regularized_solver",
    "run_cvxpy_solver",
]
