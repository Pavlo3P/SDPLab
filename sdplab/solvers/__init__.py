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

"""Solver entry points and convergence records.

First-order regularized dual solves are delegated to ``spacecore.optimize``
(:func:`spacecore.minimize_scipy` / :func:`spacecore.minimize_optax`) through
:func:`run_regularized_solver`.
"""

from ._common import OptimizeResult
from ._regularized import run_regularized_solver
from ._cvxpy import run_cvxpy_solver

__all__ = [
    "OptimizeResult",
    "run_regularized_solver",
    "run_cvxpy_solver",
]
