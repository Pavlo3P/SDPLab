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

"""Shared helpers and the result record for regularized dual solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spacecore import Context

from ..regularization import RegularizedSDPDualFunctional


@dataclass
class OptimizeResult:
    """Uniform result of :func:`sdplab.solvers.run_regularized_solver`.

    ``final_loss`` and ``loss_history`` report the *maximized* dual objective
    :math:`D_\\varepsilon`, regardless of the minimization sign handling
    inside the underlying optimizer. ``raw`` carries the untranslated result
    of the backend that ran (a SciPy ``OptimizeResult``, a spacecore
    ``OptaxResult``, or a ``PredCorrResult``).
    """

    dual: Any
    converged: bool
    num_iters: int
    final_loss: float
    final_grad_norm: float
    elapsed_seconds: float
    loss_history: list[float] | None = None
    grad_norm_history: list[float] | None = None
    primal: Any = None
    raw: Any = None
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a one-paragraph human-readable summary."""
        status = "converged" if self.converged else "not converged"
        return (
            f"{status} in {self.num_iters} iterations "
            f"({self.elapsed_seconds:.2f} s): "
            f"D = {self.final_loss:+.8e}, ||grad|| = {self.final_grad_norm:.3e}"
        )


def loop_functional(problem: RegularizedSDPDualFunctional) -> RegularizedSDPDualFunctional:
    """Return an equivalent functional with runtime membership checks disabled.

    Compiled and autodiff loops evaluate the functional many times; the
    ``check_level="none"`` context skips the per-array Hermitian/shape
    validation (an ``allclose`` that also cannot trace under ``jit``) while
    preserving backend and dtype.
    """
    ctx = Context(problem.ops, dtype=problem.dtype, check_level="none")
    return problem.convert(ctx)


def problem_summary(problem: RegularizedSDPDualFunctional, eps: float) -> str:
    """Return a compact problem summary for solver logs."""
    return (
        f"{type(problem).__name__} with "
        f"{type(problem.regularizer).__name__}(eps={float(eps)})"
    )


__all__ = ["OptimizeResult", "loop_functional", "problem_summary"]
