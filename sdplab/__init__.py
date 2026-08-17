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

r"""Semidefinite and conic programs over spacecore Euclidean Jordan algebras.

.. math::

    \min_X \quad \langle C, X\rangle
    \quad \text{s.t.} \quad \mathcal{A}X = b,\quad X \succeq 0

``X`` is any Jordan-algebra element -- Hermitian matrix, nonnegative vector, or
a tree of such blocks -- and :math:`X \succeq 0` means a nonnegative spectrum.
Build with ``SDPProblem(C, A, b)``; solve with :func:`run_cvxpy_solver`, or
smooth it: a :class:`Regularizer` through
:class:`RegularizedSDPDualFunctional`, ``.bind(eps)``, then
:func:`run_regularized_solver`.

Smoothing adds :math:`\varepsilon \operatorname{Tr}[\varphi(X)]` to the primal,
giving the differentiable dual

.. math::

    D_\varepsilon(y) = \langle b, y\rangle - \varepsilon
      \operatorname{Tr}\!\big[\psi\big(
      (\mathcal{A}^\dagger y - C)/\varepsilon\big)\big],

maximized over ``y`` in ``cod``, with :math:`\psi'` of the dual slack
recovering the primal. :math:`\varepsilon` is per-call, so continuation needs
no rebuild.

Subpackages: :mod:`~sdplab.problem`, :mod:`~sdplab.regularization`,
:mod:`~sdplab.solvers`, :mod:`~sdplab.examples`, :mod:`~sdplab.linalg`,
:mod:`~sdplab.special`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .problem import (
    ConstraintOp,
    Cost,
    DenseConstraintOp,
    MatrixFreeConstraintOp,
    SDPProblem,
    SparseConstraintOp,
)
from .regularization import (
    BoundDualFunctional,
    EntropyReg,
    QuadraticReg,
    RegularizedSDPDualFunctional,
    Regularizer,
)
from ._version import __version__

# Solvers pull in CVXPY (~1.2s), making an optional backend a hard requirement
# for anyone who only builds problems. PEP 562 keeps them reachable as
# ``sdplab.run_*`` while deferring the cost to first use.
_LAZY = {
    "run_regularized_solver": "sdplab.solvers",
    "run_cvxpy_solver": "sdplab.solvers",
}

if TYPE_CHECKING:  # type checkers and IDEs do not run __getattr__
    from .solvers import run_cvxpy_solver, run_regularized_solver


def __getattr__(name: str):
    """Resolve the lazily-exported solver entry points on first access."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value          # cache, so this runs once per name
    return value


def __dir__() -> list[str]:
    """Include the lazy names in ``dir(sdplab)`` and tab completion."""
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "__version__",
    # problem data
    "SDPProblem",
    "Cost",
    "ConstraintOp",
    "DenseConstraintOp",
    "SparseConstraintOp",
    "MatrixFreeConstraintOp",
    # regularization and the smoothed dual
    "Regularizer",
    "EntropyReg",
    "QuadraticReg",
    "RegularizedSDPDualFunctional",
    "BoundDualFunctional",
    # solvers (lazy)
    "run_regularized_solver",
    "run_cvxpy_solver",
]
