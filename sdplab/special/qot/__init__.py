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

r"""Quantum optimal transport spaces, operators, examples, and solvers.

QOT constraints use the partial-trace map
:math:`\mathcal{A}\Gamma = (\operatorname{Tr}^k[\Gamma])_k`
from ``Herm(d^N)`` to ``Herm(d)^N`` and its Kronecker-sum adjoint
:math:`\mathcal{A}^\dagger`. The coupling is denoted by :math:`\Gamma`; its
one-body marginals are denoted by :math:`\gamma_k`. Here
:math:`\operatorname{Tr}^k` means the partial trace that keeps subsystem
:math:`k` and traces out all other subsystems.
"""

from ._constraint_op import QOTConstraintOp
from ._cvxpy import solve_qot_dual

__all__ = [
    "QOTConstraintOp",

    "solve_qot_dual",
]
