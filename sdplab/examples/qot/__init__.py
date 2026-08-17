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

r"""Random quantum optimal-transport instances: a coupling
:math:`\Gamma \in \operatorname{Herm}(d^N)` of minimal cost with prescribed
one-body marginals :math:`\operatorname{Tr}^k[\Gamma] = \gamma_k`.

The codomain is a stack of Hermitian blocks, so the dual is a tuple of matrices
rather than a vector of scalars. :func:`generate_random_qot` derives the
marginals from a reference coupling, so the returned ``(problem, state)``
carries a feasible point.
"""

from ...special.qot import QOTConstraintOp
from ._random import generate_random_qot

__all__ = [
    "QOTConstraintOp",
    "generate_random_qot",
]
