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

r"""State tomography as feasibility: find :math:`X \succeq 0` with
:math:`\operatorname{Tr}[M_i X] = b_i` and :math:`\operatorname{Tr}[X] = 1`.

Unit trace is appended as one more measurement row. See :mod:`._build` for the
transpose the dense operator needs.
"""

from ._build import generate_qubit_tomography

__all__ = ["generate_qubit_tomography"]
