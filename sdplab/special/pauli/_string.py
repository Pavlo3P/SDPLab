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

r"""Convenience functions for materializing Pauli observables.

Pauli strings are tensor products of the one-qubit matrices ``I``, ``X``,
``Y``, and ``Z``. They are common observables in quantum tomography and other
quantum SDP models.
"""

from __future__ import annotations

from spacecore import Context, DenseArray

from ._operators import PauliString, PauliSum


def generate_single_pauli_string(s: str, ctx: Context | str | None = None) -> DenseArray:
    r"""Return the dense matrix for a Pauli tensor product.

    For example, ``"IXZ"`` means :math:`I \otimes X \otimes Z` and returns
    an ``8 x 8`` dense matrix.
    """
    return PauliString(s, ctx=ctx).materialize()


def generate_pauli_observables(observables: list[str], ctx: Context | str | None = None) -> DenseArray:
    r"""Stack dense Pauli observable matrices along a leading axis.

    If ``observables`` contains ``m`` labels on ``n`` qubits, the output has
    shape ``(m, 2**n, 2**n)``. Each slice is one observable matrix that can be
    used in constraints such as :math:`\operatorname{Tr}[M_i X] = b_i`.
    """
    if not observables:
        raise ValueError("observables must be non-empty")
    mats = [PauliString(s, ctx=ctx).materialize() for s in observables]
    return PauliString(observables[0], ctx=ctx).ctx.ops.stack(mats, axis=0)
