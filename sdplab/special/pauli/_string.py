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
