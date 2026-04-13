from __future__ import annotations

from spacecore import Context, DenseArray

from ._operators import PauliString, PauliSum


def generate_single_pauli_string(s: str, ctx: Context | str | None = None) -> DenseArray:
    return PauliString(s, ctx=ctx).materialize()


def generate_pauli_observables(observables: list[str], ctx: Context | str | None = None) -> DenseArray:
    if not observables:
        raise ValueError("observables must be non-empty")
    mats = [PauliString(s, ctx=ctx).materialize() for s in observables]
    return PauliString(observables[0], ctx=ctx).ctx.ops.stack(mats, axis=0)
