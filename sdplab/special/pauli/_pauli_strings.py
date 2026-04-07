from functools import reduce

from spacecore import Context, DenseArray
from spacecore._contextual import ctx_manager

from ._matrices import pauli_matrices

def generate_single_pauli_string(s: str, ctx: Context | str | None = None) -> DenseArray:
    ctx = ctx_manager.normalize_context(ctx)

    mats = [ctx.ops.asarray(pauli_matrices[ch]) for ch in s]
    return reduce(ctx.ops.kron, mats)


def generate_pauli_observables(observables: list[str], ctx: Context | str | None = None) -> DenseArray:
    ctx = ctx_manager.normalize_context(ctx)

    pauli_strings = []
    for s in observables:
        pauli_string = generate_single_pauli_string(s, ctx)
        pauli_strings.append(pauli_string)

    return ctx.ops.stack(pauli_strings, axis=0)
