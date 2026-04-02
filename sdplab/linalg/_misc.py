from typing import Any, Callable
from spacecore import Context, DenseArray

MVP = Callable[[Any], Any]

def make_projector(ctx: Context, vec: DenseArray) -> DenseArray:
    """
    Compute the rank-1 projector P = |v⟩⟨v| using Context.

    Parameters
    ----------
    ctx : Context
        Backend context defining numerical backend, dtype, and checks.
    vec : Any
        1-D dense array representing a vector.

    Returns
    -------
    DenseArray
        Hermitian rank-1 projector with shape (n, n).
    """
    ops = ctx.ops

    # Ensure dense array + sanitize dtype
    v = ctx.assert_dense(ctx.asarray(vec))

    # P_ij = v_i * conj(v_j)
    # einsum is the most portable representation of outer products
    return ops.einsum("i,j->ij", v, ops.conj(v))
