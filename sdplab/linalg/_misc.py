from typing import Any, Callable
from qotlib.core import BackendContext, DenseArray

MVP = Callable[[Any], Any]

def make_projector(ctx: BackendContext, vec: DenseArray) -> DenseArray:
    """
    Compute the rank-1 projector P = |v⟩⟨v| using BackendContext.

    Parameters
    ----------
    ctx : BackendContext
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
