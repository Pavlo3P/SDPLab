from functools import reduce
from typing import Tuple, Sequence

from qotlib.core import ArrayLike, DenseArray, Space, BackendContext


def kron_all(ctx: BackendContext, factors: list[DenseArray]) -> DenseArray:
    """Left-folded Kronecker product of a Python list of matrices."""
    ops = ctx.ops

    return reduce(lambda A, B: ops.kron(A, B), factors)


def kron_sum(ctx: BackendContext, blocks: DenseArray) -> DenseArray:
    ops = ctx.ops
    dtype = ctx.dtype

    N, d = blocks.shape[:2]
    I = ops.eye(d, dtype=dtype)
    D = d ** N
    K = ops.zeros((D, D), dtype=dtype)
    for k in range(N):
        factors = [I] * N
        factors[k] = blocks[k]
        K = K + kron_all(ctx, factors)
    return K


def make_perm(i: int, N: int) -> Tuple[int, ...]:
    """bra_i | bra_rest | ket_i | ket_rest  (length 2N)"""
    return (
        (i,) +
        tuple(j for j in range(N) if j != i) +
        (N + i,) +
        tuple(N + j for j in range(N) if j != i)
    )


def _compute_ptraces(ctx: BackendContext, X: DenseArray, *, d: int, N: int, perms: Sequence[Tuple[int, ...]]) -> DenseArray:
    ops = ctx.ops
    D_rest = d ** (N - 1)

    # Reshape ρ to rank-2N tensor: (a0…aN-1, b0…bN-1)
    ρ = X.reshape((d,) * (2 * N))

    def _single_ptrace(perm: Tuple[int, ...]) -> DenseArray:
        ρp = ops.transpose(ρ, perm)  # (d, d^(N-1), d, d^(N-1))
        ρp = ρp.reshape(d, D_rest, d, D_rest)
        return ops.einsum("arbr->ab", ρp)  # trace over r

    return ops.stack([_single_ptrace(perm) for perm in perms], axis=0)


def compute_ptraces(ctx: BackendContext, X: DenseArray, *, d: int, N: int) -> DenseArray:
    perms = tuple(make_perm(i, N) for i in range(N))
    return _compute_ptraces(ctx, X, d=d, N=N, perms=perms)
