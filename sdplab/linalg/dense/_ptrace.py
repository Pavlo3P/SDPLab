from typing import Tuple, Sequence

from spacecore import DenseArray, Context


def make_perm(i: int, N: int) -> Tuple[int, ...]:
    r"""Return a tensor-axis permutation for the ``i``-th partial trace.

    A matrix on :math:`(\mathbb{C}^d)^{\otimes N}` is reshaped as a tensor
    with ``N`` bra indices followed by ``N`` ket indices. This permutation
    moves bra index ``i`` and ket index ``i`` to the front so the remaining
    indices can be traced out.
    """
    return (
        (i,) +
        tuple(j for j in range(N) if j != i) +
        (N + i,) +
        tuple(N + j for j in range(N) if j != i)
    )


def _compute_ptraces(ctx: Context, X: DenseArray, *, d: int, N: int, perms: Sequence[Tuple[int, ...]]) -> DenseArray:
    r"""Compute :math:`\mathcal{A}\Gamma = (\operatorname{Tr}^k[\Gamma])_k`."""
    ops = ctx.ops
    D_rest = d ** (N - 1)

    # Reshape the coupling Gamma to a rank-2N tensor:
    # (a0, ..., aN-1, b0, ..., bN-1).
    Gamma = X.reshape((d,) * (2 * N))

    def _single_ptrace(perm: Tuple[int, ...]) -> DenseArray:
        Gamma_p = ops.transpose(Gamma, perm)  # (d, d^(N-1), d, d^(N-1))
        Gamma_p = Gamma_p.reshape(d, D_rest, d, D_rest)
        return ops.einsum("arbr->ab", Gamma_p)  # trace over r

    return ops.stack([_single_ptrace(perm) for perm in perms], axis=0)


def compute_ptraces(ctx: Context, X: DenseArray, *, d: int, N: int) -> DenseArray:
    r"""Compute all one-body partial traces of an ``N``-body coupling.

    ``X`` is the numerical array representing
    :math:`\Gamma \in \operatorname{dom}(\mathcal{A})`. It is interpreted as
    an operator on :math:`(\mathbb{C}^d)^{\otimes N}`, reshaped into one bra
    and one ket index per subsystem. The result lies in
    :math:`\operatorname{cod}(\mathcal{A})` and has shape ``(N, d, d)``; block
    ``k`` is
    :math:`(\mathcal{A}\Gamma)_k = \operatorname{Tr}^k[\Gamma]`.
    """
    perms = tuple(make_perm(i, N) for i in range(N))
    return _compute_ptraces(ctx, X, d=d, N=N, perms=perms)