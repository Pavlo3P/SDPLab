from __future__ import annotations

from functools import reduce
from typing import Sequence

from spacecore import Context, DenseArray, HermitianSpace


def kron_all(ctx: Context, factors: Sequence[DenseArray]) -> DenseArray:
    r"""
    Return the left-folded Kronecker product.

    For matrices \(A_1,\ldots,A_n\), this computes

    \[
        A_1 \otimes A_2 \otimes \cdots \otimes A_n.
    \]
    """
    return reduce(lambda a, b: ctx.ops.kron(a, b), factors)


def log_trace_exp(space: HermitianSpace, x: DenseArray) -> DenseArray:
    r"""
    Return the log-partition function of a Hermitian matrix.

    \[
        \log \operatorname{Tr}(\exp(x)).
    \]
    """
    evals, _ = space.eigh(x)
    return space.ops.logsumexp(evals)
