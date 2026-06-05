from __future__ import annotations

from functools import reduce
from typing import Sequence

from spacecore import Context, DenseArray


def kron_all(ctx: Context, factors: Sequence[DenseArray]) -> DenseArray:
    r"""
    Return the left-folded Kronecker product.

    For matrices :math:`A_1,\ldots,A_n`, this computes

    .. math::

        A_1 \otimes A_2 \otimes \cdots \otimes A_n.

    If ``A_i`` acts on a local vector space ``V_i``, the result acts on the
    tensor product space :math:`V_1 \otimes \cdots \otimes V_n`.
    """
    return reduce(lambda a, b: ctx.ops.kron(a, b), factors)
