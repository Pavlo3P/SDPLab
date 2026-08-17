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

from __future__ import annotations

from functools import reduce
from typing import Sequence

from spacecore import Context, DenseArray


def kron_prod(ctx: Context, factors: Sequence[DenseArray]) -> DenseArray:
    r"""
    Return the left-folded Kronecker product.

    For matrices :math:`A_1,\ldots,A_n`, this computes

    .. math::

        A_1 \otimes A_2 \otimes \cdots \otimes A_n.

    If ``A_i`` acts on a local vector space ``V_i``, the result acts on the
    tensor product space :math:`V_1 \otimes \cdots \otimes V_n`.
    """
    return reduce(lambda a, b: ctx.ops.kron(a, b), factors)


def kron_sum(ctx: Context, blocks: DenseArray) -> DenseArray:
    r"""Return :math:`\mathcal{A}^\dagger y` for one-body blocks ``blocks``.

    If ``blocks[k]`` represents :math:`y_k \in \operatorname{Herm}(d)`, then
    ``blocks`` represents
    :math:`y = (y_0, \ldots, y_{N-1}) \in \operatorname{cod}(\mathcal{A})`.
    The result is the matrix in :math:`\operatorname{dom}(\mathcal{A})`

    .. math::

        \mathcal{A}^\dagger y
        =
        y_0 \oplus \cdots \oplus y_{N-1}
        =
        \sum_k I \otimes \cdots \otimes y_k \otimes \cdots \otimes I.
    """
    ops = ctx.ops
    dtype = ctx.dtype

    N, d = blocks.shape[:2]
    I = ops.eye(d, dtype=dtype)
    D = d ** N
    K = ops.zeros((D, D), dtype=dtype)
    for k in range(N):
        factors = [I] * N
        factors[k] = blocks[k]
        K = K + kron_prod(ctx, factors)
    return K
