from __future__ import annotations

import numpy as np
from spacecore import Context, HermitianSpace, LinOp, DenseArray, VectorSpace, NumpyOps

from sdplab.sdp import SDPDenseProblem


class MaxCutOperator(LinOp):
    def apply(self, x: DenseArray) -> DenseArray:
        if self._enable_checks:
            self.dom._check_member(x)

        return self.ops.diag(x)

    def rapply(self, y: DenseArray) -> DenseArray:
        if self._enable_checks:
            self.cod._check_member(y)

        return self.ops.diag(y)

    def to_dense(self) -> DenseArray:
        n = self.cod.shape[0]
        idx = self.ops.arange(n)
        A = (idx[:, None, None] == idx[None, :, None]) & (
        idx[:, None, None] == idx[None, None, :])
        return A

    @property
    def A(self) -> DenseArray:
        return self.to_dense()

    def _convert(self, new_ctx: Context) -> MaxCutOperator:
        return MaxCutOperator(self.dom.convert(new_ctx), self.cod.convert(new_ctx), new_ctx)


def generate_erdos_renyi_graph_laplacian(
    n: int,
    p: float = 0.3,
    seed: int | None = None,
    weighted: bool = True,
    weight_low: float = 0.0,
    weight_high: float = 1.0,
) -> np.ndarray:
    """
    Generate an undirected Erdős–Rényi graph G(n, p).

    Returns
    -------
    L:
        Graph Laplacian L = D - W in S^n,
        where D[i, i] = sum_j W[i, j].
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1].")
    if weight_low < 0 or weight_high < weight_low:
        raise ValueError("Require 0 <= weight_low <= weight_high.")

    rng = np.random.default_rng(seed)

    # Upper-triangular edge mask, excluding diagonal.
    edge_mask = rng.random((n, n)) < p
    edge_mask = np.triu(edge_mask, k=1)

    if weighted:
        weights = rng.uniform(weight_low, weight_high, size=(n, n))
        W_upper = edge_mask * weights
    else:
        W_upper = edge_mask.astype(float)

    W = W_upper + W_upper.T

    degrees = W.sum(axis=1)
    L = np.diag(degrees) - W

    return L


def generate_max_cut(
    n: int,
    p: float = 0.3,
    seed: int | None = None,
    weighted: bool = True,
    weight_low: float = 0.0,
    weight_high: float = 1.0,
    atol: float = 0.0,
    rtol: float = 0.0,
    enforce_herm: bool = True,
    ctx: Context | str | None = None
):
    np_ctx = Context(NumpyOps(), dtype='float64', enable_checks=False)

    L = generate_erdos_renyi_graph_laplacian(n, p, seed, weighted, weight_low, weight_high)
    C = -L / 4  # Minus to turn maximization into minimization
    b = np.ones(n)

    dom = HermitianSpace(n, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=np_ctx)
    cod = VectorSpace((n,), ctx=np_ctx)
    A = MaxCutOperator(dom, cod, np_ctx)

    sdp = SDPDenseProblem(C, A, b, ctx=np_ctx)

    return sdp.convert(ctx)
