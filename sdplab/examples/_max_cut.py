from __future__ import annotations

import numpy as np
from spacecore import (
    Context,
    DenseArray,
    DenseVectorSpace,
    HermitianSpace,
    LinOp,
    NumpyOps,
    checked_method,
    jax_pytree_class,
)

from sdplab.problem import SDPProblem


@jax_pytree_class
class MaxCutOperator(LinOp[HermitianSpace, DenseVectorSpace]):
    r"""Diagonal-extraction operator :math:`\mathcal{A}` for the Max-Cut SDP.

    The linear map is
    :math:`\mathcal{A}: \operatorname{Herm}(n) \to \mathbb{R}^n` with
    :math:`(\mathcal{A}X)_i = X_{ii}`, so the constraint
    :math:`\mathcal{A}X = \mathbf{1}` fixes every diagonal entry to one. Its
    adjoint sends a vector to the matrix with that diagonal,
    :math:`\mathcal{A}^\dagger y = \operatorname{diag}(y)`, which satisfies
    :math:`\operatorname{Tr}[(\mathcal{A}X)y] = \operatorname{Tr}[X \operatorname{diag}(y)]`.
    """

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, X: DenseArray) -> DenseArray:
        r"""Return :math:`\mathcal{A}X = \operatorname{diag}(X)`, the diagonal of ``X``."""
        return self.ops.diag(X)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: DenseArray) -> DenseArray:
        r"""Return :math:`\mathcal{A}^\dagger y = \operatorname{diag}(y)`."""
        return self.ops.diag(y)

    def to_dense(self) -> DenseArray:
        r"""Return the operator tensor of shape ``codomain.shape + domain.shape``.

        The entry ``T[i, p, q]`` is one iff ``i == p == q``, so that
        ``sum_{p,q} T[i, p, q] X[p, q] = X[i, i]``.
        """
        n = self.cod.shape[0]
        idx = self.ops.arange(n)
        mask = (idx[:, None, None] == idx[None, :, None]) & (
            idx[:, None, None] == idx[None, None, :]
        )
        return self.ops.asarray(mask, dtype=self.dtype)

    def tree_flatten(self):
        """The operator carries no arrays; spaces and context are static aux."""
        return (), (self.dom, self.cod, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild the operator from JAX PyTree data."""
        dom, cod, ctx = aux
        return cls(dom, cod, ctx)

    def _convert(self, new_ctx: Context) -> MaxCutOperator:
        """Return an equivalent operator in ``new_ctx``."""
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
    np_ctx = Context(NumpyOps(), dtype="float64", check_level="none")

    L = generate_erdos_renyi_graph_laplacian(n, p, seed, weighted, weight_low, weight_high)
    C = -L / 4  # Minus to turn maximization into minimization
    b = np.ones(n)

    dom = HermitianSpace(n, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=np_ctx)
    cod = DenseVectorSpace((n,), ctx=np_ctx)
    A = MaxCutOperator(dom, cod, np_ctx)

    sdp = SDPProblem(C, A, b, ctx=np_ctx)

    if ctx is not None:
        sdp = sdp.convert(ctx)
    return sdp
