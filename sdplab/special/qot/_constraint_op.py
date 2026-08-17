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

r"""Linear operator for quantum optimal transport marginal constraints.

For local dimension ``d`` and ``N`` subsystems, the global Hilbert space is
:math:`\mathcal{H} = (\mathbb{C}^d)^{\otimes N}`. The QOT coupling is a
Hermitian matrix
:math:`\Gamma \in \operatorname{dom}(\mathcal{A}) = \operatorname{Herm}(d^N)`.
The constraint operator maps this coupling to its one-body marginals:

.. math::

    \mathcal{A}\Gamma
    =
    (\operatorname{Tr}^0[\Gamma], \ldots,
     \operatorname{Tr}^{N-1}[\Gamma])
    \in \operatorname{cod}(\mathcal{A}) = \operatorname{Herm}(d)^N.

Here :math:`\operatorname{Tr}^k` means the partial trace that keeps subsystem
:math:`k` and traces out every subsystem :math:`j \ne k`.

The adjoint maps block variables
:math:`U = (U_0, \ldots, U_{N-1}) \in \operatorname{cod}(\mathcal{A})` back to
:math:`\operatorname{dom}(\mathcal{A})` by the Kronecker sum

.. math::

    \mathcal{A}^\dagger U
    =
    U_0 \oplus \cdots \oplus U_{N-1}
    =
    \sum_k I \otimes \cdots \otimes U_k \otimes \cdots \otimes I.
"""

from __future__ import annotations

from typing import Any

from spacecore import DenseArray, HermitianSpace, SparseArray, StackedSpace, jax_pytree_class, Context, checked_method

from ...linalg import kron_sum
from ...linalg.dense._ptrace import make_perm, _compute_ptraces
from ...problem import MatrixFreeConstraintOp

@jax_pytree_class
class QOTConstraintOp(MatrixFreeConstraintOp):
    r"""Partial-trace operator :math:`\mathcal{A}` for quantum optimal transport.

    This is the linear map

    .. math::

        \mathcal{A}: \operatorname{Herm}(d^N) \to \operatorname{Herm}(d)^N.

    Its domain :math:`\operatorname{dom}(\mathcal{A})` contains global
    Hermitian matrices on ``N`` tensor factors. Its codomain
    :math:`\operatorname{cod}(\mathcal{A})` contains ``N`` Hermitian
    ``d x d`` matrices, one per subsystem.

    If :math:`\Gamma \in \operatorname{dom}(\mathcal{A})` is a feasible QOT
    coupling and :math:`\gamma_k` is the prescribed marginal for site ``k``,
    then the equality constraint is

    .. math::

        (\mathcal{A}\Gamma)_k
        = \operatorname{Tr}^k[\Gamma]
        = \gamma_k.
    """

    def __init__(self,
                 *,
                 d: int,
                 N: int,
                 atol: float = 0.0,
                 rtol: float = 0.0,
                 enforce_herm: bool = True,
                 ctx: Context | str | None = None
                 ):
        r"""Create :math:`\mathcal{A}: \operatorname{Herm}(d^N) \to \operatorname{Herm}(d)^N`.

        Args:
            d: Local Hilbert-space dimension.
            N: Number of tensor factors/subsystems.
            atol: Absolute tolerance for Hermitian membership checks.
            rtol: Relative tolerance for Hermitian membership checks.
            enforce_herm: Whether domain and codomain require Hermitian input.
            ctx: Optional backend context.
        """
        if d <= 0 or type(d) is not int:
            raise ValueError("d must be positive integer.")
        if N <= 0 or type(N) is not int:
            raise ValueError("N must be positive integer.")

        atol = float(atol)
        rtol = float(rtol)
        enforce_herm = bool(enforce_herm)

        dom = HermitianSpace(d ** N, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=ctx)
        cod = StackedSpace(HermitianSpace(d, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=ctx), N, ctx=ctx)
        super(QOTConstraintOp, self).__init__(dom, cod, ctx)

        self.d = d
        self.N = N
        self.perms = tuple(make_perm(i, self.N) for i in range(self.N))

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, X: DenseArray) -> DenseArray:
        r"""Return :math:`\mathcal{A}\Gamma`, the one-body marginals of ``X``.

        ``X`` is the numerical array representing
        :math:`\Gamma \in \operatorname{dom}(\mathcal{A})`. It has shape
        ``(d^N, d^N)``, and the return value lies in
        :math:`\operatorname{cod}(\mathcal{A})` with shape ``(N, d, d)``.
        The ``k``-th block is
        :math:`(\mathcal{A}\Gamma)_k = \operatorname{Tr}^k[\Gamma]`.
        """
        return _compute_ptraces(self.ctx, X, d=self.d, N=self.N, perms=self.perms)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: DenseArray) -> Any:
        r"""Apply the adjoint :math:`\mathcal{A}^\dagger` as a Kronecker sum.

        For :math:`y = (y_0, \ldots, y_{N-1}) \in \operatorname{cod}(\mathcal{A})`,
        the adjoint is

        .. math::

            \mathcal{A}^\dagger y
            =
            y_0 \oplus \cdots \oplus y_{N-1}
            =
            \sum_k I \otimes \cdots \otimes y_k \otimes \cdots \otimes I,

        as an element of :math:`\operatorname{dom}(\mathcal{A})`. This identity
        is characterized by

        .. math::

            \operatorname{Tr}[(\mathcal{A}\Gamma)y]
            =
            \operatorname{Tr}[\Gamma(\mathcal{A}^\dagger y)].

        The decorator already asserts codomain membership on entry and domain
        membership on exit, so no explicit check is repeated here.
        """
        return kron_sum(self.ctx, y)

    def _site_index(self, k: int):
        r"""Return ``(base, right)`` for subsystem ``k``.

        A ``D``-index splits as ``(l, s_k, r)`` with ``l < d^k`` the higher
        subsystems, ``s_k < d`` the kept one, and ``r < d^{N-1-k}`` the lower
        ones. ``base`` enumerates every ``(l, r)`` pair as the flat ``D``-index
        with ``s_k = 0``, so ``base + s * right`` is the index with
        ``s_k = s``. The ``d^{N-1}`` entries of ``base`` are exactly the
        configurations traced over at site ``k``.
        """
        import numpy as np

        d, N = self.d, self.N
        block = d ** (N - k)          # stride of subsystem k in a D-index
        right = d ** (N - 1 - k)      # d-index stride of subsystem k
        left = d ** k                 # number of higher-subsystem configs
        hi = np.arange(left)[:, None]
        lo = np.arange(right)[None, :]
        return (hi * block + lo).ravel(), right

    def to_sparse(self) -> SparseArray:
        r"""Materialize :math:`\mathcal{A}` as a sparse coordinate matrix.

        The shape is ``(prod(cod.shape), prod(dom.shape)) = (N d^2, d^{2N})``,
        matching :meth:`~spacecore.LinOp.to_matrix`: row :math:`(k, a, b)` is
        the flattened tensor slice reading off :math:`\operatorname{Tr}^k` at
        marginal entry :math:`(a, b)`,

        .. math::

            (\mathcal{A}\Gamma)_{k,ab}
            = \sum_{l,r} \Gamma_{(l,a,r),(l,b,r)},

        so that row carries exactly :math:`d^{N-1}` unit entries -- the
        configurations of the traced-out subsystems. Storage is therefore
        :math:`N d^{N+1}` nonzeros instead of the :math:`N d^{2N+2}` of the
        dense form. The base class raises ``NotImplementedError`` here; the
        partial trace is a 0/1 incidence matrix, so it is worth providing.
        """
        import numpy as np
        from scipy.sparse import coo_matrix

        d, N = self.d, self.N
        D = d ** N
        rows, cols = [], []
        for k in range(N):
            base, right = self._site_index(k)
            for a in range(d):
                for b in range(d):
                    row = (k * d + a) * d + b          # flat cod coordinate
                    R = base + a * right               # bra keeps subsystem k = a
                    C = base + b * right               # ket keeps subsystem k = b
                    rows.append(np.full(R.shape, row))
                    cols.append(R * D + C)             # flat dom coordinate
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.ones(rows.shape, dtype=self.dtype)
        coo = coo_matrix((data, (rows, cols)), shape=(N * d * d, D * D))
        return self.ctx.assparse(coo)

    def tree_flatten(self):
        """Return children and auxiliary data for JAX PyTree flattening."""
        aux = (
            self.d,
            self.N,
            self.dom.atol,
            self.dom.rtol,
            self.dom.enforce_herm,
            self.cod.ctx,
        )
        return (), aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild the operator from JAX PyTree data."""
        d, N, atol, rtol, enforce_herm, ctx = aux
        return cls(d=d, N=N, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=ctx)

    def _convert(self, new_ctx: Context) -> QOTConstraintOp:
        """Return an equivalent operator in ``new_ctx``."""
        return QOTConstraintOp(d=self.d, N=self.N, atol=self.dom.atol, rtol=self.dom.rtol, enforce_herm=self.dom.enforce_herm, ctx=new_ctx)

    def _herm_generators(self):
        r"""Yield the real-coordinate Hermitian generators of a ``d x d`` block.

        Each generator is a Hermitian matrix ``H`` paired with the marginal
        coordinate its constraint reads off, so that for a Hermitian marginal
        :math:`\gamma`, :math:`\operatorname{Tr}[H \gamma]` is **real**. Enumerated
        over the upper triangle ``i <= j``:

        * diagonal ``E_{ii}``               -> :math:`\gamma_{ii}` (real);
        * real off-diagonal ``(E_{ij}+E_{ji})/2``   -> :math:`\operatorname{Re}\gamma_{ij}`;
        * imag off-diagonal ``i(E_{ij}-E_{ji})/2``  -> :math:`\operatorname{Im}\gamma_{ij}`.

        The imaginary generators are emitted only for a complex context: a real
        context gives ``d(d+1)/2`` generators per block, a complex one ``d^2``.

        Yields:
            Tuples ``(entries, coord)`` where ``entries`` is a list of
            ``(a, b, value)`` non-zeros of ``H`` and ``coord`` is ``("re"|"im", i, j)``
            naming the marginal coordinate (used to build the real ``b``).
        """
        d = self.d
        complex_ctx = self.ops.is_complex_dtype(self.dtype)
        for i in range(d):
            yield [(i, i, 1.0)], ("re", i, i)
            for j in range(i + 1, d):
                yield [(i, j, 0.5), (j, i, 0.5)], ("re", i, j)
                if complex_ctx:
                    yield [(i, j, 0.5j), (j, i, -0.5j)], ("im", i, j)

    def to_cvxpy(self) -> list[SparseArray]:
        r"""Return the QOT constraints as a list of per-constraint sparse matrices.

        This adapts the dense Kronecker construction of the QOT-to-SDP proof
        (th. 3.1 of https://arxiv.org/abs/2105.06922) into standard-form
        constraint matrices for a general SDP solver such as the CVXPY backend,
        whose equalities read :math:`\operatorname{Re}\operatorname{Tr}[A_i \Gamma] = b_i`.

        Constraint ``i = (k, \alpha)`` is the Hermitian matrix

        .. math::

            A_i = \mathcal{A}^\dagger(H_\alpha^{(k)})
                = I \otimes \cdots \otimes H_\alpha \otimes \cdots \otimes I,

        where :math:`H_\alpha` ranges over the real-coordinate Hermitian
        generators of the ``k``-th ``d x d`` block (see :meth:`_herm_generators`).
        Because each :math:`H_\alpha` is Hermitian, the matching right-hand side
        :math:`b_i = \operatorname{Tr}[H_\alpha^{(k)} \gamma_k]` is **real** even
        though the marginals :math:`\gamma_k` are complex Hermitian -- naively
        flattening the marginal entries would instead give a complex ``b``. The
        matching ``b`` is produced by :meth:`rhs_to_cvxpy` and the dual is
        reassembled by :meth:`dual_from_cvxpy`.

        The returned list has ``m = N d(d+1)/2`` entries for a real context and
        ``m = N d^2`` for a complex one, each a sparse ``(d^N, d^N)`` matrix so a
        solver can form ``trace(A_i @ Gamma)`` directly. Each generator embeds
        over ``d^{N-1}`` configurations of the traced-out subsystems, so storage
        stays sparse rather than the dense ``m d^{2N}``.
        """
        import numpy as np
        from scipy.sparse import coo_matrix

        d, N = self.d, self.N
        D = d ** N
        is_complex = self.ops.is_complex_dtype(self.dtype)
        dtype = np.complex128 if is_complex else np.float64

        generators = list(self._herm_generators())

        mats: list[SparseArray] = []
        for k in range(N):
            base, right = self._site_index(k)         # D-index shared by bra/ket

            for entries, _coord in generators:
                rows, cols, data = [], [], []
                for a, b, value in entries:
                    R = base + a * right              # bra keeps subsystem k = a
                    C = base + b * right              # ket keeps subsystem k = b
                    rows.append(R)
                    cols.append(C)
                    data.append(np.full(R.shape, value))
                coo = coo_matrix(
                    (
                        np.concatenate(data).astype(dtype),
                        (np.concatenate(rows), np.concatenate(cols)),
                    ),
                    shape=(D, D),
                )
                mats.append(self.ctx.assparse(coo))
        return mats

    def _generator_matrices(self) -> DenseArray:
        r"""Return the generators of :meth:`_herm_generators` as ``(G, d, d)``.

        Densified from the sparse ``(a, b, value)`` entry lists once, on the
        operator's own context, so :meth:`dual_from_cvxpy` can contract them in
        a single ``einsum`` rather than a Python accumulation. The generators
        depend only on ``d`` and the context's field, so the result is cached.
        """
        cached = getattr(self, "_generator_cache", None)
        if cached is not None:
            return cached

        d = self.d
        rows = []
        for entries, _coord in self._herm_generators():
            H = [[0.0 for _ in range(d)] for _ in range(d)]
            for a, b, value in entries:
                H[a][b] += value
            rows.append(H)
        # The imaginary generators are complex, so build in the widest dtype the
        # context offers and let the caller take the real part on a real field.
        self._generator_cache = self.ops.asarray(rows, dtype=self.dtype)
        return self._generator_cache

    def dual_from_cvxpy(self, y: DenseArray) -> DenseArray:
        r"""Reassemble marginal dual blocks from per-constraint scalar duals.

        Inverse of the row layout of :meth:`to_cvxpy` / :meth:`rhs_to_cvxpy`:
        constraint ``i = (k, \alpha)`` reads off generator :math:`H_\alpha` of
        block ``k``, so the marginal dual is
        :math:`U_k = \sum_\alpha y_{(k,\alpha)} H_\alpha`, a Hermitian ``d x d``
        block. The result is the stacked ``(N, d, d)`` codomain element. The
        caller supplies ``y`` already carrying the intended dual sign.
        """
        ops = self.ops
        H = self._generator_matrices()
        coeffs = ops.reshape(ops.asarray(y, dtype=self.dtype), (self.N, -1))
        return ops.einsum("kg,gab->kab", coeffs, H)

    def rhs_to_cvxpy(self, rhs: DenseArray) -> DenseArray:
        r"""Return the real right-hand side ``b`` matching :meth:`to_cvxpy`.

        ``rhs`` is the stacked codomain array ``(N, d, d)`` of Hermitian
        one-body marginals :math:`\gamma_k`. The returned real vector ``b`` has
        length ``m`` (the number of matrices from :meth:`to_cvxpy`) with
        :math:`b_{(k,\alpha)} = \operatorname{Tr}[H_\alpha^{(k)} \gamma_k]`, laid
        out in the same generator order so that the SDP equality
        :math:`\operatorname{Re}\operatorname{Tr}[A_i \Gamma] = b_i` holds.

        Equivalently :math:`b = \operatorname{Re}\langle H_\alpha, \gamma_k\rangle`
        read off the generator matrices, which is how it is evaluated: the
        per-entry ``re``/``im`` selection of :meth:`_herm_generators` is exactly
        what tracing against the generator performs.
        """
        ops = self.ops
        self.cod.check_member(rhs)
        gamma = ops.reshape(ops.asarray(rhs, dtype=self.dtype), (self.N, self.d, self.d))
        H = self._generator_matrices()
        # Tr[H_a gamma_k] = sum_{ab} H[a,b] gamma[b,a]; real by construction
        # because every generator is Hermitian and every gamma_k is.
        b = ops.einsum("gab,kba->kg", H, gamma)
        # Keep b real even on a complex context, so a solver reads real
        # equalities rather than complex-with-zero-imag ones.
        return ops.asarray(ops.real(ops.reshape(b, (-1,))),
                           dtype=ops.real_dtype(self.dtype))
