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

from spacecore import DenseArray, HermitianSpace, SparseArray, StackedSpace, jax_pytree_class, Context, LinOp, checked_method

from ...linalg import kron_sum
from ...linalg.dense._ptrace import make_perm, _compute_ptraces

@jax_pytree_class
class QOTConstraintOp(LinOp[HermitianSpace, StackedSpace]):
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
        return _compute_ptraces(self.dom.ctx, X, d=self.d, N=self.N, perms=self.perms)

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
        """
        self.cod.check_member(y)
        return kron_sum(self.cod.ctx, y)

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

    def to_sparse(self) -> SparseArray:
        r"""Return the QOT constraints as a real-SDP sparse constraint stack.

        This adapts the dense Kronecker construction of the QOT-to-SDP proof
        (th. 3.1 of https://arxiv.org/abs/2105.06922) into a sparse standard-form
        stack suitable for a general SDP solver such as the CVXPY backend, whose
        equalities read :math:`\operatorname{Re}\operatorname{Tr}[A_i \Gamma] = b_i`.

        Row ``i = (k, \alpha)`` is the flattened Hermitian constraint matrix

        .. math::

            A_i = \mathcal{A}^\dagger(H_\alpha^{(k)})
                = I \otimes \cdots \otimes H_\alpha \otimes \cdots \otimes I,

        where :math:`H_\alpha` ranges over the real-coordinate Hermitian
        generators of the ``k``-th ``d x d`` block (see :meth:`_herm_generators`).
        Because each :math:`H_\alpha` is Hermitian, the matching right-hand side
        :math:`b_i = \operatorname{Tr}[H_\alpha^{(k)} \gamma_k]` is **real** even
        though the marginals :math:`\gamma_k` are complex Hermitian -- naively
        flattening the marginal entries would instead give a complex ``b``. The
        matching ``b`` is produced by :meth:`marginals_to_rhs`.

        The returned matrix has shape ``(m, d^{2N})``, where
        ``m = N d(d+1)/2`` for a real context and ``m = N d^2`` for a complex
        one. Row ``i`` reshaped to ``(d^N, d^N)`` recovers :math:`A_i` (row-major),
        so a solver can form ``trace(A_i @ Gamma)`` directly. Each generator
        embeds over ``d^{N-1}`` configurations of the traced-out subsystems, so
        storage stays sparse rather than the dense ``m d^{2N}``.
        """
        import numpy as np
        from scipy.sparse import coo_matrix

        d, N = self.d, self.N
        D = d ** N
        dom_size = D * D

        generators = list(self._herm_generators())
        m_block = len(generators)                 # d(d+1)/2 real, d^2 complex
        m = N * m_block
        is_complex = self.ops.is_complex_dtype(self.dtype)

        rows_all, cols_all, data_all = [], [], []
        for k in range(N):
            block = d ** (N - k)          # stride of subsystem k in a D-index
            right = d ** (N - 1 - k)      # d-index stride of subsystem k
            left = d ** k                 # number of higher-subsystem configs

            hi = np.arange(left)[:, None]
            lo = np.arange(right)[None, :]
            base = (hi * block + lo).ravel()          # D-index shared by bra/ket

            for g, (entries, _coord) in enumerate(generators):
                row_id = k * m_block + g
                for a, b, value in entries:
                    R = base + a * right              # bra keeps subsystem k = a
                    C = base + b * right              # ket keeps subsystem k = b
                    col = R * D + C
                    rows_all.append(np.full(col.shape, row_id))
                    cols_all.append(col)
                    data_all.append(np.full(col.shape, value))

        rows = np.concatenate(rows_all)
        cols = np.concatenate(cols_all)
        data = np.concatenate(data_all).astype(np.complex128 if is_complex else np.float64)

        coo = coo_matrix((data, (rows, cols)), shape=(m, dom_size))
        return self.ctx.assparse(coo)

    def rhs_to_cvxpy(self, rhs: DenseArray) -> DenseArray:
        r"""Return the real right-hand side ``b`` matching :meth:`to_sparse`.

        ``marginals`` is the stacked codomain array ``(N, d, d)`` of Hermitian
        one-body marginals :math:`\gamma_k`. The returned real vector ``b`` has
        length ``m`` (the row count of :meth:`to_sparse`) with
        :math:`b_{(k,\alpha)} = \operatorname{Tr}[H_\alpha^{(k)} \gamma_k]`, laid
        out in the same generator order so that the SDP equality
        :math:`\operatorname{Re}\operatorname{Tr}[A_i \Gamma] = b_i` holds.
        """
        import numpy as np

        self.cod.check_member(rhs)
        gamma = np.asarray(rhs).reshape(self.N, self.d, self.d)
        generators = list(self._herm_generators())

        b = np.empty(self.N * len(generators), dtype=np.float64)
        for k in range(self.N):
            for g, (_entries, (part, i, j)) in enumerate(generators):
                entry = gamma[k, i, j]
                b[k * len(generators) + g] = entry.real if part == "re" else entry.imag
        # b is real by construction, even for a complex context: keep it real so
        # a solver reads real equalities rather than complex-with-zero-imag ones.
        real_dtype = self.ops.real_dtype(self.dtype)
        return self.ops.asarray(b, dtype=real_dtype)
