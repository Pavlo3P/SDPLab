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

from spacecore import DenseArray, HermitianSpace, jax_pytree_class, Context, LinOp

from ._linalg import _compute_ptraces, kron_sum, make_perm
from ._block_space import BlockMatrixSpace


@jax_pytree_class
class QOTConstraintOp(LinOp[HermitianSpace, BlockMatrixSpace]):
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
        cod = BlockMatrixSpace(d=d, N=N, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=ctx)
        super(QOTConstraintOp, self).__init__(dom, cod, ctx)

        self.d = d
        self.N = N
        self.perms = tuple(make_perm(i, self.N) for i in range(self.N))

    def apply(self, X: DenseArray) -> DenseArray:
        r"""Return :math:`\mathcal{A}\Gamma`, the one-body marginals of ``X``.

        ``X`` is the numerical array representing
        :math:`\Gamma \in \operatorname{dom}(\mathcal{A})`. It has shape
        ``(d^N, d^N)``, and the return value lies in
        :math:`\operatorname{cod}(\mathcal{A})` with shape ``(N, d, d)``.
        The ``k``-th block is
        :math:`(\mathcal{A}\Gamma)_k = \operatorname{Tr}^k[\Gamma]`.
        """
        self.dom.check_member(X)
        return _compute_ptraces(self.dom.ctx, X, d=self.d, N=self.N, perms=self.perms)

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
