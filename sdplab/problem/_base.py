r"""Abstract base class for semidefinite programming problems.

This module uses the following mathematical model for an SDP. The base data
are a cost element :math:`C \in \operatorname{dom}(\mathcal{A})`, a linear
operator

.. math::

    \mathcal{A} : \operatorname{dom}(\mathcal{A})
    \to \operatorname{cod}(\mathcal{A}),

and a right-hand side :math:`b \in \operatorname{cod}(\mathcal{A})`. Together
they define the primal problem

.. math::

    \min_{X \in \operatorname{dom}(\mathcal{A})}\ &\operatorname{Tr}[C X] \\
    \text{s.t.}\quad &\mathcal{A}X = b, \\
                 &X \succeq 0.

In the usual dense case, :math:`\operatorname{dom}(\mathcal{A})` is the real
space of symmetric matrices or the complex space of Hermitian matrices. The
objective element :math:`C` and the primal variable :math:`X` both belong to
:math:`\operatorname{dom}(\mathcal{A})`. The expression :math:`X \succeq 0`
means that :math:`X` is positive semidefinite in the Loewner order.

The value :math:`\mathcal{A}X` and the right-hand side :math:`b` both lie in
:math:`\operatorname{cod}(\mathcal{A})`. If
:math:`\operatorname{cod}(\mathcal{A})` is finite-dimensional, the equation
:math:`\mathcal{A}X = b` represents scalar affine equality constraints, for
example :math:`\operatorname{Tr}[A_i X] = b_i`.

The dual variable satisfies :math:`y \in \operatorname{cod}(\mathcal{A})`. The
adjoint operator

.. math::

    \mathcal{A}^\dagger : \operatorname{cod}(\mathcal{A})
    \to \operatorname{dom}(\mathcal{A})

is defined by

.. math::

    \operatorname{Tr}[(\mathcal{A}X)y]
    =
    \operatorname{Tr}[X(\mathcal{A}^\dagger y)].

The expression :math:`\mathcal{A}^\dagger y - C` is the dual slack expression
used by this package's dual and regularized-dual routines.

Practical checklist:

    1. Decide what the matrix variable :math:`X` is. This determines
       :math:`\operatorname{dom}(\mathcal{A})`.
    2. Decide what numbers, vectors, or blocks must equal prescribed values.
       This determines :math:`\operatorname{cod}(\mathcal{A})` and :math:`b`.
    3. Implement or build a linear operator :math:`\mathcal{A}` that computes
       those quantities from :math:`X`.
    4. Put the cost matrix in :math:`C`.
    5. The problem is represented by the triple :math:`(C, \mathcal{A}, b)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spacecore import (
    ArrayLike, Context, Space,
    ContextBound, resolve_context_priority,
    EuclideanJordanAlgebraSpace,
    HermitianSpace, InnerProductSpace,
)
from spacecore.linop import LinOp

from ._hermitian import HermitianCost
from ..variables import SDPPrimal, SDPDual


def _dispatch_cost(
        C: Any,
        matrix_space: EuclideanJordanAlgebraSpace,
        ctx: Context
) -> HermitianCost:
    if not isinstance(matrix_space, HermitianSpace):
        raise TypeError('Dense cost must have Hermitian space.')

    ops = ctx.ops
    if ops.is_dense(C):
        C = ctx.assert_dense(C)
        return HermitianCost.from_dense(C, matrix_space, ctx)
    elif ops.is_sparse(C):
        C = ctx.assert_sparse(C)
        return HermitianCost.from_sparse(C, matrix_space, ctx)
    elif isinstance(C, HermitianCost):
        if not (C.matrix_space == matrix_space):
            raise TypeError('Linear operator domain and HermitianCost matrix domain should coincide.')
        return C.convert(ctx)
    else:
        raise TypeError('Unknown cost type.')


@dataclass(init=False)
class SDPProblem(ContextBound):
    r"""Base representation of an SDP with linear equality constraints.

    An instance stores the triple :math:`(C, \mathcal{A}, b)` from the
    standard primal SDP

    .. math::

        \min_{X \in \operatorname{dom}(\mathcal{A})}\ &\operatorname{Tr}[C X] \\
        \text{s.t.}\quad &\mathcal{A}X = b, \\
                     &X \succeq 0.

    Here :math:`C \in \operatorname{dom}(\mathcal{A})`, the linear constraint
    operator is
    :math:`\mathcal{A} : \operatorname{dom}(\mathcal{A}) \to
    \operatorname{cod}(\mathcal{A})`, and
    :math:`b \in \operatorname{cod}(\mathcal{A})`. A primal variable
    :math:`X \in \operatorname{dom}(\mathcal{A})` is represented by
    ``SDPPrimal`` and a dual variable
    :math:`y \in \operatorname{cod}(\mathcal{A})` is represented by
    ``SDPDual``. Concrete subclasses specify how :math:`\operatorname{Tr}[C X]`
    is evaluated and how to diagonalize the dual slack expression
    :math:`\mathcal{A}^\dagger y - C`.

    Think of this class as the common language between modeling code and
    solvers. It does not decide which algorithm to use. It only stores the
    mathematical data and the operations that every solver needs.
    """

    def __init__(self,
                 C: HermitianCost | ArrayLike,
                 A: LinOp,
                 b: ArrayLike,
                 ctx: Context | str | None = None,
                 ):
        r"""Create the SDP data :math:`(C, \mathcal{A}, b)`.

        Args:
            C: Objective element in :math:`\operatorname{dom}(\mathcal{A})`.
                For matrix SDPs this is a symmetric or Hermitian matrix with
                the same shape as :math:`X`.
            A: Linear constraint operator
                :math:`\mathcal{A} : \operatorname{dom}(\mathcal{A}) \to
                \operatorname{cod}(\mathcal{A})`.
            b: Right-hand side in :math:`\operatorname{cod}(\mathcal{A})`.
            ctx: Optional backend context. When supplied, ``A`` and array data
                are converted to this context.

        Raises:
            TypeError or ValueError: If :math:`C` is not a member of
                :math:`\operatorname{dom}(\mathcal{A})` or :math:`b` is not a
                member of :math:`\operatorname{cod}(\mathcal{A})`. This catches
                many modeling mistakes early: wrong matrix size, wrong dtype,
                or a constraint vector with the wrong length.
        """
        ctx = resolve_context_priority(ctx, C, A, b)
        super(SDPProblem, self).__init__(ctx)

        if not isinstance(A.dom, EuclideanJordanAlgebraSpace):
            raise TypeError()
        if not A.dom.is_euclidean:
            raise NotImplementedError(
                "SDP problem currently supports only Euclidean matrix spaces."
            )
        if not isinstance(A.cod, InnerProductSpace):
            raise TypeError()
        if not A.cod.is_euclidean:
            raise NotImplementedError(
                "SDP problem currently supports only Euclidean matrix spaces."
            )

        self.A = A.convert(ctx)
        self.A.cod.check_member(b)
        self.C = _dispatch_cost(C, A.dom, ctx)
        self.b = self.dual_from_array(b)

    @property
    def dom(self) -> Space:
        r"""Return :math:`\operatorname{dom}(\mathcal{A})`, the primal space containing :math:`C` and :math:`X`.

        In a dense real SDP, this is usually the space of symmetric
        :math:`n \times n` matrices. In a dense complex SDP, it is usually the
        space of Hermitian :math:`n \times n` matrices.
        """
        return self.A.dom

    @property
    def cod(self) -> Space:
        r"""Return :math:`\operatorname{cod}(\mathcal{A})`, the space containing :math:`\mathcal{A}X`, :math:`b`, and :math:`y`.

        If the SDP has :math:`m` scalar equality constraints,
        :math:`\operatorname{cod}(\mathcal{A})` is typically a vector space of
        shape ``(m,)``. Structured problems may use product or block spaces
        instead.
        """
        return self.A.cod

    def primal_objective(self, primal: SDPPrimal) -> float:
        r"""Evaluate the primal objective :math:`\operatorname{Tr}[C X]`.

        In dense symmetric or Hermitian matrix spaces this is the trace
        objective :math:`\operatorname{Re}\operatorname{Tr}[C X]`.
        """
        raise self.ops.real(self.C.inner(primal.X))

    def dual_objective(self, dual: SDPDual) -> float:
        r"""Evaluate the linear dual objective term :math:`\operatorname{Tr}[b\ y]` in :math:`\operatorname{cod}(\mathcal{A})`."""
        return self.ops.real(self.b.inner(dual))

    def A_apply(self, primal: SDPPrimal) -> SDPDual:
        r"""Return :math:`\mathcal{A}X` wrapped as an ``SDPDual``-space value.

        Use this to check equality feasibility: a primal candidate :math:`X` is
        feasible for the affine constraints when ``A_apply(X).y`` equals
        :math:`b` in :math:`\operatorname{cod}(\mathcal{A})`.
        """
        return self.dual_from_array(self.A.apply(primal.X))

    def AT_apply(self, dual: SDPDual) -> SDPPrimal:
        r"""Return :math:`\mathcal{A}^\dagger y` wrapped as an ``SDPPrimal``-space value.

        The adjoint :math:`\mathcal{A}^\dagger` is taken with respect to the
        inner products of :math:`\operatorname{dom}(\mathcal{A})` and
        :math:`\operatorname{cod}(\mathcal{A})`. It moves dual variables back
        into the matrix space where they can be compared with :math:`C`.
        """
        return self.primal_from_array(self.A.rapply(dual.y))

    def dual_from_array(self, array: ArrayLike) -> SDPDual:
        r"""Interpret ``array`` in :math:`\operatorname{cod}(\mathcal{A})` as a dual variable :math:`y`."""
        return SDPDual(self.cod, array, ctx=self.ctx)

    def primal_from_array(self, array: ArrayLike) -> SDPPrimal:
        r"""Interpret ``array`` in :math:`\operatorname{dom}(\mathcal{A})` as a primal variable :math:`X`."""
        return SDPPrimal(self.dom, array, ctx=self.ctx)
