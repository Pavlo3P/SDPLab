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

from abc import ABC, abstractmethod
from dataclasses import dataclass

from spacecore import (
    ArrayLike, DenseArray, BackendOps, Context, Space,
    ContextBound, resolve_context_priority
)
from spacecore.linop import LinOp

from ...sdp.variables import SDPPrimal, SDPDual


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
                 C: ArrayLike,
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
        ctx = resolve_context_priority(ctx, A)
        super(SDPProblem, self).__init__(ctx)

        self.A = A.convert(ctx)
        self.A.dom.check_member(C)
        self.A.cod.check_member(b)
        self.C = C
        self.b = b

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

    @abstractmethod
    def primal_objective(self, primal: SDPPrimal) -> float:
        r"""Evaluate the primal objective :math:`\operatorname{Tr}[C X]`.

        In dense symmetric or Hermitian matrix spaces this is the trace
        objective :math:`\operatorname{Re}\operatorname{Tr}[C X]`.
        """
        raise NotImplementedError()

    def dual_objective(self, dual: SDPDual) -> float:
        r"""Evaluate the linear dual objective term :math:`\operatorname{Tr}[b\ y]` in :math:`\operatorname{cod}(\mathcal{A})`."""
        return self.ops.real(self.cod.inner(self.b, dual.y))

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

    def primal_from_eigendecomp(self, eigvals: DenseArray, eigvecs: DenseArray) -> SDPPrimal:
        r"""Build :math:`X = V \operatorname{diag}(\lambda) V^\dagger` as a primal variable.

        Subclasses that represent dense matrix cones should use ``eigvecs`` as
        columns of :math:`V` and ``eigvals`` as the corresponding spectrum
        :math:`\lambda`.
        """
        raise NotImplementedError()

    @abstractmethod
    def dual_constr_eig_decomp(self, dual: SDPDual, k: int = None) -> tuple[DenseArray, DenseArray]:
        r"""Return eigenpairs of the dual slack expression :math:`\mathcal{A}^\dagger y - C`.

        This eigendecomposition is used to check or manipulate the
        semidefinite dual constraint and to recover primal matrices in
        regularized formulations.

        Args:
            dual: Dual variable used to form the slack expression.
            k: Number of first eigenpairs to return. If ``None``, return the
                full eigendecomposition.
        """
        raise NotImplementedError()
