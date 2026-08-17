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

r"""Conic problem data over a Euclidean Jordan algebra.

This module uses the following mathematical model for an SDP. The base data
are a cost element :math:`C \in \operatorname{dom}(\mathcal{A})`, a linear
operator

.. math::

    \mathcal{A} : \operatorname{dom}(\mathcal{A})
    \to \operatorname{cod}(\mathcal{A}),

and a right-hand side :math:`b \in \operatorname{cod}(\mathcal{A})`. Together
they define the primal problem

.. math::

    \min_{X \in \operatorname{dom}(\mathcal{A})}\ &\langle C, X\rangle \\
    \text{s.t.}\quad &\mathcal{A}X = b, \\
                 &X \succeq 0.

The domain is any Euclidean Jordan algebra space: the dense space of
symmetric or Hermitian matrices (classic SDP), an elementwise Jordan space
(linear programming over the nonnegative orthant), or a
:class:`~spacecore.TreeSpace` of such leaves (block-structured problems).
The pairing :math:`\langle C, X\rangle` is the domain inner product — the
trace pairing :math:`\operatorname{Tr}[C X]` in the Hermitian case — and
:math:`X \succeq 0` means the spectrum of :math:`X` is nonnegative in the
Jordan-algebraic sense.

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
:math:`\langle \mathcal{A}X, y\rangle = \langle X, \mathcal{A}^\dagger y\rangle`.
The expression :math:`\mathcal{A}^\dagger y - C` is the dual slack used by
this package's dual and regularized-dual routines.

Practical checklist:

    1. Decide what the conic variable :math:`X` is. This determines
       :math:`\operatorname{dom}(\mathcal{A})`.
    2. Decide what numbers, vectors, or blocks must equal prescribed values.
       This determines :math:`\operatorname{cod}(\mathcal{A})` and :math:`b`.
    3. Implement or build a linear operator :math:`\mathcal{A}` that computes
       those quantities from :math:`X`.
    4. Put the cost element in :math:`C`.
    5. The problem is represented by the triple :math:`(C, \mathcal{A}, b)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spacecore import (
    ArrayLike, Context,
    ContextBound, resolve_context_priority,
    DenseLinOp, SparseLinOp,
    EuclideanJordanAlgebraSpace, HermitianSpace,
    InnerProductSpace, TreeElement, TreeSpace,
    jax_pytree_class,
)
from spacecore.linop import LinOp

from ._constraint import (
    ConstraintOp,
    DenseConstraintOp,
    SparseConstraintOp,
    WrappedConstraintOp,
)
from ._cost import Cost, ElementCost, HermitianCost


def as_member(space: InnerProductSpace, x: Any, ctx: Context) -> Any:
    """Return ``x`` as a raw member of ``space`` represented in ``ctx``.

    Accepts backend arrays, array-likes, raw trees, and bound
    :class:`~spacecore.TreeElement` values; validates membership according to
    the context's check level.
    """
    if isinstance(x, TreeElement):
        x = x.value
    if isinstance(space, TreeSpace):
        # Structural flatten, then move each leaf onto the target backend;
        # convert_element would validate the leaves before converting them.
        leaves = space.flatten_tree(x)
        x = space.unflatten_tree(tuple(ctx.asarray(leaf) for leaf in leaves))
    else:
        if ctx.ops.is_sparse(x):
            raise TypeError(
                "Sparse data is not supported here; densify it first "
                "(e.g. ops.to_dense)."
            )
        x = ctx.asarray(x)
    space.check_member(x)
    return x


def _dispatch_cost(
    C: Any,
    domain: EuclideanJordanAlgebraSpace,
    ctx: Context,
) -> Cost:
    """Wrap ``C`` as a :class:`Cost` on ``domain``.

    A pre-built :class:`Cost` is converted and checked against the domain;
    dense/sparse matrices on a Hermitian domain become operator-backed
    Hermitian costs; anything else (including raw trees) becomes an
    :class:`ElementCost`.
    """
    if isinstance(C, Cost):
        C = C.convert(ctx)
        if not (C.space == domain):
            raise TypeError(
                "Cost space and linear operator domain must coincide."
            )
        return C
    if isinstance(domain, HermitianSpace):
        if ctx.ops.is_sparse(C):
            return HermitianCost.from_sparse(ctx.assparse(C), domain, ctx)
        return HermitianCost.from_dense(ctx.asarray(C), domain, ctx)
    return ElementCost(C, domain, ctx)


def _dispatch_constraint(A: LinOp, ctx: Context) -> ConstraintOp:
    """Resolve a user-supplied constraint operator into a :class:`ConstraintOp`.

    A pre-built :class:`~sdplab.problem.ConstraintOp` (including the QOT
    operator) is converted and returned. A stored-tensor
    :class:`~spacecore.DenseLinOp` / :class:`~spacecore.SparseLinOp` is wrapped
    in the matching constraint operator. Any other :class:`~spacecore.LinOp`
    (e.g. a hand-written matrix-free operator) is wrapped by delegation and
    stays matrix-free; its per-constraint matrices are materialized lazily, only
    if the cvxpy backend asks for them.
    """
    if isinstance(A, ConstraintOp):
        return A.convert(ctx)
    if isinstance(A, DenseLinOp):
        return DenseConstraintOp.from_linop(A.convert(ctx))
    if isinstance(A, SparseLinOp):
        return SparseConstraintOp.from_linop(A.convert(ctx))
    if isinstance(A, LinOp):
        return WrappedConstraintOp.from_linop(A.convert(ctx))
    raise TypeError(
        f"SDPProblem requires a LinOp constraint operator; got {type(A).__name__}."
    )


@jax_pytree_class
@dataclass(init=False)
class SDPProblem(ContextBound):
    r"""Base representation of a conic problem with linear equality constraints.

    An instance stores the triple :math:`(C, \mathcal{A}, b)` from the
    standard primal problem

    .. math::

        \min_{X \in \operatorname{dom}(\mathcal{A})}\ &\langle C, X\rangle \\
        \text{s.t.}\quad &\mathcal{A}X = b, \\
                     &X \succeq 0.

    Here :math:`C \in \operatorname{dom}(\mathcal{A})`, the linear constraint
    operator is
    :math:`\mathcal{A} : \operatorname{dom}(\mathcal{A}) \to
    \operatorname{cod}(\mathcal{A})`, and
    :math:`b \in \operatorname{cod}(\mathcal{A})`. Both :math:`C` and
    :math:`b` are stored as plain elements of their spaces; primal and dual
    variables are likewise plain elements of ``dom`` and ``cod``.

    Think of this class as the common language between modeling code and
    solvers. It does not decide which algorithm to use. It only stores the
    mathematical data and the operations that every solver needs.
    """

    def __init__(self,
                 C: Cost | ArrayLike,
                 A: LinOp,
                 b: ArrayLike,
                 ctx: Context | str | None = None,
                 ):
        r"""Create the problem data :math:`(C, \mathcal{A}, b)`.

        Args:
            C: Objective in :math:`\operatorname{dom}(\mathcal{A})` — a
                prepared :class:`~sdplab.problem.Cost`, a dense or sparse
                matrix on a Hermitian domain, or a plain domain element (a
                tree of blocks on a tree domain). Arrays are wrapped through
                :func:`_dispatch_cost`.
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
            raise TypeError(
                "SDPProblem requires a EuclideanJordanAlgebraSpace domain; "
                f"got {type(A.dom).__name__}."
            )
        if not A.dom.is_euclidean:
            raise NotImplementedError(
                "SDPProblem currently supports only Euclidean domains."
            )
        if not isinstance(A.cod, InnerProductSpace):
            raise TypeError(
                "SDPProblem requires an InnerProductSpace codomain; "
                f"got {type(A.cod).__name__}."
            )
        if not A.cod.is_euclidean:
            raise NotImplementedError(
                "SDPProblem currently supports only Euclidean codomains."
            )

        self.A = _dispatch_constraint(A, ctx)
        self.C = _dispatch_cost(C, self.A.dom, ctx)
        self.b = as_member(self.A.cod, b, ctx)

    @property
    def dom(self) -> EuclideanJordanAlgebraSpace:
        r"""Return :math:`\operatorname{dom}(\mathcal{A})`, the primal space containing :math:`C` and :math:`X`."""
        return self.A.dom

    @property
    def cod(self) -> InnerProductSpace:
        r"""Return :math:`\operatorname{cod}(\mathcal{A})`, the space containing :math:`\mathcal{A}X`, :math:`b`, and :math:`y`."""
        return self.A.cod

    def primal_objective(self, X: Any) -> Any:
        r"""Evaluate the primal objective :math:`\langle C, X\rangle` at a ``dom`` element.

        In dense symmetric or Hermitian matrix spaces this is the trace
        objective :math:`\operatorname{Re}\operatorname{Tr}[C X]`.
        """
        return self.C.inner(X)

    def dual_objective(self, y: Any) -> Any:
        r"""Evaluate the linear dual objective term :math:`\langle b, y\rangle` at a ``cod`` element."""
        return self.ops.real(self.cod.inner(self.b, y))

    def dual_slack(self, y: Any) -> Any:
        r"""Return the dual slack :math:`\mathcal{A}^\dagger y - C` in ``dom``."""
        return self.dom.axpy(-1.0, self.C.element, self.A.rapply(y))

    def feasibility_gap(self, X: Any) -> Any:
        r"""Return :math:`\mathcal{A}X - b`, the equality-constraint residual in ``cod``."""
        return self.cod.axpy(-1.0, self.b, self.A.apply(X))

    def _convert(self, new_ctx: Context) -> SDPProblem:
        """Return an equivalent problem with data represented in ``new_ctx``.

        ``C`` and ``b`` are re-validated against the converted spaces by the
        constructor; the constructor's ``as_member`` moves them onto
        ``new_ctx`` first.
        """
        return SDPProblem(self.C, self.A, self.b, new_ctx)

    def tree_flatten(self):
        """Children are the array-bearing cost, operator, and RHS; ctx is static.

        The cost ``C``, operator ``A``, and RHS ``b`` flow through as pytree
        leaves. Reconstruction restores them directly (no re-validation),
        keeping the round-trip safe under tracing.
        """
        return (self.C, self.A, self.b), (self.ctx,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild a problem from JAX PyTree data without re-running validation."""
        C, A, b = children
        (ctx,) = aux
        obj = cls.__new__(cls)
        ContextBound.__init__(obj, ctx)
        obj.C = C
        obj.A = A
        obj.b = b
        return obj
