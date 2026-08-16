r"""Cost abstractions for conic problems.

The objective term :math:`\langle C, X\rangle` is more than a stored array:
a cost is in general an *operator* — it may act on vectors (a Hamiltonian
applied matrix-free), be stored sparsely, or only implicitly define the
pairing with the primal variable. This module keeps that concept explicit:

- :class:`Cost` — the abstract contract every solver relies on: the pairing
  :meth:`Cost.inner` and a domain-element representation :attr:`Cost.element`
  (used by dual-slack and first-order updates).
- :class:`ElementCost` — a cost that *is* a plain element of any Euclidean
  Jordan algebra domain, including :class:`~spacecore.TreeSpace` trees.
- :class:`HermitianCost` / :class:`DenseHermitianCost` /
  :class:`SparseHermitianCost` — self-adjoint matrix costs that additionally
  act as operators on the underlying vector space (``matvec``), enabling
  matrix-free algorithms on the cost.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from spacecore import (
    Context,
    ContextBound,
    DenseArray,
    DenseLinOp,
    DenseVectorSpace,
    EuclideanJordanAlgebraSpace,
    HermitianSpace,
    InnerProductSpace,
    LinOp,
    SparseArray,
    SparseLinOp,
    TreeElement,
    TreeSpace,
    jax_pytree_class,
    resolve_context_priority,
)


MSpace = TypeVar("MSpace", bound=EuclideanJordanAlgebraSpace)


class Cost(ContextBound, ABC, Generic[MSpace]):
    r"""Abstract cost :math:`X \mapsto \langle C, X\rangle` on a Jordan domain.

    The two operations every solver needs:

    - :meth:`inner` evaluates the (real) objective pairing
      :math:`\langle C, X\rangle`.
    - :attr:`element` returns the cost as a plain element of :attr:`space`,
      used to form the dual slack :math:`\mathcal{A}^\dagger y - C` and
      first-order primal updates.
    """

    space: MSpace

    def __init__(
        self,
        space: MSpace,
        ctx: Context | str | None = None,
    ) -> None:
        resolved_ctx = resolve_context_priority(ctx, space)
        super().__init__(resolved_ctx)

        self.space = space.convert(resolved_ctx)

        if not self.space.is_euclidean:
            raise NotImplementedError(
                "Cost currently supports only Euclidean Jordan domains."
            )

    @abstractmethod
    def inner(self, X: Any) -> Any:
        r"""Return the real pairing :math:`\langle C, X\rangle` as a backend scalar.

        The result is a backend array (not a Python float), so the pairing can
        be evaluated inside compiled loops and under autodiff.
        """

    @property
    @abstractmethod
    def element(self) -> Any:
        """Return the cost as a plain element of :attr:`space`."""

    @abstractmethod
    def to_cvxpy(self) -> Any:
        r"""Return the cost matrix ``C`` in cvxpy-ready form.

        The objective handed to cvxpy is
        :math:`\operatorname{Re}\operatorname{Tr}[C\, X]`, so ``C`` is the
        Hermitian cost matrix (dense or sparse). Only matrix (Hermitian) costs
        support this; element/tree costs do not.
        """

    def to_dense(self) -> Any:
        """Return a dense representation; the default is :attr:`element`."""
        return self.element

    def to_sparse(self) -> SparseArray:
        """Return a sparse representation; the default sparsifies :meth:`to_dense`."""
        return self.ops.assparse(self.to_dense())


@jax_pytree_class
class ElementCost(Cost[EuclideanJordanAlgebraSpace]):
    """A cost stored directly as a domain element (works on any EJA, incl. trees)."""

    def __init__(
        self,
        value: Any,
        space: EuclideanJordanAlgebraSpace,
        ctx: Context | str | None = None,
    ) -> None:
        resolved_ctx = resolve_context_priority(ctx, space)
        super().__init__(space, resolved_ctx)

        if isinstance(value, TreeElement):
            value = value.value
        if isinstance(self.space, TreeSpace):
            # Structural flatten, then move each leaf onto the target backend;
            # convert_element would validate the leaves before converting them.
            leaves = self.space.flatten_tree(value)
            value = self.space.unflatten_tree(
                tuple(self.ctx.asarray(leaf) for leaf in leaves)
            )
        else:
            value = self.ctx.asarray(value)
        self.space.check_member(value)
        self._value = value

    @property
    def element(self) -> Any:
        return self._value

    def to_cvxpy(self) -> Any:
        raise NotImplementedError(
            "ElementCost has no cvxpy matrix form; the cvxpy backend supports "
            "only Hermitian matrix costs."
        )

    def inner(self, X: Any) -> Any:
        return self.ops.real(self.space.inner(self._value, X))

    def _convert(self, new_ctx: Context) -> ElementCost:
        return ElementCost(self._value, self.space.convert(new_ctx), new_ctx)

    def tree_flatten(self):
        """The stored element is the pytree child; space and ctx are static."""
        if isinstance(self.space, TreeSpace):
            return tuple(self.space.flatten_tree(self._value)), (self.space, self.ctx)
        return (self._value,), (self.space, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild without re-running membership validation (jit-safe)."""
        space, ctx = aux
        obj = cls.__new__(cls)
        ContextBound.__init__(obj, ctx)
        obj.space = space
        if isinstance(space, TreeSpace):
            obj._value = space.unflatten_tree(tuple(children))
        else:
            (obj._value,) = children
        return obj


class HermitianCost(Cost[HermitianSpace]):
    """Self-adjoint SDP cost represented as an operator and matrix-space element.

    Beyond the :class:`Cost` contract, a Hermitian cost acts on vectors of the
    underlying Hilbert space through :attr:`operator` (``matvec``), which is
    what matrix-free spectral algorithms consume.
    """

    operator: LinOp

    def __init__(
        self,
        space: HermitianSpace,
        ctx: Context | str | None = None,
    ) -> None:
        """Bind the matrix space, rejecting domains that are not Hermitian.

        The :class:`Cost` base only requires a Euclidean Jordan domain, but a
        Hermitian cost additionally reads ``space.n`` and acts on vectors of the
        underlying Hilbert space, so the domain must be a
        :class:`~spacecore.HermitianSpace` (real symmetric or complex Hermitian).
        """
        super().__init__(space, ctx)

        if not isinstance(self.space, HermitianSpace):
            raise TypeError(
                "HermitianCost requires a HermitianSpace matrix domain; got "
                f"{type(self.space).__name__}. Use ElementCost for other "
                "Euclidean Jordan domains."
            )

    @property
    def matrix_space(self) -> HermitianSpace:
        """The Hermitian matrix space this cost lives in (alias of :attr:`space`)."""
        return self.space

    @property
    def vector_space(self) -> InnerProductSpace:
        """Vector space on which the cost acts."""
        return self.operator.domain

    def _validate_operator(self) -> None:
        if self.operator.domain != self.operator.codomain:
            raise ValueError(
                "HermitianCost requires operator.domain == operator.codomain."
            )

        if not isinstance(self.vector_space, InnerProductSpace):
            raise TypeError(
                "HermitianCost requires an InnerProductSpace operator domain."
            )

        if not self.vector_space.is_euclidean:
            raise NotImplementedError(
                "HermitianCost currently supports only Euclidean vector spaces."
            )

        if self.vector_space.size != self.space.n:
            raise ValueError(
                "HermitianCost operator acts on vectors of dimension "
                f"{self.vector_space.size}, but the matrix space is "
                f"{self.space.n} x {self.space.n}."
            )

        if self.operator.is_hermitian() is False:
            raise ValueError(
                "The supplied matrix is not Hermitian/self-adjoint."
            )

    def matvec(self, x: Any) -> Any:
        """Apply the cost to one vector."""
        return self.operator.apply(x)

    def matvec_batch(self, xs: Any) -> Any:
        """Apply the cost independently over leading batch axes."""
        return self.operator.vapply(xs)

    @classmethod
    def from_dense(
        cls,
        matrix: DenseArray,
        matrix_space: HermitianSpace,
        ctx: Context | str | None = None,
    ) -> DenseHermitianCost:
        return DenseHermitianCost(matrix, matrix_space, ctx)

    @classmethod
    def from_sparse(
        cls,
        matrix: SparseArray,
        matrix_space: HermitianSpace,
        ctx: Context | str | None = None,
    ) -> SparseHermitianCost:
        return SparseHermitianCost(matrix, matrix_space, ctx)


@jax_pytree_class
class DenseHermitianCost(HermitianCost):
    """Hermitian cost backed by a dense matrix."""

    def __init__(
        self,
        matrix: DenseArray,
        matrix_space: HermitianSpace,
        ctx: Context | str | None = None,
    ) -> None:
        resolved_ctx = resolve_context_priority(ctx, matrix_space)
        matrix = resolved_ctx.asarray(matrix)

        super().__init__(matrix_space, resolved_ctx)

        expected_shape = (self.space.n, self.space.n)
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                f"Expected matrix shape {expected_shape}, got {matrix.shape}."
            )

        self.space.check_member(matrix)
        self._matrix = matrix
        self._build_operator()

    def _build_operator(self) -> None:
        vector_space = DenseVectorSpace((self.space.n,), ctx=self.ctx)
        self.operator = DenseLinOp(self._matrix, vector_space, vector_space, self.ctx)
        self._validate_operator()

    @property
    def matrix(self) -> DenseArray:
        """Stored dense matrix."""
        return self._matrix

    @property
    def element(self) -> DenseArray:
        return self._matrix

    def to_cvxpy(self) -> DenseArray:
        """Return the dense Hermitian cost matrix for ``Re Tr[C X]``."""
        return self._matrix

    def inner(self, X: Any) -> Any:
        return self.ops.real(self.space.inner(self._matrix, X))

    def to_dense(self) -> DenseArray:
        return self._matrix

    def to_sparse(self) -> SparseArray:
        return self.ops.assparse(self._matrix)

    def _convert(self, new_ctx: Context) -> DenseHermitianCost:
        return type(self)(
            new_ctx.asarray(self._matrix),
            self.space.convert(new_ctx),
            new_ctx,
        )

    def tree_flatten(self):
        """The matrix is the pytree child; space and ctx are static."""
        return (self._matrix,), (self.space, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild without membership validation (illegal under jit tracing)."""
        space, ctx = aux
        obj = cls.__new__(cls)
        ContextBound.__init__(obj, ctx)
        obj.space = space
        (obj._matrix,) = children
        vector_space = DenseVectorSpace((space.n,), ctx=ctx)
        obj.operator = DenseLinOp(obj._matrix, vector_space, vector_space, ctx)
        return obj


@jax_pytree_class
class SparseHermitianCost(HermitianCost):
    """Hermitian cost backed by a sparse matrix.

    :attr:`element` densifies the stored matrix; algorithms that only need the
    pairing (:meth:`inner`) or the vector action (:meth:`matvec`) never pay
    that cost.
    """

    def __init__(
        self,
        matrix: SparseArray,
        matrix_space: HermitianSpace,
        ctx: Context | str | None = None,
    ) -> None:
        resolved_ctx = resolve_context_priority(ctx, matrix_space)
        resolved_ctx.assert_sparse(matrix)

        super().__init__(matrix_space, resolved_ctx)

        expected_shape = (self.space.n, self.space.n)
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                f"Expected matrix shape {expected_shape}, got {matrix.shape}."
            )

        self._matrix = matrix
        self._build_operator()

    def _build_operator(self) -> None:
        vector_space = DenseVectorSpace((self.space.n,), ctx=self.ctx)
        self.operator = SparseLinOp(self._matrix, vector_space, vector_space, self.ctx)
        self._validate_operator()

    @property
    def matrix(self) -> SparseArray:
        """Stored sparse matrix."""
        return self._matrix

    @property
    def element(self) -> DenseArray:
        return self.to_dense()

    def to_cvxpy(self) -> SparseArray:
        """Return the sparse Hermitian cost matrix for ``Re Tr[C X]``."""
        return self._matrix

    def inner(self, X: Any) -> Any:
        product = self.ops.sparse_matmul(self._matrix, X)
        return self.ops.real(self.ops.sum(self.ops.diagonal(product)))

    def to_dense(self) -> DenseArray:
        return self.operator.to_dense()

    def to_sparse(self) -> SparseArray:
        return self._matrix

    def _convert(self, new_ctx: Context) -> SparseHermitianCost:
        return type(self)(
            new_ctx.assparse(self._matrix),
            self.space.convert(new_ctx),
            new_ctx,
        )

    def tree_flatten(self):
        """The sparse matrix is the pytree child; space and ctx are static."""
        return (self._matrix,), (self.space, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild without membership validation (illegal under jit tracing)."""
        space, ctx = aux
        obj = cls.__new__(cls)
        ContextBound.__init__(obj, ctx)
        obj.space = space
        (obj._matrix,) = children
        vector_space = DenseVectorSpace((space.n,), ctx=ctx)
        obj.operator = SparseLinOp(obj._matrix, vector_space, vector_space, ctx)
        return obj
