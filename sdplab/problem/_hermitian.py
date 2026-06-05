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
    resolve_context_priority,
)


MSpace = TypeVar("MSpace", bound=EuclideanJordanAlgebraSpace)


class HermitianCost(ContextBound, ABC, Generic[MSpace]):
    """Self-adjoint SDP cost represented as an operator and matrix-space element."""

    matrix_space: MSpace
    operator: LinOp

    def __init__(
        self,
        matrix_space: MSpace,
        ctx: Context | str | None = None,
    ) -> None:
        resolved_ctx = resolve_context_priority(ctx, matrix_space)
        super().__init__(resolved_ctx)

        self.matrix_space = matrix_space.convert(resolved_ctx)

        if not self.matrix_space.is_euclidean:
            raise NotImplementedError(
                "HermitianCost currently supports only Euclidean matrix spaces."
            )

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

    @abstractmethod
    def inner(self, X: Any) -> float:
        """Return the real matrix-space inner product with ``X``."""

    @abstractmethod
    def to_dense(self) -> DenseArray:
        """Return a dense matrix representation."""

    @abstractmethod
    def to_sparse(self) -> SparseArray:
        """Return a sparse matrix representation."""

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


class DenseHermitianCost(HermitianCost[HermitianSpace]):
    """Hermitian cost backed by a dense matrix."""

    def __init__(
        self,
        matrix: DenseArray,
        matrix_space: HermitianSpace,
        ctx: Context | str | None = None,
    ) -> None:
        resolved_ctx = resolve_context_priority(ctx, matrix_space)
        resolved_ctx.assert_dense(matrix)

        super().__init__(matrix_space, resolved_ctx)

        expected_shape = (self.matrix_space.n, self.matrix_space.n)
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                f"Expected matrix shape {expected_shape}, got {matrix.shape}."
            )

        self.matrix_space.check_member(matrix)

        self._matrix = matrix

        vector_space = DenseVectorSpace(
            (self.matrix_space.n,),
            ctx=resolved_ctx,
        )
        self.operator = DenseLinOp(
            matrix,
            vector_space,
            vector_space,
            resolved_ctx,
        )

        self._validate_operator()

    @property
    def matrix(self) -> DenseArray:
        """Stored dense matrix."""
        return self._matrix

    def inner(self, X: Any) -> float:
        value = self.matrix_space.inner(self.matrix, X)
        return float(self.ops.real(value))

    def to_dense(self) -> DenseArray:
        return self.matrix

    def to_sparse(self) -> SparseArray:
        return self.ops.assparse(self.matrix)

    def _convert(self, new_ctx: Context) -> DenseHermitianCost:
        return type(self)(
            new_ctx.asarray(self.matrix),
            self.matrix_space.convert(new_ctx),
            new_ctx,
        )


class SparseHermitianCost(HermitianCost[HermitianSpace]):
    """Hermitian cost backed by a sparse matrix."""

    def __init__(
        self,
        matrix: SparseArray,
        matrix_space: HermitianSpace,
        ctx: Context | str | None = None,
    ) -> None:
        resolved_ctx = resolve_context_priority(ctx, matrix_space)
        resolved_ctx.assert_sparse(matrix)

        super().__init__(matrix_space, resolved_ctx)

        expected_shape = (self.matrix_space.n, self.matrix_space.n)
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                f"Expected matrix shape {expected_shape}, got {matrix.shape}."
            )

        self._matrix = matrix

        vector_space = DenseVectorSpace(
            (self.matrix_space.n,),
            ctx=resolved_ctx,
        )
        self.operator = SparseLinOp(
            matrix,
            vector_space,
            vector_space,
            resolved_ctx,
        )

        self._validate_operator()

    @property
    def matrix(self) -> SparseArray:
        """Stored sparse matrix."""
        return self._matrix

    def inner(self, X: Any) -> float:
        self.matrix_space.check_member(X)

        product = self.ops.sparse_matmul(self.matrix, X)
        value = self.ops.sum(self.ops.diagonal(product))

        return float(self.ops.real(value))

    def to_dense(self) -> DenseArray:
        return self.operator.to_dense()

    def to_sparse(self) -> SparseArray:
        return self.matrix

    def _convert(self, new_ctx: Context) -> SparseHermitianCost:
        return type(self)(
            new_ctx.assparse(self.matrix),
            self.matrix_space.convert(new_ctx),
            new_ctx,
        )