from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from ._base import LinOp
from ..space import ProductSpace
from ..backend import jax_pytree_class


def assert_spaces_identical(spaces):
    """
    Raise if spaces are not semantically identical.

    Semantic identity means:
      - same concrete Space class
      - same shape
      - same backend ops object
      - same dtype
    """
    if not spaces:
        raise ValueError("spaces must be non-empty")

    ref = spaces[0]
    for i, sp in enumerate(spaces[1:], start=1):
        if type(sp) is not type(ref):
            raise TypeError(
                f"Space {i} has type {type(sp).__name__}, "
                f"expected {type(ref).__name__}."
            )
        if sp.shape != ref.shape:
            raise ValueError(
                f"Space {i} has shape {sp.shape}, expected {ref.shape}."
            )
        if sp.ctx.ops.family != ref.ctx.ops.family:
            raise TypeError(
                f"Space {i} has different backend ops."
            )
        if sp.ctx.dtype != ref.ctx.dtype:
            raise TypeError(
                f"Space {i} has dtype {sp.ctx.dtype}, expected {ref.ctx.dtype}."
            )


@jax_pytree_class
@dataclass(slots=True)
class BlockDiagonalLinOp(LinOp):
    """
    Block-diagonal operator between product spaces.

    dom = X1 × ... × Xk
    cod = Y1 × ... × Yk

    ops[i] : Xi -> Yi
    """
    ops: Tuple[LinOp, ...]

    def __post_init__(self) -> None:
        self._check_backends()

        if not isinstance(self.dom, ProductSpace) or not isinstance(self.cod, ProductSpace):
            raise TypeError("BlockDiagonalLinOp expects dom and cod to be ProductSpace.")

        if len(self.ops) != len(self.dom.spaces) or len(self.ops) != len(self.cod.spaces):
            raise ValueError("Number of component ops must match product arity.")

        for i, A in enumerate(self.ops):
            try:
                assert_spaces_identical((A.dom, self.dom.spaces[i]))
                assert_spaces_identical((A.cod, self.cod.spaces[i]))
            except (TypeError, ValueError) as e:
                raise TypeError(f"Component op {i} has incompatible dom/cod spaces.") from e

    def apply(self, x: Any) -> Any:
        self.assert_domain(x)
        return tuple(A.apply(xi) for A, xi in zip(self.ops, x))

    def rapply(self, y: Any) -> Any:
        self.assert_codomain(y)
        return tuple(A.rapply(yi) for A, yi in zip(self.ops, y))

    def tree_flatten(self):
        aux = (self.dom, self.cod)
        children = self.ops
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        dom, cod = aux
        ops = children
        return cls(dom, cod, ops)

    @classmethod
    def from_ops(cls, ops: Tuple[LinOp, ...]) -> "BlockDiagonalLinOp":
        if not ops:
            raise ValueError("ops must be non-empty")

        dom = ProductSpace(tuple(op.dom for op in ops))
        cod = ProductSpace(tuple(op.cod for op in ops))
        return cls(dom, cod, ops)


@jax_pytree_class
@dataclass(slots=True)
class SumToSingleLinOp(LinOp):
    """
    Sum of component operators from a product domain into a single codomain.

    dom = X1 × ... × Xk
    cod = Y

    ops[i] : Xi -> Y
    apply(x)  = sum_i ops[i](x_i)
    rapply(y) = (ops[i]^*(y))_i
    """
    ops: Tuple[LinOp, ...]

    def __post_init__(self) -> None:
        self._check_backends()

        if not isinstance(self.dom, ProductSpace):
            raise TypeError("SumToSingleLinOp expects dom to be ProductSpace.")

        if len(self.ops) != len(self.dom.spaces):
            raise ValueError("Number of ops must match product arity.")

        for i, A in enumerate(self.ops):
            try:
                assert_spaces_identical((A.dom, self.dom.spaces[i]))
                assert_spaces_identical((A.cod, self.cod))
            except (ValueError, TypeError) as e:
                raise TypeError(f"Component op {i} must map dom.spaces[{i}] -> cod.") from e

    def apply(self, x: Any) -> Any:
        self.assert_domain(x)
        acc = None
        for A, xi in zip(self.ops, x):
            yi = A.apply(xi)
            acc = yi if acc is None else self.cod.add(yi, acc)
        return acc

    def rapply(self, y: Any) -> Any:
        self.assert_codomain(y)
        return tuple(A.rapply(y) for A in self.ops)

    def tree_flatten(self):
        aux = (self.dom, self.cod)
        children = self.ops
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        dom, cod = aux
        ops = children
        return cls(dom, cod, ops)

    @classmethod
    def from_ops(cls, ops: Tuple[LinOp, ...]) -> "SumToSingleLinOp":
        if not ops:
            raise ValueError("ops must be non-empty")

        dom = ProductSpace(tuple(op.dom for op in ops))
        cod = ops[0].cod

        return cls(dom, cod, ops)


@jax_pytree_class
@dataclass(slots=True)
class StackedLinOp(LinOp):
    """
    Stack of operators from a single domain into a product codomain.

    dom = X
    cod = Y1 × ... × Yk

    ops[i] : X -> Yi
    apply(x)  = (ops[i](x))_i
    rapply(y) = sum_i ops[i]^*(y_i)
    """
    ops: Tuple[LinOp, ...]

    def __post_init__(self) -> None:
        self._check_backends()

        if not isinstance(self.cod, ProductSpace):
            raise TypeError("StackedLinOp expects cod to be ProductSpace.")

        if len(self.ops) != len(self.cod.spaces):
            raise ValueError("Number of ops must match codomain product arity.")

        for i, A in enumerate(self.ops):
            try:
                assert_spaces_identical((A.dom, self.dom))
                assert_spaces_identical((A.cod, self.cod.spaces[i]))
            except (ValueError, TypeError) as e:
                raise TypeError(f"Component op {i} must map dom -> cod.spaces[{i}].") from e

    def apply(self, x: Any) -> Any:
        self.assert_domain(x)
        return tuple(A.apply(x) for A in self.ops)

    def rapply(self, y: Any) -> Any:
        self.assert_codomain(y)
        acc = None
        for A, yi in zip(self.ops, y):
            xi = A.rapply(yi)
            acc = xi if acc is None else self.dom.add(xi, acc)
        return acc

    def tree_flatten(self):
        aux = (self.dom, self.cod)
        children = self.ops
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        dom, cod = aux
        ops = children
        return cls(dom, cod, ops)

    @classmethod
    def from_ops(cls, ops: Tuple[LinOp, ...]) -> "StackedLinOp":
        if not ops:
            raise ValueError("ops must be non-empty")

        cod = ProductSpace(tuple(op.cod for op in ops))
        dom = ops[0].dom

        return cls(dom, cod, ops)
