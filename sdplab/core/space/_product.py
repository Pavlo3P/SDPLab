from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple, List

from ._base import Space
from ..backend import BackendContext
from ..types import DenseArray


def _prod_int(shape: Tuple[int, ...]) -> int:
    p = 1
    for d in shape:
        p *= int(d)
    return int(p)


@dataclass
class ProductSpace(Space):
    """
    Cartesian product space X = X1 × ... × Xk.

    Elements are tuples:
        x = (x1, ..., xk) with xi ∈ Xi

    Canonical dense coordinates:
        flatten(x) = concat(flatten_i(xi))

    Notes:
      - `shape` for this space is the *1D coordinate length* of the concatenated flattening.
      - `eigh` has no canonical meaning here and raises by default.
    """

    ctx: BackendContext = field(init=False)
    shape: Tuple[int, ...] = field(init=False)

    spaces: Tuple[Space, ...] = field(default_factory=tuple)

    _dims: Tuple[int, ...] = field(init=False, repr=False)
    _offsets: Tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spaces, tuple):
            self.spaces = tuple(self.spaces)

        if len(self.spaces) == 0:
            raise ValueError("ProductSpace requires at least one subspace.")

        # Enforce a single backend context / ops family
        ctx0 = self.spaces[0].ctx
        ops0 = ctx0.ops

        for i, s in enumerate(self.spaces[1:], start=1):
            if s.ctx.ops.family != ops0.family:
                raise ValueError(
                    f"Backend family mismatch in ProductSpace: "
                    f"spaces[0]={ops0.family} but spaces[{i}]={s.ctx.ops.family}"
                )
            if s.ctx.ops is not ops0:
                raise ValueError(
                    "All subspaces must share the same BackendOps instance (ctx.ops). "
                )

        object.__setattr__(self, "ctx", ctx0)

        dims = tuple(_prod_int(s.shape) for s in self.spaces)
        offsets: List[int] = [0]
        for d in dims:
            offsets.append(offsets[-1] + d)

        object.__setattr__(self, "_dims", dims)
        object.__setattr__(self, "_offsets", tuple(offsets))  # length k+1
        object.__setattr__(self, "shape", (offsets[-1],))

    @property
    def arity(self) -> int:
        return len(self.spaces)

    def _check_member(self, x: Any) -> None:
        if not isinstance(x, tuple):
            raise TypeError(f"ProductSpace element must be a tuple, got {type(x).__name__}")
        if len(x) != self.arity:
            raise ValueError(f"Expected tuple of length {self.arity}, got {len(x)}")

        for i, (si, xi) in enumerate(zip(self.spaces, x)):
            try:
                si.check_member(xi)
            except Exception as e:
                raise type(e)(f"Invalid component {i} for spaces[{i}] ({type(si).__name__}): {e}") from e

    def zeros(self) -> Tuple[Any, ...]:
        return tuple(s.zeros() for s in self.spaces)

    def add(self, x: Tuple[Any, ...], y: Tuple[Any, ...]) -> Tuple[Any, ...]:
        self.check_member(x)
        self.check_member(y)
        return tuple(s.add(xi, yi) for s, xi, yi in zip(self.spaces, x, y))

    def scale(self, a: Any, x: Tuple[Any, ...]) -> Tuple[Any, ...]:
        self.check_member(x)
        return tuple(s.scale(a, xi) for s, xi in zip(self.spaces, x))

    def inner(self, x: Tuple[Any, ...], y: Tuple[Any, ...]) -> Any:
        self.check_member(x)
        self.check_member(y)

        # Accumulate via backend ops (vdot works for scalars too, but sum is enough)
        acc = None
        for s, xi, yi in zip(self.spaces, x, y):
            v = s.inner(xi, yi)
            acc = v if acc is None else (acc + v)
        return acc

    def eigh(self, x: Any, k: int = None) -> Any:
        raise NotImplementedError(
            "ProductSpace.eigh is not defined. "
            "Call eigh on a specific component space, or define a custom convention."
        )

    def flatten(self, x: Tuple[Any, ...]) -> DenseArray:
        self.check_member(x)

        parts = []
        for s, xi in zip(self.spaces, x):
            vi = s.flatten(xi)
            vi = self.ctx.assert_dense(vi)
            parts.append(self.ctx.ops.ravel(vi))

        if len(parts) == 1:
            return parts[0]

        return self.ctx.ops.concatenate(parts, axis=0)

    def unflatten(self, v: DenseArray) -> Tuple[Any, ...]:
        v = self.ctx.assert_dense(v)
        v1 = self.ctx.ops.ravel(v)

        xs: List[Any] = []
        for i, s in enumerate(self.spaces):
            a = self._offsets[i]
            b = self._offsets[i + 1]
            vi = v1[a:b]
            xs.append(s.unflatten(vi))

        return tuple(xs)

