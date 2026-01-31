from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence, Tuple, Callable, TypeVar, Optional

from ._family import BackendFamily
from ..types import DenseArray, SparseArray, DType, ArrayLike, Index, T, X, Y, R, Carry


class BackendOps(ABC):
    """
    Backend-agnostic numerical ops interface (portable core).

    Contract:
      - This base class exposes only the portable subset used by library internals.
      - Concrete backends (NumPy/JAX/Torch) may extend these methods with additional
        optional keyword parameters (e.g., `order=`, `out=`, `where=`, `like=`, ...).
    """

    family: BackendFamily

    @abstractmethod
    def sanitize_dtype(self, dtype: DType | None) -> DType | None:
        ...

    @abstractmethod
    def is_dense(self, x: Any) -> bool:
        ...

    @abstractmethod
    def is_sparse(self, x: Any) -> bool:
        ...

    @property
    @abstractmethod
    def inf(self) -> DenseArray:
        """Positive infinity (backend scalar)."""

    @property
    @abstractmethod
    def nan(self) -> DenseArray:
        """NaN (backend scalar)."""

    @property
    @abstractmethod
    def pi(self) -> DenseArray:
        """π as backend scalar."""

    @property
    @abstractmethod
    def e(self) -> DenseArray:
        """Euler's number as backend scalar."""

    @property
    @abstractmethod
    def eps(self) -> DenseArray:
        """Machine epsilon for default float dtype."""

    @abstractmethod
    def asarray(self, x: Any, dtype: DType | None = None) -> DenseArray:
        ...

    @abstractmethod
    def empty(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseArray:
        ...

    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseArray:
        ...

    @abstractmethod
    def ones(self, shape: Tuple[int, ...], dtype: DType | None = None) -> DenseArray:
        ...

    @abstractmethod
    def arange(self, start: int, stop: int | None = None, step: int | None = None, dtype: DType | None = None) -> DenseArray:
        ...

    @abstractmethod
    def full(self, shape: Tuple[int, ...], fill_value: Any, dtype: DType | None = None) -> DenseArray:
        ...

    @abstractmethod
    def eye(self, n: int, m: int | None = None, dtype: DType | None = None) -> DenseArray:
        ...

    @abstractmethod
    def ravel(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def reshape(self, x: DenseArray, shape: Tuple[int, ...] | int) -> DenseArray:
        ...

    @abstractmethod
    def transpose(self, x: DenseArray, axes: Sequence[int] | None = None) -> DenseArray:
        ...

    @abstractmethod
    def stack(self, arrays: Sequence[DenseArray], axis: int = 0) -> DenseArray:
        ...

    @abstractmethod
    def conj(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def real(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def imag(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def abs(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def sign(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def sqrt(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def sum(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        dtype: DType | None = None,
    ) -> DenseArray:
        ...

    @abstractmethod
    def prod(
        self,
        x: DenseArray,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        dtype: DType | None = None,
    ) -> DenseArray:
        ...

    @abstractmethod
    def trace(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def argsort(self, x: DenseArray, axis: int = -1) -> DenseArray:
        ...

    @abstractmethod
    def sort(self, x: DenseArray, axis: int = -1) -> DenseArray:
        ...

    @abstractmethod
    def argmin(self, x: DenseArray, axis: int | None = None, keepdims: bool = False) -> DenseArray:
        ...

    @abstractmethod
    def argmax(self, x: DenseArray, axis: int | None = None, keepdims: bool = False) -> DenseArray:
        ...

    @abstractmethod
    def vdot(self, x: DenseArray, y: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def matmul(self, a: DenseArray, b: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def sparse_matmul(self, a: SparseArray, b: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def kron(self, a: DenseArray, b: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def einsum(self, subscripts: str, *operands: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def eigh(self, x: DenseArray) -> tuple[DenseArray, DenseArray]:
        ...

    @abstractmethod
    def logsumexp(self, a: DenseArray, axis: int | Sequence[int] | None = None, b: DenseArray = None,
                  keepdims: bool = False, return_sign: bool = False) -> DenseArray:
        ...

    @abstractmethod
    def exp(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def log(self, x: DenseArray) -> DenseArray:
        ...

    @abstractmethod
    def where(self, condition: DenseArray | bool, x: ArrayLike, y: ArrayLike) -> DenseArray:
        ...

    @abstractmethod
    def maximum(self, x: ArrayLike, y: ArrayLike) -> DenseArray:
        ...

    @abstractmethod
    def concatenate(self, arrays: Sequence[DenseArray], axis: int = 0, dtype: DType = None) -> DenseArray:
        ...

    @abstractmethod
    def index_set(
            self,
            x: DenseArray,
            index: Index,
            values: DenseArray,
            *,
            copy: bool = True,
    ) -> DenseArray:
        """
        Set `values` at `index` in `x`.

        If copy=True:
            Return a new array y with y[index] = values and y != x.

        If copy=False:
            Mutate x in-place if the backend supports it.
            If the backend does NOT support mutation, raise NotImplementedError.
        """

    @abstractmethod
    def index_add(
            self,
            x: DenseArray,
            index: Index,
            values: DenseArray,
            *,
            copy: bool = True,
    ) -> DenseArray:
        ...

    @abstractmethod
    def ix_(self, *args: Any) -> Any:
        ...

    @abstractmethod
    def fori_loop(
            self,
            lower: int,
            upper: int,
            body_fun: Callable[[int, T], T],
            init_val: T,
    ) -> T:
        """
        Signature matches jax.lax.fori_loop(lower, upper, body_fun, init_val).
        """

    @abstractmethod
    def while_loop(
            self,
            cond_fun: Callable[[T], bool],
            body_fun: Callable[[T], T],
            init_val: T,
    ) -> T:
        """
        Signature matches jax.lax.while_loop(cond_fun, body_fun, init_val).
        """

    @abstractmethod
    def scan(
            self,
            f: Callable[[Carry, X], Tuple[Carry, Y]],
            init: Carry,
            xs: X,
            length: Optional[int] = None,
            reverse: bool = False,
            unroll: int = 1,
    ) -> Tuple[Carry, Y]:
        """
        Signature matches jax.lax.scan(f, init, xs, length=None, reverse=False, unroll=1).
        Note: in JAX, `xs` is a pytree with a leading axis; `ys` matches `xs` structure
        (per-step outputs stacked along the leading axis).
        """

    @abstractmethod
    def cond(
            self,
            pred: bool,
            true_fun: Callable[[T], R],
            false_fun: Callable[[T], R],
            *operands: Any,
    ) -> R:
        ...
