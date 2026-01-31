from __future__ import annotations

from typing import Any, Sequence, Literal, Tuple, Callable, Optional
import inspect

from .._family import BackendFamily
from .._ops import BackendOps
from ...types import DenseArray, SparseArray, DType, Index, X, T, Y, R, Carry


class JaxOps(BackendOps):
    """
    BackendOps implementation for the JAX ecosystem.

    Dense arrays:
      - jax.Array

    Sparse arrays:
      - jax.experimental.sparse.BCOO / BCSR

    Each method mirrors the corresponding JAX public API signature and delegates
    to `jax.numpy` / `jax.numpy.linalg` / `jax.experimental.sparse`.

    Notes:
      - Some parameters are documented as "unused by JAX" (e.g. `out` for argmin/argmax);
        these are still accepted to match the JAX signature and keep call sites uniform.
      - Array-creation routines often accept `device` and/or `out_sharding` for explicit
        placement/sharding.
    """

    def __init__(self) -> None:
        import jax
        import jax.numpy as jnp
        from jax.experimental import sparse as jsparse

        self._jax = jax
        self._jnp = jnp
        self._jsparse = jsparse

        self._Array = getattr(jax, "Array", ())
        self._BCOO = getattr(jsparse, "BCOO", ())
        self._BCSR = getattr(jsparse, "BCSR", ())

        self._reshape_supports_copy = "copy" in inspect.signature(jnp.reshape).parameters
        self._reshape_supports_out_sharding = "out_sharding" in inspect.signature(jnp.reshape).parameters
        self._ravel_supports_out_sharding = "out_sharding" in inspect.signature(jnp.ravel).parameters
        self._zeros_supports_out_sharding = "out_sharding" in inspect.signature(jnp.zeros).parameters
        self._empty_supports_out_sharding = "out_sharding" in inspect.signature(jnp.empty).parameters

        self.family = BackendFamily.JAX

    def sanitize_dtype(self, dtype: DType | None) -> DType | None:
        """
        Normalize and validate dtype for JAX.

        Policy:
          - `dtype=None` is allowed and returned as-is.
          - Otherwise, dtype must be a valid JAX dtype specifier.
          - dtype must be supported by the active backend/device.
          - dtype must NOT be implicitly canonicalized under current JAX config
            (e.g., float64 -> float32 when jax_enable_x64=False).
        """
        if dtype is None:
            return None

        jax = self._jax
        jnp = self._jnp

        try:
            dt = jnp.dtype(dtype)
        except Exception as e:
            raise TypeError(f"Invalid dtype specifier for JAX: {dtype!r}.") from e

        # Ensure dtype is actually usable on this backend/device
        try:
            jnp.empty((), dtype=dt)
        except Exception as e:
            raise TypeError(
                f"Dtype {dt!r} is not supported by the active JAX backend/device."
            ) from e

        # Forbid implicit coercion under current JAX configuration
        dt_canon = jax.dtypes.canonicalize_dtype(dt)
        if dt_canon != dt:
            x64_enabled = bool(jax.config.read("jax_enable_x64"))
            raise TypeError(
                f"Dtype {dt} is not permitted under current JAX configuration: "
                f"it would be canonicalized to {dt_canon}. "
                f"(jax_enable_x64={x64_enabled!r})"
            )

        return dt

    def is_dense(self, x: Any) -> bool:
        """Return True iff `x` is a JAX dense array (jax.Array)."""
        return bool(self._Array) and isinstance(x, self._Array)

    def is_sparse(self, x: Any, /, *args, **kwargs) -> bool:
        """Return True iff `x` is a JAX sparse array (BCOO or BCSR)."""
        return (bool(self._BCOO) and isinstance(x, self._BCOO)) or (bool(self._BCSR) and isinstance(x, self._BCSR))

    @property
    def inf(self):
        return self._jnp.array(self._jnp.inf)

    @property
    def nan(self):
        return self._jnp.array(self._jnp.nan)

    @property
    def pi(self):
        return self._jnp.array(self._jnp.pi)

    @property
    def e(self):
        return self._jnp.array(self._jnp.e)

    @property
    def eps(self):
        return self._jnp.array(self._jnp.finfo(self._jnp.float64).eps)

    def asarray(
        self,
        a: Any,
        dtype: DType | None = None,
        order: Literal["C", "F", "A", "K"] | None = None,
        *,
        copy: bool | None = None,
        device: Any | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.asarray`."""
        return self._jnp.asarray(a, dtype=dtype, order=order, copy=copy, device=device)

    def empty(
        self,
        shape: int | Tuple[int, ...],
        dtype: DType | None = None,
        *,
        device: Any | None = None,
        out_sharding: Any | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.empty`."""
        if self._empty_supports_out_sharding:
            return self._jnp.empty(shape, dtype=dtype, device=device, out_sharding=out_sharding)
        return self._jnp.empty(shape, dtype=dtype, device=device)

    def zeros(
            self,
            shape: int | Tuple[int, ...],
            dtype: DType | None = None,
            *,
            device: Any | None = None,
            out_sharding: Any | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.zeros`."""
        return self._jnp.zeros(shape, dtype=dtype, device=device, out_sharding=out_sharding)

    def ones(
        self,
        shape: int | Tuple[int, ...],
        dtype: DType | None = None,
        *,
        device: Any | None = None,
        out_sharding: Any | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.ones`."""
        return self._jnp.ones(shape, dtype=dtype, device=device, out_sharding=out_sharding)

    def arange(self,
               start: int,
               stop: int | None = None,
               step: int | None = None,
               dtype: DType | None = None,
               *,
               device: Any | None = None,
               out_sharding: Any | None = None,
               ) -> DenseArray:
        """See: `jax.numpy.arange`."""
        return self._jnp.arange(start, stop, step, dtype=dtype, device=device, out_sharding=out_sharding)

    def full(
        self,
        shape: int | Tuple[int, ...],
        fill_value: Any,
        dtype: DType | None = None,
        *,
        device: Any | None = None,
    ) -> DenseArray:
        """
        See: `jax.numpy.full`.

        Note:
          - JAX exposes `device` for placement; `out_sharding` is not part of the public signature.
        """
        return self._jnp.full(shape, fill_value, dtype=dtype, device=device)

    def eye(
        self,
        N: int,
        M: int | None = None,
        k: int = 0,
        dtype: DType | None = None,
        *,
        device: Any | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.eye`."""
        return self._jnp.eye(N=N, M=M, k=k, dtype=dtype, device=device)

    def ravel(
        self,
        a: DenseArray,
        order: Literal["C", "F", "A", "K"] = "C",
        *,
        out_sharding: Any | None = None,
    ) -> DenseArray:
        """
        See: `jax.numpy.ravel`.
        """
        if self._ravel_supports_out_sharding:
            return self._jnp.ravel(a, order=order, out_sharding=out_sharding)
        return self._jnp.ravel(a, order=order)

    def reshape(
        self,
        a: DenseArray,
        shape: int | Tuple[int, ...],
        order: Literal["C", "F", "A"] = "C",
        *,
        copy: bool | None = None,
        out_sharding: Any | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.reshape`."""
        kwargs: dict[str, Any] = {"order": order}
        if self._reshape_supports_copy:
            kwargs["copy"] = copy
        if self._reshape_supports_out_sharding:
            kwargs["out_sharding"] = out_sharding
        return self._jnp.reshape(a, shape, **kwargs)

    def transpose(
        self,
        x: DenseArray,
        axes: Sequence[int] | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.transpose`."""
        return self._jnp.transpose(x, axes=axes)

    def stack(
        self,
        arrays: Sequence[DenseArray],
        axis: int = 0,
        out: Any | None = None,
        dtype: DType | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.stack`."""
        return self._jnp.stack(arrays, axis=axis, out=out, dtype=dtype)

    def conj(self, x: DenseArray) -> DenseArray:
        """See: `jax.numpy.conj`."""
        return self._jnp.conj(x)

    def real(self, x: DenseArray) -> DenseArray:
        """See: `jax.numpy.real`."""
        return self._jnp.real(x)

    def imag(self, x: DenseArray) -> DenseArray:
        """See: `jax.numpy.imag`."""
        return self._jnp.imag(x)

    def abs(self, x: DenseArray) -> DenseArray:
        """See: `jax.numpy.abs`."""
        return self._jnp.abs(x)

    def sign(self, x: DenseArray) -> DenseArray:
        """See: `jax.numpy.sign`."""
        return self._jnp.sign(x)

    def sqrt(self, x: DenseArray) -> DenseArray:
        """See: `jax.numpy.sqrt`."""
        return self._jnp.sqrt(x)

    def sum(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        dtype: DType | None = None,
        out: Any | None = None,
        keepdims: bool = False,
        initial: DenseArray | None = None,
        where: DenseArray | None = None,
        promote_integers: bool = True,
    ) -> DenseArray:
        """See: `jax.numpy.sum`."""
        return self._jnp.sum(
            a,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
            promote_integers=promote_integers,
        )

    def prod(
        self,
        a: DenseArray,
        axis: int | Sequence[int] | None = None,
        dtype: DType | None = None,
        out: Any | None = None,
        keepdims: bool = False,
        initial: DenseArray | None = None,
        where: DenseArray | None = None,
        promote_integers: bool = True,
    ) -> DenseArray:
        """See: `jax.numpy.prod`."""
        return self._jnp.prod(
            a,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
            promote_integers=promote_integers,
        )

    def trace(
        self,
        a: DenseArray,
        offset: int | Any = 0,
        axis1: int = 0,
        axis2: int = 1,
        dtype: DType | None = None,
        out: DenseArray | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.trace`."""
        return self._jnp.trace(a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out)

    def argsort(
        self,
        a: DenseArray,
        axis: int | None = -1,
        kind: None = None,
        order: None = None,
        stable: bool = True,
        descending: bool = False
    ) -> DenseArray:
        """
        See: `jax.numpy.argsort`.

        Notes:
          - `kind` is deprecated in JAX; kept for signature compatibility.
          - `order` is not supported by JAX; kept for signature compatibility.
        """
        return self._jnp.argsort(a, axis=axis, kind=kind, order=order, stable=stable, descending=descending)

    def sort(
        self,
        a: DenseArray,
        axis: int | None = -1,
        kind: None = None,
        order: None = None,
        stable: bool = True,
        descending: bool = False
    ) -> DenseArray:
        """
        See: `jax.numpy.sort`.

        Notes:
          - `kind` is deprecated in JAX; kept for signature compatibility.
          - `order` is not supported by JAX; kept for signature compatibility.
        """
        return self._jnp.sort(a, axis=axis, kind=kind, order=order, stable=stable, descending=descending)

    def argmin(
        self,
        a: DenseArray,
        axis: int | None = None,
        out: Any | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """See: `jax.numpy.argmin` (note: `out` is unused by JAX)."""
        return self._jnp.argmin(a, axis=axis, out=out, keepdims=keepdims)

    def argmax(
        self,
        a: DenseArray,
        axis: int | None = None,
        out: Any | None = None,
        keepdims: bool = False,
    ) -> DenseArray:
        """See: `jax.numpy.argmax` (note: `out` is unused by JAX)."""
        return self._jnp.argmax(a, axis=axis, out=out, keepdims=keepdims)

    def vdot(
        self,
        a: DenseArray,
        b: DenseArray,
        *,
        precision: Any | None = None,
        preferred_element_type: DType | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.vdot`."""
        return self._jnp.vdot(a, b, precision=precision, preferred_element_type=preferred_element_type)

    def matmul(
        self,
        a: DenseArray,
        b: DenseArray,
        *,
        precision: Any | None = None,
        preferred_element_type: DType | None = None,
        out_sharding: Any | None = None
    ) -> DenseArray:
        """See: `jax.numpy.matmul`."""
        return self._jnp.matmul(
            a,
            b,
            precision=precision,
            preferred_element_type=preferred_element_type,
            out_sharding=out_sharding,
        )

    def sparse_matmul(self, a: SparseArray, b: DenseArray) -> DenseArray:
        """Sparse @ dense. See: `jax.experimental.sparse` (BCOO/BCSR matmul via `@`)."""
        return a @ b

    def kron(self, a: DenseArray, b: DenseArray) -> DenseArray:
        """See: `jax.numpy.kron`."""
        return self._jnp.kron(a, b)

    def einsum(
        self,
        subscripts: str,
        /,
        *operands: DenseArray,
        out: Any | None = None,
        optimize: str | bool | list[Tuple[int, ...]] = "auto",
        precision: Any | None = None,
        preferred_element_type: DType | None = None,
        _dot_general: Any | None = None,
        out_sharding: Any | None = None,
    ) -> DenseArray:
        """See: `jax.numpy.einsum`."""
        return self._jnp.einsum(
            subscripts,
            *operands,
            out=out,
            optimize=optimize,
            precision=precision,
            preferred_element_type=preferred_element_type,
            # _dot_general=_dot_general,
            out_sharding=out_sharding,
        )

    def eigh(
        self,
        x: DenseArray,
        UPLO: Literal["L", "U"] = "L",
        symmetrize_input: bool = True
    ) -> Tuple[DenseArray, DenseArray]:
        """See: `jax.numpy.linalg.eigh`."""
        if self.is_sparse(x):
            raise TypeError("eigh requires a dense array; sparse input is not supported.")
        return self._jnp.linalg.eigh(x, UPLO=UPLO, symmetrize_input=symmetrize_input)

    def logsumexp(self, a: DenseArray, axis: int | Sequence[int] | None = None, b: DenseArray = None, keepdims: bool = False,
                  return_sign: bool = False, where: DenseArray | None = None) -> DenseArray | Tuple[DenseArray, DenseArray]:
        """ See: jax.scipy.special.logsumexp. """
        return self._jax.scipy.special.logsumexp(a, axis=axis, b=b, keepdims=keepdims, return_sign=return_sign, where=where)

    def exp(self, x: DenseArray) -> DenseArray:
        """See: `jax.numpy.exp`."""
        return self._jnp.exp(x)

    def log(self, x: DenseArray) -> DenseArray:
        """See: `jax.numpy.log`."""
        return self._jnp.log(x)

    def maximum(self, x: DenseArray, y: DenseArray) -> DenseArray:
        """See: `jax.numpy.maximum`."""
        return self._jnp.maximum(x, y)

    def minimum(self, x: DenseArray, y: DenseArray) -> DenseArray:
        """See: `jax.numpy.minimum`."""
        return self._jnp.minimum(x, y)

    def where(self, condition: DenseArray | bool, x: DenseArray | None = None, y: DenseArray | None = None, *,
              size: int | None = None, fill_value: DenseArray | None = None) -> DenseArray:
        """See: `jax.numpy.where`."""
        return self._jnp.where(condition, x, y, size=size, fill_value=fill_value)

    def concatenate(self, arrays: Sequence[DenseArray], axis: int = 0, dtype: DType = None) -> DenseArray:
        """See: `jax.numpy.concatenate`."""
        return self._jnp.concatenate(arrays, axis=axis, dtype=dtype)

    def index_set(self, x: DenseArray, index: Index, values: DenseArray, *, copy: bool = True):
        if not copy:
            raise NotImplementedError(
                "JAX arrays are immutable; copy=False is not supported."
            )
        return x.at[index].set(values)

    def ix_(self, *args: Any) -> Any:
        """ See: jax.numpy.ix_. """
        return self._jnp.ix_(*args)

    def fori_loop(
        self,
        lower: int,
        upper: int,
        body_fun: Callable[[int, T], T],
        init_val: T,
        *,
        unroll: int | bool | None = None,
    ) -> T:
        """See: `jax.lax.fori_loop`."""
        return self._jax.lax.fori_loop(lower, upper, body_fun, init_val, unroll=unroll)

    def while_loop(
        self,
        cond_fun: Callable[[T], bool],
        body_fun: Callable[[T], T],
        init_val: T,
    ) -> T:
        """See: `jax.lax.while_loop`."""
        return self._jax.lax.while_loop(cond_fun, body_fun, init_val)

    def scan(
        self,
        f: Callable[[Carry, X], Tuple[Carry, Y]],
        init: Carry,
        xs: X,
        length: Optional[int] = None,
        reverse: bool = False,
        unroll: int = 1,
        _split_transpose: bool = False,
    ) -> Tuple[Carry, Y]:
        """See: `jax.lax.scan`."""
        return self._jax.lax.scan(f, init, xs, length=length, reverse=reverse, unroll=unroll, _split_transpose=_split_transpose)

    def cond(
            self,
            pred: bool,
            true_fun: Callable[[T], R],
            false_fun: Callable[[T], R],
            *operands: Any,
    ) -> R:
        return self._jax.lax.cond(pred, true_fun, false_fun, *operands)

    def index_add(self, x: DenseArray, index: Index, values: DenseArray, *, copy: bool = True):
        """
        Scatter-add (accumulating) update.

        Semantics:
          y = x
          y[index] += values   (with accumulation for repeated indices)

        Notes:
          - JAX arrays are immutable; copy=False is not supported (same as index_set).
          - Uses JAX's scatter-add: x.at[index].add(values)
        """
        if not copy:
            raise NotImplementedError(
                "JAX arrays are immutable; copy=False is not supported."
            )
        return x.at[index].add(values)
