from dataclasses import dataclass
from typing import Any

from ._ops import BackendOps
from ..types import DenseArray, SparseArray, DType

@dataclass
class BackendContext:
    ops: BackendOps
    allow_sparse: bool = True
    dtype: DType | None = None
    enable_checks: bool = True

    def __post_init__(self):
        self.dtype = self.ops.sanitize_dtype(self.dtype)

    def assert_dense(self, x: Any) -> DenseArray:
        if self.enable_checks:
            if not self.ops.is_dense(x):
                raise TypeError(f"Expected dense array for {self.ops.family}, got {type(x).__name__}")
            return x
        else:
            return x

    def assert_sparse(self, x: Any) -> SparseArray:
        if self.enable_checks:
            if not self.allow_sparse:
                raise TypeError("Sparse objects are disallowed by this context.")
            if not self.ops.is_sparse(x):
                raise TypeError(f"Expected sparse array for {self.ops.family}, got {type(x).__name__}")
            return x
        else:
            return x

    def asarray(self, x: Any) -> DenseArray:
        return self.ops.asarray(x, dtype=self.dtype)
