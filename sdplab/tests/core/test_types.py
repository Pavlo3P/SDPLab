import numpy as np

from qotlib.core.types import ArrayLike, DenseArray


class DummyDense:
    def __init__(self):
        self._data = np.array([1.0])

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def real(self):
        return self

    @property
    def imag(self):
        return self

    def conj(self):
        return self

    @property
    def T(self):
        return self

    def reshape(self, shape):
        return self

    def astype(self, dtype):
        return self

    def __add__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __matmul__(self, other):
        return self

    def __rmatmul__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def __rtruediv__(self, other):
        return self

    def __getitem__(self, item):
        return self

    def max(self):
        return 0.0

    def min(self):
        return 0.0

    def sum(self):
        return 0.0

    def abs(self):
        return self


class DummyArrayLike:
    def __init__(self):
        self._data = np.array([1.0])

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def conj(self):
        return self

    @property
    def T(self):
        return self


def test_dense_array_protocol_runtime_check():
    """Verify DenseArray protocol runtime conformance using dummy type."""
    assert isinstance(DummyDense(), DenseArray)


def test_array_like_protocol_runtime_check():
    """Verify ArrayLike protocol runtime conformance using dummy type."""
    assert isinstance(DummyArrayLike(), ArrayLike)
