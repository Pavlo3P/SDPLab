"""Dense Pauli-basis decomposition utilities for small quantum systems."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np

_PAULI_MATRICES: dict[str, np.ndarray] = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _validate_label(label: str, n_qubits: int | None = None) -> str:
    """Validate and normalize a Pauli label."""

    if not isinstance(label, str):
        raise TypeError(f"Pauli label must be a string, got {type(label)!r}")
    label = label.upper()
    if n_qubits is not None and len(label) != n_qubits:
        raise ValueError(
            f"Expected Pauli label of length {n_qubits}, got {label!r}"
        )
    bad = sorted(set(label) - set(_PAULI_MATRICES))
    if bad:
        raise ValueError(
            f"Invalid Pauli characters {bad!r}; expected only I, X, Y, Z."
        )
    return label


def _n_qubits_from_dimension(dim: int) -> int:
    """Infer the number of qubits from a power-of-two dimension."""

    if dim <= 0 or dim & (dim - 1):
        raise ValueError(f"Matrix dimension must be a power of two, got {dim}")
    return int(dim.bit_length() - 1)


def _validate_square_matrix(
    matrix: np.ndarray,
    *,
    n_qubits: int | None = None,
    name: str = "matrix",
) -> tuple[np.ndarray, int]:
    """Validate a square matrix and return it with its qubit count."""

    arr = np.asarray(matrix, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    inferred = _n_qubits_from_dimension(arr.shape[0])
    if n_qubits is not None and inferred != n_qubits:
        raise ValueError(
            f"{name} has dimension {arr.shape[0]}, expected {2**n_qubits}"
        )
    return arr, inferred


def _validate_hermitian(
    matrix: np.ndarray,
    *,
    atol: float,
    name: str = "matrix",
) -> None:
    """Validate that a matrix is Hermitian up to tolerance."""

    if not np.allclose(matrix, matrix.conj().T, atol=atol):
        raise ValueError(f"{name} must be Hermitian")


def pauli_string_dense(label: str) -> np.ndarray:
    r"""Build the dense matrix for one Pauli string.

    The label convention is mathematical left-to-right tensor order. For
    example, ``"ZI"`` means :math:`Z \otimes I`.

    Parameters
    ----------
    label : str
        Pauli label containing only ``"I"``, ``"X"``, ``"Y"``, and ``"Z"``.

    Returns
    -------
    numpy.ndarray
        Dense matrix :math:`P_0 \otimes \cdots \otimes P_{K-1}`.

    Raises
    ------
    TypeError
        If ``label`` is not a string.
    ValueError
        If ``label`` contains non-Pauli characters.

    Examples
    --------
    >>> from sdplab.special.quantum.pauli import pauli_string_dense
    >>> pauli_string_dense("Z").diagonal().tolist()
    [(1+0j), (-1+0j)]
    """

    label = _validate_label(label)
    out = np.array([[1]], dtype=complex)
    for char in label:
        out = np.kron(out, _PAULI_MATRICES[char])
    return out


@dataclass(frozen=True)
class PauliDecomposition:
    r"""Store a sparse Pauli expansion ``Q = sum_l c_l P_l``.

    A label such as ``"IXYZ"`` denotes
    :math:`I \otimes X \otimes Y \otimes Z`, where the first character acts on
    the leftmost tensor factor in the dense Kronecker-product basis.

    The sampling distribution used by :class:`PauliSamplingEstimator` is
    :math:`p_l = |c_l| / \|c\|_1`. For the zero observable,
    :math:`\|c\|_1=0`; this object returns all-zero probabilities and the
    estimator returns an exact zero estimate without sampling.

    Parameters
    ----------
    n_qubits : int
        Number of qubits :math:`K`.
    labels : tuple[str, ...]
        Pauli labels with one character per qubit.
    coeffs : array-like
        Real coefficients :math:`c_l` for the corresponding labels.
    atol : float, optional
        Tolerance for treating coefficients as real. Default is ``1e-12``.

    Attributes
    ----------
    n_qubits : int
        Number of qubits :math:`K`.
    labels : tuple[str, ...]
        Pauli labels in mathematical tensor order.
    coeffs : numpy.ndarray
        Real Pauli coefficients.
    atol : float
        Numerical tolerance used for coefficient validation.

    Examples
    --------
    >>> import numpy as np
    >>> from sdplab.special.quantum.pauli import PauliDecomposition
    >>> decomp = PauliDecomposition(1, ("Z",), np.array([2.0]))
    >>> decomp.l1_norm
    2.0
    >>> decomp.probabilities.tolist()
    [1.0]
    """

    n_qubits: int
    labels: tuple[str, ...]
    coeffs: np.ndarray
    atol: float = 1e-12

    def __post_init__(self) -> None:
        """Validate labels and coerce coefficients to real values."""

        if self.n_qubits < 0:
            raise ValueError("n_qubits must be nonnegative")
        labels = tuple(_validate_label(label, self.n_qubits) for label in self.labels)
        coeffs = np.asarray(self.coeffs)
        if coeffs.ndim != 1:
            raise ValueError("coeffs must be a one-dimensional array")
        if len(labels) != len(coeffs):
            raise ValueError("labels and coeffs must have the same length")
        if np.max(np.abs(coeffs.imag), initial=0.0) > self.atol:
            raise ValueError("coeffs must be real up to atol")
        coeffs = coeffs.real.astype(float)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "coeffs", np.asarray(coeffs, dtype=float))

    @property
    def l1_norm(self) -> float:
        r"""Return :math:`\|c\|_1 = \sum_l |c_l|`."""

        return float(np.sum(np.abs(self.coeffs)))

    @property
    def probabilities(self) -> np.ndarray:
        r"""Return :math:`p_l = |c_l| / \|c\|_1` for Pauli sampling."""

        l1_norm = self.l1_norm
        if l1_norm == 0.0:
            return np.zeros_like(self.coeffs, dtype=float)
        return np.abs(self.coeffs) / l1_norm

    @property
    def signs(self) -> np.ndarray:
        """Return ``sign(c_l)`` for the signed estimator samples."""

        return np.sign(self.coeffs)

    def reconstruct_dense(self) -> np.ndarray:
        r"""Reconstruct the dense observable :math:`Q`.

        Returns
        -------
        numpy.ndarray
            Dense matrix :math:`Q=\sum_l c_l P_l`.

        Examples
        --------
        >>> import numpy as np
        >>> from sdplab.special.quantum.pauli import PauliDecomposition
        >>> decomp = PauliDecomposition(1, ("I",), np.array([1.0]))
        >>> np.allclose(decomp.reconstruct_dense(), np.eye(2))
        True
        """

        dim = 2**self.n_qubits
        out = np.zeros((dim, dim), dtype=complex)
        for label, coeff in zip(self.labels, self.coeffs, strict=True):
            out += coeff * pauli_string_dense(label)
        return out

    def to_qiskit_sparse_pauli_op(self):
        """Convert to ``qiskit.quantum_info.SparsePauliOp``.

        Qiskit's string convention indexes qubit 0 from the right side of the
        computational basis. Labels are therefore reversed so that
        ``SparsePauliOp(...).to_matrix()`` matches :meth:`reconstruct_dense`.

        Returns
        -------
        qiskit.quantum_info.SparsePauliOp
            Qiskit sparse Pauli representation with matching dense matrix.

        Raises
        ------
        ImportError
            If Qiskit is unavailable.

        Examples
        --------
        >>> import numpy as np
        >>> from sdplab.special.quantum.pauli import PauliDecomposition
        >>> decomp = PauliDecomposition(1, ("Z",), np.array([1.0]))
        >>> op = decomp.to_qiskit_sparse_pauli_op()
        >>> op.num_qubits
        1
        """

        try:
            from qiskit.quantum_info import SparsePauliOp
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Qiskit is required for PauliDecomposition.to_qiskit_sparse_pauli_op(). "
                "Install sdplab[quantum]."
            ) from exc

        return SparsePauliOp.from_list(
            [(label[::-1], coeff) for label, coeff in zip(self.labels, self.coeffs)]
        )


def _all_pauli_labels(n_qubits: int) -> Iterable[str]:
    """Generate all Pauli labels on ``n_qubits`` qubits."""

    for factors in product("IXYZ", repeat=n_qubits):
        yield "".join(factors)


def decompose_pauli_dense(
    Q: np.ndarray,
    *,
    n_qubits: int | None = None,
    coeff_atol: float = 1e-12,
    hermitian_atol: float = 1e-10,
    drop_zero: bool = True,
) -> PauliDecomposition:
    r"""Decompose a dense Hermitian observable into Pauli strings.

    The returned decomposition satisfies
    :math:`Q \in \mathbb{H}_{2^K}` and
    :math:`Q=\sum_l c_lP_l` with
    :math:`c_l = 2^{-K}\operatorname{Tr}(QP_l)`.

    This enumerates ``4**K`` Pauli strings and materializes dense matrices, so
    it is intended for small systems, debugging, and validation. For larger
    systems, pass a sparse Pauli representation directly.

    Parameters
    ----------
    Q : array-like
        Dense Hermitian observable.
    n_qubits : int or None, optional
        Expected number of qubits. If omitted, inferred from the matrix
        dimension.
    coeff_atol : float, optional
        Tolerance for dropping zero coefficients and validating real
        coefficients. Default is ``1e-12``.
    hermitian_atol : float, optional
        Tolerance for Hermiticity validation. Default is ``1e-10``.
    drop_zero : bool, optional
        Whether to omit coefficients with absolute value below ``coeff_atol``.
        Default is ``True``.

    Returns
    -------
    PauliDecomposition
        Sparse Pauli expansion of ``Q``.

    Raises
    ------
    ValueError
        If ``Q`` is not square, has non-power-of-two dimension, is not
        Hermitian, or yields a non-real coefficient above tolerance.

    Examples
    --------
    >>> import numpy as np
    >>> from sdplab.special.quantum import decompose_pauli_dense
    >>> Z = np.array([[1, 0], [0, -1]], dtype=complex)
    >>> decomp = decompose_pauli_dense(2 * Z)
    >>> decomp.labels
    ('Z',)
    >>> decomp.coeffs.tolist()
    [2.0]
    """

    Q_arr, inferred_n_qubits = _validate_square_matrix(Q, n_qubits=n_qubits, name="Q")
    _validate_hermitian(Q_arr, atol=hermitian_atol, name="Q")

    dim = Q_arr.shape[0]
    labels: list[str] = []
    coeffs: list[float] = []
    for label in _all_pauli_labels(inferred_n_qubits):
        P = pauli_string_dense(label)
        coeff_real = _real_pauli_coefficient(np.trace(Q_arr @ P) / dim, label, coeff_atol)
        if drop_zero and abs(coeff_real) < coeff_atol:
            continue
        labels.append(label)
        coeffs.append(coeff_real)

    return PauliDecomposition(
        inferred_n_qubits,
        tuple(labels),
        np.asarray(coeffs, dtype=float),
        atol=coeff_atol,
    )


def from_qiskit_sparse_pauli_op(op, *, coeff_atol: float = 1e-12) -> PauliDecomposition:
    """Convert a Qiskit ``SparsePauliOp`` into local label convention.

    Parameters
    ----------
    op : qiskit.quantum_info.SparsePauliOp
        Sparse Pauli operator using Qiskit's label ordering.
    coeff_atol : float, optional
        Tolerance for dropping zero coefficients and validating real
        coefficients. Default is ``1e-12``.

    Returns
    -------
    PauliDecomposition
        Pauli decomposition using mathematical tensor-order labels.

    Raises
    ------
    ValueError
        If any coefficient has a non-negligible imaginary part.

    Examples
    --------
    >>> from qiskit.quantum_info import SparsePauliOp
    >>> from sdplab.special.quantum.pauli import from_qiskit_sparse_pauli_op
    >>> decomp = from_qiskit_sparse_pauli_op(SparsePauliOp.from_list([("Z", 1.0)]))
    >>> decomp.labels
    ('Z',)
    """

    labels: list[str] = []
    coeffs: list[float] = []
    for qiskit_label, coeff in zip(op.paulis.to_labels(), op.coeffs, strict=True):
        label = qiskit_label[::-1]
        coeff_real = _real_pauli_coefficient(complex(coeff), label, coeff_atol)
        if abs(coeff_real) < coeff_atol:
            continue
        labels.append(label)
        coeffs.append(coeff_real)

    return PauliDecomposition(
        int(op.num_qubits),
        tuple(labels),
        np.asarray(coeffs, dtype=float),
        atol=coeff_atol,
    )


def _real_pauli_coefficient(coeff: complex, label: str, atol: float) -> float:
    """Validate and return the real part of a Pauli coefficient."""

    coeff = complex(coeff)
    if abs(coeff.imag) > atol:
        raise ValueError(
            f"Pauli coefficient for {label!r} has non-negligible imaginary "
            f"part {coeff.imag}"
        )
    return float(coeff.real)


__all__ = [
    "PauliDecomposition",
    "decompose_pauli_dense",
    "from_qiskit_sparse_pauli_op",
    "pauli_string_dense",
]
