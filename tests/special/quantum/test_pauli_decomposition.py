"""Tests for dense Pauli decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from sdplab.special.quantum.pauli import decompose_pauli_dense, pauli_string_dense

IDENTITY = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


@pytest.mark.parametrize(
    ("label", "matrix"),
    [("I", IDENTITY), ("X", X), ("Y", Y), ("Z", Z)],
)
def test_one_qubit_pauli_matrices_decompose_correctly(label, matrix):
    decomposition = decompose_pauli_dense(matrix)

    assert decomposition.labels == (label,)
    np.testing.assert_allclose(decomposition.coeffs, [1.0])
    np.testing.assert_allclose(decomposition.reconstruct_dense(), matrix)


@pytest.mark.parametrize("label", ["XI", "YZ", "ZX", "II"])
def test_two_qubit_tensor_products_decompose_correctly(label):
    matrix = pauli_string_dense(label)
    decomposition = decompose_pauli_dense(matrix)

    assert decomposition.labels == (label,)
    np.testing.assert_allclose(decomposition.coeffs, [1.0])
    np.testing.assert_allclose(decomposition.reconstruct_dense(), matrix)


@pytest.mark.parametrize("n_qubits", [1, 2, 3])
def test_random_hermitian_reconstructs(n_qubits):
    rng = np.random.default_rng(1234 + n_qubits)
    dim = 2**n_qubits
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    Q = 0.5 * (A + A.conj().T)

    decomposition = decompose_pauli_dense(Q, coeff_atol=0.0, drop_zero=False)

    np.testing.assert_allclose(decomposition.reconstruct_dense(), Q, atol=1e-10)


def test_signed_coefficients_are_handled_correctly():
    Q = 0.5 * X - 1.25 * Z

    decomposition = decompose_pauli_dense(Q)

    assert set(decomposition.labels) == {"X", "Z"}
    coeffs = dict(zip(decomposition.labels, decomposition.coeffs, strict=True))
    assert coeffs["X"] == pytest.approx(0.5)
    assert coeffs["Z"] == pytest.approx(-1.25)
    np.testing.assert_allclose(decomposition.reconstruct_dense(), Q)


def test_zero_coefficients_are_dropped():
    decomposition = decompose_pauli_dense(Z, coeff_atol=1e-12, drop_zero=True)

    assert decomposition.labels == ("Z",)


def test_non_hermitian_input_raises():
    Q = np.array([[0, 1], [0, 0]], dtype=complex)

    with pytest.raises(ValueError, match="Hermitian"):
        decompose_pauli_dense(Q)


def test_non_power_of_two_dimension_raises():
    Q = np.eye(3, dtype=complex)

    with pytest.raises(ValueError, match="power of two"):
        decompose_pauli_dense(Q)
