"""Tests for the optional Qiskit Aer measurement backend."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("qiskit") is None
    or importlib.util.find_spec("qiskit_aer") is None,
    reason="qiskit and qiskit-aer are not installed",
)

from sdplab.special.quantum import (  # noqa: E402
    DirectDensityMatrixMeasurementBackend,
    PauliSamplingEstimator,
    QiskitDensityMatrixMeasurementBackend,
)

IDENTITY = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def ket_density(vector):
    vector = np.asarray(vector, dtype=complex)
    return np.outer(vector, vector.conj())


def qiskit_estimator():
    return PauliSamplingEstimator(QiskitDensityMatrixMeasurementBackend())


def direct_estimator():
    return PauliSamplingEstimator(DirectDensityMatrixMeasurementBackend())


def test_qiskit_agrees_with_direct_for_one_qubit_z():
    rho0 = ket_density([1, 0])

    qiskit = qiskit_estimator().estimate(Z, rho0, n_sample=50, seed=10)
    direct = direct_estimator().estimate(Z, rho0, n_sample=50, seed=10)

    assert qiskit.estimate == pytest.approx(direct.estimate)


def test_qiskit_agrees_with_direct_for_one_qubit_x_after_basis_rotation():
    rho_plus = ket_density([1 / np.sqrt(2), 1 / np.sqrt(2)])

    qiskit = qiskit_estimator().estimate(X, rho_plus, n_sample=50, seed=11)
    direct = direct_estimator().estimate(X, rho_plus, n_sample=50, seed=11)

    assert qiskit.estimate == pytest.approx(direct.estimate)


def test_qiskit_agrees_with_direct_for_one_qubit_y_after_basis_rotation():
    rho_y_plus = ket_density([1 / np.sqrt(2), 1j / np.sqrt(2)])

    qiskit = qiskit_estimator().estimate(Y, rho_y_plus, n_sample=50, seed=12)
    direct = direct_estimator().estimate(Y, rho_y_plus, n_sample=50, seed=12)

    assert qiskit.estimate == pytest.approx(direct.estimate)


def test_qiskit_agrees_for_two_qubit_zz():
    rho00 = ket_density([1, 0, 0, 0])
    ZZ = np.kron(Z, Z)

    result = qiskit_estimator().estimate(ZZ, rho00, n_sample=50, seed=13)

    assert result.estimate == pytest.approx(1.0)


def test_identity_terms_do_not_break_measurement():
    rho0 = ket_density([1, 0])
    Q = IDENTITY + Z

    result = qiskit_estimator().estimate(Q, rho0, n_sample=50, seed=14)

    assert result.estimate == pytest.approx(2.0)
    assert result.exact_value == pytest.approx(2.0)


def test_bit_ordering_for_asymmetric_two_qubit_strings():
    backend = QiskitDensityMatrixMeasurementBackend()
    rho10 = ket_density([0, 0, 1, 0])

    samples = backend.measure_pauli_expectation_samples(
        rho10,
        ["ZI", "IZ"],
        {"ZI": 20, "IZ": 20},
        seed=15,
    )

    np.testing.assert_array_equal(samples["ZI"], -np.ones(20, dtype=int))
    np.testing.assert_array_equal(samples["IZ"], np.ones(20, dtype=int))
