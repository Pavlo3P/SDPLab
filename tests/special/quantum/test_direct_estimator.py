"""Tests for the direct Pauli-sampling estimator."""

from __future__ import annotations

from math import ceil, log

import numpy as np
import pytest

from sdplab.special.quantum import (
    DirectDensityMatrixMeasurementBackend,
    PauliSamplingEstimator,
    pauli_sample_size,
)

IDENTITY = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def ket_density(vector):
    vector = np.asarray(vector, dtype=complex)
    return np.outer(vector, vector.conj())


def estimator():
    return PauliSamplingEstimator(DirectDensityMatrixMeasurementBackend())


def assert_stochastic_close(result):
    assert abs(result.estimate - result.exact_value) < 5 * result.standard_error + 0.05


def test_identity_observable_gives_one():
    rho0 = ket_density([1, 0])

    result = estimator().estimate(IDENTITY, rho0, n_sample=10, seed=1)

    assert result.estimate == pytest.approx(1.0)
    assert result.exact_value == pytest.approx(1.0)


def test_zero_observable_gives_zero():
    rho0 = ket_density([1, 0])

    result = estimator().estimate(np.zeros((2, 2), dtype=complex), rho0, n_sample=10)

    assert result.estimate == pytest.approx(0.0)
    assert result.exact_value == pytest.approx(0.0)
    assert result.standard_error == pytest.approx(0.0)


def test_z_on_zero_state_gives_one():
    rho0 = ket_density([1, 0])

    result = estimator().estimate(Z, rho0, n_sample=50, seed=2)

    assert result.estimate == pytest.approx(1.0)
    assert result.exact_value == pytest.approx(1.0)


def test_x_on_zero_state_is_near_zero():
    rho0 = ket_density([1, 0])

    result = estimator().estimate(X, rho0, n_sample=3000, seed=3)

    assert_stochastic_close(result)


def test_signed_observable_minus_z_gives_minus_one():
    rho0 = ket_density([1, 0])

    result = estimator().estimate(-Z, rho0, n_sample=50, seed=4)

    assert result.estimate == pytest.approx(-1.0)
    assert result.exact_value == pytest.approx(-1.0)


def test_two_qubit_zz_on_zero_zero_state_gives_one():
    rho00 = ket_density([1, 0, 0, 0])
    ZZ = np.kron(Z, Z)

    result = estimator().estimate(ZZ, rho00, n_sample=50, seed=5)

    assert result.estimate == pytest.approx(1.0)
    assert result.exact_value == pytest.approx(1.0)


def test_sample_size_helper_uses_expected_formula():
    l1_norm = 2.5
    epsilon = 0.1
    delta = 0.05

    assert pauli_sample_size(l1_norm, epsilon, delta) == ceil(
        2 * l1_norm**2 * log(2 / delta) / epsilon**2
    )


def test_sample_size_zero_l1_norm_returns_zero():
    assert pauli_sample_size(0.0, 0.1, 0.05) == 0
