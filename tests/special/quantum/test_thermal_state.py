"""Tests for dense thermal-state construction."""

from __future__ import annotations

import numpy as np
import pytest

from sdplab.special.quantum import build_linear_hamiltonian, build_thermal_state

IDENTITY = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def test_thermal_state_is_valid_density_matrix():
    H = 0.7 * Z + 0.2 * X

    rho = build_thermal_state(H, beta=1.3)

    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
    assert np.trace(rho) == pytest.approx(1.0)
    assert np.linalg.eigvalsh(rho).min() >= -1e-12


def test_zero_hamiltonian_gives_maximally_mixed_state():
    H = np.zeros((4, 4), dtype=complex)

    rho = build_thermal_state(H)

    np.testing.assert_allclose(rho, np.eye(4) / 4)


def test_one_qubit_z_expectation_is_minus_tanh_beta():
    beta = 0.8
    rho = build_thermal_state(Z, beta=beta)

    expectation = np.trace(Z @ rho).real

    assert expectation == pytest.approx(-np.tanh(beta))


def test_build_linear_hamiltonian():
    H = build_linear_hamiltonian(IDENTITY, [X, Z], np.array([0.2, -0.5]))

    np.testing.assert_allclose(H, IDENTITY + 0.2 * X - 0.5 * Z)
