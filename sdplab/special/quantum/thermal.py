"""Thermal density-matrix construction for dense quantum systems."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .pauli import _validate_hermitian, _validate_square_matrix


def build_linear_hamiltonian(
    H0: np.ndarray,
    observables: Sequence[np.ndarray],
    alpha: np.ndarray,
    *,
    hermitian_atol: float = 1e-10,
) -> np.ndarray:
    r"""Build ``H(alpha) = H0 + sum_j alpha_j Q_j``."""

    H, n_qubits = _validate_square_matrix(H0, name="H0")
    _validate_hermitian(H, atol=hermitian_atol, name="H0")

    alpha_arr = np.asarray(alpha, dtype=float)
    if alpha_arr.ndim != 1:
        raise ValueError("alpha must be one-dimensional")
    if len(observables) != len(alpha_arr):
        raise ValueError("observables and alpha must have the same length")

    out = H.copy()
    for j, (coeff, observable) in enumerate(zip(alpha_arr, observables, strict=True)):
        Qj, _ = _validate_square_matrix(
            observable, n_qubits=n_qubits, name=f"observables[{j}]"
        )
        _validate_hermitian(Qj, atol=hermitian_atol, name=f"observables[{j}]")
        out = out + coeff * Qj

    _validate_hermitian(out, atol=hermitian_atol, name="H(alpha)")
    return out


def build_thermal_state(
    H: np.ndarray,
    *,
    beta: float = 1.0,
    hermitian_atol: float = 1e-10,
) -> np.ndarray:
    r"""Build ``rho = exp(-beta H) / Tr(exp(-beta H))`` by eigendecomposition."""

    H_arr, _ = _validate_square_matrix(H, name="H")
    _validate_hermitian(H_arr, atol=hermitian_atol, name="H")
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")

    evals, evecs = np.linalg.eigh(H_arr)
    shift = float(evals.min())
    weights = np.exp(-float(beta) * (evals - shift))
    partition = float(weights.sum())
    if partition <= 0.0 or not np.isfinite(partition):
        raise ValueError("thermal partition function is not finite and positive")

    rho = (evecs * weights) @ evecs.conj().T / partition
    rho = 0.5 * (rho + rho.conj().T)

    if not np.allclose(np.trace(rho), 1.0, atol=hermitian_atol):
        raise ValueError("thermal state trace is not numerically one")
    if np.linalg.eigvalsh(rho).min(initial=0.0) < -hermitian_atol:
        raise ValueError("thermal state is not numerically positive semidefinite")
    return rho


__all__ = ["build_linear_hamiltonian", "build_thermal_state"]
