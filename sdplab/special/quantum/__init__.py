"""Pauli-sampling estimators for dense quantum observables."""

from .direct_backend import DirectDensityMatrixMeasurementBackend
from .estimator import (
    PauliEstimationResult,
    PauliSamplingEstimator,
    estimate_thermal_observable,
    pauli_sample_size,
)
from .pauli import PauliDecomposition, decompose_pauli_dense
from .qiskit_backend import QiskitDensityMatrixMeasurementBackend
from .thermal import build_linear_hamiltonian, build_thermal_state

__all__ = [
    "DirectDensityMatrixMeasurementBackend",
    "PauliDecomposition",
    "PauliEstimationResult",
    "PauliSamplingEstimator",
    "QiskitDensityMatrixMeasurementBackend",
    "build_linear_hamiltonian",
    "build_thermal_state",
    "decompose_pauli_dense",
    "estimate_thermal_observable",
    "pauli_sample_size",
]
