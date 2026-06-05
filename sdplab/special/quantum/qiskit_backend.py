"""Qiskit Aer density-matrix measurement backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


class QiskitDensityMatrixMeasurementBackend:
    r"""Sample Pauli measurements with Qiskit Aer's density-matrix simulator.

    For each Pauli label, this backend prepares the supplied density matrix
    :math:`\rho \in \mathcal{D}_{2^K}` with ``set_density_matrix``, rotates
    non-identity Pauli factors into the computational basis, and samples
    eigenvalues in ``{-1, +1}``.

    Parameters
    ----------
    simulator_method : str, optional
        Aer simulator method. Default is ``"density_matrix"``.
    shots_batching : bool, optional
        Stored for API compatibility. The estimator already batches shots by
        Pauli label before calling this backend. Default is ``True``.

    Attributes
    ----------
    shots_batching : bool
        Whether shot batching is enabled by configuration.

    Raises
    ------
    ImportError
        If Qiskit or Qiskit Aer is unavailable.

    Notes
    -----
    Labels use mathematical tensor order, where ``"ZI"`` means
    :math:`Z \otimes I`. Qiskit indexes qubit 0 as the rightmost tensor
    factor, so label-to-qubit conversion is handled internally.

    Examples
    --------
    Sample deterministic ``Z`` measurements on the ``|0><0|`` state:

    >>> import numpy as np
    >>> from sdplab.special.quantum import QiskitDensityMatrixMeasurementBackend
    >>> rho = np.array([[1, 0], [0, 0]], dtype=complex)
    >>> backend = QiskitDensityMatrixMeasurementBackend()
    >>> samples = backend.measure_pauli_expectation_samples(
    ...     rho, ["Z"], {"Z": 2}, seed=0
    ... )
    >>> samples["Z"].tolist()
    [1, 1]
    """

    def __init__(
        self,
        *,
        simulator_method: str = "density_matrix",
        shots_batching: bool = True,
    ) -> None:
        """Initialize the Qiskit Aer simulator backend."""

        QuantumCircuit, DensityMatrix, SetDensityMatrix, AerSimulator = _import_qiskit()
        self._QuantumCircuit = QuantumCircuit
        self._DensityMatrix = DensityMatrix
        self._SetDensityMatrix = SetDensityMatrix
        self._simulator = AerSimulator(method=simulator_method)
        self.shots_batching = bool(shots_batching)

    def measure_pauli_expectation_samples(
        self,
        rho: np.ndarray,
        labels: Sequence[str],
        shots_per_label: Mapping[str, int],
        *,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        r"""Sample Pauli eigenvalues with Qiskit Aer.

        Parameters
        ----------
        rho : array-like
            Density matrix :math:`\rho \in \mathcal{D}_{2^K}` prepared with
            ``set_density_matrix``.
        labels : sequence of str
            Pauli labels to measure, using mathematical tensor order.
        shots_per_label : mapping
            Number of measurement shots requested for each label.
        seed : int or None, optional
            Seed passed to the Aer simulator. Default is unseeded.

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping from each label to an array of eigenvalues in ``{-1, +1}``.

        Raises
        ------
        ValueError
            If ``rho`` is not square, a shot count is negative, or a label has
            incompatible dimension for ``rho``.

        Examples
        --------
        >>> import numpy as np
        >>> from sdplab.special.quantum import QiskitDensityMatrixMeasurementBackend
        >>> rho = np.array([[1, 0], [0, 0]], dtype=complex)
        >>> backend = QiskitDensityMatrixMeasurementBackend()
        >>> samples = backend.measure_pauli_expectation_samples(
        ...     rho, ["I"], {"I": 2}, seed=0
        ... )
        >>> samples["I"].tolist()
        [1, 1]
        """

        rho_arr = _validate_square_density_matrix(rho)
        samples: dict[str, np.ndarray] = {}
        for offset, label in enumerate(labels):
            n_shots = _shots_for_label(shots_per_label, label)
            if n_shots == 0:
                samples[label] = np.empty(0, dtype=int)
                continue
            if all(char == "I" for char in label):
                samples[label] = np.ones(n_shots, dtype=int)
                continue

            circuit = self._build_measurement_circuit(rho_arr, label)
            run_seed = None if seed is None else int(seed) + offset
            result = self._simulator.run(
                circuit,
                shots=n_shots,
                seed_simulator=run_seed,
            ).result()
            counts = result.get_counts(circuit)
            samples[label] = self._counts_to_eigenvalue_samples(
                counts,
                label,
                n_shots,
            )

        return samples

    def _build_measurement_circuit(self, rho: np.ndarray, label: str):
        """Build one Qiskit circuit for measuring a Pauli label."""

        n_qubits = len(label)
        _validate_label_matches_density_matrix(rho, label)
        active_positions = _active_pauli_positions(label)
        circuit = self._QuantumCircuit(n_qubits, len(active_positions))
        state = self._DensityMatrix(rho)
        circuit.append(self._SetDensityMatrix(state), circuit.qubits)
        # circuit.set_density_matrix(self._DensityMatrix(rho))

        self._apply_pauli_basis_rotations_and_measure(
            circuit,
            label,
            active_positions,
        )
        return circuit

    def _apply_pauli_basis_rotations_and_measure(
        self,
        circuit,
        label: str,
        active_positions: Sequence[int],
    ) -> None:
        """Apply Pauli-basis rotations and measurements to a circuit."""

        n_qubits = len(label)
        for classical_bit, math_position in enumerate(active_positions):
            pauli = label[math_position]
            qiskit_qubit = _qiskit_qubit_index(n_qubits, math_position)
            self._apply_pauli_basis_rotation(circuit, qiskit_qubit, pauli)
            circuit.measure(qiskit_qubit, classical_bit)

    @staticmethod
    def _apply_pauli_basis_rotation(circuit, qiskit_qubit: int, pauli: str) -> None:
        """Rotate one qubit so the requested Pauli is measured in the Z basis."""

        if pauli == "X":
            circuit.h(qiskit_qubit)
        elif pauli == "Y":
            circuit.sdg(qiskit_qubit)
            circuit.h(qiskit_qubit)
        elif pauli != "Z":
            raise ValueError(f"Invalid Pauli character {pauli!r}; expected I, X, Y, Z")

    @staticmethod
    def _counts_to_eigenvalue_samples(
        counts: Mapping[str, int],
        label: str,
        n_shots: int,
    ) -> np.ndarray:
        """Convert Qiskit counts into Pauli eigenvalue samples."""

        out = np.empty(n_shots, dtype=int)
        start = 0
        measured_positions = _active_pauli_positions(label)
        for bitstring, count in counts.items():
            eigenvalue = _bitstring_to_pauli_eigenvalue(bitstring, measured_positions)
            stop = start + int(count)
            out[start:stop] = eigenvalue
            start = stop
        if start != n_shots:
            raise ValueError("Qiskit counts did not sum to the requested shot count")
        return out


def _import_qiskit():
    """Import Qiskit objects used by the backend."""

    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import DensityMatrix
        from qiskit_aer import AerSimulator
        from qiskit_aer.library import SetDensityMatrix
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Qiskit Aer is required for QiskitDensityMatrixMeasurementBackend. "
            "Install sdplab[quantum]."
        ) from exc
    return QuantumCircuit, DensityMatrix, SetDensityMatrix, AerSimulator


def _validate_square_density_matrix(rho: np.ndarray) -> np.ndarray:
    """Validate and return a square density matrix array."""

    rho_arr = np.asarray(rho, dtype=complex)
    if rho_arr.ndim != 2 or rho_arr.shape[0] != rho_arr.shape[1]:
        raise ValueError("rho must be a square density matrix")
    return rho_arr


def _validate_label_matches_density_matrix(rho: np.ndarray, label: str) -> None:
    """Validate that a label acts on the density-matrix dimension."""

    n_qubits = len(label)
    expected_shape = (2**n_qubits, 2**n_qubits)
    if rho.shape != expected_shape:
        raise ValueError(
            f"rho has shape {rho.shape}, but label {label!r} acts on "
            f"dimension {expected_shape[0]}"
        )


def _shots_for_label(shots_per_label: Mapping[str, int], label: str) -> int:
    """Return the nonnegative shot count for one Pauli label."""

    n_shots = int(shots_per_label.get(label, 0))
    if n_shots < 0:
        raise ValueError("shots_per_label values must be nonnegative")
    return n_shots


def _active_pauli_positions(label: str) -> list[int]:
    """Return mathematical tensor positions with non-identity Pauli factors."""

    return [position for position, pauli in enumerate(label) if pauli != "I"]


def _qiskit_qubit_index(n_qubits: int, math_position: int) -> int:
    """Convert mathematical tensor position to Qiskit's qubit index."""

    # Labels use mathematical tensor order: "ZI" means Z on the left tensor
    # factor. Qiskit qubit 0 is the rightmost computational-basis factor.
    return n_qubits - 1 - math_position


def _bitstring_to_pauli_eigenvalue(
    bitstring: str,
    measured_positions: Sequence[int],
) -> int:
    """Convert a measured bitstring into the product Pauli eigenvalue."""

    # Qiskit returns bitstrings in classical-register display order. The
    # product Pauli eigenvalue depends only on the parity of measured 1 bits,
    # so the display order does not affect the final +/-1 value. Keeping this
    # conversion isolated makes bit-ordering assumptions easy to test.
    clean = bitstring.replace(" ", "")
    if len(clean) != len(measured_positions):
        raise ValueError(
            f"Expected {len(measured_positions)} measured bits, got {bitstring!r}"
        )
    parity = clean.count("1") % 2
    return -1 if parity else 1


__all__ = ["QiskitDensityMatrixMeasurementBackend"]
