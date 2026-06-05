"""Direct density-matrix Pauli measurement sampler."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .pauli import pauli_string_dense


class DirectDensityMatrixMeasurementBackend:
    r"""Sample Pauli measurements directly from a density matrix.

    This backend does not build circuits. For a Pauli observable
    :math:`P_\ell` and state :math:`\rho \in \mathcal{D}_{2^K}`, it computes
    :math:`\mu=\operatorname{Tr}(P_\ell\rho)` and samples eigenvalues
    :math:`m\in\{-1,+1\}` with the corresponding ideal probabilities.

    Notes
    -----
    This backend is useful for validation and fast tests. It implements ideal
    projective Pauli measurement statistics without simulator overhead.

    Examples
    --------
    Sample deterministic ``Z`` measurements on the ``|0><0|`` state:

    >>> import numpy as np
    >>> from sdplab.special.quantum import DirectDensityMatrixMeasurementBackend
    >>> rho = np.array([[1, 0], [0, 0]], dtype=complex)
    >>> backend = DirectDensityMatrixMeasurementBackend()
    >>> samples = backend.measure_pauli_expectation_samples(
    ...     rho, ["Z"], {"Z": 3}, seed=0
    ... )
    >>> samples["Z"].tolist()
    [1, 1, 1]
    """

    def measure_pauli_expectation_samples(
        self,
        rho: np.ndarray,
        labels: Sequence[str],
        shots_per_label: Mapping[str, int],
        *,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        r"""Sample Pauli measurement eigenvalues from ``rho``.

        Parameters
        ----------
        rho : array-like
            Density matrix :math:`\rho \in \mathcal{D}_{2^K}`.
        labels : sequence of str
            Pauli labels to measure, using mathematical tensor order.
        shots_per_label : mapping
            Number of measurement shots requested for each label.
        seed : int or None, optional
            Random seed for the measurement samples. Default is unseeded.

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping from each label to an array of eigenvalues in ``{-1, +1}``.

        Raises
        ------
        ValueError
            If a shot count is negative or a Pauli label has incompatible
            dimension for ``rho``.

        Examples
        --------
        >>> import numpy as np
        >>> from sdplab.special.quantum import DirectDensityMatrixMeasurementBackend
        >>> rho = np.eye(2, dtype=complex) / 2
        >>> backend = DirectDensityMatrixMeasurementBackend()
        >>> samples = backend.measure_pauli_expectation_samples(
        ...     rho, ["I"], {"I": 2}, seed=0
        ... )
        >>> samples["I"].tolist()
        [1, 1]
        """

        rho_arr = np.asarray(rho, dtype=complex)
        rng = np.random.default_rng(seed)
        samples: dict[str, np.ndarray] = {}

        for label in labels:
            n_shots = _shots_for_label(shots_per_label, label)
            if n_shots == 0:
                samples[label] = np.empty(0, dtype=int)
                continue
            if all(char == "I" for char in label):
                samples[label] = np.ones(n_shots, dtype=int)
                continue

            expectation = _pauli_expectation(rho_arr, label)
            samples[label] = _sample_pm_one(expectation, n_shots, rng)

        return samples


def _shots_for_label(shots_per_label: Mapping[str, int], label: str) -> int:
    """Return the nonnegative shot count for one Pauli label."""

    n_shots = int(shots_per_label.get(label, 0))
    if n_shots < 0:
        raise ValueError("shots_per_label values must be nonnegative")
    return n_shots


def _pauli_expectation(rho: np.ndarray, label: str, *, atol: float = 1e-10) -> float:
    """Compute ``Tr(P_l rho)`` for one Pauli label."""

    pauli = pauli_string_dense(label)
    if pauli.shape != rho.shape:
        raise ValueError(
            f"rho has shape {rho.shape}, but label {label!r} acts on "
            f"dimension {pauli.shape[0]}"
        )

    value = complex(np.trace(pauli @ rho))
    if abs(value.imag) > atol:
        raise ValueError(
            f"Pauli expectation for {label!r} has non-negligible "
            f"imaginary part {value.imag}"
        )
    return float(value.real)


def _sample_pm_one(
    expectation: float,
    n_shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``{-1, +1}`` outcomes with mean ``expectation``."""

    # Measuring a Pauli observable P gives eigenvalues +/-1. If E[P] = mu,
    # then P(+1) = (1 + mu) / 2.
    p_plus = 0.5 * (1.0 + float(expectation))

    # Clip only for floating-point roundoff near the physical interval.
    p_plus = float(np.clip(p_plus, 0.0, 1.0))
    return rng.choice(np.array([-1, 1], dtype=int), size=n_shots, p=[1 - p_plus, p_plus])


__all__ = ["DirectDensityMatrixMeasurementBackend"]
