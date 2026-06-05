"""Monte Carlo Pauli-sampling estimator for dense quantum observables."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, log, sqrt
from typing import Any, Literal, Protocol

import numpy as np

from .pauli import PauliDecomposition, decompose_pauli_dense
from .thermal import build_linear_hamiltonian, build_thermal_state


class MeasurementBackend(Protocol):
    r"""Define the protocol for Pauli measurement samplers.

    A backend receives a density matrix :math:`\rho \in \mathcal{D}_{2^K}`
    and Pauli labels from the decomposition of an observable
    :math:`Q \in \mathbb{H}_{2^K}`. It returns measurement eigenvalues in
    ``{-1, +1}``; the estimator handles coefficient sampling and signs.

    Notes
    -----
    Implementations sample measurements of Pauli observables in the supplied
    state. They do not choose which Pauli strings to measure and do not apply
    Pauli coefficients; that is handled by :class:`PauliSamplingEstimator`.

    Examples
    --------
    Use the direct backend as an implementation of this protocol:

    >>> import numpy as np
    >>> from sdplab.special.quantum import DirectDensityMatrixMeasurementBackend
    >>> rho = np.array([[1, 0], [0, 0]], dtype=complex)
    >>> backend = DirectDensityMatrixMeasurementBackend()
    >>> samples = backend.measure_pauli_expectation_samples(
    ...     rho, ["Z"], {"Z": 2}, seed=0
    ... )
    >>> samples["Z"].tolist()
    [1, 1]
    """

    def measure_pauli_expectation_samples(
        self,
        rho: np.ndarray,
        labels: Sequence[str],
        shots_per_label: Mapping[str, int],
        *,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return one ``{-1, +1}`` eigenvalue sample array per Pauli label."""


@dataclass(frozen=True)
class PauliEstimationResult:
    r"""Store the result of a Pauli-sampling estimate.

    ``estimate`` is the Monte Carlo Pauli-sampling approximation. When
    requested, ``exact_value`` is the dense value computed directly from
    :math:`Q` and :math:`\rho` for validation.

    Attributes
    ----------
    estimate : float
        Monte Carlo approximation to :math:`\operatorname{Tr}[Q\rho]`.
    exact_value : float or None
        Dense validation value, if requested.
    absolute_error : float or None
        Absolute difference between ``estimate`` and ``exact_value``, if the
        exact value was computed.
    n_sample : int
        Number of Pauli-string measurements requested.
    l1_norm : float
        Coefficient norm :math:`\|c\|_1` of the Pauli decomposition.
    empirical_variance : float
        Sample variance of the weighted estimator samples.
    standard_error : float
        Estimated standard error of the Monte Carlo mean.
    decomposition : PauliDecomposition
        Pauli decomposition used by the estimator.
    sampled_labels : tuple[str, ...] or None
        Sampled Pauli labels in trial order when ``store_samples=True``.
    sampled_counts : dict[str, int]
        Number of measurements requested for each sampled label.
    metadata : dict[str, Any]
        Backend and implementation metadata.

    Examples
    --------
    Create a deterministic result through the identity observable:

    >>> import numpy as np
    >>> from sdplab.special.quantum import (
    ...     DirectDensityMatrixMeasurementBackend, PauliSamplingEstimator
    ... )
    >>> rho = np.array([[1, 0], [0, 0]], dtype=complex)
    >>> identity = np.eye(2, dtype=complex)
    >>> estimator = PauliSamplingEstimator(DirectDensityMatrixMeasurementBackend())
    >>> result = estimator.estimate(identity, rho, n_sample=3, seed=0)
    >>> result.estimate
    1.0
    """

    estimate: float
    exact_value: float | None
    absolute_error: float | None
    n_sample: int
    l1_norm: float
    empirical_variance: float
    standard_error: float
    decomposition: PauliDecomposition
    sampled_labels: tuple[str, ...] | None
    sampled_counts: dict[str, int]
    metadata: dict[str, Any]


def pauli_sample_size(l1_norm: float, epsilon: float, delta: float) -> int:
    r"""Compute the Hoeffding sample-size bound for Pauli sampling.

    If ``l1_norm == 0``, the observable is zero in the Pauli basis and no
    samples are required; this helper returns ``0``.

    Parameters
    ----------
    l1_norm : float
        Pauli coefficient norm :math:`\|c\|_1`.
    epsilon : float
        Desired absolute precision.
    delta : float
        Failure probability. Must satisfy ``0 < delta < 1``.

    Returns
    -------
    int
        Smallest integer sample count satisfying
        :math:`N \ge 2\|c\|_1^2\log(2/\delta)/\epsilon^2`.

    Raises
    ------
    ValueError
        If ``l1_norm`` is negative, ``epsilon`` is nonpositive, or ``delta`` is
        outside ``(0, 1)``.

    Examples
    --------
    >>> from sdplab.special.quantum import pauli_sample_size
    >>> pauli_sample_size(0.0, epsilon=0.1, delta=0.05)
    0
    >>> pauli_sample_size(1.0, epsilon=1.0, delta=0.5)
    3
    """

    _validate_l1_norm(l1_norm)
    _validate_epsilon_delta(epsilon, delta)
    if l1_norm == 0:
        return 0
    return int(ceil(2.0 * l1_norm**2 * log(2.0 / delta) / epsilon**2))


class PauliSamplingEstimator:
    r"""Estimate :math:`\operatorname{Tr}[Q\rho]` by Pauli-string sampling.

    The estimator samples Pauli strings from the decomposition
    :math:`Q=\sum_\ell c_\ell P_\ell`, not directly from the density matrix.
    The backend is responsible only for measuring sampled Pauli observables in
    :math:`\rho \in \mathcal{D}_{2^K}` and returning eigenvalues
    :math:`m \in \{-1,+1\}`.

    Parameters
    ----------
    measurement_backend : MeasurementBackend
        Backend used to sample Pauli measurement eigenvalues from ``rho``.
    rng : numpy.random.Generator or None, optional
        Generator used when an estimator call does not supply ``seed``.
        Default is a fresh NumPy generator.

    Attributes
    ----------
    measurement_backend : MeasurementBackend
        Backend used by :meth:`estimate`.
    rng : numpy.random.Generator
        Generator used for unseeded estimator calls.

    Notes
    -----
    Sampling uses :math:`p_\ell = |c_\ell|/\|c\|_1`. The sign of each
    coefficient is restored in the random variable
    :math:`Y=\|c\|_1\operatorname{sign}(c_\ell)m`, where ``m`` is a backend
    measurement eigenvalue.

    Examples
    --------
    Estimate the expectation of ``Z`` in the one-qubit ``|0><0|`` state:

    >>> import numpy as np
    >>> from sdplab.special.quantum import (
    ...     DirectDensityMatrixMeasurementBackend, PauliSamplingEstimator
    ... )
    >>> Z = np.array([[1, 0], [0, -1]], dtype=complex)
    >>> rho = np.array([[1, 0], [0, 0]], dtype=complex)
    >>> estimator = PauliSamplingEstimator(DirectDensityMatrixMeasurementBackend())
    >>> result = estimator.estimate(Z, rho, n_sample=4, seed=0)
    >>> result.estimate
    1.0
    """

    def __init__(
        self,
        measurement_backend: MeasurementBackend,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize the estimator with a measurement backend."""

        self.measurement_backend = measurement_backend
        self.rng = rng if rng is not None else np.random.default_rng()

    def estimate(
        self,
        Q: np.ndarray | PauliDecomposition,
        rho: np.ndarray,
        *,
        n_sample: int | None = None,
        epsilon: float | None = None,
        delta: float | None = None,
        exact: bool = True,
        seed: int | None = None,
        store_samples: bool = False,
    ) -> PauliEstimationResult:
        r"""Estimate :math:`\operatorname{Tr}[Q\rho]`.

        ``Q`` may be dense or already decomposed into Pauli strings. The Monte
        Carlo samples are drawn from the Pauli coefficients of ``Q``; the
        measurement backend then samples Pauli eigenvalues from ``rho``.

        Parameters
        ----------
        Q : array-like or PauliDecomposition
            Hermitian observable :math:`Q \in \mathbb{H}_{2^K}`, or its Pauli
            decomposition.
        rho : array-like
            Density matrix :math:`\rho \in \mathcal{D}_{2^K}`.
        n_sample : int or None, optional
            Number of Pauli-string measurements. If omitted, ``epsilon`` and
            ``delta`` are used to compute a Hoeffding sample count.
        epsilon : float or None, optional
            Desired absolute precision used when ``n_sample`` is omitted.
        delta : float or None, optional
            Failure probability used when ``n_sample`` is omitted.
        exact : bool, optional
            Whether to compute the dense validation value. Default is ``True``.
        seed : int or None, optional
            Seed for this estimator call. Default uses :attr:`rng`.
        store_samples : bool, optional
            Whether to store the sampled Pauli labels in the result. Default is
            ``False``.

        Returns
        -------
        PauliEstimationResult
            Estimate, optional exact value, error diagnostics, and sampling
            metadata.

        Raises
        ------
        ValueError
            If ``rho`` has incompatible shape, ``n_sample`` is negative, or
            ``epsilon`` and ``delta`` are invalid.

        Examples
        --------
        >>> import numpy as np
        >>> from sdplab.special.quantum import (
        ...     DirectDensityMatrixMeasurementBackend, PauliSamplingEstimator
        ... )
        >>> X = np.array([[0, 1], [1, 0]], dtype=complex)
        >>> rho = np.array([[1, 0], [0, 0]], dtype=complex)
        >>> estimator = PauliSamplingEstimator(DirectDensityMatrixMeasurementBackend())
        >>> result = estimator.estimate(X, rho, n_sample=8, seed=1)
        >>> result.exact_value
        0.0
        """

        decomposition = self._as_decomposition(Q)
        rho_arr = _validate_density_matrix(rho, n_qubits=decomposition.n_qubits)
        n_sample = self._resolve_n_sample(n_sample, epsilon, delta, decomposition)
        exact_value = self._compute_exact_value(Q, decomposition, rho_arr) if exact else None

        if n_sample == 0 or decomposition.l1_norm == 0.0:
            return self._zero_result(
                decomposition,
                n_sample,
                exact_value,
                store_samples=store_samples,
            )

        rng = self._rng_for_call(seed)

        # We sample Pauli strings with probability proportional to |c_l|.
        # The coefficient sign is restored later in the estimator sample Y_n.
        sampled_indices = self._sample_pauli_indices(decomposition, n_sample, rng)

        # Batching by label is statistically equivalent to one circuit per
        # trial, but avoids running duplicate measurements many times.
        sampled_labels, shots_per_label = self._count_sampled_labels(
            decomposition,
            sampled_indices,
        )
        labels = tuple(shots_per_label)
        backend_samples = self.measurement_backend.measure_pauli_expectation_samples(
            rho_arr,
            labels,
            shots_per_label,
            seed=self._backend_seed(rng),
        )

        # Each backend sample is an eigenvalue m in {-1, +1}. The unbiased
        # estimator sample is ||c||_1 * sign(c_l) * m.
        sample_values = self._assemble_weighted_samples(
            decomposition,
            sampled_indices,
            backend_samples,
            labels,
        )
        estimate = self._compute_estimate(sample_values)

        return self._build_result(
            estimate,
            exact_value,
            sample_values,
            decomposition,
            n_sample,
            sampled_labels=sampled_labels if store_samples else None,
            sampled_counts=dict(shots_per_label),
        )

    @staticmethod
    def _as_decomposition(Q: np.ndarray | PauliDecomposition) -> PauliDecomposition:
        """Return ``Q`` as a :class:`PauliDecomposition`."""

        return Q if isinstance(Q, PauliDecomposition) else decompose_pauli_dense(Q)

    @staticmethod
    def _resolve_n_sample(
        n_sample: int | None,
        epsilon: float | None,
        delta: float | None,
        decomposition: PauliDecomposition,
    ) -> int:
        """Resolve the explicit or bound-derived sample count."""

        if n_sample is None:
            if epsilon is None or delta is None:
                raise ValueError(
                    "Either n_sample or both epsilon and delta must be provided"
                )
            n_sample = pauli_sample_size(decomposition.l1_norm, epsilon, delta)
        return _validate_n_sample(n_sample)

    def _rng_for_call(self, seed: int | None) -> np.random.Generator:
        """Return the random generator for one estimator call."""

        return np.random.default_rng(seed) if seed is not None else self.rng

    @staticmethod
    def _sample_pauli_indices(
        decomposition: PauliDecomposition,
        n_sample: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample Pauli coefficient indices according to ``|c_l| / ||c||_1``."""

        return rng.choice(
            len(decomposition.labels),
            size=n_sample,
            p=decomposition.probabilities,
        )

    @staticmethod
    def _count_sampled_labels(
        decomposition: PauliDecomposition,
        sampled_indices: np.ndarray,
    ) -> tuple[tuple[str, ...], Counter[str]]:
        """Count sampled Pauli labels for backend batching."""

        sampled_labels = tuple(decomposition.labels[index] for index in sampled_indices)
        return sampled_labels, Counter(sampled_labels)

    @staticmethod
    def _backend_seed(rng: np.random.Generator) -> int:
        """Draw a seed for backend measurement sampling."""

        return int(rng.integers(0, np.iinfo(np.uint32).max))

    @staticmethod
    def _assemble_weighted_samples(
        decomposition: PauliDecomposition,
        sampled_indices: np.ndarray,
        backend_samples: Mapping[str, np.ndarray],
        labels: Sequence[str],
    ) -> np.ndarray:
        """Assemble weighted estimator samples from backend eigenvalues."""

        next_sample_position = {label: 0 for label in labels}
        sample_values = np.empty(len(sampled_indices), dtype=float)

        for output_position, decomp_index in enumerate(sampled_indices):
            label = decomposition.labels[decomp_index]
            label_position = next_sample_position[label]
            eigenvalues = backend_samples[label]
            if label_position >= len(eigenvalues):
                raise ValueError(
                    f"Backend returned too few samples for Pauli label {label!r}"
                )
            next_sample_position[label] += 1
            sample_values[output_position] = (
                decomposition.l1_norm
                * decomposition.signs[decomp_index]
                * float(eigenvalues[label_position])
            )

        return sample_values

    @staticmethod
    def _compute_estimate(sample_values: np.ndarray) -> float:
        """Compute the Monte Carlo mean from weighted samples."""

        return float(np.mean(sample_values)) if len(sample_values) else 0.0

    @staticmethod
    def _compute_exact_value(
        Q: np.ndarray | PauliDecomposition,
        decomposition: PauliDecomposition,
        rho: np.ndarray,
    ) -> float:
        """Compute the dense validation value."""

        Q_arr = Q.reconstruct_dense() if isinstance(Q, PauliDecomposition) else np.asarray(Q)
        value = complex(np.trace(Q_arr @ rho))
        if abs(value.imag) > 1e-10:
            raise ValueError(
                "Exact trace value has a non-negligible imaginary part "
                f"{value.imag}"
            )
        return float(value.real)

    def _zero_result(
        self,
        decomposition: PauliDecomposition,
        n_sample: int,
        exact_value: float | None,
        *,
        store_samples: bool,
    ) -> PauliEstimationResult:
        """Build the exact zero-sampling result."""

        return self._build_result(
            0.0,
            exact_value,
            np.empty(0, dtype=float),
            decomposition,
            n_sample,
            sampled_labels=tuple() if store_samples else None,
            sampled_counts={},
        )

    def _build_result(
        self,
        estimate: float,
        exact_value: float | None,
        sample_values: np.ndarray,
        decomposition: PauliDecomposition,
        n_sample: int,
        *,
        sampled_labels: tuple[str, ...] | None,
        sampled_counts: dict[str, int],
    ) -> PauliEstimationResult:
        """Build a :class:`PauliEstimationResult` from samples and metadata."""

        empirical_variance = (
            float(np.var(sample_values, ddof=1)) if len(sample_values) > 1 else 0.0
        )
        standard_error = sqrt(empirical_variance / n_sample) if n_sample > 0 else 0.0
        absolute_error = None if exact_value is None else float(abs(estimate - exact_value))

        return PauliEstimationResult(
            estimate=estimate,
            exact_value=exact_value,
            absolute_error=absolute_error,
            n_sample=n_sample,
            l1_norm=decomposition.l1_norm,
            empirical_variance=empirical_variance,
            standard_error=standard_error,
            decomposition=decomposition,
            sampled_labels=sampled_labels,
            sampled_counts=sampled_counts,
            metadata={"backend": type(self.measurement_backend).__name__},
        )


def _validate_density_matrix(
    rho: np.ndarray,
    *,
    n_qubits: int,
    atol: float = 1e-10,
) -> np.ndarray:
    """Validate and return a density matrix with the expected dimension."""

    rho_arr = np.asarray(rho, dtype=complex)
    expected_shape = (2**n_qubits, 2**n_qubits)
    if rho_arr.shape != expected_shape:
        raise ValueError(f"rho must have shape {expected_shape}, got {rho_arr.shape}")
    if not np.allclose(rho_arr, rho_arr.conj().T, atol=atol):
        raise ValueError("rho must be Hermitian")
    return rho_arr


def _validate_n_sample(n_sample: int) -> int:
    """Validate and return a nonnegative sample count."""

    n_sample = int(n_sample)
    if n_sample < 0:
        raise ValueError("n_sample must be nonnegative")
    return n_sample


def _validate_l1_norm(l1_norm: float) -> None:
    """Validate a nonnegative Pauli coefficient l1 norm."""

    if l1_norm < 0:
        raise ValueError("l1_norm must be nonnegative")


def _validate_epsilon_delta(epsilon: float, delta: float) -> None:
    """Validate Hoeffding precision and failure-probability parameters."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must satisfy 0 < delta < 1")


def estimate_thermal_observable(
    Q: np.ndarray | PauliDecomposition,
    H: np.ndarray | None = None,
    *,
    H0: np.ndarray | None = None,
    observables: Sequence[np.ndarray] | None = None,
    alpha: np.ndarray | None = None,
    beta: float = 1.0,
    n_sample: int | None = None,
    epsilon: float | None = None,
    delta: float | None = None,
    backend: Literal["direct", "qiskit"] = "qiskit",
    seed: int | None = None,
) -> PauliEstimationResult:
    r"""Build a thermal state and estimate :math:`\operatorname{Tr}[Q\rho]`.

    The Hamiltonian may be supplied directly as ``H`` or assembled as
    :math:`H(\alpha)=H_0+\sum_j \alpha_j Q_j`. The thermal state is

    .. math::

        \rho = \frac{\exp(-\beta H)}{\operatorname{Tr}\exp(-\beta H)}.

    Parameters
    ----------
    Q : array-like or PauliDecomposition
        Observable :math:`Q \in \mathbb{H}_{2^K}`, or its Pauli
        decomposition.
    H : array-like or None, optional
        Dense Hamiltonian. If omitted, ``H0``, ``observables``, and ``alpha``
        are required.
    H0 : array-like or None, optional
        Base Hamiltonian used when ``H`` is omitted.
    observables : sequence of array-like or None, optional
        Hamiltonian perturbation observables used when ``H`` is omitted.
    alpha : array-like or None, optional
        Coefficients for ``observables`` used when ``H`` is omitted.
    beta : float, optional
        Inverse temperature. Default is ``1.0``.
    n_sample : int or None, optional
        Number of Pauli-string measurements. If omitted, ``epsilon`` and
        ``delta`` determine the sample count.
    epsilon : float or None, optional
        Desired absolute precision used when ``n_sample`` is omitted.
    delta : float or None, optional
        Failure probability used when ``n_sample`` is omitted.
    backend : {"direct", "qiskit"}, optional
        Measurement backend name. Default is ``"qiskit"``.
    seed : int or None, optional
        Seed for the estimator call.

    Returns
    -------
    PauliEstimationResult
        Monte Carlo estimate and dense validation value for the thermal state.

    Raises
    ------
    ValueError
        If neither ``H`` nor the triple ``H0``, ``observables``, ``alpha`` is
        supplied, or if ``backend`` is unknown.
    ImportError
        If ``backend="qiskit"`` and Qiskit Aer is unavailable.

    Examples
    --------
    Estimate a one-qubit thermal ``Z`` expectation with the direct backend:

    >>> import numpy as np
    >>> from sdplab.special.quantum import estimate_thermal_observable
    >>> Z = np.array([[1, 0], [0, -1]], dtype=complex)
    >>> result = estimate_thermal_observable(
    ...     Z, Z, n_sample=4, backend="direct", seed=0
    ... )
    >>> result.n_sample
    4
    """

    if H is None:
        if H0 is None or observables is None or alpha is None:
            raise ValueError("Provide either H or H0, observables, and alpha")
        H = build_linear_hamiltonian(H0, observables, alpha)

    rho = build_thermal_state(H, beta=beta)

    if backend == "direct":
        from .direct_backend import DirectDensityMatrixMeasurementBackend

        measurement_backend: MeasurementBackend = DirectDensityMatrixMeasurementBackend()
    elif backend == "qiskit":
        from .qiskit_backend import QiskitDensityMatrixMeasurementBackend

        measurement_backend = QiskitDensityMatrixMeasurementBackend()
    else:
        raise ValueError("backend must be 'direct' or 'qiskit'")

    estimator = PauliSamplingEstimator(measurement_backend)
    return estimator.estimate(
        Q,
        rho,
        n_sample=n_sample,
        epsilon=epsilon,
        delta=delta,
        seed=seed,
    )


__all__ = [
    "MeasurementBackend",
    "PauliEstimationResult",
    "PauliSamplingEstimator",
    "estimate_thermal_observable",
    "pauli_sample_size",
]
