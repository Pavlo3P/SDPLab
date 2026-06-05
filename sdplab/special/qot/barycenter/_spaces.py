"""Gaussian phase-space utilities for quadratic QOT barycenters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spacecore import Context, ContextBound, DenseArray, jax_pytree_class

from ._repr import backend_label, html_repr, plain_repr


def _as_float(x: Any) -> float:
    """Convert a scalar backend value to a Python float for validation."""
    try:
        return float(x)
    except TypeError:
        return float(x.item())


def _is_real_dtype(dtype: Any) -> bool:
    """Return whether ``dtype`` is a real floating dtype."""
    return np.issubdtype(np.dtype(dtype), np.floating)


def _check_no_complex_input(x: Any, name: str) -> None:
    """Reject complex-valued input before context coercion can drop imaginary parts."""
    raw = np.asarray(x)
    if np.iscomplexobj(raw):
        raise TypeError(f"{name} must have a real dtype.")


def _to_numpy_real(x: Any, name: str) -> np.ndarray:
    """Return ``x`` as a finite real NumPy array for validation-only checks."""
    raw = np.asarray(x)
    if np.iscomplexobj(raw):
        raise TypeError(f"{name} must have a real dtype.")
    arr = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _check_shape(x: Any, expected: tuple[int, ...], name: str) -> None:
    """Validate the array shape."""
    if tuple(x.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(x.shape)}.")


def _canonical_omega(m: int) -> np.ndarray:
    """Return the canonical symplectic matrix in (Q1, P1, ..., Qm, Pm) order."""
    omega = np.zeros((2 * m, 2 * m), dtype=float)
    for k in range(m):
        i = 2 * k
        omega[i, i + 1] = 1.0
        omega[i + 1, i] = -1.0
    return omega


@jax_pytree_class
@dataclass(init=False)
class GaussianPhaseSpace(ContextBound):
    r"""Represent a canonical Gaussian phase space.

    Store the coordinate convention, symplectic matrix, Planck constant, and
    validation tolerances used by Gaussian QOT barycenter objects. Coordinates
    are ordered as :math:`R = (Q_1, P_1, \ldots, Q_m, P_m)`.

    Parameters
    ----------
    m : int
        Number of bosonic modes. Must be positive.
    hbar : float, optional
        Nonnegative Planck constant in the commutation relation
        :math:`[R_i, R_j] = i \hbar \Omega_{ij}`. The value zero selects the
        classical Gaussian endpoint. Default is 1.
    atol : float, optional
        Absolute tolerance used by validation routines. Default is 1e-9.
    rtol : float, optional
        Relative tolerance used by validation routines. Default is 1e-9.
    ctx : Context, str, or None, optional
        Backend context. If ``None``, the default SpaceCore context is used.

    Attributes
    ----------
    m : int
        Number of modes.
    dim : int
        Phase-space dimension, equal to ``2 * m``.
    hbar : float
        Planck constant used in uncertainty checks and Gaussian calculus.
    omega : array-like
        Canonical symplectic matrix in ``(Q1, P1, ..., Qm, Pm)`` order.
    atol : float
        Absolute validation tolerance.
    rtol : float
        Relative validation tolerance.

    Raises
    ------
    ValueError
        If ``m`` is not positive or ``hbar`` is negative or non-finite.

    Notes
    -----
    Covariances are validated against the uncertainty condition

    .. math::

        \Gamma + \frac{i\hbar}{2}\Omega \succeq 0.

    At :math:`\hbar = 0`, this reduces to ordinary positive semidefiniteness
    of the covariance matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from spacecore import Context, NumpyOps
    >>> from sdplab.special.qot.barycenter import GaussianPhaseSpace
    >>> ctx = Context(NumpyOps(), dtype=np.float64)
    >>> space = GaussianPhaseSpace(1, hbar=0.5, ctx=ctx)
    >>> space.dim
    2
    >>> np.allclose(space.omega, [[0.0, 1.0], [-1.0, 0.0]])
    True
    """

    def __init__(
        self,
        m: int,
        *,
        hbar: float = 1.0,
        atol: float = 1e-9,
        rtol: float = 1e-9,
        ctx: Context | str | None = None,
    ) -> None:
        if type(m) is not int or m <= 0:
            raise ValueError("m must be a positive integer.")
        hbar = float(hbar)
        if not np.isfinite(hbar) or hbar < 0.0:
            raise ValueError("hbar must be nonnegative and finite.")
        super(GaussianPhaseSpace, self).__init__(ctx)
        self.m = m
        self.dim = 2 * m
        self.hbar = hbar
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.omega = self.ctx.asarray(_canonical_omega(m))

    def check_mean(self, d: DenseArray) -> None:
        """Validate a phase-space mean vector.

        Parameters
        ----------
        d : array-like
            Candidate mean vector with shape ``(dim,)``.

        Raises
        ------
        TypeError
            If ``d`` is complex-valued or not real-valued after conversion.
        ValueError
            If ``d`` has the wrong shape.
        """
        _check_no_complex_input(d, "mean")
        arr = self.ctx.asarray(d)
        _check_shape(arr, (self.dim,), "mean")
        if not _is_real_dtype(arr.dtype):
            raise TypeError("mean must have a real dtype.")

    def check_covariance(self, Gamma: DenseArray) -> None:
        r"""Validate a real symmetric covariance matrix.

        Parameters
        ----------
        Gamma : array-like
            Candidate covariance matrix with shape ``(dim, dim)``.

        Raises
        ------
        TypeError
            If ``Gamma`` is complex-valued.
        ValueError
            If ``Gamma`` has the wrong shape, contains non-finite values, or
            is not symmetric within tolerances.
        """
        G_np = _to_numpy_real(Gamma, "covariance")
        if G_np.shape != (self.dim, self.dim):
            raise ValueError(
                f"covariance must have shape {(self.dim, self.dim)}, got {G_np.shape}."
            )
        if not np.allclose(G_np, G_np.T, atol=self.atol, rtol=self.rtol):
            raise ValueError("covariance must be symmetric.")

    def check_uncertainty(self, Gamma: DenseArray) -> None:
        r"""Validate the Gaussian uncertainty principle.

        Parameters
        ----------
        Gamma : array-like
            Candidate covariance matrix.

        Raises
        ------
        TypeError
            If ``Gamma`` is complex-valued.
        ValueError
            If ``Gamma`` is not a valid covariance matrix or violates
            :math:`\Gamma + i\hbar\Omega/2 \succeq 0` within tolerances.
        """
        self.check_covariance(Gamma)
        gamma = _to_numpy_real(Gamma, "covariance")
        omega = np.asarray(self.omega, dtype=float)
        herm = gamma + 0.5j * self.hbar * omega
        eigvals = np.linalg.eigvalsh(herm)
        scale = max(1.0, float(np.linalg.norm(gamma, ord=2)))
        min_allowed = -self.atol - self.rtol * scale
        if float(eigvals.min()) < min_allowed:
            raise ValueError(
                "covariance violates the Gaussian uncertainty principle: "
                f"min eigenvalue is {float(eigvals.min()):.3e}."
            )

    def check_square_matrix(self, A: DenseArray, *, name: str = "matrix") -> None:
        """Validate a real square matrix on this phase space."""
        arr_np = _to_numpy_real(A, name)
        if arr_np.shape != (self.dim, self.dim):
            raise ValueError(f"{name} must have shape {(self.dim, self.dim)}, got {arr_np.shape}.")

    def check_symmetric_matrix(self, A: DenseArray, *, name: str = "matrix") -> None:
        """Validate a real symmetric matrix on this phase space."""
        arr_np = _to_numpy_real(A, name)
        if arr_np.shape != (self.dim, self.dim):
            raise ValueError(f"{name} must have shape {(self.dim, self.dim)}, got {arr_np.shape}.")
        if not np.allclose(arr_np, arr_np.T, atol=self.atol, rtol=self.rtol):
            raise ValueError(f"{name} must be symmetric.")

    def symmetrize_matrix(self, A: DenseArray) -> DenseArray:
        r"""Return the real symmetric part of a matrix.

        Parameters
        ----------
        A : array-like
            Square real matrix on this phase space.

        Returns
        -------
        array-like
            Matrix :math:`(A + A^T) / 2` in this space's context.
        """
        self.check_square_matrix(A)
        A = self.ctx.asarray(A)
        return 0.5 * (A + self.ops.transpose(A, (1, 0)))

    def zero_mean(self) -> DenseArray:
        """Return the zero mean vector in this phase space.

        Returns
        -------
        array-like
            Zero vector with shape ``(dim,)``.
        """
        return self.ops.zeros((self.dim,))

    def identity_covariance(self) -> DenseArray:
        """Return the identity covariance matrix.

        Returns
        -------
        array-like
            Identity matrix with shape ``(dim, dim)``.
        """
        return self.ops.eye(self.dim)

    def is_compatible(self, other: Any) -> bool:
        """Return whether another object is the same phase-space convention.

        Parameters
        ----------
        other : object
            Candidate phase space.

        Returns
        -------
        bool
            Whether ``other`` is a :class:`GaussianPhaseSpace` with matching
            mode count and ``hbar`` within tolerances.
        """
        if not isinstance(other, GaussianPhaseSpace) or self.m != other.m:
            return False
        scale = max(1.0, abs(self.hbar), abs(other.hbar))
        return abs(self.hbar - other.hbar) <= self.atol + self.rtol * scale

    def _repr_rows(self) -> tuple[tuple[str, Any], ...]:
        """Return rows shared by plain and HTML reprs."""
        return (
            ("modes", self.m),
            ("dimension", self.dim),
            ("hbar", self.hbar),
            ("coordinate order", "(Q1, P1, ..., Qm, Pm)"),
            ("tolerances", f"atol={self.atol:g}, rtol={self.rtol:g}"),
            ("backend", backend_label(self)),
        )

    def __repr__(self) -> str:
        """Return a readable representation."""
        return plain_repr(type(self).__name__, self._repr_rows())

    def _repr_html_(self) -> str:
        """Return a notebook-friendly representation."""
        return html_repr(type(self).__name__, self._repr_rows())

    def tree_flatten(self):
        """Return children and auxiliary data for JAX PyTree flattening."""
        return (), (self.m, self.hbar, self.atol, self.rtol, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild a phase space from JAX PyTree data."""
        m, hbar, atol, rtol, ctx = aux
        return cls(m, hbar=hbar, atol=atol, rtol=rtol, ctx=ctx)

    def _convert(self, new_ctx: Context) -> "GaussianPhaseSpace":
        """Convert to ``new_ctx``.

        Rebuilds the symplectic matrix by constructing a new phase space with
        the same ``m``, ``hbar``, ``atol``, and ``rtol`` under ``new_ctx``.
        """
        return GaussianPhaseSpace(
            self.m,
            hbar=self.hbar,
            atol=self.atol,
            rtol=self.rtol,
            ctx=new_ctx,
        )
