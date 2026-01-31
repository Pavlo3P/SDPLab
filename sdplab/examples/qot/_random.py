from typing import Optional, Tuple
import numpy as np

from qotlib.core import BackendContext, DenseArray
from qotlib.linalg import compute_ptraces


def generate_random_qot(
        ctx: BackendContext,
        d: int, N: int,
        proportions: tuple[float, ...],
        seed: Optional[int] = 0,
) -> Tuple[DenseArray, DenseArray, DenseArray]:
    """
    Generates a quantum optimal transport (QOT) example by constructing a Hermitian cost matrix,
    computing its eigenvalues and eigenvectors, and generating reduced density matrices for subsystems.

    Args:
        n (int): Local Hilbert space dimension.
        subsystems (tuple[int, ...]): Tuple specifying the number of subsystems.
        proportions (tuple[float, ...]): Coefficients for ground state mixture.
        key (jax.random.PRNGKey): JAX random key for reproducibility.

    Returns:
        tuple:
            - cost_matrix (jnp.ndarray): The Hermitian cost matrix.
            - ptraces (tuple[jnp.ndarray, ...]): Partial traces of the ground state.
            - gs_matrix (jnp.ndarray): The ground state density matrix.
            - E0 (float): The weighted sum of eigenvalues for the ground state.
            - dims (tuple[int, ...]): Dimensions of the composite system.
            - system_parts (tuple[tuple[int, ...], ...]): Tuple of subsystem indices.
    """

    D = d ** N
    np.random.seed(seed)
    cost_re = np.random.normal(loc=0, scale=np.eye(D), size=(D, D))
    np.random.seed(seed + 1)
    cost_im = np.random.normal(loc=0, scale=np.eye(D), size=(D, D))
    cost_matrix = cost_re + 1j * cost_im
    cost_matrix = (cost_matrix + cost_matrix.T.conj()) / 2

    # Compute eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(cost_matrix)

    # Compute ground state energy and density matrix
    gamma = sum(p * np.outer(v, v.conj()) for p, v in zip(proportions, evecs.T.conj()))
    marginals = compute_ptraces(ctx, gamma, d=d, N=N)

    # Convert according to the input backend
    cost_matrix = ctx.asarray(cost_matrix)
    gamma = ctx.asarray(gamma)
    marginals = ctx.asarray(marginals)

    return cost_matrix, marginals, gamma
