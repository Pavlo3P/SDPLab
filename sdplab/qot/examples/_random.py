from typing import Optional, Tuple
import numpy as np

from spacecore import Context, NumpyOps
from .._constraint_op import QOTConstraintOp
from ...sdp import SDPDenseProblem, SDPPrimal


def generate_random_qot(
        d: int, N: int,
        proportions: tuple[float, ...],
        seed: Optional[int] = 0,
        atol: float = 0.0,
        rtol: float = 0.0,
        enforce_herm: bool = True,
        ctx: Context | str | None = None
) -> Tuple[SDPDenseProblem, SDPPrimal]:
    """
    Generate a random dense QOT instance together with a feasible primal state.

    This function samples a random Hermitian cost matrix on
    $$
    (\mathbb{C}^d)^{\otimes N},
    $$
    builds a reference density matrix as a convex combination of eigenvector projectors,
    computes its one-body marginals through ``QOTConstraintOp``, and returns the
    corresponding dense SDP problem plus the same state wrapped as an ``SDPPrimal``.

    The generated primal state is feasible for the constructed constraint data by design,
    since the marginals are obtained by applying the constraint operator to that state.

    Args:
        d: Local Hilbert space dimension. The full state space is
            $$
            (\mathbb{C}^d)^{\otimes N},
            $$
            so the ambient matrix size is
            $$
            D = d^N.
            $$
        N: Number of subsystems.
        proportions: Coefficients used to form the mixed state
            $$
            \gamma = \sum_i p_i \, |v_i\rangle \langle v_i|,
            $$
            where the $$v_i$$ are eigenvectors of the sampled cost matrix.
            This is intended to be a convex combination, so typically the entries
            should be nonnegative and sum to $$1$$.
        seed: Random seed used for NumPy sampling.
        atol: Absolute tolerance passed to ``QOTConstraintOp``.
        rtol: Relative tolerance passed to ``QOTConstraintOp``.
        enforce_herm: Whether the constraint operator should enforce Hermitian outputs.
        ctx: Target context for the returned SDP problem. If provided, the generated
            problem is converted to this context before returning.

    Returns:
        A pair ``(qot, state)`` where:

        - ``qot`` is an ``SDPDenseProblem`` representing the dense QOT SDP with the
          sampled Hermitian cost matrix and marginals induced by ``gamma``.
        - ``state`` is an ``SDPPrimal`` containing the same density matrix used to
          define the marginals, now represented in the returned problem's context.

    Notes:
        - The random cost matrix is sampled entrywise in real and imaginary parts and
          then symmetrized to be Hermitian.
        - The density matrix ``gamma`` is built in the NumPy complex context
          ``Context(NumpyOps(), dtype=np.complex128)`` before optional conversion.
    """

    D = d ** N
    np.random.seed(seed)
    cost_re_im = np.random.normal(loc=0, scale=1, size=(2, D, D))
    cost_matrix = cost_re_im[0, :, :] + 1j * cost_re_im[1, :, :]
    cost_matrix = (cost_matrix + cost_matrix.T.conj()) / 2

    # Compute eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(cost_matrix)

    # Compute ground state energy and density matrix
    np_ctx = Context(NumpyOps(), dtype=np.complex128)
    gamma = sum(p * np.outer(v, v.conj()) for p, v in zip(proportions, evecs.T.conj()))
    qot_op = QOTConstraintOp(d=d, N=N, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=np_ctx)
    marginals = qot_op.apply(gamma)

    # Define QOT problem & convert to target ctx
    qot = SDPDenseProblem(cost_matrix, qot_op, marginals, np_ctx)
    qot = qot.convert(ctx)

    # Wrap example state into SDPPrimal
    state = qot.primal_from_array(gamma)

    return qot, state
