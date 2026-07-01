r"""Random QOT problem generator.

Use this module when you want a self-contained QOT test problem. It creates a
random cost matrix, builds a valid primal density matrix, computes the
marginals from that density matrix, and returns an SDP whose constraints are
therefore feasible by construction.
"""

from typing import Optional, Tuple
import numpy as np

from spacecore import Context, NumpyOps
from ...special.qot import QOTConstraintOp
from ...problem import SDPProblem
from ...variables import SDPPrimal


def generate_random_qot(
        d: int, N: int,
        proportions: tuple[float, ...],
        seed: Optional[int] = 0,
        atol: float = 0.0,
        rtol: float = 0.0,
        enforce_herm: bool = True,
        ctx: Context | str | None = None
) -> Tuple[SDPProblem, SDPPrimal]:
    r"""
    Generate a random dense QOT instance together with a feasible primal state.

    The generated SDP has the form

    .. math::

        \min_\Gamma \quad \operatorname{Tr}[C \Gamma]
        \quad \text{s.t.} \quad
        \operatorname{Tr}^k[\Gamma] = \gamma_k,\quad
        k = 0, ..., N - 1,\quad \Gamma \succeq 0.

    The function first chooses a reference coupling ``Gamma`` and then defines
    :math:`\gamma_k = \operatorname{Tr}^k[\Gamma]`. Because the right-hand side is computed
    from ``Gamma``, the returned ``state`` is guaranteed to satisfy the
    equality constraints.

    This function samples a random Hermitian cost matrix on
    :math:`(\mathbb{C}^d)^{\otimes N}`,
    builds a reference density matrix as a convex combination of eigenvector projectors,
    computes its one-body marginals through ``QOTConstraintOp``, and returns the
    corresponding dense SDP problem plus the same state wrapped as an ``SDPPrimal``.

    The generated primal state is feasible for the constructed constraint data by design,
    since the marginals are obtained by applying the constraint operator to that state.

    Args:
        d: Local Hilbert space dimension. The full state space is
            :math:`(\mathbb{C}^d)^{\otimes N}`, so the ambient matrix size is
            :math:`D = d^N`.
        N: Number of subsystems.
        proportions: Coefficients used to form the mixed state
            :math:`\Gamma = \sum_i p_i \, |v_i\rangle \langle v_i|`,
            where the :math:`v_i` are eigenvectors of the sampled cost matrix.
            This is intended to be a convex combination, so typically the entries
            should be nonnegative and sum to :math:`1`.
        seed: Random seed used for NumPy sampling.
        atol: Absolute tolerance passed to ``QOTConstraintOp``.
        rtol: Relative tolerance passed to ``QOTConstraintOp``.
        enforce_herm: Whether the constraint operator should enforce Hermitian outputs.
        ctx: Target context for the returned SDP problem. If provided, the generated
            problem is converted to this context before returning.

    Returns:
        - ``qot`` is an ``SDPDenseProblem`` representing the dense QOT SDP with the
          sampled Hermitian cost matrix and marginals induced by ``Gamma``.
        - ``state`` is an ``SDPPrimal`` containing the same density matrix used to
          define the marginals, now represented in the returned problem's context.

    Notes:
        - The random cost matrix is sampled entrywise in real and imaginary parts and
          then symmetrized to be Hermitian.
        - The density matrix ``Gamma`` is built in the NumPy complex context
          ``Context(NumpyOps(), dtype=np.complex128)`` before optional conversion.
    """

    D = d ** N

    proportions = np.asarray(proportions, dtype=float)

    if len(proportions) > D:
        raise ValueError("len(proportions) cannot exceed D = d ** N")

    if np.any(proportions < 0):
        raise ValueError("proportions must be nonnegative")

    if not np.isclose(np.sum(proportions), 1.0):
        raise ValueError("proportions must sum to 1")

    rng = np.random.default_rng(seed)
    cost_re_im = rng.normal(size=(2, D, D))
    cost_matrix = cost_re_im[0, :, :] + 1j * cost_re_im[1, :, :]
    cost_matrix = (cost_matrix + cost_matrix.T.conj()) * .5

    # Compute eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(cost_matrix)

    # Compute ground state energy and density matrix
    np_ctx = Context(NumpyOps(), dtype=np.complex128, enable_checks=False)
    Gamma = sum(p * np.outer(v, v.conj()) for p, v in zip(proportions, evecs.T))
    Gamma = (Gamma + Gamma.T.conj()) * .5
    qot_op = QOTConstraintOp(d=d, N=N, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=np_ctx)
    marginals = qot_op.apply(Gamma)

    # Define QOT problem & convert to target ctx
    qot = SDPProblem(cost_matrix, qot_op, marginals, ctx=np_ctx)
    qot = qot.convert(ctx)

    # Wrap example state into SDPPrimal
    state = qot.primal_from_array(Gamma)

    return qot, state
