import jax.numpy as jnp

from qotlib.sdp import SDPDenseProblem, SDPDual, SDPPrimal


def make_qubit_tomography_sdp(
        A: jnp.ndarray,
        b_obs: jnp.ndarray
) -> SDPDenseProblem:
    """
    Build the SDPProblem for single‐qubit tomography and
    return also the initial (primal, dual) guesses.
    """
    # dimension (should be 2 for a qubit)
    d = A.shape[1]
    # stack your M_i into A
    # cost = 0 for pure feasibility
    C = jnp.zeros((d, d))
    # tau=1 to enforce Tr[X]=1
    tau = 1.0

    # 1) the SDPProblem
    # tomography_sdp = SDPProblem(C=C, A=A, b=b_obs, tau=tau)
    # C, A, b_obs = convert_complex_sdp_to_real_sdp(C, A, b_obs)
    tomography_sdp = SDPDenseProblem(C, A, b_obs, tau)

    # 2) start primal at the maximally mixed state
    # primal0 = SDPPrimal(jnp.eye(d) * (1.0/d))

    # 3) one dual‐variable per M_i, plus one for the trace constraint
    # dual0 = SDPDual(jnp.zeros((A.shape[0] + 1,)))

    return tomography_sdp