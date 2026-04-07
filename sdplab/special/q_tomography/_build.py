from spacecore import DenseArray, Context, DenseLinOp, VectorSpace, HermitianSpace

from ...sdp import SDPDenseProblem


def make_qubit_tomography_sdp(
        A: DenseArray,
        b_obs: DenseArray,
        atol: float = 0.0,
        rtol: float = 0.0,
        enforce_herm: bool = True,
        ctx: Context | str | None = None
) -> SDPDenseProblem:
    """
    Build the SDPProblem for single‐qubit tomography and
    return also the initial (primal, dual) guesses.
    """
    n = A.shape[0]
    d = A.shape[1]

    dom = HermitianSpace(d, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=ctx)
    cod = VectorSpace((n,), ctx=ctx)
    A_op = DenseLinOp(A, dom, cod, ctx=ctx)

    # cost = 0 for pure feasibility
    C = dom.zeros()

    # tau=1 to enforce Tr[X]=1
    tau = 1.0

    # SDPProblem
    tomography_sdp = SDPDenseProblem(C, A_op, b_obs, tau, ctx=ctx)

    return tomography_sdp