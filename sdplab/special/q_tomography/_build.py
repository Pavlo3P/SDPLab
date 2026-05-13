r"""Builders for qubit tomography SDP feasibility problems.

Quantum tomography estimates an unknown density matrix
:math:`X \in \operatorname{Herm}(d)` from linear measurement constraints.
This module builds the feasibility SDP

.. math::

    \min_{X \in \operatorname{Herm}(d)}\quad 0
    \quad \text{s.t.} \quad
    \mathcal{A}X = b_{\mathrm{obs}},\quad
    \operatorname{Tr}[X] = 1,\quad
    X \succeq 0.

Here :math:`\mathcal{A}: \operatorname{Herm}(d) \to \mathbb{R}^n` is the
measurement map and :math:`b_{\mathrm{obs}} \in \mathbb{R}^n` stores the
observed expectation values.
"""

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
    r"""Build a dense SDP for qubit tomography constraints.

    The unknown is a density matrix
    :math:`X \in \operatorname{Herm}(d)`. The returned problem is

    .. math::

        \min_{X \in \operatorname{Herm}(d)}\quad 0
        \quad \text{s.t.} \quad
        \mathcal{A}X = b_{\mathrm{obs}},\quad
        \operatorname{Tr}[X] = 1,\quad
        X \succeq 0.

    The input array ``A`` is used to build a ``DenseLinOp`` representation of
    :math:`\mathcal{A}: \operatorname{Herm}(d) \to \mathbb{R}^n`. Each row is
    a linear measurement functional, typically
    :math:`X \mapsto \operatorname{Tr}[M_i X]`, and ``b_obs[i]`` is the
    measured expectation value.

    Args:
        A: Dense observation operator representing :math:`\mathcal{A}`.
        b_obs: Vector :math:`b_{\mathrm{obs}} \in \mathbb{R}^n` containing
            observed expectation values.
        atol: Absolute Hermitian membership tolerance.
        rtol: Relative Hermitian membership tolerance.
        enforce_herm: Whether the primal domain enforces Hermitian matrices.
        ctx: Optional backend context for spaces and arrays.

    Returns:
        Dense SDP with zero objective and trace-one primal constraint.
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
