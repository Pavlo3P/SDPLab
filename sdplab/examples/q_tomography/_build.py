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

Here :math:`\mathcal{A}: \operatorname{Herm}(d) \to \mathbb{R}^m` is the
measurement map :math:`(\mathcal{A}X)_i = \operatorname{Tr}[M_i X]` and
:math:`b_{\mathrm{obs}}` stores the observed expectation values. The unit-trace
constraint is appended as one extra row measuring the identity, so the returned
object is a plain :class:`~sdplab.problem.SDPProblem`.

The measurement map is a dense tensor contraction and nothing more, so it is a
:class:`~sdplab.problem.DenseConstraintOp` rather than a hand-written operator
-- see the transpose note on :func:`generate_qubit_tomography`.
"""

from __future__ import annotations

import numpy as np
from spacecore import Context, DenseArray, DenseVectorSpace, HermitianSpace, NumpyOps

from ...problem import DenseConstraintOp, SDPProblem


def generate_qubit_tomography(
    M: DenseArray,
    b_obs: DenseArray,
    atol: float = 0.0,
    rtol: float = 0.0,
    enforce_herm: bool = True,
    ctx: Context | str | None = None,
) -> SDPProblem:
    r"""Build the tomography feasibility SDP for observables ``M``.

    The unknown is a density matrix :math:`X \in \operatorname{Herm}(d)` and
    the returned problem is

    .. math::

        \min_{X \in \operatorname{Herm}(d)}\quad 0
        \quad \text{s.t.} \quad
        \operatorname{Tr}[M_i X] = b_i,\quad
        \operatorname{Tr}[X] = 1,\quad
        X \succeq 0.

    The trace-one constraint is appended as one extra row measuring the
    identity, so the constraint operator has ``m + 1`` rows.

    **Mind the transpose.** A :class:`~sdplab.problem.DenseConstraintOp` pairs
    its tensor with :math:`X` by Frobenius,
    :math:`(\mathcal{A}X)_i = \sum_{pq} T_{i,pq} X_{pq} =
    \operatorname{Tr}[T_i^{T} X]`. Handing it the observables directly would
    therefore measure :math:`\operatorname{Tr}[M_i^{T} X]` -- for a Hermitian
    observable the *conjugate* of the intended value, silently flipping the
    sign of every measurement with a nonzero imaginary part (the Pauli
    :math:`Y`, say). The stack is transposed here so that
    :math:`(\mathcal{A}X)_i = \operatorname{Tr}[M_i X]`, which also makes the
    operator's cvxpy encoding come out as the :math:`M_i` themselves.

    Args:
        M: Stack of Hermitian observables, shape ``(m, d, d)``.
        b_obs: Observed expectation values :math:`b_i = \operatorname{Tr}[M_i X]`,
            length ``m``. Taken as real -- an imaginary part would not be
            attainable by a Hermitian observable on a Hermitian state.
        atol: Absolute Hermitian membership tolerance.
        rtol: Relative Hermitian membership tolerance.
        enforce_herm: Whether the primal domain enforces Hermitian matrices.
        ctx: Optional backend context for the returned problem. The problem is
            assembled on NumPy (in a complex dtype when ``M`` is complex) and
            converted, matching :func:`~sdplab.examples.generate_max_cut`.

    Returns:
        Feasibility :class:`~sdplab.problem.SDPProblem` with a zero cost.

    Raises:
        ValueError: If ``M`` is not a stack of square matrices, if ``b_obs``
            does not match its length, or if any observable is not Hermitian.
    """
    M = np.asarray(M)
    b_obs = np.asarray(b_obs).reshape(-1)

    if M.ndim != 3 or M.shape[-1] != M.shape[-2]:
        raise ValueError(f"M must have shape (m, d, d); got {M.shape}.")
    if b_obs.shape[0] != M.shape[0]:
        raise ValueError(
            f"b_obs has length {b_obs.shape[0]}, but M has {M.shape[0]} observables."
        )
    if not np.allclose(M, np.conj(np.swapaxes(M, -1, -2)), atol=1e-10):
        raise ValueError("every observable M[i] must be Hermitian.")

    m, d = M.shape[0], M.shape[-1]
    dtype = "complex128" if np.iscomplexobj(M) else "float64"
    np_ctx = Context(NumpyOps(), dtype=dtype, check_level="none")

    # Append Tr[X] = 1 as one extra measurement of the identity.
    M_full = np.concatenate([M.astype(dtype), np.eye(d, dtype=dtype)[None]], axis=0)
    b_full = np.concatenate([b_obs.real.astype("float64"), [1.0]])

    dom = HermitianSpace(d, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=np_ctx)
    cod = DenseVectorSpace((m + 1,), ctx=np_ctx)
    A = DenseConstraintOp(np.swapaxes(M_full, -1, -2), dom, cod, np_ctx)

    sdp = SDPProblem(dom.zeros(), A, b_full, ctx=np_ctx)   # zero cost: pure feasibility

    if ctx is not None:
        sdp = sdp.convert(ctx)
    return sdp
