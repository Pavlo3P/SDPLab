"""Tests for the per-constraint CVXPY backend.

The backend asks the problem's ``ConstraintOp`` for a list of per-constraint
matrices ``[A_0, ..., A_{m-1}]`` and its ``Cost`` for the cost matrix ``C``, and
assembles the SDP as ``Re Tr[A_i X] == b_i`` with objective ``Re Tr[C X]`` over a
Hermitian/symmetric cone variable ``X >> 0``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from spacecore import Context, LinOp, NumpyOps, SparseLinOp

from sdplab.examples import generate_max_cut, generate_random_qot
from sdplab.problem import (
    SDPProblem,
    ConstraintOp,
    DenseConstraintOp,
    SparseConstraintOp,
    WrappedConstraintOp,
    ElementCost,
)
from sdplab.solvers import run_cvxpy_solver
from sdplab.special.qot import QOTConstraintOp, solve_qot_dual


def _assert_optimal(sdp, X, y, tol=1e-6):
    """Strong duality, dual feasibility, and primal feasibility."""
    p = float(sdp.primal_objective(X))
    d = float(sdp.dual_objective(y))
    assert abs(p - d) < 1e-5 * max(1.0, abs(p))
    slack = np.asarray(sdp.dual_slack(y))
    if slack.ndim == 2:
        assert np.linalg.eigvalsh(slack).max() < tol
    else:  # stacked blocks
        assert max(np.linalg.eigvalsh(s).max() for s in slack) < tol
    assert float(sdp.cod.norm(sdp.feasibility_gap(X))) < tol
    return p


# ---------------------------------------------------------------------------
# End-to-end solves
# ---------------------------------------------------------------------------


def test_real_maxcut_matches_reference():
    sdp = generate_max_cut(8, p=0.5, seed=1)
    # A plain matrix-free LinOp is wrapped by delegation and stays matrix-free.
    assert isinstance(sdp.A, WrappedConstraintOp)

    X, y = run_cvxpy_solver(sdp, solver="CLARABEL")
    p = _assert_optimal(sdp, X, y)

    reference, _ = run_cvxpy_solver(sdp, solver="CLARABEL")
    assert np.isclose(p, float(sdp.primal_objective(reference)), atol=1e-7)


def test_complex_qot_structured_codomain():
    """Complex Hermitian domain with a stacked (2,2,2) Hermitian codomain."""
    qot, _ = generate_random_qot(2, 2, (0.7, 0.3), seed=0)
    assert isinstance(qot.A, ConstraintOp)

    X, y = run_cvxpy_solver(qot, solver="CLARABEL")

    assert np.asarray(X).shape == (4, 4)
    assert np.asarray(y).shape == (2, 2, 2)
    _assert_optimal(qot, X, y, tol=1e-5)

    # Primal and dual respect the Hermitian structure.
    Xb, yb = np.asarray(X), np.asarray(y)
    assert np.allclose(Xb, Xb.conj().T)
    assert np.allclose(yb, np.conj(np.swapaxes(yb, -1, -2)))


def test_qot_generic_backend_matches_dedicated_dual_solver():
    """The generic per-constraint path reproduces the bespoke QOT dual solver."""
    qot, _ = generate_random_qot(2, 2, (0.6, 0.4), seed=3)
    X, y = run_cvxpy_solver(qot, solver="CLARABEL")
    prim_ref, dual_ref = solve_qot_dual(qot, solver="CLARABEL")

    assert np.isclose(
        float(qot.primal_objective(X)),
        float(qot.primal_objective(prim_ref)),
        atol=1e-5,
    )
    assert np.allclose(np.asarray(y), np.asarray(dual_ref), atol=1e-3)


def test_sparse_constraint_op_matches_dense():
    """A SparseLinOp constraint dispatches to SparseConstraintOp and agrees."""
    ctx = Context(NumpyOps(), dtype="float64")
    dense = generate_max_cut(6, p=0.5, seed=2).convert(ctx)

    T_mat = np.asarray(dense.A.to_matrix())  # (m, n*n)
    S = ctx.assparse(sps.csr_matrix(T_mat))
    sparse_op = SparseLinOp(S, dense.A.dom, dense.A.cod, ctx)
    sparse_sdp = SDPProblem(dense.C.to_dense(), sparse_op, dense.b, ctx=ctx)

    assert isinstance(sparse_sdp.A, SparseConstraintOp)

    Xs, ys = run_cvxpy_solver(sparse_sdp, solver="CLARABEL")
    _assert_optimal(sparse_sdp, Xs, ys)

    Xd, _ = run_cvxpy_solver(dense, solver="CLARABEL")
    assert np.isclose(
        float(sparse_sdp.primal_objective(Xs)),
        float(dense.primal_objective(Xd)),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# The per-constraint contract
# ---------------------------------------------------------------------------


def test_to_cvxpy_trace_convention_matches_apply():
    """Tr[A_i @ X] equals the operator's i-th output on a random Hermitian X."""
    qot, state = generate_random_qot(2, 2, (0.7, 0.3), seed=1)
    op = qot.A
    A_list = op.to_cvxpy()
    b = np.asarray(op.rhs_to_cvxpy(np.asarray(qot.b)))

    Gamma = np.asarray(state)
    for A_i, b_i in zip(A_list, b):
        val = np.trace(np.asarray(A_i.todense()) @ Gamma)
        assert np.isclose(val.real, b_i, atol=1e-9)
        assert abs(val.imag) < 1e-9


def test_dense_constraint_op_trace_convention():
    sdp = generate_max_cut(5, seed=4)
    op = sdp.A
    A_list = op.to_cvxpy()
    assert len(A_list) == op.cod.size

    rng = np.random.default_rng(0)
    Xm = rng.standard_normal((5, 5))
    Xm = (Xm + Xm.T) / 2
    applied = np.asarray(op.apply(sdp.ctx.asarray(Xm)))
    for i, A_i in enumerate(A_list):
        assert np.isclose(np.trace(np.asarray(A_i) @ Xm), applied[i], atol=1e-10)


def test_complex_hermitian_generic_constraints_trace_convention():
    """A complex-Hermitian (non-QOT) DenseConstraintOp obeys Tr[A_i X] == apply.

    Guards the transpose (not conjugate-transpose) orientation of to_cvxpy on
    genuinely complex constraints: for a Hermitian slice, ``A_i = T_i.T`` equals
    ``conj(T_i)`` and is correct, whereas ``T_i.conj().T == T_i`` would break the
    identity. Only QOT otherwise exercises complex constraints, and it overrides
    to_cvxpy, so the generic path would be unguarded.
    """
    from spacecore import DenseLinOp, DenseVectorSpace, HermitianSpace

    ctx = Context(NumpyOps(), dtype="complex128")
    n, m = 3, 4
    dom = HermitianSpace(n, ctx=ctx)
    cod = DenseVectorSpace((m,), ctx=ctx)
    rng = np.random.default_rng(11)

    # Genuinely complex Hermitian constraint matrices (nonzero imaginary parts).
    A_true = []
    for _ in range(m):
        M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        A_true.append((M + M.conj().T) / 2)
    A_true = np.stack(A_true)
    assert np.abs(A_true.imag).max() > 0.1

    # Store T_i = A_i^T so apply(X)_i = sum T_i[jk] X[jk] = Tr[A_i X].
    T = np.stack([A_true[i].T for i in range(m)])
    op = DenseConstraintOp.from_linop(DenseLinOp(ctx.asarray(T), dom, cod, ctx))

    # to_cvxpy recovers exactly the true Hermitian matrices.
    A_list = op.to_cvxpy()
    for i in range(m):
        assert np.allclose(np.asarray(A_list[i]), A_true[i], atol=1e-12)

    # Feasible instance, solved through the generic backend.
    X0 = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    X0 = X0 @ X0.conj().T  # PSD Hermitian
    b = np.array([np.real(np.trace(A_true[i] @ X0)) for i in range(m)])
    sdp = SDPProblem(np.eye(n, dtype=complex), op, ctx.asarray(b), ctx=ctx)

    X, y = run_cvxpy_solver(sdp, solver="CLARABEL")
    Xb = np.asarray(X)
    feas = max(abs(np.real(np.trace(A_true[i] @ Xb)) - b[i]) for i in range(m))
    assert feas < 1e-7
    _assert_optimal(sdp, X, y)


def test_rhs_and_dual_round_trip_plain():
    """For a plain scalar codomain, rhs/dual conversions are flatten/unflatten."""
    sdp = generate_max_cut(4, seed=0)
    op = sdp.A
    b = np.asarray(sdp.b, dtype=float)
    r = np.asarray(op.rhs_to_cvxpy(sdp.b))
    assert np.allclose(r, b)
    back = np.asarray(op.dual_from_cvxpy(r))
    assert np.allclose(back, b)


def test_matrix_free_default_to_cvxpy_trace_convention():
    """The MatrixFreeConstraintOp default (A_i = adjoint(e_i)) obeys the trace convention.

    Guards the base materialization path, which QOTConstraintOp overrides and so
    the suite would otherwise never exercise.
    """
    from spacecore import DenseVectorSpace, HermitianSpace, checked_method

    from sdplab.problem import MatrixFreeConstraintOp

    class DiagConstraintOp(MatrixFreeConstraintOp):
        """(A X)_i = X_ii, adjoint(y) = diag(y) -- a matrix-free MaxCut operator."""

        @checked_method(in_space="domain", out_space="codomain")
        def apply(self, X):
            return self.ops.diag(X)

        @checked_method(in_space="codomain", out_space="domain")
        def rapply(self, y):
            return self.ops.diag(y)

        def tree_flatten(self):
            return (), (self.dom, self.cod, self.ctx)

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls(*aux)

    ctx = Context(NumpyOps(), dtype="float64")
    n = 4
    op = DiagConstraintOp(HermitianSpace(n, ctx=ctx), DenseVectorSpace((n,), ctx=ctx), ctx)

    A_list = op.to_cvxpy()
    assert len(A_list) == n

    rng = np.random.default_rng(1)
    Xm = rng.standard_normal((n, n))
    Xm = (Xm + Xm.T) / 2
    applied = np.asarray(op.apply(ctx.asarray(Xm)))
    for i, A_i in enumerate(A_list):
        assert np.isclose(np.trace(np.asarray(A_i) @ Xm), applied[i], atol=1e-12)


def test_wrapped_constraint_op_delegates_and_matches_dense():
    """A plain LinOp is wrapped by delegation and agrees with the densified path."""
    ctx = Context(NumpyOps(), dtype="float64")
    dense = generate_max_cut(6, p=0.5, seed=4).convert(ctx)
    dom, cod = dense.A.dom, dense.A.cod

    class DiagLinOp(LinOp):
        """(A X)_i = X_ii, adjoint(y) = diag(y), with no stored tensor."""

        def apply(self, X):
            return self.ops.diag(X)

        def rapply(self, y):
            return self.ops.diag(y)

        def tree_flatten(self):
            return (), (self.dom, self.cod, self.ctx)

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls(*aux)

        def _convert(self, new_ctx):
            return DiagLinOp(self.dom.convert(new_ctx), self.cod.convert(new_ctx), new_ctx)

    op = DiagLinOp(dom, cod, ctx)
    sdp = SDPProblem(dense.C.to_dense(), op, dense.b, ctx=ctx)

    # Dispatched to the wrapper, not densified; apply/rapply delegate unchanged.
    assert isinstance(sdp.A, WrappedConstraintOp)
    assert sdp.A.op is not None
    rng = np.random.default_rng(5)
    Xm = rng.standard_normal((6, 6))
    Xm = ctx.asarray((Xm + Xm.T) / 2)
    assert np.allclose(np.asarray(sdp.A.apply(Xm)), np.asarray(op.apply(Xm)))
    assert np.allclose(np.asarray(sdp.A.rapply(sdp.b)), np.asarray(op.rapply(sdp.b)))

    # The lazily materialized A_i obey the trace convention.
    applied = np.asarray(op.apply(Xm))
    for i, A_i in enumerate(sdp.A.to_cvxpy()):
        assert np.isclose(np.trace(np.asarray(A_i) @ np.asarray(Xm)), applied[i], atol=1e-12)

    # Same solution as the equivalent stored-tensor problem.
    X, y = run_cvxpy_solver(sdp, solver="CLARABEL")
    _assert_optimal(sdp, X, y)
    Xd, _ = run_cvxpy_solver(dense, solver="CLARABEL")
    assert np.isclose(
        float(sdp.primal_objective(X)), float(dense.primal_objective(Xd)), atol=1e-6
    )

    # Context conversion keeps the wrapper and its operand.
    same = sdp.A.convert(ctx)
    assert isinstance(same, WrappedConstraintOp)


def test_element_cost_has_no_cvxpy_form():
    ctx = Context(NumpyOps(), dtype="float64")
    from spacecore import ElementwiseJordanSpace

    space = ElementwiseJordanSpace((3,), ctx=ctx)
    cost = ElementCost(ctx.asarray([1.0, 2.0, 3.0]), space, ctx)
    with pytest.raises(NotImplementedError):
        cost.to_cvxpy()


# ---------------------------------------------------------------------------
# QOT with marginals taken from a ground state
# ---------------------------------------------------------------------------


def _qot_from_ground_state(d, N, seed):
    """Return ``(sdp, lambda_min)`` for marginals read off the ground state of C.

    Let psi be the ground state of a random Hermitian cost C on (C^d)^{⊗N} and
    set gamma_k = Tr^k[|psi><psi|]. Then Gamma = |psi><psi| is feasible and
    attains <C, Gamma> = lambda_min(C). Conversely every feasible Gamma is psd
    with unit trace (each marginal has trace 1), so <C, Gamma> >= lambda_min(C)
    Tr[Gamma] = lambda_min(C). The QOT optimum is therefore *exactly*
    lambda_min(C) -- a closed-form reference that needs no second solver.
    """
    ctx = Context(NumpyOps(), dtype=np.complex128, check_level="none")
    D = d ** N
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    C = (M + M.conj().T) / 2

    w, V = np.linalg.eigh(C)
    lam_min, psi = float(w[0]), V[:, 0]
    rho = np.outer(psi, psi.conj())

    op = QOTConstraintOp(d=d, N=N, ctx=ctx)
    marginals = op.apply(ctx.asarray(rho))
    sdp = SDPProblem(ctx.asarray(C), op, marginals, ctx=ctx)
    return sdp, lam_min


@pytest.mark.parametrize("d, N, seed", [(2, 2, 0), (2, 3, 1), (3, 2, 2)])
def test_qot_ground_state_marginals_attain_lambda_min(d, N, seed):
    """The generic backend reproduces the closed-form optimum lambda_min(C)."""
    pytest.importorskip("cvxpy")
    sdp, lam_min = _qot_from_ground_state(d, N, seed)

    X, y = run_cvxpy_solver(sdp, solver="SCS")
    X = np.asarray(X)

    assert float(np.real(np.trace(X))) == pytest.approx(1.0, abs=1e-4)
    np.testing.assert_allclose(np.asarray(sdp.A.apply(X)), np.asarray(sdp.b), atol=1e-4)
    assert float(np.real(sdp.primal_objective(X))) == pytest.approx(lam_min, abs=1e-4)


@pytest.mark.parametrize("d, N, seed", [(2, 2, 0), (2, 3, 1), (3, 2, 2)])
def test_qot_dedicated_dual_attains_lambda_min(d, N, seed):
    """The dedicated QOT dual solver should reach the same lambda_min(C)."""
    pytest.importorskip("cvxpy")
    sdp, lam_min = _qot_from_ground_state(d, N, seed)

    X, U = solve_qot_dual(sdp, solver="SCS")
    X = np.asarray(X)

    assert float(np.real(np.trace(X))) == pytest.approx(1.0, abs=1e-4)
    assert float(np.real(sdp.primal_objective(X))) == pytest.approx(lam_min, abs=1e-4)
