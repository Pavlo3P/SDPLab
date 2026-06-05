from __future__ import annotations

import numpy as np
import pytest

from sdplab.special.qot.barycenter import (
    GaussianPhaseSpace,
    GaussianState,
    QOTGaussianBarycenterProblem,
    QuadraticOperator,
    QuadraticOperatorTuple,
)
from sdplab.special.qot.barycenter._gaussian_calculus import quadratic_gibbs_state


def test_gaussian_state_expectation_and_uncertainty(np_ctx):
    space = GaussianPhaseSpace(1, ctx=np_ctx)

    with pytest.raises(ValueError, match="uncertainty"):
        GaussianState(space, [0.0, 0.0], 0.1 * np.eye(2), ctx=np_ctx)

    mean = np_ctx.asarray([1.0, 2.0])
    cov = np_ctx.asarray([[1.0, 0.25], [0.25, 1.5]])
    state = GaussianState(space, mean, cov, ctx=np_ctx)
    op = QuadraticOperator(
        space,
        3.0,
        np_ctx.asarray([4.0, 5.0]),
        np_ctx.asarray([[2.0, 0.5], [0.5, 6.0]]),
        ctx=np_ctx,
    )

    second = cov + np.outer(mean, mean)
    expected = 3.0 + np.dot([4.0, 5.0], mean) + 0.5 * np.trace(op.quadratic @ second)
    assert np.allclose(state.expect_quadratic(op), expected)


def test_gaussian_phase_space_hbar_scales_uncertainty(np_ctx):
    space = GaussianPhaseSpace(1, hbar=0.1, ctx=np_ctx)
    state = GaussianState(space, [0.0, 0.0], 0.1 * np.eye(2), ctx=np_ctx)

    assert space.hbar == 0.1
    assert np.allclose(state.cov, 0.1 * np.eye(2))

    classical_space = GaussianPhaseSpace(1, hbar=0.0, ctx=np_ctx)
    classical_state = GaussianState(classical_space, [0.0, 0.0], 0.1 * np.eye(2), ctx=np_ctx)
    assert classical_space.hbar == 0.0
    assert np.allclose(classical_state.cov, 0.1 * np.eye(2))

    with pytest.raises(ValueError, match="hbar"):
        GaussianPhaseSpace(1, hbar=-1e-3, ctx=np_ctx)

    with pytest.raises(ValueError, match="uncertainty"):
        GaussianState(GaussianPhaseSpace(1, hbar=0.5, ctx=np_ctx), [0.0, 0.0], 0.1 * np.eye(2), ctx=np_ctx)


def test_quadratic_gibbs_state_one_mode_thermal_formula(np_ctx):
    beta = 1.3
    space = GaussianPhaseSpace(1, ctx=np_ctx)
    op = QuadraticOperator(space, 0.0, np.zeros(2), -beta * np.eye(2), ctx=np_ctx)

    state = quadratic_gibbs_state(op)

    expected_trace = 1.0 / (2.0 * np.sinh(beta / 2.0))
    expected_cov = 0.5 / np.tanh(beta / 2.0) * np.eye(2)
    assert np.allclose(state.normalization, expected_trace)
    assert np.allclose(state.mean, np.zeros(2))
    assert np.allclose(state.cov, expected_cov)


def test_quadratic_gibbs_state_respects_hbar(np_ctx):
    beta = 1.3
    hbar = 0.2
    space = GaussianPhaseSpace(1, hbar=hbar, ctx=np_ctx)
    op = QuadraticOperator(space, 0.0, np.zeros(2), -beta * np.eye(2), ctx=np_ctx)

    state = quadratic_gibbs_state(op)

    expected_trace = 1.0 / (2.0 * np.sinh(hbar * beta / 2.0))
    expected_cov = 0.5 * hbar / np.tanh(hbar * beta / 2.0) * np.eye(2)
    assert np.allclose(state.normalization, expected_trace)
    assert np.allclose(state.cov, expected_cov)


def test_quadratic_gibbs_state_classical_hbar_zero(np_ctx):
    beta = 1.3
    space = GaussianPhaseSpace(1, hbar=0.0, ctx=np_ctx)
    op = QuadraticOperator(space, 0.0, np.zeros(2), -beta * np.eye(2), ctx=np_ctx)

    state = quadratic_gibbs_state(op)

    expected_trace = 2.0 * np.pi / beta
    expected_cov = np.eye(2) / beta
    assert np.allclose(state.normalization, expected_trace)
    assert np.allclose(state.cov, expected_cov)


def _make_problem(np_ctx):
    space0 = GaussianPhaseSpace(1, ctx=np_ctx)
    space = GaussianPhaseSpace(1, ctx=np_ctx)
    joint = GaussianPhaseSpace(2, ctx=np_ctx)
    sigma = [GaussianState(space, [0.1, -0.2], np.eye(2), ctx=np_ctx)]
    cost = QuadraticOperator(joint, 0.0, np.zeros(4), 4.0 * np.eye(4), ctx=np_ctx)
    problem = QOTGaussianBarycenterProblem(
        space0,
        space,
        sigma,
        cost,
        np_ctx.asarray([1.0]),
        epsilon=1.0,
        tau=1.0,
        ctx=np_ctx,
    )
    U = QuadraticOperatorTuple(
        space0,
        np_ctx.asarray([0.2]),
        np_ctx.asarray([[0.05, -0.03]]),
        np_ctx.asarray([np.eye(2)]),
        ctx=np_ctx,
    )
    V = QuadraticOperatorTuple(
        space,
        np_ctx.asarray([-0.1]),
        np_ctx.asarray([[0.02, 0.01]]),
        np_ctx.asarray([0.1 * np.eye(2)]),
        ctx=np_ctx,
    )
    return problem, U, V


def test_qot_gaussian_problem_rejects_mismatched_hbar(np_ctx):
    space0 = GaussianPhaseSpace(1, hbar=1.0, ctx=np_ctx)
    space = GaussianPhaseSpace(1, hbar=0.5, ctx=np_ctx)
    joint = GaussianPhaseSpace(2, hbar=1.0, ctx=np_ctx)
    sigma = [GaussianState(space, [0.1, -0.2], np.eye(2), ctx=np_ctx)]
    cost = QuadraticOperator(joint, 0.0, np.zeros(4), 4.0 * np.eye(4), ctx=np_ctx)

    with pytest.raises(ValueError, match="same hbar"):
        QOTGaussianBarycenterProblem(
            space0,
            space,
            sigma,
            cost,
            np_ctx.asarray([1.0]),
            epsilon=1.0,
            tau=1.0,
            ctx=np_ctx,
        )


def test_qot_gaussian_dual_gradient_constants_match_finite_difference(np_ctx):
    problem, U, V = _make_problem(np_ctx)
    gradients = problem.gradients(U, V)
    h = 1e-5

    U_plus = U.with_params(U.constants + np_ctx.asarray([h]), U.linears, U.quadratics)
    U_minus = U.with_params(U.constants - np_ctx.asarray([h]), U.linears, U.quadratics)
    dU = (problem.dual_objective(U_plus, V) - problem.dual_objective(U_minus, V)) / (2.0 * h)

    V_plus = V.with_params(V.constants + np_ctx.asarray([h]), V.linears, V.quadratics)
    V_minus = V.with_params(V.constants - np_ctx.asarray([h]), V.linears, V.quadratics)
    dV = (problem.dual_objective(U, V_plus) - problem.dual_objective(U, V_minus)) / (2.0 * h)

    assert gradients.grad_U_constants.shape == (1,)
    assert gradients.grad_U_linears.shape == (1, 2)
    assert gradients.grad_U_quadratics.shape == (1, 2, 2)
    assert gradients.grad_V_constants.shape == (1,)
    assert gradients.grad_V_linears.shape == (1, 2)
    assert gradients.grad_V_quadratics.shape == (1, 2, 2)
    assert np.allclose(gradients.grad_U_constants[0], dU, rtol=1e-5, atol=1e-6)
    assert np.allclose(gradients.grad_V_constants[0], dV, rtol=1e-5, atol=1e-6)


def test_qot_gaussian_dual_uses_exp_conjugate_without_minus_one(np_ctx):
    problem, U, _ = _make_problem(np_ctx)

    eta = problem.dual_barycenter_state(U)

    base_trace = 1.0 / (2.0 * np.sinh(0.5))
    expected = np.exp(-0.2 + 0.5 * np.dot([0.05, -0.03], [0.05, -0.03])) * base_trace
    assert np.allclose(eta.normalization, expected)


def test_qot_barycenter_objects_have_readable_reprs(np_ctx):
    problem, U, V = _make_problem(np_ctx)
    state = problem.sigma[0]
    op = U[0]
    gradients = problem.gradients(U, V)

    objects = [problem.space0, state, op, U, problem, gradients]
    for obj in objects:
        text = repr(obj)
        html = obj._repr_html_()
        assert type(obj).__name__ in text
        assert type(obj).__name__ in html
        assert "backend" in text


def test_qot_gaussian_hamiltonian_margins_diagnose_domain_errors(np_ctx):
    problem, U, V = _make_problem(np_ctx)

    coupling_margin, eta_margin = problem.hamiltonian_margins(U, V)
    assert float(coupling_margin) > 0.0
    assert float(eta_margin) > 0.0

    bad_U = U.with_params(
        U.constants,
        U.linears,
        np_ctx.asarray([-np.eye(2)]),
    )
    _, bad_eta_margin = problem.hamiltonian_margins(bad_U, V)
    assert float(bad_eta_margin) < 0.0

    with pytest.raises(ValueError, match=r"min eig\(H\)="):
        problem.dual_barycenter_state(bad_U)


def test_qot_gaussian_log_partition_gradient_constants_match_finite_difference(np_ctx):
    problem, U, V = _make_problem(np_ctx)
    problem = QOTGaussianBarycenterProblem(
        problem.space0,
        problem.space,
        problem.sigma,
        problem.cost,
        problem.alpha,
        problem.epsilon,
        problem.tau,
        use_log_partition=True,
        ctx=np_ctx,
    )
    gradients = problem.gradients(U, V)
    h = 1e-5

    U_plus = U.with_params(U.constants + np_ctx.asarray([h]), U.linears, U.quadratics)
    U_minus = U.with_params(U.constants - np_ctx.asarray([h]), U.linears, U.quadratics)
    dU = (problem.dual_objective(U_plus, V) - problem.dual_objective(U_minus, V)) / (2.0 * h)

    V_plus = V.with_params(V.constants + np_ctx.asarray([h]), V.linears, V.quadratics)
    V_minus = V.with_params(V.constants - np_ctx.asarray([h]), V.linears, V.quadratics)
    dV = (problem.dual_objective(U, V_plus) - problem.dual_objective(U, V_minus)) / (2.0 * h)

    assert np.allclose(problem.dual_state_couplings(U, V)[0].normalization, 1.0)
    assert np.allclose(problem.dual_barycenter_state(U).normalization, 1.0)
    assert np.allclose(gradients.grad_U_constants[0], dU, rtol=1e-5, atol=1e-6)
    assert np.allclose(gradients.grad_V_constants[0], dV, rtol=1e-5, atol=1e-6)
