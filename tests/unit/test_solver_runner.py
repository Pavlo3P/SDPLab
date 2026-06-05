from __future__ import annotations

import io
import logging
import time

from sdplab.solvers import OptimizeResult, run_solver


def test_run_solver_uses_step_function_until_gradient_tolerance():
    """The generic loop owns convergence, history, and finalization."""
    calls = []

    def step_fn(state):
        next_state = state + 1
        loss = 1.0 / next_state
        grad_norm = 0.1 / next_state
        calls.append(next_state)
        return next_state, loss, grad_norm

    result = run_solver(
        init_state=0,
        step_fn=step_fn,
        finalize_fn=lambda state: f"final:{state}",
        max_iter=10,
        tol=0.026,
        verbose=0,
    )

    assert isinstance(result, OptimizeResult)
    assert result.dual == "final:4"
    assert result.converged is True
    assert result.num_iters == 4
    assert result.loss_history == [1.0, 0.5, 1 / 3, 0.25]
    assert result.grad_norm_history == [0.1, 0.05, 0.1 / 3, 0.025]
    assert len(result.step_times) == 4
    assert calls == [1, 2, 3, 4]


def test_run_solver_reports_max_iter_when_not_converged():
    result = run_solver(
        init_state=0,
        step_fn=lambda state: (state + 1, 1.0, 1.0),
        finalize_fn=lambda state: state,
        max_iter=3,
        tol=0.5,
        verbose=0,
        record_history=False,
    )

    assert result.dual == 3
    assert result.converged is False
    assert result.num_iters == 3
    assert result.loss_history is None
    assert result.grad_norm_history is None
    assert result.step_times is None


def test_run_solver_verbose_zero_produces_no_output(capsys):
    run_solver(
        init_state=0,
        step_fn=lambda state: (state + 1, 1.0, 0.0),
        finalize_fn=lambda state: state,
        max_iter=1,
        tol=0.5,
        verbose=0,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_run_solver_progress_callback_tracks_iterations():
    calls = []

    run_solver(
        init_state=0,
        step_fn=lambda state: (state + 1, 1.0 / (state + 1), 1.0),
        finalize_fn=lambda state: state,
        max_iter=3,
        tol=0.5,
        verbose=0,
        progress_callback=lambda it, loss, grad: calls.append((it, loss, grad)),
    )

    assert calls == [(0, 1.0, 1.0), (1, 0.5, 1.0), (2, 1 / 3, 1.0)]


def test_run_solver_step_time_includes_block_until_ready():
    class AsyncScalar:
        def __init__(self, value):
            self.value = value

        def block_until_ready(self):
            time.sleep(0.01)
            return self

        def item(self):
            return self.value

    result = run_solver(
        init_state=0,
        step_fn=lambda state: (state + 1, AsyncScalar(1.0), AsyncScalar(0.0)),
        finalize_fn=lambda state: state,
        max_iter=1,
        tol=0.5,
        verbose=0,
    )

    assert result.step_times[0] >= 0.01


def test_run_solver_verbose_uses_provided_logger():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("sdplab.tests.provided_solver")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    try:
        run_solver(
            init_state=0,
            step_fn=lambda state: (state + 1, 1.0, 0.0),
            finalize_fn=lambda state: state,
            max_iter=1,
            tol=0.5,
            verbose=3,
            solver_name="dummy",
            problem_summary="unit test",
            log=logger,
        )

        text = stream.getvalue()
        assert "Solver: dummy" in text
        assert "Problem: unit test" in text
        assert (
            "    iter |           loss |      grad_norm |   step_ms |     d_loss | median_ms | elapsed_s"
            in text
        )
        assert "       0 |   1.000000e+00 |   0.000000e+00" in text
        assert "[CONVERGED]" in text
    finally:
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


def test_run_solver_verbose_is_visible_when_logging_unconfigured(capsys):
    logger = logging.getLogger("sdplab.tests.unconfigured_solver")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.NOTSET)

    try:
        run_solver(
            init_state=0,
            step_fn=lambda state: (state + 1, 1.0, 0.0),
            finalize_fn=lambda state: state,
            max_iter=1,
            tol=0.5,
            verbose=1,
            solver_name="visible",
            log=logger,
        )

        captured = capsys.readouterr()
        assert "Solver: visible" in captured.out
        assert "[CONVERGED]" in captured.out
    finally:
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


def test_run_solver_verbose_two_keeps_statistics_in_table():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("sdplab.tests.verbose_two_table")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    try:
        run_solver(
            init_state=0,
            step_fn=lambda state: (state + 1, 1.0 / (state + 1), 1.0),
            finalize_fn=lambda state: state,
            max_iter=65,
            tol=0.5,
            verbose=2,
            log_every=1,
            solver_name="table-stats",
            log=logger,
        )

        text = stream.getvalue()
        assert "median_ms" in text
        assert "elapsed_s" in text
        assert "median step time" not in text
        assert "       0 |" in text
        assert "      64 |" in text
        assert "       1 |" not in text
    finally:
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


def test_run_solver_verbose_four_uses_boxed_unicode_output():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("sdplab.tests.fancy_solver")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    try:
        run_solver(
            init_state=0,
            step_fn=lambda state: (state + 1, 1.0 / (state + 1), 0.1 / (state + 1)),
            finalize_fn=lambda state: state,
            max_iter=3,
            tol=1e-9,
            verbose=4,
            log_every=1,
            solver_name="fancy",
            problem_summary="SDPRegularized with QuadraticReg(eps=1.0)",
            initial_dual_norm=0.0,
            log=logger,
            color=False,
        )

        text = stream.getvalue()
        assert "╔" in text
        assert "║ Solver: fancy" in text
        assert "QuadraticReg(ε=1.0)" in text
        assert "Initial state:" in text
        assert "‖∇‖ <" in text
        assert "Δloss" in text
        assert "median_ms" in text
        assert "elapsed_s" in text
        assert "Trajectory summary:" in text
        assert "Loss trajectory:" in text
        assert "█" in text
        assert "╚" in text
        assert "Progress:" not in text
    finally:
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


def test_run_solver_verbose_four_supports_ascii_output():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("sdplab.tests.fancy_ascii_solver")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    try:
        run_solver(
            init_state=0,
            step_fn=lambda state: (state + 1, 1.0, 0.0),
            finalize_fn=lambda state: state,
            max_iter=1,
            tol=0.5,
            verbose=4,
            solver_name="ascii",
            problem_summary="SDPRegularized with QuadraticReg(eps=1.0)",
            ascii_only=True,
            log=logger,
        )

        text = stream.getvalue()
        assert "+" in text
        assert "| Solver: ascii" in text
        assert "QuadraticReg(eps=1.0)" in text
        assert "grad_norm <" in text
        assert "d_loss" in text
        assert "median_ms" in text
        assert "elapsed_s" in text
        assert "Progress:" not in text
        assert "╔" not in text
        assert "Δ" not in text
    finally:
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


def test_run_solver_verbose_four_keeps_statistics_in_table_rows():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("sdplab.tests.fancy_table_stats")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    try:
        run_solver(
            init_state=0,
            step_fn=lambda state: (state + 1, 1.0, 1.0),
            finalize_fn=lambda state: state,
            max_iter=3,
            tol=0.5,
            verbose=4,
            log_every=1,
            solver_name="table-stats",
            log=logger,
        )

        text = stream.getvalue()
        assert "median_ms" in text
        assert "elapsed_s" in text
        assert "Progress:" not in text
        assert "\033[" not in text
        assert "     0 │" in text
        assert "     1 │" in text
        assert "     2 │" in text
    finally:
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)
