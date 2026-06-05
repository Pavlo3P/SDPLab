"""Generic gradient-solver loop and result record."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import re
import statistics
import sys
import time
from typing import Any, Callable

from ..sdp import SDPDual, SDPPrimal


logger = logging.getLogger(__name__)
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")
_VERBOSE_TWO_PRINT_EVERY = 64


@dataclass
class OptimizeResult:
    """Result returned by generic gradient-based dual solvers."""

    dual: SDPDual | None
    converged: bool
    num_iters: int
    final_loss: float
    final_grad_norm: float
    elapsed_seconds: float
    loss_history: list[float] | None = None
    grad_norm_history: list[float] | None = None
    step_times: list[float] | None = None
    primal: SDPPrimal | None = None

    def __repr__(self) -> str:
        status = "converged" if self.converged else "max_iter"
        return (
            f"OptimizeResult({status}, iters={self.num_iters}, "
            f"loss={self.final_loss:.4e}, "
            f"grad_norm={self.final_grad_norm:.4e}, "
            f"elapsed={self.elapsed_seconds:.2f}s)"
        )

    def summary(self) -> str:
        """Return a compact multi-line text summary."""
        status = "CONVERGED" if self.converged else "MAX_ITER REACHED"
        lines = [
            f"[{status}] iters={self.num_iters} "
            f"final_loss={self.final_loss:.6e} "
            f"final_grad_norm={self.final_grad_norm:.6e}",
            f"elapsed: {self.elapsed_seconds:.3f} s",
        ]
        if self.step_times:
            median = statistics.median(self.step_times) * 1000
            total_step = sum(self.step_times)
            overhead = self.elapsed_seconds - total_step
            overhead_pct = (
                100 * overhead / self.elapsed_seconds
                if self.elapsed_seconds > 0
                else 0.0
            )
            lines.append(
                f"median step: {median:.2f} ms, "
                f"total step time: {total_step:.3f} s"
            )
            lines.append(
                f"overhead: {overhead:.3f} s "
                f"({overhead_pct:.1f}% of wall time)"
            )
        return "\n".join(lines)


def run_solver(
    *,
    init_state: Any,
    step_fn: Callable[[Any], tuple[Any, Any, Any]],
    finalize_fn: Callable[[Any], SDPDual],
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: int = 1,
    log_every: int = 50,
    record_history: bool = True,
    progress_callback: Callable[[int, float, float], None] | None = None,
    solver_name: str = "solver",
    problem_summary: str | None = None,
    initial_dual_norm: float | None = None,
    initial_loss: float | None = None,
    initial_grad_norm: float | None = None,
    ascii_only: bool = False,
    color: bool | None = None,
    log: logging.Logger | None = None,
) -> OptimizeResult:
    """Run a generic gradient-based optimization loop.

    ``step_fn`` owns backend details. It receives the current state and returns
    ``(new_state, loss_value, grad_norm)``. The loop owns convergence,
    histories, progress callbacks, timing, and logging.
    """
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    log = logger if log is None else log
    verbose = int(verbose)
    if verbose >= 1:
        _ensure_info_logging(log)
    fancy = (
        _FancyPrinter(
            log=log,
            solver_name=solver_name,
            problem_summary=problem_summary,
            max_iter=max_iter,
            tol=tol,
            initial_dual_norm=initial_dual_norm,
            initial_loss=initial_loss,
            initial_grad_norm=initial_grad_norm,
            ascii_only=ascii_only,
            color=color,
        )
        if verbose >= 4
        else None
    )

    if fancy is not None:
        fancy.print_header()
    elif verbose >= 1:
        _print_header(
            log,
            solver_name,
            problem_summary,
            max_iter,
            tol,
            initial_dual_norm,
            initial_loss,
            initial_grad_norm,
        )
    if verbose >= 2 and fancy is None:
        _print_iter_table_header(log)

    losses = [] if record_history else None
    grad_norms = [] if record_history else None
    public_step_times = [] if record_history else None
    step_times: list[float] = []

    state = init_state
    loss_f = float("nan")
    grad_norm_f = float("inf")
    prev_loss: float | None = None
    converged = False
    it = -1
    overall_start = time.perf_counter()

    for it in range(max_iter):
        step_start = time.perf_counter()
        state, loss, grad_norm = step_fn(state)

        loss_f = _as_float(loss)
        grad_norm_f = _as_float(grad_norm)
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)

        if record_history:
            losses.append(loss_f)
            grad_norms.append(grad_norm_f)
            public_step_times.append(step_time)

        if progress_callback is not None:
            progress_callback(it, loss_f, grad_norm_f)

        if fancy is not None:
            fancy.print_iter(
                it,
                loss_f,
                grad_norm_f,
                step_time,
                prev_loss,
                step_times,
                overall_start,
            )
        elif verbose >= 3 or (
            verbose == 2 and it % _VERBOSE_TWO_PRINT_EVERY == 0
        ):
            _print_iter(
                log,
                it,
                loss_f,
                grad_norm_f,
                step_time,
                prev_loss,
                step_times,
                overall_start,
            )

        prev_loss = loss_f

        if grad_norm_f < tol:
            converged = True
            break

    elapsed = time.perf_counter() - overall_start
    result = OptimizeResult(
        dual=finalize_fn(state),
        converged=converged,
        num_iters=it + 1,
        final_loss=loss_f,
        final_grad_norm=grad_norm_f,
        elapsed_seconds=elapsed,
        loss_history=losses,
        grad_norm_history=grad_norms,
        step_times=public_step_times,
    )

    if fancy is not None:
        fancy.print_footer(result, step_times)
    elif verbose >= 1:
        _print_footer(log, result, step_times)

    return result


def _as_float(value: Any) -> float:
    """Convert scalar backend values to Python floats."""
    if hasattr(value, "block_until_ready"):
        value = value.block_until_ready()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _ensure_info_logging(log: logging.Logger) -> None:
    """Make verbose solver output visible when logging is unconfigured."""
    if log.getEffectiveLevel() > logging.INFO:
        log.setLevel(logging.INFO)

    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.INFO)
        handler._sdplab_solver_handler = True
        log.addHandler(handler)
        log.propagate = False


def _print_header(
    log: logging.Logger,
    name: str,
    summary: str | None,
    max_iter: int,
    tol: float,
    initial_dual_norm: float | None,
    initial_loss: float | None,
    initial_grad_norm: float | None,
) -> None:
    sep = "=" * 80
    log.info(sep)
    log.info("Solver: %s", name)
    if summary:
        log.info("Problem: %s", summary)
    if initial_dual_norm is not None:
        log.info("Initial dual norm: %.6e", float(initial_dual_norm))
    if initial_loss is not None:
        log.info("Initial loss: %.6e", float(initial_loss))
    if initial_grad_norm is not None:
        log.info("Initial gradient norm: %.6e", float(initial_grad_norm))
    log.info("Tolerance: grad_norm < %g", tol)
    log.info("Max iterations: %d", max_iter)
    log.info(sep)


def _print_iter_table_header(log: logging.Logger) -> None:
    log.info(
        "%8s | %14s | %14s | %9s | %10s | %9s | %9s",
        "iter",
        "loss",
        "grad_norm",
        "step_ms",
        "d_loss",
        "median_ms",
        "elapsed_s",
    )
    log.info("%s", "-" * 92)


def _print_iter(
    log: logging.Logger,
    it: int,
    loss: float,
    grad_norm: float,
    step_time: float,
    prev_loss: float | None,
    step_times: list[float],
    overall_start: float,
) -> None:
    d_loss = "" if prev_loss is None else f"{loss - prev_loss:+10.2e}"
    median_ms = statistics.median(step_times) * 1000
    elapsed = time.perf_counter() - overall_start
    log.info(
        "%8d | %14.6e | %14.6e | %9.2f | %10s | %9.2f | %9.3f",
        it,
        loss,
        grad_norm,
        step_time * 1000,
        d_loss,
        median_ms,
        elapsed,
    )


def _print_footer(
    log: logging.Logger,
    result: OptimizeResult,
    step_times: list[float],
) -> None:
    sep = "=" * 80
    log.info(sep)
    status = "CONVERGED" if result.converged else "MAX_ITER REACHED"
    log.info(
        "[%s] iters=%d  final_loss=%.6e  final_grad_norm=%.6e",
        status,
        result.num_iters,
        result.final_loss,
        result.final_grad_norm,
    )
    log.info("              elapsed: %.3f s", result.elapsed_seconds)
    if step_times:
        median = statistics.median(step_times) * 1000
        total_step = sum(step_times)
        overhead = result.elapsed_seconds - total_step
        overhead_pct = (
            100 * overhead / result.elapsed_seconds
            if result.elapsed_seconds > 0
            else 0.0
        )
        log.info(
            "              median step: %.2f ms, total step time: %.3f s",
            median,
            total_step,
        )
        log.info(
            "              overhead: %.3f s (%.1f%% of wall time)",
            overhead,
            overhead_pct,
        )
    log.info(sep)


class _FancyPrinter:
    """Boxed verbose output used by ``verbose=4``."""

    width = 96

    def __init__(
        self,
        *,
        log: logging.Logger,
        solver_name: str,
        problem_summary: str | None,
        max_iter: int,
        tol: float,
        initial_dual_norm: float | None,
        initial_loss: float | None,
        initial_grad_norm: float | None,
        ascii_only: bool,
        color: bool | None,
    ) -> None:
        self.log = log
        self.solver_name = solver_name
        self.problem_summary = problem_summary
        self.max_iter = max_iter
        self.tol = tol
        self.initial_dual_norm = initial_dual_norm
        self.initial_loss = initial_loss
        self.initial_grad_norm = initial_grad_norm
        self.ascii_only = ascii_only
        self.use_color = (
            bool(color)
            if color is not None
            else (not ascii_only and sys.stdout.isatty())
        )
        self.top = "+" if ascii_only else "╔"
        self.top_right = "+" if ascii_only else "╗"
        self.mid_left = "+" if ascii_only else "╠"
        self.mid_right = "+" if ascii_only else "╣"
        self.thin_left = "+" if ascii_only else "╟"
        self.thin_right = "+" if ascii_only else "╢"
        self.bottom = "+" if ascii_only else "╚"
        self.bottom_right = "+" if ascii_only else "╝"
        self.v = "|" if ascii_only else "║"
        self.h = "-" if ascii_only else "═"
        self.thin_h = "-" if ascii_only else "─"
        self.col = "|" if ascii_only else "│"
        self.grad = "grad_norm" if ascii_only else "‖∇‖"
        self.dual0 = "||y0||" if ascii_only else "‖y₀‖"
        self.delta = "d_loss" if ascii_only else "Δloss"
        self.eps = "eps" if ascii_only else "ε"
        self.check = "OK" if ascii_only else "✓"
        self.cross = "X" if ascii_only else "✗"
        self.spark_chars = "▁▂▃▄▅▆▇█" if not ascii_only else "._:-=+*#"
        self._line_count = 0

    def print_header(self) -> None:
        self._raw(self.top + self.h * (self.width - 2) + self.top_right)
        self._line(f"Solver: {self.solver_name}", style="bold")
        if self.problem_summary:
            self._line(f"Problem: {self._decorate_problem(self.problem_summary)}")
        self._line(
            f"Tolerance: {self.grad} < {_fmt_scientific(self.tol).strip()}    "
            f"Max iterations: {self.max_iter}"
        )
        self._divider()
        self._line("Initial state:")
        parts = []
        if self.initial_dual_norm is not None:
            parts.append(f"{self.dual0} = {_fmt_scientific(self.initial_dual_norm).strip()}")
        if self.initial_loss is not None:
            parts.append(f"loss0 = {_fmt_scientific(self.initial_loss).strip()}")
        if self.initial_grad_norm is not None:
            parts.append(f"{self.grad}0 = {_fmt_scientific(self.initial_grad_norm).strip()}")
        self._line("  " + ("     ".join(parts) if parts else "not evaluated"))
        self._divider()
        self._line(
            f"{'iter':>6} {self.col} {'loss':>14} {self.col} "
            f"{'grad_norm':>14} {self.col} {'step_ms':>9} {self.col} "
            f"{self.delta:>10} {self.col} {'median_ms':>9} {self.col} {'elapsed_s':>9}"
        )
        self._thin_divider()

    def print_iter(
        self,
        it: int,
        loss: float,
        grad_norm: float,
        step_time: float,
        prev_loss: float | None,
        step_times: list[float],
        overall_start: float,
    ) -> None:
        d_loss = "" if prev_loss is None else _fmt_delta(loss - prev_loss)
        median_ms = statistics.median(step_times) * 1000
        elapsed = time.perf_counter() - overall_start
        self._line(
            f"{it:6d} {self.col} {_fmt_scientific(loss)} {self.col} "
            f"{_fmt_scientific(grad_norm)} {self.col} "
            f"{step_time * 1000:9.2f} {self.col} {d_loss:>10} {self.col} "
            f"{median_ms:9.2f} {self.col} {elapsed:9.3f}"
        )

    def print_footer(self, result: OptimizeResult, step_times: list[float]) -> None:
        self._divider()
        self._trajectory_summary(result)
        self._divider()
        status = (
            f"{self.check} CONVERGED"
            if result.converged
            else f"{self.cross} MAX_ITER REACHED"
        )
        status_style = "green" if result.converged else "red"
        self._line(
            f"{status}  iters={result.num_iters}  "
            f"loss={_fmt_scientific(result.final_loss).strip()}  "
            f"{self.grad}={_fmt_scientific(result.final_grad_norm).strip()}",
            style=status_style,
        )
        if step_times:
            median = statistics.median(step_times) * 1000
            total_step = sum(step_times)
            overhead = result.elapsed_seconds - total_step
            overhead_pct = (
                100 * overhead / result.elapsed_seconds
                if result.elapsed_seconds > 0
                else 0.0
            )
            self._line(
                f"elapsed: {result.elapsed_seconds:.3f}s  "
                f"median step: {median:.2f}ms  total step: {total_step:.3f}s"
            )
            self._line(
                f"overhead: {overhead:.3f}s "
                f"({overhead_pct:.1f}% - Python loop + logging + compile)"
            )
        else:
            self._line(f"elapsed: {result.elapsed_seconds:.3f}s")
        spark = _sparkline(result.loss_history, ascii_only=self.ascii_only)
        if spark:
            self._line(f"Loss trajectory: {spark}")
        self._raw(self.bottom + self.h * (self.width - 2) + self.bottom_right)

    def _trajectory_summary(self, result: OptimizeResult) -> None:
        losses = result.loss_history
        grads = result.grad_norm_history
        if not losses and not grads:
            return
        self._line("Trajectory summary:")
        if losses and len(losses) >= 2:
            orders = _orders_changed(losses[0], losses[-1])
            mono = sum(
                1
                for prev, curr in zip(losses, losses[1:])
                if curr <= prev
            )
            self._line(
                f"  loss: {_fmt_scientific(losses[0]).strip()} -> "
                f"{_fmt_scientific(losses[-1]).strip()} "
                f"({orders})"
            )
            self._line(
                f"  monotonic loss decrease: {mono} of {len(losses) - 1} steps"
            )
        if grads and len(grads) >= 2:
            orders = _orders_changed(grads[0], grads[-1])
            self._line(
                f"  gradient: {_fmt_scientific(grads[0]).strip()} -> "
                f"{_fmt_scientific(grads[-1]).strip()} "
                f"({orders})"
            )

    def _decorate_problem(self, summary: str) -> str:
        if self.ascii_only:
            return summary
        return summary.replace("(eps=", f"({self.eps}=")

    def _divider(self) -> None:
        self._raw(self.mid_left + self.h * (self.width - 2) + self.mid_right)

    def _thin_divider(self) -> None:
        self._raw(self.thin_left + self.thin_h * (self.width - 2) + self.thin_right)

    def _line(self, text: str, *, style: str | None = None) -> None:
        self._raw(self._boxed_line(text, style=style))

    def _boxed_line(self, text: str, *, style: str | None = None) -> str:
        content_width = self.width - 4
        plain_len = len(_ANSI_RE.sub("", text))
        if plain_len > content_width:
            text = text[:content_width]
            plain_len = len(_ANSI_RE.sub("", text))
        line = f"{self.v} {text}{' ' * (content_width - plain_len)} {self.v}"
        return _colorize(line, style, self.use_color)

    def _raw(self, line: str, *, style: str | None = None) -> None:
        self.log.info("%s", _colorize(line, style, self.use_color))
        self._line_count += 1


def _fmt_scientific(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        return f"{value:>14}"
    abs_value = abs(value)
    if abs_value != 0.0 and (abs_value < 1e-99 or abs_value > 1e99):
        return f"{value:14.2e}"
    return f"{value:14.6e}"


def _fmt_delta(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        return f"{value:>10}"
    return f"{value:+10.2e}"


def _orders_changed(start: float, end: float) -> str:
    start_abs = abs(float(start))
    end_abs = abs(float(end))
    if start_abs == 0.0 or end_abs == 0.0:
        return "orders unavailable"
    orders = math.log10(start_abs) - math.log10(end_abs)
    direction = "decrease" if orders >= 0 else "increase"
    return f"{abs(orders):.1f} orders of magnitude {direction}"


def _sparkline(values: list[float] | None, *, ascii_only: bool) -> str:
    if not values:
        return ""
    chars = "._:-=+*#" if ascii_only else "▁▂▃▄▅▆▇█"
    width = min(40, len(values))
    if len(values) > width:
        indices = [
            round(i * (len(values) - 1) / (width - 1))
            for i in range(width)
        ]
        sample = [values[i] for i in indices]
    else:
        sample = list(values)
    transformed = [
        math.log10(max(abs(float(value)), 1e-300))
        for value in sample
        if math.isfinite(float(value))
    ]
    if not transformed:
        return ""
    low = min(transformed)
    high = max(transformed)
    if high == low:
        return chars[0] * len(transformed)
    return "".join(
        chars[
            min(
                len(chars) - 1,
                max(0, round((value - low) / (high - low) * (len(chars) - 1))),
            )
        ]
        for value in transformed
    )


def _colorize(line: str, style: str | None, enabled: bool) -> str:
    if not enabled or style is None:
        return line
    codes = {
        "green": "\033[32m",
        "red": "\033[31m",
        "dim": "\033[2m",
        "bold": "\033[1m",
    }
    code = codes.get(style)
    if code is None:
        return line
    return f"{code}{line}\033[0m"


__all__ = ["OptimizeResult", "run_solver"]
