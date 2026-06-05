"""Lightweight representations for Gaussian barycenter objects."""

from __future__ import annotations

from html import escape
from typing import Any, Iterable


def backend_label(obj: Any) -> str:
    """Return a compact backend label for a context-bound object.

    This helper is used by plain-text and HTML representations.

    Parameters
    ----------
    obj : object
        Object that may expose a ``ctx`` attribute.

    Returns
    -------
    str
        Human-readable backend, dtype, and validation summary.
    """
    ctx = getattr(obj, "ctx", None)
    if ctx is None:
        return "unknown"
    family = getattr(getattr(ctx, "ops", None), "family", "unknown")
    return f"{family}, dtype={ctx.dtype}, checks={ctx.enable_checks}"


def shape_dtype(x: Any) -> str:
    """Return shape and dtype for an array-like object.

    The result is intentionally compact for use inside object representations.

    Parameters
    ----------
    x : object
        Candidate array-like object.

    Returns
    -------
    str
        Compact shape and dtype description, or the Python type name when no
        shape is available.
    """
    shape = getattr(x, "shape", None)
    dtype = getattr(x, "dtype", None)
    if shape is None:
        return type(x).__name__
    if dtype is None:
        return f"shape={tuple(shape)}"
    return f"shape={tuple(shape)}, dtype={dtype}"


def safe_float(x: Any, *, precision: int = 6) -> str:
    """Format a scalar value when eager conversion is available.

    Falls back to shape and dtype information for traced or non-scalar values.

    Parameters
    ----------
    x : object
        Candidate scalar value.
    precision : int, optional
        Number of significant digits. Default is 6.

    Returns
    -------
    str
        Formatted scalar, or a shape/dtype summary if conversion fails.
    """
    try:
        return f"{float(x):.{precision}g}"
    except Exception:
        return shape_dtype(x)


def safe_array_stat(ops: Any, fn_name: str, x: Any, *, precision: int = 6) -> str | None:
    """Return a scalar array statistic when available.

    Failures are swallowed because repr generation must not force eager
    evaluation on backends that cannot provide it.

    Parameters
    ----------
    ops : object
        Backend operations object exposing ``fn_name``.
    fn_name : str
        Name of the reduction function to call.
    x : object
        Array-like input to summarize.
    precision : int, optional
        Number of significant digits for scalar formatting. Default is 6.

    Returns
    -------
    str or None
        Formatted statistic, or ``None`` if the reduction cannot be evaluated.
    """
    try:
        fn = getattr(ops, fn_name)
        return safe_float(fn(x), precision=precision)
    except Exception:
        return None


def array_summary(
    name: str,
    x: Any,
    *,
    ops: Any | None = None,
    stat: str | None = "norm",
) -> str:
    """Return a compact labeled summary for an array.

    The summary is designed for repr tables rather than numerical reporting.

    Parameters
    ----------
    name : str
        Label to include in the summary.
    x : object
        Array-like object to summarize.
    ops : object or None, optional
        Backend operations object used to compute ``stat``. Default is no
        statistic.
    stat : str or None, optional
        Reduction name to call on ``ops``. Default is ``"norm"``.

    Returns
    -------
    str
        Labeled summary containing shape, dtype, and optionally a statistic.
    """
    parts = [f"{name}: {shape_dtype(x)}"]
    if ops is not None and stat is not None:
        value = safe_array_stat(ops, stat, x)
        if value is not None:
            parts.append(f"{stat}={value}")
    return ", ".join(parts)


def plain_repr(title: str, rows: Iterable[tuple[str, Any]]) -> str:
    """Return a multi-line representation with aligned labels.

    Values are converted to strings lazily during rendering.

    Parameters
    ----------
    title : str
        Object title.
    rows : iterable of tuple
        Label-value pairs to render.

    Returns
    -------
    str
        Plain-text representation.
    """
    rows = [(str(k), str(v)) for k, v in rows]
    if not rows:
        return f"{title}()"
    width = max(len(k) for k, _ in rows)
    body = "\n".join(f"  {k:<{width}} = {v}" for k, v in rows)
    return f"{title}(\n{body}\n)"


def html_repr(title: str, rows: Iterable[tuple[str, Any]]) -> str:
    """Return a notebook-friendly HTML table.

    Labels and values are HTML-escaped before being inserted into the table.

    Parameters
    ----------
    title : str
        Object title.
    rows : iterable of tuple
        Label-value pairs to render.

    Returns
    -------
    str
        HTML fragment for rich notebook display.
    """
    body = "\n".join(
        "<tr>"
        f"<th style='text-align:left;padding:3px 12px 3px 0'>{escape(str(k))}</th>"
        f"<td style='text-align:left;padding:3px 0'>{escape(str(v))}</td>"
        "</tr>"
        for k, v in rows
    )
    return (
        "<div style='font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'>"
        f"<div style='font-weight:600;margin-bottom:4px'>{escape(title)}</div>"
        "<table>"
        f"{body}"
        "</table>"
        "</div>"
    )
