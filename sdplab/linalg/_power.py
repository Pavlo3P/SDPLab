from __future__ import annotations

from typing import Any, Callable
from qotlib.core import Space

from ._misc import MVP


def power_method(
    vector_space: Space,
    mvp: MVP,
    init_v: Any,
    *,
    n_iter: int = 100,
    eps: float = 1e-12,
) -> Any:

    ops = vector_space.ctx.ops
    vector_space.check_member(init_v)

    def normalize(v):
        return vector_space.scale(1 / ops.maximum(vector_space.norm(v), eps), v)

    v0 = normalize(init_v)

    def body(i, v):
        return normalize(mvp(v))

    return ops.fori_loop(0, n_iter, body, v0)

