r"""Primal and dual variable wrappers for SDP problems.

An SDP has two main kinds of variables:

    Primal variable ``X``:
        The matrix being optimized. It lives in ``dom`` and must satisfy
        constraints such as :math:`\mathcal{A}X = b` and
        :math:`X \succeq 0`.
    Dual variable ``y``:
        The multiplier for the equality constraint ``A X = b``. It lives in
        ``cod``, the same space as ``b``.

The wrappers make arithmetic type-aware. Adding two primal variables is fine.
Adding a primal variable to a dual variable is not mathematically meaningful,
so the classes keep those roles separate.
"""

from ._base import SDPVar
from .primal import SDPPrimal
from .dual import SDPDual

__all__ = [
    "SDPVar",
    "SDPPrimal",
    "SDPDual",
]
