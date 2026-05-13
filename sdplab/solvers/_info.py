"""Convergence metadata returned by SDP solvers.

Solvers may return a primal matrix, a dual variable, objective histories,
gradient norms, and timing information. Not every solver fills every field.
"""

from __future__ import annotations

from dataclasses import dataclass

from spacecore import DenseArray
from ..sdp import SDPPrimal, SDPDual


@dataclass
class ConvergenceInfo:
    """Container for primal/dual solver outputs and convergence diagnostics.

    Fields are optional because different algorithms expose different
    diagnostics. For example, a first-order dual solver may provide
    ``dual_obj`` and ``grad_norm`` histories, while a direct conic solver may
    return only final primal and dual variables.
    """

    dual: SDPDual = None
    primal: SDPPrimal = None
    primal_obj: DenseArray = None
    dual_obj: DenseArray = None
    grad_norm: DenseArray = None
    tol_reached: bool = None
    time: float = None
