from __future__ import annotations

from dataclasses import dataclass

from spacecore import DenseArray
from ..sdp import SDPPrimal, SDPDual


@dataclass
class ConvergenceInfo:
    dual: SDPDual = None
    primal: SDPPrimal = None
    primal_obj: DenseArray = None
    dual_obj: DenseArray = None
    grad_norm: DenseArray = None
    tol_reached: bool = None
    time: float = None
