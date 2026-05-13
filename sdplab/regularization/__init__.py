r"""Regularized SDP wrappers and spectral regularizers.

For ``X = V diag(lambda) V^dagger``, a spectral regularizer adds the trace-form
penalty :math:`\varepsilon \operatorname{Tr}[\varphi(X)]` to the primal SDP
objective :math:`\operatorname{Tr}[C X]`. The Legendre transform of
:math:`\varphi` is denoted by :math:`\psi`; in code, ``phi_star`` implements
:math:`\psi`. The dual side is evaluated on eigenvalues of the dual slack
expression :math:`A^\dagger y - C`.

Regularization changes the problem from "find a best feasible matrix" to
"find a best feasible matrix while also preferring a particular spectrum."
Entropy regularization prefers spread-out spectra. Quadratic regularization
penalizes large eigenvalues.
"""

from .regs import (
    AbstractRegularizer,
    EntropyReg,
    EntropyRegLog,
    QuadraticReg
)
from ._reg_sdp import SDPRegularized

__all__ = [
    "AbstractRegularizer",
    "EntropyRegLog",
    "EntropyReg",
    "QuadraticReg",
    "SDPRegularized",
]
