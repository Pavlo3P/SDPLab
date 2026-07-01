r"""Separable spectral regularizers for SDPs.

A :class:`Regularizer` is a pure, immutable penalty built from four elementwise
scalar formulas (:math:`\varphi`, :math:`\psi=\varphi^*`, :math:`\psi'`,
:math:`\log\psi'`). Concrete penalties are *prepared instances* returned by the
:func:`entropy` and :func:`quadratic` factories -- no subclassing needed.

The regularization strength :math:`\varepsilon` is not stored on the penalty;
it is supplied per call by :class:`SDPRegularized`, which couples a penalty to a
base SDP and owns the spectral calculus and normalization.

:class:`EntropyRegLog` (via :func:`entropy_fixed_trace`) is the fixed-trace
variant, whose conjugate couples the whole spectrum and so cannot be expressed
elementwise.
"""

from ._base import Regularizer
from .entropy import EntropyReg
from .quadratic import QuadraticReg

__all__ = [
    "Regularizer",
    "EntropyReg",
    "QuadraticReg",
]
