r"""Provide Gaussian QOT barycenter primitives.

The package exposes phase spaces, Gaussian states, quadratic operators, stacked
dual-variable containers, and the entropic Gaussian QOT barycenter problem.

Examples
--------
>>> from sdplab.special.qot.barycenter import GaussianPhaseSpace
>>> space = GaussianPhaseSpace(1)
>>> space.dim
2
"""

from ._spaces import GaussianPhaseSpace
from ._gaussian_state import GaussianState
from ._quadratic_operator import QuadraticOperator
from ._quadratic_operator_tuple import QuadraticOperatorTuple
from ._problem import QOTGaussianBarycenterProblem

__all__ = [
    "GaussianPhaseSpace",
    "GaussianState",
    "QuadraticOperator",
    "QuadraticOperatorTuple",
    "QOTGaussianBarycenterProblem",
]
