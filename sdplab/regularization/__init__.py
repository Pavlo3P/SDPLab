r"""Built-in spectral SDP regularizers.

``EntropyReg`` and ``QuadraticReg`` are separable spectral regularizers:
``phi`` implements the scalar :math:`\varphi`, ``phi_star`` implements the
scalar conjugate :math:`\psi`, and ``phi_star_prime`` implements
:math:`\psi'`.

``EntropyRegLog`` is coupled across eigenvalues. It represents
:math:`\varepsilon\log\operatorname{Tr}\exp(X/\varepsilon)` and returns
normalized exponential-weight gradients with trace one.
"""

from ._base import Regularizer, SDPRegularized
from .entropy import EntropyReg, EntropyRegLog
from .quadratic import QuadraticReg

__all__ = [
    'Regularizer',
    'SDPRegularized',
    'EntropyReg',
    'EntropyRegLog',
    'QuadraticReg',
]
