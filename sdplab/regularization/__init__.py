r"""Built-in scalar functions used as spectral SDP regularizers.

Each regularizer defines a convex scalar :math:`\varphi`, its Legendre
transform :math:`\psi`, and the derivative :math:`\psi'` used for
dual-to-primal eigenvalue recovery. In code, ``phi`` implements
:math:`\varphi`, ``phi_star`` implements :math:`\psi`, and
``phi_star_prime`` implements :math:`\psi'`.
"""

from ._base import Regularizer
from .entropy import EntropyReg, EntropyRegLog
from .quadratic import QuadraticReg

__all__ = [
    'Regularizer',
    'EntropyRegLog',
    'EntropyReg',
    'QuadraticReg',
]
