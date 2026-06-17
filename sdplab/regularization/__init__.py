r"""Built-in spectral SDP regularizers.

``EntropyReg`` and ``QuadraticReg`` are separable spectral regularizers:
``phi`` implements the scalar :math:`\varphi`, ``phi_star`` implements the
scalar conjugate :math:`\psi`, and ``phi_star_prime`` implements
:math:`\psi'`.

``EntropyRegLog`` is the fixed-trace entropy variant. For trace
:math:`\tau > 0`, it uses the coupled conjugate

.. math::

    \varepsilon\tau
    \left(
        \log\operatorname{Tr}\exp(X/\varepsilon)
        - \log\tau
    \right),

not an elementwise scalar conjugate. Its recovered gradients are normalized
exponential weights with trace :math:`\tau`.
"""

from ._base import Regularizer
from .entropy import EntropyReg, EntropyRegLog
from .quadratic import QuadraticReg

__all__ = [
    "Regularizer",
    "EntropyReg",
    "EntropyRegLog",
    "QuadraticReg",
]
