r"""Core SDP problem and variable abstractions.

Use this namespace when you want to describe the mathematical problem

.. math::

    \min_X \quad \operatorname{Tr}[C X]
    \quad \text{s.t.} \quad \mathcal{A}X = b,\quad X \succeq 0.

The objects line up with the formula directly:

    ``SDPProblem``:
        Stores ``C in dom``, ``A : dom -> cod``, and ``b in cod``.
    ``SDPPrimal``:
        Wraps a candidate primal matrix ``X in dom``.
    ``SDPDual``:
        Wraps a candidate dual vector/block object ``y in cod``.
    ``SDPDenseProblem``:
        Specializes the abstract SDP to dense symmetric/Hermitian matrix
        variables.

If you can write down ``C``, ``A``, and ``b``, then you can build an
``SDPProblem``. If you can additionally represent ``X`` as a dense symmetric
or Hermitian matrix, use ``SDPDenseProblem``.
"""

from .variables import SDPDual, SDPPrimal, SDPVar

from .problem import SDPProblem, SDPDenseProblem

__all__ = [
    "SDPVar",
    "SDPDual",
    "SDPPrimal",
    "SDPDenseProblem",
    "SDPProblem",
]
