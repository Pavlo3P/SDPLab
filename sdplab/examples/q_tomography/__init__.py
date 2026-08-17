r"""State tomography as feasibility: find :math:`X \succeq 0` with
:math:`\operatorname{Tr}[M_i X] = b_i` and :math:`\operatorname{Tr}[X] = 1`.

Unit trace is appended as one more measurement row. See :mod:`._build` for the
transpose the dense operator needs.
"""

from ._build import generate_qubit_tomography

__all__ = ["generate_qubit_tomography"]
