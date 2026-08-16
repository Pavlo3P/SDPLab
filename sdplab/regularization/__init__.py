r"""Separable spectral regularizers for conic problems.

A :class:`Regularizer` is a pure, immutable penalty defined on a Euclidean
Jordan algebra space by four elementwise scalar formulas (:math:`\varphi`,
:math:`\psi=\varphi^*`, :math:`\psi'`, :math:`\log\psi'`) applied through the
space's spectral calculus. It therefore works uniformly on Hermitian matrix
spaces, elementwise Jordan spaces, stacked spaces, and trees of such leaves.

The regularization strength :math:`\varepsilon` is not stored on the penalty;
it is supplied per call. :class:`RegularizedSDPDualFunctional` couples a
penalty to a base :class:`~sdplab.problem.SDPProblem` as a spacecore
:class:`~spacecore.Functional` over the constraint codomain; its
:meth:`~RegularizedSDPDualFunctional.bind` method fixes :math:`\varepsilon`
for the single-argument optimizer adapters in ``spacecore.optimize``.

Every regularizer's smoothed dual term (:meth:`Regularizer.legendre`) has a
``normalized`` flag: ``False`` gives the free separable conjugate, ``True``
gives the fixed-trace (unit-trace primal) log-partition form, which is
globally bounded and has a unit-trace gradient. For the entropy regularizer
this fixed-trace form is exactly the log-partition entropy dual
:math:`\varepsilon \log \operatorname{Tr}\exp` — the bounded target for the
entropy dual at small :math:`\varepsilon`. The flag is surfaced end-to-end by
:class:`BoundDualFunctional`.
"""

from ._base import Regularizer
from ._functional import BoundDualFunctional, RegularizedSDPDualFunctional
from .entropy import EntropyReg
from .quadratic import QuadraticReg

__all__ = [
    "Regularizer",
    "RegularizedSDPDualFunctional",
    "BoundDualFunctional",
    "EntropyReg",
    "QuadraticReg",
]
