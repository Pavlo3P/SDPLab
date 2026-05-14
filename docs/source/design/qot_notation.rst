QOT notation
============

Quantum optimal transport uses notation specialized to tensor-product
systems. SDPLab reserves :math:`\Gamma` for the global coupling:

.. math::

   \Gamma \in \operatorname{Herm}(d^N),
   \qquad
   \Gamma \succeq 0.

The marginal on subsystem :math:`k` is denoted by :math:`\gamma_k`.
The partial trace that keeps subsystem :math:`k` is denoted by
:math:`\operatorname{Tr}^k`:

.. math::

   \gamma_k = \operatorname{Tr}^k[\Gamma].

This differs from the common notation :math:`\operatorname{Tr}_{\neg k}`,
but it is the convention used throughout SDPLab's QOT documentation: the
superscript names the subsystem that remains.

Constraint operator
-------------------

The QOT constraint operator is

.. math::

   \mathcal{A}\Gamma
   =
   (\operatorname{Tr}^0[\Gamma], \ldots,
   \operatorname{Tr}^{N-1}[\Gamma]).

The codomain is a block matrix space:

.. math::

   \mathcal{A}\Gamma \in \operatorname{Herm}(d)^N.

The adjoint is a Kronecker sum. SDPLab denotes Kronecker sums by
:math:`\oplus`:

.. math::

   \mathcal{A}^\dagger U
   =
   U_0 \oplus \cdots \oplus U_{N-1}
   =
   \sum_k I \otimes \cdots \otimes U_k \otimes \cdots \otimes I.
