Quantum optimal transport
=========================

SDPLab includes helpers for dense quantum optimal transport (QOT) problems.
For local dimension :math:`d` and :math:`N` subsystems, the coupling is a
positive semidefinite matrix

.. math::

   \Gamma \in \operatorname{Herm}(d^N),
   \qquad
   \Gamma \succeq 0.

The one-body marginals are denoted by :math:`\gamma_k`. SDPLab uses
:math:`\operatorname{Tr}^k` for the partial trace that keeps subsystem
:math:`k` and traces out all other subsystems:

.. math::

   \gamma_k = \operatorname{Tr}^k[\Gamma].

QOT SDP
-------

The dense QOT problem has the form

.. math::

   \min_\Gamma \quad \operatorname{Tr}[C \Gamma]
   \quad \text{s.t.} \quad
   \operatorname{Tr}^k[\Gamma] = \gamma_k,\quad
   k = 0,\ldots,N-1,\quad
   \Gamma \succeq 0.

The constraint operator is

.. math::

   \mathcal{A}\Gamma
   =
   (\operatorname{Tr}^0[\Gamma], \ldots,
   \operatorname{Tr}^{N-1}[\Gamma]).

Its adjoint maps block variables :math:`U = (U_0,\ldots,U_{N-1})` back to the
global space. This adjoint is a Kronecker sum, denoted by :math:`\oplus`:

.. math::

   \mathcal{A}^\dagger U
   =
   U_0 \oplus \cdots \oplus U_{N-1}
   =
   \sum_k I \otimes \cdots \otimes U_k \otimes \cdots \otimes I.

Using the helpers
-----------------

.. code-block:: python

   from sdplab.special.qot import QOTConstraintOp, generate_random_qot

   qot_op = QOTConstraintOp(d=2, N=3)
   qot, state = generate_random_qot(d=2, N=3, proportions=(1.0,))

The returned ``qot`` is an :class:`sdplab.sdp.SDPDenseProblem`. The returned
``state`` is a feasible :class:`sdplab.sdp.SDPPrimal` whose array represents
the coupling :math:`\Gamma`.
