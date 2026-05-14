QOT
===

Quantum optimal transport helpers use :math:`\Gamma` for the coupling and
:math:`\gamma_k = \operatorname{Tr}^k[\Gamma]` for the one-body marginals.

.. autosummary::
   :nosignatures:

   sdplab.special.qot.QOTConstraintOp
   sdplab.special.qot.BlockMatrixSpace
   sdplab.special.qot.compute_ptraces
   sdplab.special.qot.kron_sum
   sdplab.special.qot.solve_qot_dual
   sdplab.special.qot.generate_random_qot

Spaces and operators
--------------------

.. autoclass:: sdplab.special.qot.QOTConstraintOp
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: sdplab.special.qot.BlockMatrixSpace
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

Linear algebra
--------------

.. autofunction:: sdplab.special.qot.compute_ptraces

.. autofunction:: sdplab.special.qot.kron_sum

Solvers and examples
--------------------

.. autofunction:: sdplab.special.qot.solve_qot_dual

.. autofunction:: sdplab.special.qot.generate_random_qot
