Solvers API
===========

Solver functions return :class:`sdplab.sdp.SDPPrimal` and
:class:`sdplab.sdp.SDPDual` objects when the solve succeeds.

.. autosummary::
   :nosignatures:

   sdplab.solvers.run_cvxpy_solver
   sdplab.solvers.run_optax_solver
   sdplab.solvers.ConvergenceInfo

Entry points
------------

.. autofunction:: sdplab.solvers.run_cvxpy_solver

.. autofunction:: sdplab.solvers.run_optax_solver

Diagnostics
-----------

.. autoclass:: sdplab.solvers.ConvergenceInfo
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
