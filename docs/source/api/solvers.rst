Solvers API
===========

Entry points. First-order solves are delegated to ``spacecore.optimize``;
``run_cvxpy_solver`` is the reference backend.

.. autosummary::
   :nosignatures:

   sdplab.solvers.OptimizeResult
   sdplab.solvers.run_regularized_solver
   sdplab.solvers.run_cvxpy_solver

.. autoclass:: sdplab.solvers.OptimizeResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: sdplab.solvers.run_regularized_solver

.. autofunction:: sdplab.solvers.run_cvxpy_solver
