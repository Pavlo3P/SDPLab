Solvers API
===========

Solver functions return solver-specific result records. Smooth regularized
dual solvers return :class:`sdplab.solvers.OptimizeResult`.

.. autosummary::
   :nosignatures:

   sdplab.solvers.run_cvxpy_solver
   sdplab.solvers.run_regularized_solver
   sdplab.solvers.solve_optax
   sdplab.solvers.solve_torch
   sdplab.solvers.solve_scipy
   sdplab.solvers.OptimizeResult

Entry points
------------

.. autofunction:: sdplab.solvers.run_cvxpy_solver

.. autofunction:: sdplab.solvers.run_regularized_solver

.. autofunction:: sdplab.solvers.solve_optax

.. autofunction:: sdplab.solvers.solve_torch

.. autofunction:: sdplab.solvers.solve_scipy

Diagnostics
-----------

.. autoclass:: sdplab.solvers.OptimizeResult
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
