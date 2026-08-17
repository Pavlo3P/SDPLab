SDPLab documentation
====================

SDPLab is a Python library for constructing, regularizing, and solving
semidefinite programs. It builds on `spacecore
<https://pypi.org/project/spacecore/>`_ spaces and linear operators, so the
mathematical objects in an SDP are represented directly in code, and delegates
every first-order optimization loop to ``spacecore.optimize``.

* :doc:`tutorials/index` explains the main ideas with worked examples.
* :doc:`api/index` provides object-level reference pages, generated from each
  package's public exports.
* :doc:`release_notes` records user-visible changes.

Core model
----------

SDPLab represents an SDP in trace form:

.. math::

   \min_X \quad \langle C, X\rangle
   \quad \text{s.t.} \quad
   \mathcal{A}X = b,\quad
   X \succeq 0.

Here :math:`C, X \in \mathrm{dom}`, :math:`\mathcal{A}: \mathrm{dom} \to
\mathrm{cod}` is a linear constraint operator, and :math:`b \in \mathrm{cod}`
is the right-hand side. ``dom`` may be any Euclidean Jordan algebra space, so
:math:`X \succeq 0` means a nonnegative Jordan spectrum -- positive
semidefiniteness for a Hermitian matrix, nonnegativity for a vector.

Quick example
-------------

.. code-block:: python

   import numpy as np
   from sdplab import EntropyReg, RegularizedSDPDualFunctional, run_regularized_solver
   from sdplab.examples import generate_max_cut
   from sdplab.solvers import run_cvxpy_solver

   problem = generate_max_cut(8, seed=0, unit_trace=True)

   # Reference solve through CVXPY.
   X, y = run_cvxpy_solver(problem, solver="CLARABEL")

   # Or smooth it and optimize the dual.
   dual = RegularizedSDPDualFunctional(problem, EntropyReg(problem.dom))
   result = run_regularized_solver(dual.bind(0.1), verbose=0)
   X_eps = dual.primal_from_dual(result.dual, 0.1)

See :doc:`tutorials/building_problems` to assemble a problem from your own
cost and constraint operator.

.. toctree::
   :maxdepth: 2
   :hidden:

   tutorials/index
   api/index
   release_notes
