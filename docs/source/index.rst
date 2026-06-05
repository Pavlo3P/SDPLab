SDPLab documentation
====================

SDPLab is a Python library for constructing, regularizing, and solving
semidefinite programs. It builds on SpaceCore-style spaces and linear
operators so that the mathematical objects in an SDP are represented directly
in code.

The documentation has four layers:

* :doc:`tutorials/index` explains the main ideas with worked examples.
* :doc:`design/index` describes the mathematical and API policies behind the
  library.
* :doc:`special/index` covers specialized workflows such as quantum estimators.
* :doc:`api/index` provides explicit object-level API reference pages.
* :doc:`release_notes` records user-visible changes.

Core model
----------

SDPLab represents an SDP in trace form:

.. math::

   \min_X \quad \operatorname{Tr}[C X]
   \quad \text{s.t.} \quad
   \mathcal{A}X = b,\quad
   X \succeq 0.

Here :math:`C, X \in \mathrm{dom}`, the cost matrix :math:`C` is symmetric
or Hermitian, :math:`\mathcal{A}: \mathrm{dom} \to \mathrm{cod}` is a linear
constraint operator, and :math:`b \in \mathrm{cod}` is the right-hand side.
The constraint :math:`X \succeq 0` means that :math:`X` is positive
semidefinite.

Quick example
-------------

.. code-block:: python

   import spacecore as sc
   from sdplab.sdp import SDPDenseProblem
   from sdplab.solvers import run_cvxpy_solver

   ctx = sc.Context(sc.NumpyOps())

   # dom contains the Hermitian primal matrix X and cost matrix C.
   dom = sc.HermitianSpace(n, ctx=ctx)

   # cod contains A X, b, and the dual variable y.
   cod = sc.VectorSpace((m,), ctx=ctx)

   # A_op should be a SpaceCore linear operator dom -> cod.
   problem = SDPDenseProblem(C, A_op, b, tau=None, ctx=ctx)
   primal, dual = run_cvxpy_solver(problem)

.. toctree::
   :maxdepth: 2
   :hidden:

   tutorials/index
   design/index
   special/index
   api/index
   release_notes
