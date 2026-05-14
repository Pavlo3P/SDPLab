Building a dense SDP
====================

The class :class:`sdplab.sdp.SDPDenseProblem` stores dense SDP data. It is the
right problem class when the primal matrix is represented explicitly as a
symmetric or Hermitian matrix.

Problem data
------------

A dense problem has the form

.. math::

   \min_X \quad \operatorname{Re}\operatorname{Tr}[C X]
   \quad \text{s.t.} \quad
   \mathcal{A}X = b,\quad
   X \succeq 0.

Optionally, the dense problem also enforces

.. math::

   \operatorname{Tr}[X] = \tau.

The trace constraint is useful when :math:`X` is a density matrix, where
:math:`\tau = 1`.

Construction pattern
--------------------

.. code-block:: python

   import spacecore as sc
   from sdplab.sdp import SDPDenseProblem

   ctx = sc.Context(sc.NumpyOps())
   dom = sc.HermitianSpace(n, ctx=ctx)
   cod = sc.VectorSpace((m,), ctx=ctx)

   # A_op must be a SpaceCore LinOp with A_op.dom == dom and A_op.cod == cod.
   problem = SDPDenseProblem(C, A_op, b, tau=1.0, ctx=ctx)

After construction:

* ``problem.C`` is the cost matrix :math:`C`.
* ``problem.A`` is the linear operator :math:`\mathcal{A}`.
* ``problem.b`` is the right-hand side :math:`b`.
* ``problem.tau`` is either ``None`` or the trace value :math:`\tau`.

Primal and dual wrappers
------------------------

Use :class:`sdplab.sdp.SDPPrimal` for primal matrices and
:class:`sdplab.sdp.SDPDual` for dual variables:

.. code-block:: python

   primal = problem.primal_from_array(X)
   dual = problem.dual_from_array(y)

The wrappers preserve the mathematical roles of the arrays. A primal stores
:math:`X \in \mathrm{dom}`. A dual stores :math:`y \in \mathrm{cod}`.

Solving with CVXPY
------------------

.. code-block:: python

   from sdplab.solvers import run_cvxpy_solver

   primal, dual = run_cvxpy_solver(problem, solver="MOSEK")

The CVXPY solver returns both primal and dual wrappers. The primal objective
is evaluated as

.. math::

   \operatorname{Re}\operatorname{Tr}[C X].
