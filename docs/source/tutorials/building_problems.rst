Building a problem
==================

:class:`sdplab.problem.SDPProblem` holds the triple :math:`(C, \mathcal{A}, b)`
and nothing else: the objective, the constraint operator, and the right-hand
side. It is data plus the handful of operations every solver needs, so the same
object is handed to the CVXPY backend or lifted into a smoothed dual.

Choose the spaces first
-----------------------

Two spaces determine everything else. ``dom`` is where the unknown :math:`X`
and the cost :math:`C` live; ``cod`` is where :math:`\mathcal{A}X`, :math:`b`,
and the dual variable :math:`y` live.

.. code-block:: python

   import numpy as np
   from spacecore import Context, DenseLinOp, DenseVectorSpace, HermitianSpace, NumpyOps

   n, m = 4, 3
   ctx = Context(NumpyOps())                 # optional; see the spacecore docs

   dom = HermitianSpace(n, ctx=ctx)          # X and C: n x n Hermitian
   cod = DenseVectorSpace((m,), ctx=ctx)     # A X and b: length-m real vectors

``dom`` may be any spacecore Euclidean Jordan algebra space -- a Hermitian
matrix space as here, an elementwise (orthant) space for an LP, or a
:class:`~spacecore.TreeSpace` of such blocks. Nothing below assumes a single
dense matrix.

Build the operator, then the problem
------------------------------------

A stored tensor becomes a :class:`~sdplab.problem.DenseConstraintOp` or
:class:`~sdplab.problem.SparseConstraintOp`; an operator defined only by
``apply``/``rapply`` stays matrix-free. Any :class:`~spacecore.LinOp` works --
:class:`~sdplab.problem.SDPProblem` wraps it as needed.

.. code-block:: python

   from sdplab.problem import SDPProblem

   rng = np.random.default_rng(0)
   A_mats = rng.normal(size=(m, n, n))
   A_mats = A_mats + np.swapaxes(A_mats, -1, -2)      # each A_i symmetric

   C = np.diag([1.0, 2.0, 3.0, 4.0])
   b = np.array([1.0, 0.0, 0.5])

   A = DenseLinOp(A_mats, dom, cod, ctx=ctx)
   problem = SDPProblem(C, A, b, ctx=ctx)

After construction ``problem.A`` is the operator, ``problem.b`` the right-hand
side, and ``problem.dom`` / ``problem.cod`` the two spaces. The cost is a
:class:`~sdplab.problem.Cost` -- an operator in general, not merely a stored
matrix, so a dense or sparse Hermitian cost can also act on vectors.

Everything is a plain space element
-----------------------------------

There are no primal or dual wrapper objects. A primal is whatever array or tree
``dom`` holds, a dual is whatever ``cod`` holds, and the problem exposes the
operations that pair them:

.. code-block:: python

   X = np.eye(n) / n
   y = np.zeros(m)

   problem.primal_objective(X)      # <C, X>
   problem.dual_objective(y)        # <b, y>
   problem.feasibility_gap(X)       # A X - b, an element of cod
   problem.dual_slack(y)            # A^dagger y - C, an element of dom

The dual slack is the object most algorithms actually work with: semidefinite
constraints are constraints on eigenvalues, and the slack is what gets
diagonalized.

Solving
-------

.. code-block:: python

   from sdplab.solvers import run_cvxpy_solver

   X, y = run_cvxpy_solver(problem, solver="CLARABEL")

The backend asks the constraint operator for its per-constraint matrices
(``to_cvxpy``) and assembles :math:`\operatorname{Re}\langle A_i, X\rangle = b_i`.
For the smoothed-dual route instead, see :doc:`regularization`.

A complex codomain -- stacked Hermitian blocks, say -- is fine for the problem
itself, but :func:`spacecore.minimize_scipy` needs a real domain, so the scipy
route refuses one. :func:`spacecore.realify` (spacecore 0.4.3 and later) bridges
that: it returns a functional unchanged when the domain is already real, and
otherwise presents it over stacked real coordinates.
