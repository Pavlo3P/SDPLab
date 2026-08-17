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

   \min_\Gamma \quad \langle C, \Gamma\rangle
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

Building an instance
--------------------

:func:`~sdplab.examples.generate_random_qot` derives the marginals from a
reference coupling, so the returned ``state`` is feasible by construction:

.. code-block:: python

   import numpy as np
   from sdplab.examples import generate_random_qot
   from sdplab.solvers import run_cvxpy_solver

   qot, state = generate_random_qot(d=2, N=2, proportions=(0.6, 0.4), seed=0)

   np.linalg.norm(np.asarray(qot.feasibility_gap(state)))   # ~0, state is feasible

   X, y = run_cvxpy_solver(qot, solver="CLARABEL")

``qot`` is a plain :class:`~sdplab.problem.SDPProblem` and ``state`` a plain
array -- there are no wrapper objects. To build an instance with marginals of
your own, use the operator directly:

.. code-block:: python

   from spacecore import Context, NumpyOps
   from sdplab.problem import SDPProblem
   from sdplab.special.qot import QOTConstraintOp

   ctx = Context(NumpyOps(), dtype=np.complex128, check_level="none")
   op = QOTConstraintOp(d=2, N=2, ctx=ctx)

   gamma = op.apply(ctx.asarray(rho))      # marginals of a chosen state rho
   sdp = SDPProblem(ctx.asarray(H), op, gamma, ctx=ctx)

A dedicated dual solver, :func:`~sdplab.special.qot.solve_qot_dual`, models the
dual directly in CVXPY instead of going through the per-constraint encoding.

What makes QOT the structured case
----------------------------------

The codomain is a *stack* of Hermitian blocks rather than a flat vector, so the
dual variable is a tuple of matrices. Two consequences worth knowing:

* The scipy route rejects it. :func:`spacecore.minimize_scipy` needs a real
  codomain, and this one is complex, so
  :func:`~sdplab.solvers.run_regularized_solver` raises rather than failing
  inside SciPy. Either use the optax route on a JAX backend, or wrap the bound
  functional with :func:`spacecore.realify` (spacecore 0.4.3 and later), which
  returns it unchanged on a real domain and otherwise presents it over stacked
  real coordinates:

  .. code-block:: python

     from spacecore import minimize_scipy, realify

     F = realify(-dual.bind(eps))          # idempotent; no branch on the field
     result = minimize_scipy(F, F.domain.zeros())
* The adjoint has a gauge direction. Shifting :math:`U_0` by :math:`cI` and
  :math:`U_1` by :math:`-cI` leaves :math:`\mathcal{A}^\dagger U` unchanged, so
  the dual optimum is not unique -- compare recovered primals, not duals.
