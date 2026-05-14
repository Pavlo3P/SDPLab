SDP basics
==========

A semidefinite program is an optimization problem whose unknown is a matrix.
SDPLab uses the following primal form:

.. math::

   \min_X \quad \operatorname{Tr}[C X]
   \quad \text{s.t.} \quad
   \mathcal{A}X = b,\quad
   X \succeq 0.

Each symbol has a direct coding meaning:

* :math:`X \in \mathrm{dom}` is the unknown primal matrix.
* :math:`C \in \mathrm{dom}` is the symmetric or Hermitian cost matrix.
* :math:`\operatorname{Tr}[C X]` is the scalar objective value.
* :math:`\mathcal{A}: \mathrm{dom} \to \mathrm{cod}` is a linear constraint
  operator.
* :math:`b \in \mathrm{cod}` is the desired constraint value.
* :math:`X \succeq 0` means that :math:`X` is positive semidefinite.

Why the trace appears
---------------------

For dense real or complex matrices, the expression
:math:`\operatorname{Tr}[C X]` is the matrix analogue of a dot product. If
:math:`C` is the cost matrix, then entries of :math:`X` aligned with large
positive entries of :math:`C` make the objective larger; entries aligned with
negative directions make it smaller.

Linear constraints
------------------

The equation :math:`\mathcal{A}X = b` should be read as "apply the constraint
operator to the matrix." In the common dense-vector case, this means a list of
trace equations:

.. math::

   (\mathcal{A}X)_i = \operatorname{Tr}[A_i X] = b_i,
   \qquad i = 0,\ldots,m-1.

The matrices :math:`A_i` are not stored by :class:`sdplab.sdp.SDPProblem`
itself. They live inside the SpaceCore linear operator that represents
:math:`\mathcal{A}`.

Dual variables
--------------

The dual variable :math:`y` lives in the same space as :math:`b`. The adjoint
operator

.. math::

   \mathcal{A}^\dagger: \mathrm{cod} \to \mathrm{dom}

moves :math:`y` back into matrix space. SDPLab solvers often use the dual
slack expression

.. math::

   \mathcal{A}^\dagger y - C.

Its eigenvalues are important because semidefinite constraints are spectral:
they are constraints on eigenvalues.
