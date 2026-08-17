Regularized SDPs
================

Regularization adds a spectral penalty to the primal SDP. The unregularized
problem is

.. math::

   \min_X \quad \langle C, X\rangle
   \quad \text{s.t.} \quad
   \mathcal{A}X = b,\quad X \succeq 0.

Write the eigendecomposition of the primal as

.. math::

   X = V \operatorname{diag}(\lambda) V^\dagger.

A scalar convex :math:`\varphi` is applied to :math:`X` spectrally, so the
penalty is a sum over eigenvalues,

.. math::

   \operatorname{Tr}[\varphi(X)] = \sum_i \varphi(\lambda_i),

and the regularized primal objective at strength :math:`\varepsilon > 0` is

.. math::

   P_\varepsilon(X)
   =
   \langle C, X\rangle + \varepsilon \operatorname{Tr}[\varphi(X)].

Dual-side view
--------------

Write the dual slack as

.. math::

   S = \mathcal{A}^\dagger y - C,
   \qquad
   S = W \operatorname{diag}(s) W^\dagger.

The regularized dual is unconstrained and differentiable,

.. math::

   D_\varepsilon(y)
   =
   \langle b, y\rangle
   - \varepsilon \operatorname{Tr}\!\left[\psi\!\left(\frac{S}{\varepsilon}\right)\right],
   \qquad
   \psi(s) = \sup_t \{s t - \varphi(t)\},

where :math:`\psi` is the Legendre transform of :math:`\varphi`. The map
``primal_from_dual`` inverts the first-order relation in the eigenbasis of
:math:`S`,

.. math::

   \lambda_i(X) = \psi'\!\left(\frac{s_i}{\varepsilon}\right).

Built-in regularizers
---------------------

.. autosummary::
   :nosignatures:

   sdplab.regularization.EntropyReg
   sdplab.regularization.QuadraticReg

Entropy regularization uses

.. math::

   \varphi(t) =
   \begin{cases}
   t(\log t - 1), & t > 0,\\
   0, & t = 0,\\
   +\infty, & t < 0,
   \end{cases}
   \qquad
   \psi(s) = \exp(s).

Quadratic regularization uses

.. math::

   \varphi(t) = \frac{t^2}{2} + \iota_{[0,\infty)}(t),
   \qquad
   \psi(s) = \frac{\max(s, 0)^2}{2}.

In both cases, the scalar formulas are lifted to matrices through the
eigenvalues of Hermitian matrices.

Fixed-trace recovery
--------------------

Every method that recovers a primal takes ``normalized``. It selects the
*fixed-trace* conjugate -- the supremum taken over unit-trace primals only --
rather than the free one, so it changes the objective and not merely the
reported :math:`X`:

.. math::

   F(S) = \sup_{X \succeq 0,\ \operatorname{Tr}[X] = 1}
          \ \langle S, X\rangle - \varepsilon \operatorname{Tr}[\varphi(X)].

The trace constraint carries a multiplier that enters *additively in the
argument*, :math:`X = \psi'((S - \theta)/\varepsilon)` with :math:`\theta`
fixed by :math:`\sum_i \psi'((s_i - \theta)/\varepsilon) = 1`. Dividing
:math:`\psi'(S/\varepsilon)` by its trace is not the same operation: a
division preserves the ratios :math:`\lambda_i/\lambda_j` and the zero
pattern, while the shift changes both. For entropy the two coincide -- a shift
rescales :math:`\exp` globally, so the answer is
:math:`\operatorname{softmax}(S/\varepsilon)`, the Gibbs state. For the
quadratic penalty the shift moves the clip point instead, so the answer is the
projection onto the simplex and is genuinely low rank.

Solving
-------

Couple a problem to a regularizer, bind :math:`\varepsilon`, and hand the
result to the solver. ``bind`` returns a standard single-argument
:class:`spacecore.Functional`, so the ``spacecore.optimize`` drivers consume it
directly.

.. code-block:: python

   from sdplab import EntropyReg, RegularizedSDPDualFunctional, run_regularized_solver
   from sdplab.examples import generate_max_cut

   problem = generate_max_cut(8, seed=0, unit_trace=True)
   dual = RegularizedSDPDualFunctional(problem, EntropyReg(problem.dom))

   result = run_regularized_solver(dual.bind(0.1), verbose=0)
   X = dual.primal_from_dual(result.dual, 0.1)

   problem.primal_objective(X)     # -5.0859, against -5.0990 from CVXPY

:math:`\varepsilon` is supplied per call rather than stored, so a continuation
schedule can lower it without rebuilding the functional. The instance is built
with ``unit_trace=True`` because ``primal_from_dual`` normalizes to
:math:`\operatorname{Tr}[X] = 1`: on a problem whose feasible set has a different
trace the recovered :math:`X` would not be feasible for it.
