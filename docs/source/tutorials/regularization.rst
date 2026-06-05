Regularized SDPs
================

Regularization adds a spectral penalty to the primal SDP. The unregularized
problem is

.. math::

   \min_X \quad \operatorname{Tr}[C X]
   \quad \text{s.t.} \quad
   \mathcal{A}X = b,\quad X \succeq 0.

Let

.. math::

   X = V \operatorname{diag}(\lambda) V^\dagger.

A separable spectral regularizer is written in trace form as

.. math::

   R(X) = \varepsilon \operatorname{Tr}[\varphi(X)].

The scalar function :math:`\varphi` is applied to :math:`X` spectrally, so

.. math::

   \operatorname{Tr}[\varphi(X)]
   =
   \sum_i \varphi(\lambda_i).

The regularized primal objective is

.. math::

   P_\varepsilon(X)
   =
   \operatorname{Tr}[C X] + R(X).

Dual-Side View
--------------

For separable regularizers, regularized solvers operate with the dual
expression

.. math::

   D_\varepsilon(y)
   =
   \operatorname{Tr}[b y]
   - \varepsilon \operatorname{Tr}\left[\psi\left(\frac{A^\dagger y - C}{\varepsilon}\right)\right],

where :math:`\psi` is the Legendre transform of :math:`\varphi`. The map
``primal_from_dual`` uses the first-order relation
:math:`\lambda_i(X) = \psi'(s_i / \varepsilon)` in the eigenbasis of
:math:`A^\dagger y - C`, :math:`s_i`.

.. math::

   \psi(s)
   =
   \sup_t \{s t - \varphi(t)\}.

Built-in regularizers
---------------------

.. autosummary::
   :nosignatures:

   sdplab.regularization.EntropyReg
   sdplab.regularization.EntropyRegLog
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

The ``EntropyRegLog`` variant is coupled across eigenvalues. It represents

.. math::

   F^*(S)
   =
   \varepsilon \log\operatorname{Tr}\exp(S / \varepsilon)
   =
   \varepsilon \log\left(\sum_i \exp(s_i / \varepsilon)\right).

This is the entropy regularization variant for trace-normalized problems with
:math:`\tau = 1`, such as density-matrix SDPs. Its gradient has normalized
eigenvalues

.. math::

   \frac{\exp(s_i / \varepsilon)}
        {\sum_j \exp(s_j / \varepsilon)},

so the gradient matrix has trace one. Because the conjugate is coupled,
``EntropyRegLog.phi_star`` is intentionally not an elementwise operation.

Quadratic regularization uses

.. math::

   \varphi(t) = \frac{t^2}{2} + \iota_{[0,\infty)}(t),
   \qquad
   \psi(s) = \frac{\max(s, 0)^2}{2}.

In both cases, the scalar formulas are lifted to matrices through the
eigenvalues of Hermitian matrices.
