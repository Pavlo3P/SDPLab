Regularization policy
=====================

SDPLab regularizers are spectral. They act on a Hermitian matrix through its
eigenvalues:

.. math::

   X = V \operatorname{diag}(\lambda) V^\dagger.

For separable regularizers, a scalar convex function :math:`\varphi` is lifted
to the matrix penalty

.. math::

   R(X) = \varepsilon \operatorname{Tr}[\varphi(X)].

The trace notation means

.. math::

   \operatorname{Tr}[\varphi(X)]
   =
   \sum_i \varphi(\lambda_i).

The Legendre transform of :math:`\varphi` is denoted by :math:`\psi`:

.. math::

   \psi(s)
   =
   \sup_t \{s t - \varphi(t)\}.

The smooth separable regularized dual objective is

.. math::

   D_\varepsilon(y) =
   \operatorname{Tr}[b y]
   - \varepsilon \operatorname{Tr}\left[\psi\left(\frac{A^\dagger y - C}{\varepsilon}\right)\right],

where :math:`\psi` is the Legendre transform of :math:`\varphi`. The map
``primal_from_dual`` uses the first-order relation
:math:`\lambda_i(X) = \psi'(s_i / \varepsilon)` in the eigenbasis of
:math:`A^\dagger y - C`, :math:`s_i`.

The log-trace-exponential entropy variant is not separable on the dual side.
It uses the coupled function

.. math::

   F^*(S)
   =
   \varepsilon \log\operatorname{Tr}\exp(S / \varepsilon)
   =
   \varepsilon \log\left(\sum_i \exp(s_i / \varepsilon)\right).

Its gradient has eigenvalues

.. math::

   \frac{\exp(s_i / \varepsilon)}
        {\sum_j \exp(s_j / \varepsilon)},

so the recovered matrix always has trace one.

Why spectral regularizers
-------------------------

Semidefinite constraints are spectral constraints. The condition
:math:`X \succeq 0` says that all eigenvalues of :math:`X` are nonnegative.
Regularizing eigenvalues therefore matches the geometry of the SDP cone.

Primal recovery
---------------

Let :math:`s_i` be eigenvalues of :math:`\mathcal{A}^\dagger y - C`. Regularized
dual methods recover primal eigenvalue weights by applying
:math:`\psi'` to scaled slack eigenvalues:

.. math::

   \lambda_i(X)
   =
   \psi'\left(\frac{s_i}{\varepsilon}\right).

The recovered eigenvectors are the eigenvectors of
:math:`\mathcal{A}^\dagger y - C`. For separable regularizers, ``phi_star``
implements :math:`\psi`, ``phi_star_prime`` implements :math:`\psi'`, and
``log_phi_star_prime`` implements :math:`\log(\psi')`. For ``EntropyRegLog``,
``phi_star`` is intentionally unavailable because the conjugate is coupled;
its derivative methods return normalized exponential weights for the full
spectrum.

Built-in policies
-----------------

Entropy regularization maps slack eigenvalues through an exponential rule.
Quadratic regularization maps them through a nonnegative clipping rule.
Both are implemented by subclasses of
:class:`sdplab.regularization.Regularizer`.

The entropy-log variant changes the dual regularization from
:math:`\operatorname{Tr}[\psi(S)]` to
:math:`\log(\operatorname{Tr}[\psi(S)])`. This logarithmic dual term is for
trace-normalized problems with :math:`\tau = 1`.
