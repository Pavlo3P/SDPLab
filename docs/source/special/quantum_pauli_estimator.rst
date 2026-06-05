Pauli-Sampling Quantum Estimator
================================

``sdplab.special.quantum`` provides a small dense simulation workflow for
estimating thermal observable values of the form
:math:`\operatorname{Tr}[Q\rho(\alpha)]`.

Pauli Decomposition
-------------------

For ``K`` qubits, a dense Hermitian observable is expanded as

.. math::

   Q = \sum_{\ell \in \{I,X,Y,Z\}^K} c_\ell P_\ell,
   \qquad
   c_\ell = 2^{-K}\operatorname{Tr}(Q P_\ell).

Labels use mathematical tensor order: ``"ZI"`` means :math:`Z \otimes I`.
The dense helper ``decompose_pauli_dense`` enumerates all :math:`4^K` strings,
so it is intended for small systems, debugging, and validation unless the
observable is already sparse in the Pauli basis.

Signed Sampling
---------------

Generic Pauli coefficients may be negative. The sampling distribution therefore
uses absolute values,

.. math::

   p_\ell = |c_\ell| / \|c\|_1.

After measuring the sampled Pauli string and receiving
:math:`m \in \{-1,+1\}`, the estimator uses the signed random variable

.. math::

   Y = \|c\|_1 \operatorname{sign}(c_\ell) m.

Including the sign is what makes the estimator unbiased for
:math:`\operatorname{Tr}[Q\rho]`.

Thermal States
--------------

``build_thermal_state`` constructs

.. math::

   \rho = \frac{\exp(-\beta H)}{\operatorname{Tr}\exp(-\beta H)}

with a Hermitian eigendecomposition and a spectral shift, avoiding a generic
matrix exponential. ``build_linear_hamiltonian`` can form
:math:`H(\alpha)=H_0+\sum_j\alpha_jQ_j`.

Backends
--------

The estimator separates Pauli-string sampling from measurement simulation.

* The exact dense value is computed as :math:`\operatorname{Re}\operatorname{Tr}(Q\rho)`.
* ``DirectDensityMatrixMeasurementBackend`` samples from the exact Pauli
  expectation :math:`\mu_P=\operatorname{Tr}(P\rho)` without circuits.
* ``QiskitDensityMatrixMeasurementBackend`` uses Qiskit Aer, prepares the
  provided density matrix with ``set_density_matrix``, and applies manual
  ``X``/``Y`` basis rotations before computational-basis measurement.

Scaling
-------

The dense state dimension is :math:`d=2^K`, and the dense Pauli decomposition
contains up to :math:`4^K` strings. This module is therefore for small and
medium dense quantum simulations, or for observables that are already sparse in
the Pauli basis.
