Spaces and operators
====================

SDPLab deliberately delegates array ownership, validation, and linear-operator
structure to SpaceCore. This keeps the SDP code close to the mathematical
model:

.. math::

   C \in \mathrm{dom},
   \qquad
   \mathcal{A}: \mathrm{dom} \to \mathrm{cod},
   \qquad
   b \in \mathrm{cod}.

Problem construction validates these relationships:

* ``C`` must belong to ``A.dom``.
* ``b`` must belong to ``A.cod``.
* Primal arrays must belong to the problem domain.
* Dual arrays must belong to the problem codomain.

Context conversion
------------------

Most public SDPLab objects inherit SpaceCore's context-bound behavior. Calling
``convert(ctx)`` returns the same mathematical object represented in a new
backend context.

For example, a dense SDP problem can be converted from a NumPy context to a
JAX context while preserving the same symbolic data:

.. code-block:: python

   converted = problem.convert("jax")

The conversion changes array representation, not the mathematical problem.

Trace pairing
-------------

Dense objectives are evaluated using the trace pairing

.. math::

   \operatorname{Re}\operatorname{Tr}[C X].

The real part appears because complex Hermitian arithmetic may contain tiny
imaginary roundoff even when the mathematical value is real.
