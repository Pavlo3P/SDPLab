Regularization API
==================

Spectral regularizers and the smoothed dual functional they induce.
A :class:`~sdplab.regularization.Regularizer` supplies a scalar convex
``phi`` applied spectrally; ``bind(eps)`` yields the single-argument
functional the solvers consume.

.. autosummary::
   :nosignatures:

   sdplab.regularization.Regularizer
   sdplab.regularization.RegularizedSDPDualFunctional
   sdplab.regularization.BoundDualFunctional
   sdplab.regularization.EntropyReg
   sdplab.regularization.QuadraticReg

.. autoclass:: sdplab.regularization.Regularizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: sdplab.regularization.RegularizedSDPDualFunctional
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: sdplab.regularization.BoundDualFunctional
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: sdplab.regularization.EntropyReg
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: sdplab.regularization.QuadraticReg
   :members:
   :undoc-members:
   :show-inheritance:
