Regularization API
==================

Regularizers define scalar spectral formulas and
:class:`sdplab.regularization.SDPRegularized` lifts them to SDP objectives.
``EntropyReg`` and ``QuadraticReg`` are separable. ``EntropyRegLog`` is the
coupled log-trace-exponential variant whose gradients are normalized to trace
one.

.. autosummary::
   :nosignatures:

   sdplab.regularization.SDPRegularized
   sdplab.regularization.Regularizer
   sdplab.regularization.EntropyReg
   sdplab.regularization.EntropyRegLog
   sdplab.regularization.QuadraticReg

Regularized problem wrapper
---------------------------

.. autoclass:: sdplab.regularization.SDPRegularized
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

Scalar spectral regularizers
----------------------------

.. autoclass:: sdplab.regularization.Regularizer
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: sdplab.regularization.EntropyReg
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: sdplab.regularization.EntropyRegLog
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: sdplab.regularization.QuadraticReg
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
