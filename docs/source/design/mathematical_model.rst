Mathematical model
==================

SDPLab treats an SDP as data for the problem

.. math::

   \min_X \quad \operatorname{Tr}[C X]
   \quad \text{s.t.} \quad
   \mathcal{A}X = b,\quad X \succeq 0.

The object :class:`sdplab.sdp.SDPProblem` stores exactly the triple
:math:`(C, \mathcal{A}, b)`.

Domains and codomains
---------------------

The domain :math:`\mathrm{dom}` is the space containing the primal matrix and
the cost matrix:

.. math::

   C, X \in \mathrm{dom}.

The codomain :math:`\mathrm{cod}` is the space containing constraint values:

.. math::

   \mathcal{A}X, b, y \in \mathrm{cod}.

The linear constraint operator has type

.. math::

   \mathcal{A}: \mathrm{dom} \to \mathrm{cod}.

This mirrors SpaceCore's operator model. SDPLab does not need to know whether
:math:`\mathrm{cod}` is a vector space, a block matrix space, or another
structured space. It only requires that :math:`\mathcal{A}` has ``dom`` and
``cod`` and provides an adjoint.

Dual slack
----------

The adjoint

.. math::

   \mathcal{A}^\dagger: \mathrm{cod} \to \mathrm{dom}

maps dual variables back into matrix space. The dual slack expression used
throughout the library is

.. math::

   \mathcal{A}^\dagger y - C.

For dense Hermitian SDPs, eigenvalues of this expression determine the
semidefinite order and the regularized primal recovery.
