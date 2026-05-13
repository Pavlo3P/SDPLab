r"""Primal SDP variable wrapper.

The primal variable is the optimization variable in

.. math::

    \min_{X \in \operatorname{dom}(\mathcal{A})}\ &\operatorname{Tr}[C X] \\
    \text{s.t.}\quad &\mathcal{A}X = b, \\
                    &X \succeq 0.

Here :math:`C \in \operatorname{dom}(\mathcal{A})` and
:math:`b \in \operatorname{cod}(\mathcal{A})`. For dense SDPs,
:math:`X \in \operatorname{dom}(\mathcal{A})` is a symmetric or Hermitian
matrix and :math:`X \succeq 0` means positive semidefinite.
"""

from __future__ import annotations

from typing import Tuple
from spacecore import Space, jax_pytree_class, Context, ArrayLike, DenseArray
from ._base import SDPVar

@jax_pytree_class
class SDPPrimal(SDPVar):
    r"""Primal matrix variable associated with an SDP domain space.

    ``space`` is interpreted as :math:`\operatorname{dom}(\mathcal{A})`, the
    domain of the constraint operator. The stored value is
    :math:`X \in \operatorname{dom}(\mathcal{A})`. In matrix SDPs this means a
    symmetric or Hermitian matrix. The equality constraints are expressed by
    applying :math:`\mathcal{A}` to this object, and solvers additionally
    enforce the cone constraint :math:`X \succeq 0`.

    A useful way to read a primal variable is:

        "This is my candidate matrix X."

    You can ask the problem for :math:`\mathcal{A}X` to check constraints, ask
    the problem for :math:`\operatorname{Tr}[C X]` to check objective value, or
    diagonalize :math:`X` to inspect positive semidefiniteness.
    """

    def __init__(
        self,
        space: Space,
        X: ArrayLike,
        ctx: Context | str | None = None,
    ):
        r"""Create a primal variable by validating :math:`X \in \operatorname{dom}(\mathcal{A})`."""
        super(SDPPrimal, self).__init__(space, ctx)
        self.space.check_member(X)
        self.X = self.space.ctx.asarray(X)

    @property
    def val(self) -> ArrayLike:
        """Return the coordinate representation of the primal variable ``X``.

        For dense matrix SDPs, this is the actual matrix array.
        """
        return self.X

    def _new_like(self, new_val: ArrayLike) -> SDPPrimal:
        r"""Return another primal variable in the same space :math:`\operatorname{dom}(\mathcal{A})`."""
        return SDPPrimal(self.space, new_val)

    def eigh(self, k: int | None = None) -> Tuple[DenseArray, ArrayLike]:
        r"""Return eigenvalues and eigenvectors of the matrix represented by :math:`X`.

        For :math:`X = V \operatorname{diag}(\lambda)V^\dagger`, this returns
        :math:`\lambda` and :math:`V`. The optional ``k`` is forwarded to the
        space-specific eigensolver.
        """
        return self.space.eigh(self.X, k)

    def tree_flatten(self):
        """Return children and auxiliary data for JAX PyTree flattening."""
        return (self.X,), (self.space,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild a primal variable from JAX PyTree data."""
        (X,) = children
        (space,) = aux
        return cls(space, X)

    def _convert(self, new_ctx: Context) -> SDPPrimal:
        """Return this primal variable converted to ``new_ctx``."""
        return SDPPrimal(self.space, self.X, new_ctx)
