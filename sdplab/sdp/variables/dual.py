r"""Dual SDP variable wrapper.

For an SDP with
:math:`\mathcal{A} : \operatorname{dom}(\mathcal{A}) \to \operatorname{cod}(\mathcal{A})`,
the dual variable :math:`y \in \operatorname{cod}(\mathcal{A})` corresponds
to the equality constraint :math:`\mathcal{A}X = b`. The adjoint expression
:math:`\mathcal{A}^\dagger y - C \in \operatorname{dom}(\mathcal{A})` is the
matrix object used by dual feasibility and regularized primal recovery.
"""

from __future__ import annotations


from spacecore import Space, jax_pytree_class, Context, ArrayLike
from ._base import SDPVar


@jax_pytree_class
class SDPDual(SDPVar):
    r"""Dual variable associated with the SDP equality constraints.

    If
    :math:`\mathcal{A} : \operatorname{dom}(\mathcal{A}) \to \operatorname{cod}(\mathcal{A})`
    and :math:`b \in \operatorname{cod}(\mathcal{A})`, then this wrapper stores
    :math:`y \in \operatorname{cod}(\mathcal{A})`. Its linear objective
    contribution is :math:`\operatorname{Tr}[b\ y]`. Applying the adjoint
    operator gives
    :math:`\mathcal{A}^\dagger y \in \operatorname{dom}(\mathcal{A})`, which can
    be compared with :math:`C \in \operatorname{dom}(\mathcal{A})` or
    diagonalized through :math:`\mathcal{A}^\dagger y - C`.

    A useful way to read a dual variable is:

        "This is my vector of prices or multipliers for the equalities."

    The dual variable rewards or penalizes violation of
    :math:`\mathcal{A}X = b` through the trace pairing
    :math:`\operatorname{Tr}[(\mathcal{A}X - b)y]`.
    """

    def __init__(
            self,
            space: Space,
            y: ArrayLike,
            ctx: Context | str | None = None,
    ):
        r"""Create a dual variable by validating :math:`y \in \operatorname{cod}(\mathcal{A})`."""
        super(SDPDual, self).__init__(space, ctx)
        self.space.check_member(y)
        self.y = self.space.ctx.asarray(y)

    @property
    def val(self) -> ArrayLike:
        """Return the coordinate representation of the dual variable ``y``."""
        return self.y

    def _new_like(self, new_val: ArrayLike) -> SDPDual:
        r"""Return another dual variable in the same space :math:`\operatorname{cod}(\mathcal{A})`."""
        return SDPDual(self.space, new_val)

    def tree_flatten(self):
        """Return children and auxiliary data for JAX PyTree flattening."""
        return (self.y,), (self.space,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild a dual variable from JAX PyTree data."""
        (y,) = children
        (space,) = aux
        return cls(space, y)

    def _convert(self, new_ctx: Context) -> SDPDual:
        """Return this dual variable converted to ``new_ctx``."""
        return SDPDual(self.space, self.y, new_ctx)
