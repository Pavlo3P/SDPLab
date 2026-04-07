from __future__ import annotations


from spacecore import Space, jax_pytree_class, Context, ArrayLike
from ._base import SDPVar


@jax_pytree_class
class SDPDual(SDPVar):
    def __init__(
            self,
            space: Space,
            y: ArrayLike,
            ctx: Context | str | None = None,
    ):
        super(SDPDual, self).__init__(space, ctx)
        self.space.check_member(y)
        self.y = self.space.ctx.asarray(y)

    @property
    def val(self) -> ArrayLike:
        return self.y

    def _new_like(self, new_val: ArrayLike) -> SDPDual:
        return SDPDual(self.space, new_val)

    def tree_flatten(self):
        return (self.y,), (self.space,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (y,) = children
        (space,) = aux
        return cls(space, y)

    def _convert(self, new_ctx: Context) -> SDPDual:
        return SDPDual(self.space, self.y, new_ctx)