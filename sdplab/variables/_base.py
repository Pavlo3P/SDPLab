r"""Shared arithmetic for SDP variables.

An SDP variable is an element of a vector space used by an SDP model.
Primal variables are elements :math:`X \in \operatorname{dom}(\mathcal{A})`,
while dual variables are elements
:math:`y \in \operatorname{cod}(\mathcal{A})`.

This base class delegates addition, scaling, inner products, and norms to the
underlying ``space`` object, so the same arithmetic works for NumPy, JAX, real,
complex, dense, and product spaces.
"""

from __future__ import annotations

from typing import Any, Tuple
from abc import abstractmethod

from spacecore import ArrayLike, Space, Context, ContextBound, resolve_context_priority


class SDPVar(ContextBound):
    r"""Base class for typed elements of SDP vector spaces.

    The wrapper records the mathematical space containing the variable. Two
    variables can be added or paired by an inner product only when they have
    the same concrete SDP variable type. This prevents accidentally combining
    primal elements :math:`X \in \operatorname{dom}(\mathcal{A})` with dual
    elements :math:`y \in \operatorname{cod}(\mathcal{A})`.

    This class is deliberately thin. The actual geometry, such as Hermitian
    matrix inner products or product-space norms, belongs to ``space``.
    """
    
    def __init__(
        self,
        space: Space,
        ctx: Context | str | None = None,
    ):
        """Bind this variable to its mathematical ``space`` and backend context."""
        ctx = resolve_context_priority(ctx, space)
        super(SDPVar, self).__init__(ctx)
        self.space = space.convert(ctx)

    @property
    @abstractmethod
    def val(self) -> ArrayLike:
        """Return the underlying coordinate representation of the variable."""
        ...

    def __post_init__(self):
        """Validate that ``val`` is a member of ``space``."""
        self.space.check_member(self.val)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the coordinate-array shape of this variable."""
        return self.val.shape

    @abstractmethod
    def _new_like(self, new_val: ArrayLike) -> SDPVar:
        """Create another variable of the same SDP-variable type and space."""
        ...

    def __add__(self, other: Any):
        if type(other) is not type(self):
            return NotImplemented
        y = self.space.add(self.val, other.val)
        return self._new_like(y)

    def __radd__(self, other: Any):
        # For sum(..., start=0) patterns; treat 0 as neutral if desired.
        if other == 0:
            return self
        return NotImplemented

    def __sub__(self, other: Any):
        if type(other) is not type(self):
            return NotImplemented
        y = self.space.add(self.val, self.space.scale(-1, other.val))
        return self._new_like(y)

    def __neg__(self):
        return self._new_like(self.space.scale(-1, self.val))

    def __mul__(self, a: Any):
        y = self.space.scale(a, self.val)
        return self._new_like(y)

    def __rmul__(self, a: Any):
        y = self.space.scale(a, self.val)
        return self._new_like(y)

    def __truediv__(self, a: Any):
        y = self.space.scale(1 / a, self.val)
        return self._new_like(y)

    def axpy(self, a: Any, y: SDPVar):
        """Return ``a * self + y`` in this variable's vector space."""
        if type(y) is not type(self):
            raise TypeError(f"axpy expects {type(self).__name__}, got {type(y).__name__}")
        res = self.space.axpy(a, self.val, y.val)
        return self._new_like(res)

    def inner(self, other: SDPVar):
        """Return the inner product induced by the underlying vector space."""
        if type(other) is not type(self):
            raise TypeError(f"inner expects {type(self).__name__}, got {type(other).__name__}")
        return self.space.inner(self.val, other.val)

    def norm(self):
        """Return the norm induced by the underlying vector space."""
        return self.space.norm(self.val)
