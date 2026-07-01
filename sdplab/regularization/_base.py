r"""Base class for spectral regularizers for SDPs.

... (docstring unchanged) ...
"""
from __future__ import annotations

from typing import Any, Callable, Tuple
from abc import abstractmethod
from dataclasses import dataclass

from spacecore import (
    DenseArray,
    ArrayLike,
    Context,
    jax_pytree_class,
    ContextBound,
    EuclideanJordanAlgebraSpace,
    TreeSpace,
    TreeSpectralDecomposition,
    resolve_context_priority
)

from ._eval_space import create_eigval_space

_UNSET = object()
SpaceElement = Any


@jax_pytree_class
@dataclass(init=False)
class Regularizer(ContextBound):
    r"""Base class for scalar spectral regularizers.  ... (docstring unchanged) ..."""

    def __init__(
        self,
        space: EuclideanJordanAlgebraSpace,
        ctx: Context | str | None = None,
    ):
        ctx = resolve_context_priority(ctx, space)
        super(Regularizer, self).__init__(ctx)
        if space is None:
            raise ValueError("Regularizer requires a domain space.")
        if not space.is_euclidean:
            raise NotImplementedError(
                "Regularization currently supports only Euclidean Jordan spaces."
            )
        self.space: EuclideanJordanAlgebraSpace = space.convert(self.ctx)
        self.eigval_space = create_eigval_space(self.space)

    # ---- scalar spectral operations (subclass contract) ---------------------
    @abstractmethod
    def phi(self, x: DenseArray) -> DenseArray:
        r"""Scalar convex penalty :math:`\varphi` applied to primal eigenvalues."""

    @abstractmethod
    def phi_star(self, x: DenseArray) -> DenseArray:
        r"""Legendre transform :math:`\psi` applied to scaled slack eigenvalues."""

    @abstractmethod
    def phi_star_prime(self, x: DenseArray) -> DenseArray:
        r"""Derivative :math:`\psi'` recovering primal eigenvalues."""

    @abstractmethod
    def log_phi_star_prime(self, x: DenseArray) -> DenseArray:
        r"""Log-form :math:`\log\psi'` for stable normalization."""

    # ---- structure-safe spectral plumbing -----------------------------------
    def _eigvals(self, X: SpaceElement) -> SpaceElement:
        """Eigenvalues of ``X`` as a member of ``self.eigval_space`` (no frame).

        On trees this reduces per leaf instead of via ``TreeSpace.spectrum``,
        which concatenates leaf spectra along the last axis and fails when leaves
        have different spectrum rank (e.g. a stacked leaf).
        """
        space = self.space
        if isinstance(space, TreeSpace):
            parts = tuple(
                leaf.spectrum(component)
                for leaf, component in zip(space.leaf_spaces, space._components(X))
            )
            return self.eigval_space.unflatten_tree(parts)
        return space.spectrum(X)

    def _decompose(self, X: SpaceElement) -> Tuple[SpaceElement, Any]:
        """Return ``(eigvals, recon)`` — eigenvalues as an ``eigval_space`` member
        plus the opaque frame data that ``_reconstruct`` needs."""
        space = self.space
        decomposition = space.spectral_decompose(X)
        if isinstance(space, TreeSpace):
            # TreeSpace returns a TreeSpectralDecomposition, not an (evals, frame) pair.
            eigvals = self.eigval_space.unflatten_tree(decomposition.eigvals)
            return eigvals, decomposition
        eigvals, frame = decomposition
        return eigvals, frame

    def _reconstruct(self, eigvals: SpaceElement, recon: Any) -> SpaceElement:
        """Rebuild a ``self.space`` element from transformed eigenvalues."""
        space = self.space
        if isinstance(space, TreeSpace):
            recon = TreeSpectralDecomposition(
                eigvals=self.eigval_space.flatten_tree(eigvals),
                frames=recon.frames,
            )
            return space.from_spectrum(recon)
        return space.from_spectrum(eigvals, recon)

    def _trace(self, eigvals: SpaceElement) -> DenseArray:
        """Additive sum of every eigenvalue across the (structured) eigval space."""
        return self.ops.sum(self.eigval_space.flatten(eigvals))

    # ---- public API ---------------------------------------------------------
    def __call__(self, X: SpaceElement, val: float) -> DenseArray:
        r"""Evaluate :math:`\varepsilon \operatorname{Tr}[\varphi(X)]`."""
        eigvals = self._eigvals(X)
        eigvals = self.eigval_space.spectral_apply(eigvals, self.phi)
        return self._trace(eigvals) * val

    def legendre(self, X: SpaceElement, val: float) -> DenseArray:
        r"""Evaluate :math:`\varepsilon \operatorname{Tr}[\psi(X/\varepsilon)]`."""
        eigvals = self._eigvals(X)
        eigvals = self.eigval_space.scale(1.0 / val, eigvals)
        eigvals = self.eigval_space.spectral_apply(eigvals, self.phi_star)
        return self._trace(eigvals) * val

    def legendre_and_grad(
        self, X: SpaceElement, val: float, normalized: bool = False
    ) -> Tuple[DenseArray, SpaceElement]:
        eigvals, recon = self._decompose(X)
        scaled = self.eigval_space.scale(1.0 / val, eigvals)
        legendre = self._trace(self.eigval_space.spectral_apply(scaled, self.phi_star)) * val
        if normalized:
            grad_eigvals = self._robust_normalization(scaled)
        else:
            grad_eigvals = self.eigval_space.spectral_apply(scaled, self.phi_star_prime)
        return legendre, self._reconstruct(grad_eigvals, recon)

    def phi_star_prime_matrix(
        self, X: ArrayLike, val: float, normalized: bool = True
    ) -> ArrayLike:
        r"""Return the primal matrix :math:`(\varphi^*)'(X / \varepsilon)`."""
        eigvals, recon = self._decompose(X)
        scaled = self.eigval_space.scale(1.0 / val, eigvals)
        if normalized:
            grad_eigvals = self._robust_normalization(scaled)
        else:
            grad_eigvals = self.eigval_space.spectral_apply(scaled, self.phi_star_prime)
        return self._reconstruct(grad_eigvals, recon)

    def _robust_normalization(self, eigvals: SpaceElement) -> SpaceElement:
        r"""Softmax of :math:`\log\psi'(s_i/\varepsilon)`; the recovered element
        has unit trace. ``eigvals`` holds the already-scaled slack eigenvalues.

        The log-sum-exp is a *global* reduction over the whole spectrum, computed
        on the flattened vector — routing it through ``spectral_apply`` would
        raise, since that method forbids shape-reducing maps. Only the final
        ``exp`` is entrywise.
        """
        log_psi_prime = self.eigval_space.spectral_apply(eigvals, self.log_phi_star_prime)
        lse = self.ops.logsumexp(self.eigval_space.flatten(log_psi_prime), axis=-1)
        return self.eigval_space.spectral_apply(
            log_psi_prime, lambda ev: self.ops.exp(ev - lse)
        )

    # ---- context / copy / pytree -------------------------------------------
    def _convert(self, new_ctx: Context) -> "Regularizer":
        """Return this regularizer represented in ``new_ctx``."""
        return self._copy_with(ctx=new_ctx)

    def with_space(self, space: EuclideanJordanAlgebraSpace) -> "Regularizer":
        return self._copy_with(space=space, ctx=space.ctx)

    def _extra_dynamic_children(self) -> Tuple[Any, ...]:
        """Return subclass backend-array PyTree children."""
        return ()

    def _extra_static_aux(self) -> Tuple[Any, ...]:
        """Return subclass static PyTree auxiliary state."""
        return ()

    def _restore_extra_state(
        self, dynamic_children: Tuple[Any, ...], static_aux: Tuple[Any, ...]
    ) -> None:
        """Restore subclass state from PyTree/copy hooks."""

    def _convert_extra_dynamic_children(
        self, dynamic_children: Tuple[Any, ...], new_ctx: Context
    ) -> Tuple[Any, ...]:
        return tuple(new_ctx.ops.asarray(child) for child in dynamic_children)

    def _copy_with(
        self,
        *,
        space: EuclideanJordanAlgebraSpace | object = _UNSET,
        ctx: Context | None = None,
    ) -> "Regularizer":
        new_space = self.space if space is _UNSET else space
        new_ctx = self.ctx if ctx is None else ctx

        obj = type(self).__new__(type(self))
        obj.space = new_space.convert(new_ctx)
        obj._ctx = obj.space.ctx                       # normalized Context
        obj.eigval_space = create_eigval_space(obj.space)

        extra_dynamic = self._extra_dynamic_children()
        if obj._ctx != self.ctx:
            extra_dynamic = self._convert_extra_dynamic_children(extra_dynamic, obj._ctx)
        obj._restore_extra_state(extra_dynamic, self._extra_static_aux())
        return obj

    def tree_flatten(self):
        """Children are subclass dynamic arrays; ε is a per-call argument, not state."""
        return self._extra_dynamic_children(), (
            self.space,
            self.ctx,
            self._extra_static_aux(),
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild a regularizer from JAX PyTree data."""
        space, ctx, extra_static = aux
        obj = cls.__new__(cls)
        obj._ctx = ctx
        obj.space = space
        obj.eigval_space = create_eigval_space(space)   # was missing before
        obj._restore_extra_state(tuple(children), extra_static)
        return obj
