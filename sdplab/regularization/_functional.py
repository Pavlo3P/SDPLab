from __future__ import annotations

from typing import Any, Self, Tuple

from spacecore import Context, Functional, jax_pytree_class

from ._base import Regularizer
from ..problem import SDPProblem


def _logsumexp(x):
    import numpy as np

    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(np.asarray(x) - m))))


@jax_pytree_class
class RegularizedSDPDualFunctional(Functional):
    r"""Smooth regularized dual objective of a conic problem as a :class:`Functional`.

    For a base problem with cost :math:`C \in \operatorname{dom}(\mathcal{A})`,
    operator :math:`\mathcal{A}`, and RHS :math:`b`, coupled to a spectral
    penalty with Legendre transform :math:`\psi`, this evaluates

    .. math::

        D_\varepsilon(y) =
        \langle b, y\rangle
        - \varepsilon \operatorname{Tr}\!\left[
            \psi\!\left(\tfrac{\mathcal{A}^\dagger y - C}{\varepsilon}\right)
          \right],
        \qquad y \in \operatorname{cod}(\mathcal{A}).

    Arguments and results are plain space elements (arrays, or raw trees on a
    :class:`~spacecore.TreeSpace` codomain). This always reports the
    *maximization* objective; sign handling for minimizing optimizers lives on
    the bound view. :math:`\varepsilon` is supplied per call so a continuation
    schedule can vary it without rebuilding the functional; :meth:`bind` fixes
    it and yields a standard single-argument functional.
    """

    def __init__(
        self,
        problem: SDPProblem,
        regularizer: Regularizer,
        ctx: Context | str | None = None,
    ):
        super(RegularizedSDPDualFunctional, self).__init__(problem.cod, ctx)
        self.problem = problem.convert(self.ctx)
        self.regularizer = regularizer.convert(self.ctx)

    def slack(self, y: Any) -> Any:
        r"""Return the dual slack :math:`\mathcal{A}^\dagger y - C` in ``dom``."""
        return self.problem.dual_slack(y)

    def value(
        self, y: Any, eps_val: float, normalized: bool = False, *args: Any, **kwargs: Any
    ) -> Any:
        r"""Return :math:`D_\varepsilon(y)` for strength ``eps_val``.

        ``normalized`` selects the unit-trace primal recovery, which affects
        only the gradient; it is accepted here so ``value`` and ``grad`` share
        a call surface. The trailing ``*args``/``**kwargs`` keep this
        compatible with :class:`~spacecore.Functional` (``__call__``).
        """
        dual_val = self.problem.dual_objective(y)
        return dual_val - self.regularizer.legendre(self.slack(y), eps_val)

    def value_and_grad(
        self, y: Any, eps_val: float, normalized: bool = False, *args: Any, **kwargs: Any
    ) -> Tuple[Any, Any]:
        r"""Return :math:`(D_\varepsilon(y), \nabla_y D_\varepsilon(y))`.

        The gradient is
        :math:`b - \mathcal{A}\,\psi'\!\big((\mathcal{A}^\dagger y - C)/\varepsilon\big)`:
        the Legendre gradient lives in :math:`\operatorname{dom}(\mathcal{A})`
        and is pushed through :math:`\mathcal{A}` before combining with ``b``.
        """
        cod = self.problem.cod
        dual_val = self.problem.dual_objective(y)
        reg_val, reg_grad = self.regularizer.legendre_and_grad(
            self.slack(y), eps_val, normalized
        )
        val = dual_val - reg_val
        grad = cod.axpy(-1.0, self.problem.A.apply(reg_grad), self.problem.b)
        return val, grad

    def grad(
        self, y: Any, eps_val: float, normalized: bool = False, *args: Any, **kwargs: Any
    ) -> Any:
        r"""Return :math:`\nabla_y D_\varepsilon(y)` alone."""
        return self.value_and_grad(y, eps_val, normalized, *args, **kwargs)[1]

    def primal_from_dual(self, y: Any, eps_val: float, normalized: bool = True) -> Any:
        r"""Recover the primal element from a dual iterate.

        With :math:`\mathcal{A}^\dagger y - C = V \operatorname{diag}(s) V^\dagger`
        the eigenvalues are :math:`\lambda_i = \psi'(s_i / \varepsilon)` (or, when
        ``normalized``, their unit-trace normalization), giving
        :math:`X = V \operatorname{diag}(\lambda) V^\dagger` as a plain ``dom``
        element -- the first-order map used to read a primal certificate off a
        dual optimum.
        """
        return self.regularizer.phi_star_prime_matrix(
            self.slack(y), eps_val, normalized
        )

    def bind(
        self,
        eps_val: float,
        *,
        normalized: bool = False,
    ) -> BoundDualFunctional:
        """Return a single-argument functional with ``eps_val`` baked in.

        The result satisfies the plain :class:`~spacecore.Functional` contract
        and can be handed to :func:`spacecore.minimize_scipy` /
        :func:`spacecore.minimize_optax`. To hand this maximization objective to
        a minimizer, negate through the functional algebra: ``-problem.bind(eps)``.
        """
        return BoundDualFunctional(self, eps_val, normalized=normalized)

    def __call__(self, y: Any, *args: Any, **kwargs: Any) -> Any:
        r"""Evaluate ``value`` while forwarding ``eps_val`` and other extras."""
        return self.value(y, *args, **kwargs)

    def _convert(self, new_ctx: Context) -> "RegularizedSDPDualFunctional":
        """Return this functional represented in ``new_ctx``."""
        return RegularizedSDPDualFunctional(
            self.problem, self.regularizer, ctx=new_ctx
        )

    def tree_flatten(self) -> tuple[tuple[Any, ...], Any]:
        """Children are the array-bearing problem and regularizer; ctx is static."""
        return (self.problem, self.regularizer), (self.ctx, )

    @classmethod
    def tree_unflatten(cls, aux: Any, children: Any) -> Self:
        """Rebuild the functional from JAX PyTree data."""
        problem, regularizer = children
        (ctx, ) = aux
        obj = cls.__new__(cls)
        obj.dom = problem.cod
        obj._ctx = ctx
        obj.problem = problem
        obj.regularizer = regularizer
        return obj


@jax_pytree_class
class BoundDualFunctional(Functional):
    r"""A :class:`RegularizedSDPDualFunctional` with ε fixed.

    Satisfies the single-argument :class:`~spacecore.Functional` contract
    expected by the ``spacecore.optimize`` adapters, reporting the same
    maximization objective as its base (use ``-bound`` for a minimization
    view). ``eps_val`` is a pytree leaf, so continuation schedules can rebuild
    bound functionals per ε without retriggering ``jax.jit`` compilation.
    """

    def __init__(
        self,
        base: RegularizedSDPDualFunctional,
        eps_val: float,
        *,
        normalized: bool = False,
    ):
        super(BoundDualFunctional, self).__init__(base.domain, base.ctx)
        self.base = base
        self.eps_val = eps_val
        self.normalized = normalized

    def value(self, y: Any, *args: Any, **kwargs: Any) -> Any:
        return self.base.value(y, self.eps_val, self.normalized)

    def grad(self, y: Any, *args: Any, **kwargs: Any) -> Any:
        return self.base.grad(y, self.eps_val, self.normalized)

    def value_and_grad(self, y: Any, *args: Any, **kwargs: Any) -> Tuple[Any, Any]:
        return self.base.value_and_grad(y, self.eps_val, self.normalized)

    def primal_from_dual(self, y: Any, normalized: bool = True) -> Any:
        """Recover the primal element from a dual iterate at the bound ε."""
        return self.base.primal_from_dual(y, self.eps_val, normalized)

    def _convert(self, new_ctx: Context) -> "BoundDualFunctional":
        return BoundDualFunctional(
            self.base.convert(new_ctx), self.eps_val, normalized=self.normalized
        )

    def tree_flatten(self) -> tuple[tuple[Any, ...], Any]:
        """Children are the base functional and ε; the normalization flag is static."""
        return (self.base, self.eps_val), (self.normalized,)

    @classmethod
    def tree_unflatten(cls, aux: Any, children: Any) -> Self:
        base, eps_val = children
        (normalized,) = aux
        obj = cls.__new__(cls)
        obj.dom = base.dom
        obj._ctx = base.ctx
        obj.base = base
        obj.eps_val = eps_val
        obj.normalized = normalized
        return obj
