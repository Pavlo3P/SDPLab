from __future__ import annotations

from spacecore import (
    ArrayLike,
    Context,
    DenseArray,
    EuclideanJordanAlgebraSpace,
    jax_pytree_class,
)

from ._base import Regularizer, _validate_positive_scalar


@jax_pytree_class
class EntropyReg(Regularizer):
    r"""Separable entropy regularizer.

    This class uses

    .. math::

        \varphi(t) = t(\log t - 1) + \iota_{[0,\infty)}(t),

    with the convention ``0 * log(0) = 0``. Its scalar conjugate is
    ``phi_star(s) = exp(s)``.
    """

    def phi(self, x: DenseArray) -> DenseArray:
        r"""Return ``x * (log(x) - 1)`` on the nonnegative domain."""
        ops = self.ops
        safe_x = ops.where(x > 0.0, x, 1.0)
        positive = safe_x * (ops.log(safe_x) - 1.0)
        return ops.where(
            x > 0.0,
            positive,
            ops.where(x == 0.0, 0.0, ops.inf),
        )

    def phi_star(self, x: DenseArray) -> DenseArray:
        r"""Return ``exp(x)`` elementwise."""
        return self.ops.exp(x)

    def phi_star_prime(self, x: DenseArray) -> DenseArray:
        r"""Return ``exp(x)`` elementwise."""
        return self.ops.exp(x)

    def log_phi_star_prime(self, x: DenseArray) -> DenseArray:
        r"""Return ``log(phi_star_prime(x)) = x`` elementwise."""
        return x


@jax_pytree_class
class EntropyRegLog(EntropyReg):
    r"""Fixed-trace entropy regularizer.

    For ``tau > 0`` this class represents

    .. math::

        F_\tau(X)
        =
        \varepsilon\operatorname{Tr}(X\log X)
        + \iota_{\{X\succeq0,\ \operatorname{Tr}X=\tau\}}(X).

    Its conjugate is

    .. math::

        F_\tau^*(Y)
        =
        \varepsilon\tau
        \left(
            \log\operatorname{Tr}\exp(Y/\varepsilon)
            - \log\tau
        \right).

    The conjugate gradient is positive semidefinite and has trace ``tau``.
    """

    def __init__(
        self,
        val: DenseArray | float,
        space: EuclideanJordanAlgebraSpace,
        ctx: Context | str | None = None,
        *,
        tau: DenseArray | float = 1.0,
    ) -> None:
        super().__init__(val, space, ctx)
        self.tau = _validate_positive_scalar(tau, self.ctx)

    def phi(self, x: DenseArray) -> DenseArray:
        r"""Return ``x * log(x)`` on the nonnegative domain."""
        ops = self.ops
        safe_x = ops.where(x > 0.0, x, 1.0)
        positive = safe_x * ops.log(safe_x)
        return ops.where(
            x > 0.0,
            positive,
            ops.where(x == 0.0, 0.0, ops.inf),
        )

    def __call__(self, X: ArrayLike) -> DenseArray:
        r"""Evaluate the fixed-trace entropy penalty.

        Inputs outside the fixed-trace domain ``Tr(X) = tau`` receive value
        ``+inf``. The trace is computed spectrally as the sum of eigenvalues.
        """
        if self.space is None:
            raise ValueError("Matrix evaluation requires a regularizer space.")

        eigvals = self.ops.real(self.space.spectrum(X))
        value = self.val * self._phi(eigvals)
        trace = self.ops.sum(eigvals)
        tau = self.ops.asarray(self.tau)

        # Use a backend-compatible tolerance test. This is intentionally not a
        # Python bool, so the method remains usable inside compiled code.
        atol = self.ops.asarray(1e-8)
        rtol = self.ops.asarray(1e-7)
        ok = self.ops.abs(trace - tau) <= atol + rtol * self.ops.abs(tau)
        return self.ops.where(ok, value, self.ops.inf)

    def phi_star(self, x: DenseArray) -> DenseArray:
        raise NotImplementedError(
            "EntropyRegLog has no elementwise conjugate; its conjugate is "
            "coupled across the complete spectrum."
        )

    def _phi_star(self, eigvals: DenseArray) -> DenseArray:
        tau = self.ops.asarray(self.tau)
        return tau * (self.ops.logsumexp(eigvals) - self.ops.log(tau))

    def log_phi_star_prime(self, eigvals: DenseArray) -> DenseArray:
        tau = self.ops.asarray(self.tau)
        return self.ops.log(tau) + eigvals - self.ops.logsumexp(eigvals)

    def phi_star_prime(self, eigvals: DenseArray) -> DenseArray:
        return self.ops.exp(self.log_phi_star_prime(eigvals))

    def _robust_normalization(self, eigvals: DenseArray) -> DenseArray:
        """Return stable exponential weights summing to ``tau``."""
        return self.phi_star_prime(eigvals)

    def _convert(self, new_ctx: Context) -> EntropyRegLog:
        space = self.space.convert(new_ctx) if self.space is not None else None
        return type(self)(
            self.val,
            space,
            ctx=new_ctx,
            tau=self.tau,
        )

    def tree_flatten(self):
        return (self.val, self.tau), (self.space, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        val, tau = children
        space, ctx = aux

        obj = cls.__new__(cls)
        obj._ctx = ctx
        obj.space = space
        obj.val = val
        obj.tau = tau
        return obj
