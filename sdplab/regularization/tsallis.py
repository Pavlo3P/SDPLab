r"""Tsallis (:math:`q`-deformed) spectral regularizer for regularized SDPs.

.. math::

    R_\varepsilon(X) = \varepsilon\operatorname{Tr}[\varphi_q(X)],
    \qquad
    \varphi_q(t) = \frac{t^q - t}{q-1} + \iota_{[0,\infty)}(t),

interpolating von Neumann entropy (:math:`q\to1`) and the quadratic penalty
(:math:`q=2`). The conjugate derivative is the **q-exponential**

.. math::

    \psi_q'(s) = \left[\frac{\max\{1+(q-1)s,\,0\}}{q}\right]^{1/(q-1)},
    \qquad \psi_q(s) = \psi_q'(s)^q,

i.e. the :math:`\alpha`-entmax family: softmax at :math:`q=1`, sparsemax at
:math:`q=2`. For every :math:`q>1` it has **compact support**, so the recovered
primal is exactly low rank rather than full rank with an
:math:`O(e^{-\Delta/\varepsilon})` tail.
"""

from spacecore import DenseArray, jax_pytree_class

from ._base import NEG_EIG_TOL, Regularizer

__all__ = ["TsallisReg"]


@jax_pytree_class
class TsallisReg(Regularizer):
    r"""Tsallis entropy of index ``q``.

    Args:
        space: the primal Euclidean Jordan algebra space.
        q: Tsallis index, ``q > 1``. Values in ``(1, 2]`` give a
            compactly-supported (exactly sparse) primal map; ``q = 2`` is the
            quadratic/sparsemax end. For ``q = 1`` use
            :class:`~sdplab.regularization.EntropyReg` -- the formulas here are
            singular there, though they converge to it as ``q -> 1+``.
        normalization: how ``normalized=True`` imposes the unit trace.
            ``"theta"`` (default) is the chemical potential :meth:`_theta`, the
            exact fixed-trace primal; ``"softmax"`` is the base class's
            :math:`g_i/\sum_j g_j`, cheaper but only correct at :math:`q=1`.
            See the *unit-trace primal* section below.
    """

    _NORMALIZATIONS = ("softmax", "theta")

    def __init__(self, space, q: float = 1.5, normalization: str = "theta", ctx=None):
        super(TsallisReg, self).__init__(space, ctx)
        q = float(q)
        if q <= 1.0:
            raise ValueError(
                f"TsallisReg needs q > 1 (got {q}); use EntropyReg for q = 1"
            )
        if normalization not in self._NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {self._NORMALIZATIONS} "
                f"(got {normalization!r})"
            )
        self.q = q
        self.normalization = normalization

    # ---- scalar spectral operations ----------------------------------------

    def phi(self, x: DenseArray) -> DenseArray:
        r"""Return :math:`(x^q - x)/(q-1)` on :math:`x\ge0`, else :math:`+\infty`.

        Round-off-negative eigenvalues (down to ``-NEG_EIG_TOL``) evaluate at
        the limit :math:`\varphi_q(0)=0` rather than out of domain.
        """
        ops = self.ctx.ops
        q = self.q
        safe = ops.maximum(x, 0.0)
        return ops.where(
            x >= -NEG_EIG_TOL, (safe ** q - safe) / (q - 1.0), float("inf")
        )

    def _base_(self, x: DenseArray) -> DenseArray:
        r"""Return :math:`\max\{1+(q-1)x, 0\}/q`, the q-exponential's base."""
        ops = self.ctx.ops
        return ops.maximum(1.0 + (self.q - 1.0) * x, 0.0) / self.q

    def phi_star(self, x: DenseArray) -> DenseArray:
        r"""Return :math:`\psi_q(x) = \psi_q'(x)^q`."""
        return self.phi_star_prime(x) ** self.q

    def phi_star_prime(self, x: DenseArray) -> DenseArray:
        r"""Return the q-exponential :math:`\psi_q'(x)`, exactly zero off support."""
        return self._base_(x) ** (1.0 / (self.q - 1.0))

    def log_phi_star_prime(self, x: DenseArray) -> DenseArray:
        r"""Return :math:`\log\psi_q'(x)`, with :math:`-\infty` off support.

        Evaluated in the log domain so the unit-trace normalization stays
        finite; off-support entries contribute exactly zero weight, which is
        what makes the recovered primal exactly sparse.
        """
        ops = self.ctx.ops
        b = self._base_(x)
        safe = ops.where(b > 0.0, b, 1.0)
        return ops.where(b > 0.0, ops.log(safe) / (self.q - 1.0), float("-inf"))

    # ---- pytree plumbing ----------------------------------------------------

    def _extra_static_aux(self):
        return (self.q, self.normalization)

    def _restore_extra_state(self, dynamic_children, static_aux):
        self.q, self.normalization = static_aux

    # ---- unit-trace primal --------------------------------------------------
    #
    # The base class imposes the unit trace as a softmax over ``log psi'``,
    # i.e. ``g_i / sum_j g_j``. Dividing by the sum can only rescale, so it
    # preserves the ratios g_i/g_j and the zero pattern; the trace constraint's
    # multiplier enters *additively in the argument*, ``psi'((s_i - theta)/eps)``,
    # which changes both. The two coincide only when a shift acts as a global
    # rescale -- ``log psi'`` affine, i.e. the entropy family. For q > 1 the
    # shift also moves the support threshold ``1 + (q-1)x > 0``, so ``"softmax"``
    # is not invariant under ``S -> S + cI`` and its rank depends on that shift;
    # ``"theta"`` is, which is why it is the default; ``"softmax"`` is kept as
    # the cheaper opt-in that matches the base class.

    def _theta(self, u, iters: int = 40):
        r"""Scaled chemical potential ``tau`` with ``sum_i psi'(u_i - tau) = 1``.

        ``tau`` is the dual variable of :math:`\operatorname{Tr}X=1`. Handling
        the trace inside the regularizer is the right place when it is already
        implied by the affine constraints -- an explicit trace row would be
        linearly dependent on them and add a null direction to the Hessian.

        Safeguarded Newton on a monotone scalar equation, run for a **fixed**
        ``iters`` trips of :meth:`ops.fori_loop` so the whole thing traces: no
        host scalars, no data-dependent trip count, no dynamic shapes. With
        :math:`\psi''=\psi'^{\,2-q}/q` the Newton step is closed-form.

        The bracket is closed-form too (a search loop would not trace). Since
        :math:`\psi_q'(1)=1`, the top term alone already reaches the target at
        :math:`\tau=u_{\max}-1`, so that is a valid ``lo``; and every term is
        :math:`\le 1/n` once :math:`x \le (q n^{1-q}-1)/(q-1)`, giving ``hi``.
        Both limits keep the top eigenvalue inside the support, so the log-sum
        below is never empty. As :math:`q\to1` they collapse to
        :math:`[u_{\max}-1,\;u_{\max}-1+\log n]`, the entropy bracket.
        """
        ops = self.ops
        q, e = self.q, 1.0 / (self.q - 1.0)
        n = int(self.ops.shape(u)[-1])

        def log_res_and_slope(t):
            """Return ``(log sum_i psi'(u_i - t), d/dt log sum)``.

            Everything stays in the log domain: the exponent ``1/(q-1)`` is
            ~1000 at q = 1.001, so materializing the sum overflows. ``sum psi'
            = 1`` is solved as ``logsumexp = 0``, and the Newton slope is a
            ratio of two log-sum-exps, hence O(1). Off-support entries are
            masked to ``-inf`` *after* the exponent is applied, never
            multiplied by it -- ``(2 - q) * -inf`` is a NaN at ``q = 2``.
            """
            b = ops.maximum(1.0 + (q - 1.0) * (u - t), 0.0) / q
            pos = b > 0.0
            lg_raw = e * ops.log(ops.where(pos, b, 1.0))
            lg = ops.where(pos, lg_raw, -float("inf"))
            m2 = ops.where(pos, (2.0 - q) * lg_raw, -float("inf"))
            lse = ops.logsumexp(lg, axis=-1)
            lse2 = ops.logsumexp(m2, axis=-1)
            return lse, -ops.exp(lse2 - lse) / q

        u_max = ops.max(u)
        lo = u_max - 1.0
        hi = u_max - (q * n ** (1.0 - q) - 1.0) / (q - 1.0)

        def body(_i, state):
            lo, hi, tau = state
            res, slope = log_res_and_slope(tau)
            # the sum is non-increasing in tau, so log-sum crosses 0 once
            lo = ops.where(res > 0.0, tau, lo)
            hi = ops.where(res > 0.0, hi, tau)
            cand = tau - ops.where(slope < 0.0, res / slope, 0.0)
            inside = (cand > lo) & (cand < hi)
            return lo, hi, ops.where(inside, cand, 0.5 * (lo + hi))  # safeguard

        _lo, _hi, tau = ops.fori_loop(0, iters, body, (lo, hi, 0.5 * (lo + hi)))
        return tau

    def _grad_robust_normalization(self, scaled):
        r"""Unit-trace primal eigenvalues, by ``self.normalization``."""
        if self.normalization == "softmax":
            return super(TsallisReg, self)._grad_robust_normalization(scaled)
        tau = self._theta(self.eigval_space.flatten(scaled))
        return self.eigval_space.spectral_apply(
            scaled, lambda ev: self.phi_star_prime(ev - tau)
        )
