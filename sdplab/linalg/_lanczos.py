from __future__ import annotations

from typing import Tuple, Callable

from ..core import BackendContext, DenseArray
from ._misc import MVP


def stochastic_lanczos(
    ctx: BackendContext,
    mvp: MVP,
    initial_vector: DenseArray,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[DenseArray, DenseArray]:
    """
    Backend-agnostic Lanczos for smallest eigenpair approximation.

    Fixes:
      - safe init (no NaNs if initial_vector is zero)
      - remove numpy import (use ops.arange)
      - refine returned eigenvalue via Rayleigh quotient of reconstructed Ritz vector
        (much more reliable than eigvals[0] of padded tridiagonal in practice)
    """
    ops = ctx.ops

    # Ensure dense + cast to context dtype
    v0 = ctx.asarray(initial_vector)
    v0 = ctx.assert_dense(v0)

    n = v0.shape[0]

    # Allocate
    V = ops.zeros((max_iter + 1, n), dtype=ctx.dtype)
    alphas = ops.zeros((max_iter,), dtype=ctx.dtype)
    betas = ops.zeros((max_iter + 1,), dtype=ctx.dtype)

    # Scalars
    tol_s = ops.asarray(tol)
    eps_s = ops.asarray(1e-12)

    # Safe normalize initial vector: if ||v0|| <= eps -> use e0
    v0_norm = ops.sqrt(ops.real(ops.vdot(v0, v0)))

    e0 = ops.zeros((n,), dtype=ctx.dtype)
    e0 = ops.index_set(e0, (0,), ops.asarray(1.0), copy=True)

    v0_unit = ops.cond(
        v0_norm > eps_s,
        lambda _: v0 / v0_norm,
        lambda _: e0,
        ops.asarray(0.0),  # dummy operand
    )
    V = ops.index_set(V, (0, slice(None)), v0_unit, copy=True)

    beta0 = ops.asarray(1.0)  # enter loop
    i0 = 0

    # Precompute indices for masks (static shape)
    full_indices = ops.arange(max_iter + 1)
    idx = ops.arange(max_iter)

    def cond_fun(state):
        i, V_, alphas_, betas_, beta = state
        return (i < max_iter) & ((i == 0) | (beta >= tol_s))

    def body_fun(state):
        i, V_, alphas_, betas_, beta = state

        v_i = V_[i]  # (n,)
        w = mvp(v_i)
        w = ctx.asarray(w)
        w = ctx.assert_dense(w)

        # alpha = <v_i, w>
        alpha = ops.real(ops.vdot(v_i, w))
        alphas_ = ops.index_set(alphas_, (i,), alpha, copy=True)

        # w <- w - alpha*v_i - (i>0)*beta_i*v_{i-1}
        w = ops.cond(
            i == 0,
            lambda w_in: w_in - alpha * v_i,
            lambda w_in: w_in - alpha * v_i - betas_[i] * V_[i - 1],
            w,
        )

        # --- Reorthogonalize against {v_0,...,v_i} using masking ---
        valid = full_indices < (i + 1)
        mask = ops.where(valid, ops.asarray(1.0), ops.asarray(0.0)).astype(w.dtype)

        coeffs_full = ops.einsum("jn,n->j", ops.conj(V_), w)  # (max_iter+1,)
        coeffs_valid = coeffs_full * mask
        proj = ops.sum(coeffs_valid[:, None] * V_, axis=0)
        w = w - proj

        beta_new = ops.sqrt(ops.real(ops.vdot(w, w)))
        betas_ = ops.index_set(betas_, (i + 1,), beta_new, copy=True)

        # Store next vector only if beta_new is safe
        def _set_next(V_in):
            return ops.index_set(V_in, (i + 1, slice(None)), w / beta_new, copy=True)

        V_ = ops.cond(beta_new >= tol_s, _set_next, lambda V_in: V_in, V_)

        return (i + 1, V_, alphas_, betas_, beta_new)

    i_final, V, alphas, betas, beta_final = ops.while_loop(
        cond_fun, body_fun, (i0, V, alphas, betas, beta0)
    )
    m = i_final  # number of Lanczos steps performed (dynamic under JAX)

    # --- Pad unused parts (fixed shapes) ---
    mask_alpha = idx < m
    alphas_full = ops.where(mask_alpha, alphas, ops.asarray(1e10))

    # Decouple boundary: beta_m must be 0 so T doesn't connect into padded part
    betas_full = ops.where(full_indices == m, ops.asarray(0.0), betas)

    # Build dense tridiagonal T of shape (max_iter, max_iter)
    T = ops.zeros((max_iter, max_iter), dtype=ctx.dtype)

    def fill_diag(ii, T_in):
        return ops.index_set(T_in, (ii, ii), alphas_full[ii], copy=True)

    T = ops.fori_loop(0, max_iter, fill_diag, T)

    def fill_off(ii, T_in):
        b = betas_full[ii + 1]  # beta_{ii+1}
        T_in = ops.index_set(T_in, (ii, ii + 1), b, copy=True)
        T_in = ops.index_set(T_in, (ii + 1, ii), b, copy=True)
        return T_in

    T = ops.fori_loop(0, max_iter - 1, fill_off, T)

    eigvals, eigvecs = ops.eigh(T)  # ascending for Hermitian
    y_full = eigvecs[:, 0]          # Ritz vector in Krylov basis

    # Zero out padded components of y
    mask_y = ops.where(idx < m, ops.asarray(1.0), ops.asarray(0.0)).astype(y_full.dtype)
    y_valid = y_full * mask_y

    # Reconstruct approx eigenvector x = sum_j y_j v_j
    V_reduced = V[:max_iter, :]  # static slice length
    x = ops.einsum("j,jn->n", y_valid, V_reduced)

    # Normalize x safely (helps Rayleigh quotient stability)
    x_norm = ops.sqrt(ops.real(ops.vdot(x, x)))
    x = ops.cond(
        x_norm > eps_s,
        lambda _: x / x_norm,
        lambda _: e0,
        ops.asarray(0.0),
    )

    # Rayleigh refinement: λ = <x, A x> / <x, x>
    Ax = mvp(x)
    Ax = ctx.asarray(Ax)
    Ax = ctx.assert_dense(Ax)

    num = ops.real(ops.vdot(x, Ax))
    den = ops.real(ops.vdot(x, x))
    lam = num / den

    return lam, x
