from __future__ import annotations

import numpy as np
import pytest

from qotlib.core.backend import BackendContext
from qotlib.core.backend.numpy import NumpyOps
from qotlib.core.low_rank import LowRankHermitianMatrixSpace, LowRankMatrix


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def ctx_np():
    return BackendContext(
        ops=NumpyOps(),
        dtype=np.complex128,
        enable_checks=True,
    )


@pytest.fixture
def space(ctx_np):
    n = 12
    r = 4
    return LowRankHermitianMatrixSpace(
        ctx=ctx_np,
        shape=(n, n),
        n=n,
        max_rank=r,
    )


# -----------------------------------------------------------------------------
# Helpers (match LowRankMatrix: eigvecs is (n, r))
# -----------------------------------------------------------------------------
def _col_orthonormal_eigvecs(n: int, r: int, seed: int, dtype) -> np.ndarray:
    """
    Return V with shape (n, r) with orthonormal columns: V^H V = I_r.
    """
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((n, r)) + 1j * rng.standard_normal((n, r))).astype(dtype)
    Q, _ = np.linalg.qr(A)  # (n, r), columns orthonormal
    assert np.allclose(Q.conj().T @ Q, np.eye(r), atol=1e-10)
    return Q


def make_lowrank(ctx: BackendContext, n: int, r: int, seed: int) -> LowRankMatrix:
    V = _col_orthonormal_eigvecs(n, r, seed=seed, dtype=ctx.dtype)

    rng = np.random.default_rng(seed + 123)
    # For Hermitian: eigenvalues should be real; dtype should typically match V.real.dtype.
    s = rng.standard_normal(r).astype(V.real.dtype)

    return LowRankMatrix(ctx=ctx, max_rank=r, eigvals=s, eigvecs=V)


def dense_hs_inner(A: np.ndarray, B: np.ndarray) -> complex:
    # Hilbert–Schmidt / Frobenius inner product
    return np.vdot(A, B)


# -----------------------------------------------------------------------------
# Construction invariants
# -----------------------------------------------------------------------------
def test_space_construct_rejects_nonpositive(ctx_np):
    """Reject invalid dimensions and max_rank in space construction."""
    with pytest.raises(ValueError):
        LowRankHermitianMatrixSpace(ctx=ctx_np, shape=(1, 1), n=0, max_rank=2)
    with pytest.raises(ValueError):
        LowRankHermitianMatrixSpace(ctx=ctx_np, shape=(1, 1), n=2, max_rank=0)


# -----------------------------------------------------------------------------
# Membership: accept / reject
# -----------------------------------------------------------------------------
def test_accepts_valid_lowrank(space):
    """Accept valid LowRankMatrix members."""
    x = make_lowrank(space.ctx, space.n, space.max_rank, seed=1)
    space.check_member(x)  # should not raise


def test_rejects_wrong_type(space):
    """Reject non-LowRankMatrix members."""
    bad = np.zeros((space.n, space.n), dtype=space.ctx.dtype)
    with pytest.raises(TypeError):
        space.check_member(bad)


def test_rejects_wrong_ctx(space):
    """Reject members with mismatched backend context."""
    other_ctx = BackendContext(ops=NumpyOps(), dtype=space.ctx.dtype, enable_checks=True)
    x = make_lowrank(other_ctx, space.n, space.max_rank, seed=2)

    # Your space should enforce ctx coincidence.
    with pytest.raises(TypeError, match="ctx"):
        space.check_member(x)


def test_rejects_wrong_n_dimension(space):
    """Reject members whose dimension does not match the space."""
    # eigvecs shape is (n+1, r), so LowRankMatrix itself is valid,
    # but space membership should reject due to n mismatch.
    x = make_lowrank(space.ctx, space.n + 1, space.max_rank, seed=3)
    with pytest.raises(TypeError, match=r"n"):
        space.check_member(x)


def test_rejects_incompatible_eigval_dtype(space):
    """Reject incompatible eigval dtype relative to eigvecs."""
    # Keep eigvecs dtype correct but make eigvals dtype incompatible with your policy.
    V = _col_orthonormal_eigvecs(space.n, space.max_rank, seed=4, dtype=space.ctx.dtype)

    # Typical policy: eigvals.dtype == eigvecs.real.dtype (float64 here).
    s_bad = np.ones((space.max_rank,), dtype=np.float32)

    x = LowRankMatrix(ctx=space.ctx, max_rank=space.max_rank, eigvals=s_bad, eigvecs=V)
    with pytest.raises(TypeError):
        space.check_member(x)


# -----------------------------------------------------------------------------
# zeros()
# -----------------------------------------------------------------------------
def test_zeros_contract(space):
    """Ensure zeros returns a valid LowRankMatrix with correct shapes."""
    z = space.zeros()
    space.check_member(z)

    assert isinstance(z, LowRankMatrix)
    assert z.ctx is space.ctx
    assert z.max_rank == space.max_rank
    assert tuple(z.eigvals.shape) == (space.max_rank,)
    assert len(z.eigvecs.shape) == 2
    assert z.eigvecs.shape[1] == space.max_rank
    assert z.eigvecs.shape[0] == space.n


# -----------------------------------------------------------------------------
# add(): policy is to materialize dense
# -----------------------------------------------------------------------------
def test_add_materializes_dense(space):
    """Verify add materializes a dense Hermitian matrix."""
    x = make_lowrank(space.ctx, space.n, space.max_rank, seed=10)
    y = make_lowrank(space.ctx, space.n, space.max_rank, seed=11)

    Z = space.add(x, y)

    assert isinstance(Z, np.ndarray)
    assert Z.shape == (space.n, space.n)
    assert np.allclose(Z, Z.conj().T)


# -----------------------------------------------------------------------------
# scale()
# -----------------------------------------------------------------------------
def test_scale_real_ok(space):
    """Allow real scaling for low-rank matrices."""
    x = make_lowrank(space.ctx, space.n, space.max_rank, seed=12)
    y = space.scale(2.0, x)
    space.check_member(y)

    assert np.allclose(y.to_dense(), 2.0 * x.to_dense())


def test_scale_complex_rejected(space):
    """Reject complex scaling for Hermitian low-rank matrices."""
    x = make_lowrank(space.ctx, space.n, space.max_rank, seed=13)
    with pytest.raises(TypeError):
        space.scale(1.0 + 1.0j, x)


# -----------------------------------------------------------------------------
# inner(): matches dense Hilbert–Schmidt; conjugate symmetry
# -----------------------------------------------------------------------------
def test_inner_matches_dense(space):
    """Ensure inner product matches dense Hilbert–Schmidt product."""
    x = make_lowrank(space.ctx, space.n, space.max_rank, seed=20)
    y = make_lowrank(space.ctx, space.n, space.max_rank, seed=21)

    got = space.inner(x, y)
    exp = dense_hs_inner(x.to_dense(), y.to_dense())

    assert np.allclose(got, exp)
    assert np.allclose(got, np.conj(space.inner(y, x)))


# -----------------------------------------------------------------------------
# eigh(): returns the stored eigenpairs in this representation
# -----------------------------------------------------------------------------
def test_eigh_returns_stored(space):
    """Ensure eigh returns stored eigenpairs."""
    x = make_lowrank(space.ctx, space.n, space.max_rank, seed=22)
    w, V = space.eigh(x)

    assert np.allclose(w, x.eigvals)
    assert np.allclose(V, x.eigvecs)


# -----------------------------------------------------------------------------
# flatten / unflatten
# -----------------------------------------------------------------------------
def test_flatten_shape(space):
    """Ensure flatten returns expected vector shape."""
    x = make_lowrank(space.ctx, space.n, space.max_rank, seed=30)
    v = space.flatten(x)

    assert isinstance(v, np.ndarray)
    assert v.shape == (space.n * space.n,)


def test_unflatten_returns_lowrank(space):
    """Ensure unflatten returns a valid LowRankMatrix."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((space.n, space.n)) + 1j * rng.standard_normal((space.n, space.n))
    H = (A + A.conj().T) / 2.0
    v = H.ravel()

    x = space.unflatten(v)

    assert isinstance(x, LowRankMatrix)
    assert x.ctx is space.ctx
    assert x.max_rank == space.max_rank
    space.check_member(x)

    Xd = x.to_dense()
    assert Xd.shape == (space.n, space.n)
    assert np.allclose(Xd, Xd.conj().T)


def test_unflatten_truncates_largest_abs_eigs(space):
    """Ensure unflatten keeps largest-magnitude eigenvalues."""
    """
    If your unflatten policy is "keep max_rank eigenpairs with largest |eigval|",
    this test locks that in.
    """
    n, r = space.n, space.max_rank
    rng = np.random.default_rng(123)

    # Random unitary U via QR
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    U, _ = np.linalg.qr(A)

    # Distinct magnitudes
    w = np.linspace(10.0, 1.0, n).astype(np.float64)
    H = U @ np.diag(w) @ U.conj().T
    H = (H + H.conj().T) / 2.0

    x = space.unflatten(H.ravel())
    space.check_member(x)

    expected = np.sort(np.abs(w))[-r:]
    got = np.sort(np.abs(np.asarray(x.eigvals)))
    assert np.allclose(got, expected, atol=1e-10, rtol=1e-10)


# -----------------------------------------------------------------------------
# Checks disabled policy: ctx mismatch should be skipped
# -----------------------------------------------------------------------------
def test_checks_disabled_skips_ctx_validation():
    """Ensure checks-disabled skips context validation."""
    ctx = BackendContext(ops=NumpyOps(), dtype=np.complex128, enable_checks=False)
    n, r = 6, 2
    S = LowRankHermitianMatrixSpace(ctx=ctx, shape=(n, n), n=n, max_rank=r)

    other_ctx = BackendContext(ops=NumpyOps(), dtype=np.complex128, enable_checks=True)
    x = make_lowrank(other_ctx, n, r, seed=40)

    # With checks disabled, space.check_member is a no-op; boundary ops should run.
    v = S.flatten(x)
    assert v.shape == (n * n,)
