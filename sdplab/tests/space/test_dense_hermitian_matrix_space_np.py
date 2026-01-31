from __future__ import annotations

import numpy as np
import pytest

from qotlib.core.backend import BackendContext
from qotlib.core.backend.numpy import NumpyOps
from qotlib.core.space import DenseHermitianMatrixSpace

@pytest.fixture
def ctx():
    return BackendContext(
        ops=NumpyOps(),
        dtype=np.complex128,
        enable_checks=True,
    )


@pytest.fixture
def space(ctx):
    n = 5
    return DenseHermitianMatrixSpace(
        ctx=ctx,
        n=n,
        atol=1e-12,
        rtol=0.0,
    )


def hermitian(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return (A + A.conj().T) / 2


# ---------------------------------------------------------------------
# Construction & membership
# ---------------------------------------------------------------------
def test_rejects_nonpositive_dimension(ctx):
    """Reject Hermitian spaces with non-positive dimension."""
    with pytest.raises(ValueError):
        DenseHermitianMatrixSpace(ctx=ctx, n=0)


def test_accepts_valid_hermitian(space):
    """Accept valid Hermitian matrices as members."""
    X = hermitian(space.n)
    space.check_member(X)  # should not raise


def test_rejects_wrong_shape(space):
    """Reject matrices with wrong shape."""
    bad = np.zeros((space.n, space.n + 1), dtype=space.ctx.dtype)
    with pytest.raises(TypeError):
        space.check_member(bad)


def test_rejects_non_hermitian(space):
    """Reject matrices that violate Hermitian symmetry."""
    X = hermitian(space.n)
    X[0, 1] += 1e-6
    with pytest.raises(TypeError):
        space.check_member(X)


# ---------------------------------------------------------------------
# Hermitian logic
# ---------------------------------------------------------------------
def test_is_hermitian_tolerance(space):
    """Validate Hermitian tolerance thresholds."""
    X = hermitian(space.n)
    X[0, 1] += 5e-13
    assert space.is_hermitian(X)

    X[0, 1] += 1e-6
    assert not space.is_hermitian(X)


def test_symmetrize_is_projection(space):
    """Ensure symmetrize projects to Hermitian space."""
    A = np.random.randn(space.n, space.n) + 1j * np.random.randn(space.n, space.n)
    H = space.symmetrize(A)

    assert np.allclose(H, H.conj().T)
    assert np.allclose(space.symmetrize(H), H)


# ---------------------------------------------------------------------
# Algebra
# ---------------------------------------------------------------------
def test_zeros_identity(space):
    """Ensure zeros is the additive identity."""
    Z = space.zeros()
    X = hermitian(space.n)

    assert np.allclose(space.add(X, Z), X)
    assert np.allclose(space.add(Z, X), X)


def test_add_commutes(space):
    """Check commutativity of addition."""
    X = hermitian(space.n, seed=1)
    Y = hermitian(space.n, seed=2)

    assert np.allclose(space.add(X, Y), space.add(Y, X))


def test_scale_real_ok_complex_rejected(space):
    """Allow real scaling and reject complex scaling."""
    X = hermitian(space.n)

    Y = space.scale(2.0, X)
    space.check_member(Y)

    with pytest.raises(TypeError):
        space.scale(1.0 + 1.0j, X)


# ---------------------------------------------------------------------
# Inner product & norm
# ---------------------------------------------------------------------
def test_inner_conjugate_symmetry(space):
    """Verify conjugate symmetry of inner product."""
    X = hermitian(space.n, seed=3)
    Y = hermitian(space.n, seed=4)

    xy = space.inner(X, Y)
    yx = space.inner(Y, X)

    assert np.allclose(xy, np.conj(yx))


def test_norm_consistency(space):
    """Check norm matches inner product consistency."""
    X = hermitian(space.n)
    lhs = space.norm(X) ** 2
    rhs = np.real(space.inner(X, X))

    assert np.allclose(lhs, rhs)


# ---------------------------------------------------------------------
# Flatten / unflatten
# ---------------------------------------------------------------------
def test_flatten_shape(space):
    """Ensure flatten produces the expected vector length."""
    X = hermitian(space.n)
    v = space.flatten(X)
    assert v.shape == (space.n * space.n,)


def test_unflatten_returns_hermitian(space):
    """Ensure unflatten returns a Hermitian matrix."""
    X = hermitian(space.n)
    v = space.flatten(X)
    Y = space.unflatten(v)

    assert Y.shape == (space.n, space.n)
    assert np.allclose(Y, Y.conj().T)
    assert np.allclose(X, Y)


# ---------------------------------------------------------------------
# Eigen-decomposition
# ---------------------------------------------------------------------
def test_eigh_reconstruction(space):
    """Reconstruct a Hermitian matrix from its eigendecomposition."""
    X = hermitian(space.n)
    w, U = space.eigh(X)

    X_rec = U @ np.diag(w) @ U.conj().T
    assert np.allclose(X_rec, X)


# ---------------------------------------------------------------------
# Checks disabled policy
# ---------------------------------------------------------------------
def test_checks_disabled_allows_non_hermitian():
    ctx = BackendContext(
        ops=NumpyOps(),
        dtype=np.complex128,
        enable_checks=False,
    )
    n = 4
    S = DenseHermitianMatrixSpace(ctx=ctx, n=n)

    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    # No membership checks → should not raise
    B = S.add(A, A)
    assert B.shape == (n, n)
