from __future__ import annotations

import numpy as np
import pytest

from qotlib.core.backend import BackendContext
from qotlib.core.backend.numpy import NumpyOps
from qotlib.core.space import DenseVectorSpace


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def ctx_np():
    return BackendContext(
        ops=NumpyOps(),
        dtype=np.float64,
        enable_checks=True,
    )


@pytest.fixture
def space(ctx_np):
    n = 7
    return DenseVectorSpace(ctx=ctx_np, shape=(n,), n=n)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def rand_vec(n: int, seed: int = 0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(dtype)
    return x


def rand_cvec(n: int, seed: int = 0, dtype=np.complex128):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    return (a + 1j * b).astype(dtype)


# ---------------------------------------------------------------------
# Construction & membership
# ---------------------------------------------------------------------
def test_construct_rejects_nonpositive_n(ctx_np):
    """Reject vector spaces with non-positive dimension."""
    with pytest.raises(ValueError, match="n must be positive"):
        DenseVectorSpace(ctx=ctx_np, shape=(0,), n=0)


def test_check_member_accepts_correct_shape(space):
    """Accept members with matching dense shape."""
    x = np.zeros((space.n,), dtype=space.ctx.dtype)
    space.check_member(x)  # should not raise


def test_check_member_rejects_wrong_shape(space):
    """Reject members with incorrect shape."""
    x = np.zeros((space.n, 1), dtype=space.ctx.dtype)
    with pytest.raises(TypeError, match=r"Expected shape"):
        space.check_member(x)


def test_check_member_rejects_non_dense(space):
    """Reject non-dense vector members."""
    with pytest.raises(TypeError):
        space.check_member([0.0] * space.n)  # not a numpy.ndarray


# ---------------------------------------------------------------------
# Zeros and algebra
# ---------------------------------------------------------------------
def test_zeros_is_member_and_identity(space):
    """Ensure zeros is a member and additive identity."""
    z = space.zeros()
    x = rand_vec(space.n, seed=1, dtype=space.ctx.dtype)

    space.check_member(z)
    assert np.allclose(space.add(x, z), x)
    assert np.allclose(space.add(z, x), x)


def test_add_commutative_and_associative(space):
    """Check add commutativity and associativity."""
    x = rand_vec(space.n, seed=2, dtype=space.ctx.dtype)
    y = rand_vec(space.n, seed=3, dtype=space.ctx.dtype)
    z = rand_vec(space.n, seed=4, dtype=space.ctx.dtype)

    assert np.allclose(space.add(x, y), space.add(y, x))
    assert np.allclose(space.add(space.add(x, y), z), space.add(x, space.add(y, z)))


def test_scale_distributive(space):
    """Check scale distributivity over addition."""
    x = rand_vec(space.n, seed=5, dtype=space.ctx.dtype)
    y = rand_vec(space.n, seed=6, dtype=space.ctx.dtype)
    a = 2.25

    lhs = space.scale(a, space.add(x, y))
    rhs = space.add(space.scale(a, x), space.scale(a, y))
    assert np.allclose(lhs, rhs)


def test_axpy_matches_add_scale(space):
    """Ensure axpy matches scale+add semantics."""
    x = rand_vec(space.n, seed=7, dtype=space.ctx.dtype)
    y = rand_vec(space.n, seed=8, dtype=space.ctx.dtype)
    a = -0.75

    got = space.axpy(a, x, y)
    exp = space.add(space.scale(a, x), y)
    assert np.allclose(got, exp)


# ---------------------------------------------------------------------
# Inner product & norm
# ---------------------------------------------------------------------
def test_inner_matches_numpy_vdot(space):
    """Confirm inner product matches NumPy vdot."""
    x = rand_vec(space.n, seed=9, dtype=space.ctx.dtype)
    y = rand_vec(space.n, seed=10, dtype=space.ctx.dtype)

    got = space.inner(x, y)
    exp = np.vdot(x, y)
    assert np.allclose(got, exp)


def test_inner_conjugate_symmetry_complex_dtype():
    """Verify conjugate symmetry for complex inner products."""
    ctx = BackendContext(ops=NumpyOps(), dtype=np.complex128, enable_checks=True)
    n = 6
    S = DenseVectorSpace(ctx=ctx, shape=(n,), n=n)

    x = rand_cvec(n, seed=11, dtype=ctx.dtype)
    y = rand_cvec(n, seed=12, dtype=ctx.dtype)

    xy = S.inner(x, y)
    yx = S.inner(y, x)
    assert np.allclose(xy, np.conj(yx))


def test_norm_consistency(space):
    """Check norm consistency with inner product."""
    x = rand_vec(space.n, seed=13, dtype=space.ctx.dtype)
    lhs = space.norm(x) ** 2
    rhs = np.real(space.inner(x, x))
    assert np.allclose(lhs, rhs)


# ---------------------------------------------------------------------
# Flatten / unflatten
# ---------------------------------------------------------------------
def test_flatten_is_identity(space):
    """Ensure flatten returns the same dense vector."""
    x = rand_vec(space.n, seed=14, dtype=space.ctx.dtype)
    v = space.flatten(x)
    assert v is x  # DenseVectorSpace.flatten returns x as-is
    assert np.allclose(v, x)


def test_unflatten_reshapes(space):
    """Ensure unflatten reshapes vectors correctly."""
    v = rand_vec(space.n, seed=15, dtype=space.ctx.dtype)
    x = space.unflatten(v)
    assert tuple(x.shape) == (space.n,)
    assert np.allclose(x, v)


# ---------------------------------------------------------------------
# eigh is not defined
# ---------------------------------------------------------------------
def test_eigh_raises(space):
    """Ensure vector spaces reject eigendecomposition."""
    x = rand_vec(space.n, seed=16, dtype=space.ctx.dtype)
    with pytest.raises(TypeError, match=r"eigh is not defined for vector spaces"):
        space.eigh(x)


# ---------------------------------------------------------------------
# Checks disabled policy
# ---------------------------------------------------------------------
def test_checks_disabled_skips_membership_validation():
    """Verify operations proceed when checks are disabled."""
    ctx = BackendContext(ops=NumpyOps(), dtype=np.float64, enable_checks=False)
    n = 5
    S = DenseVectorSpace(ctx=ctx, shape=(n,), n=n)

    # Wrong shape, but check_member is gated → operations can proceed (policy choice)
    bad = np.zeros((n, 1), dtype=ctx.dtype)

    # add/scale will operate at NumPy level; shape follows NumPy broadcasting rules.
    out = S.scale(2.0, bad)
    assert out.shape == (n, 1)
