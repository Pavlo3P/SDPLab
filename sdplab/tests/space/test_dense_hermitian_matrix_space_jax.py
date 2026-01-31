from __future__ import annotations

import pytest


def _jax_import():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


from qotlib.core.backend import BackendContext
from qotlib.core.backend.jax import JaxOps
from qotlib.core.space import DenseHermitianMatrixSpace


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def jax_jnp():
    return _jax_import()


@pytest.fixture
def ctx_jax(jax_jnp):
    _, jnp = jax_jnp
    return BackendContext(
        ops=JaxOps(),
        dtype=jnp.complex128,
        enable_checks=True,
    )


@pytest.fixture
def ctx_jax_no_checks(jax_jnp):
    _, jnp = jax_jnp
    return BackendContext(
        ops=JaxOps(),
        dtype=jnp.complex128,
        enable_checks=False,
    )


@pytest.fixture
def space(ctx_jax):
    n = 5
    return DenseHermitianMatrixSpace(
        ctx=ctx_jax,
        n=n,
        atol=1e-12,
        rtol=0.0,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def hermitian(jax, jnp, n: int, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    a = jax.random.normal(key, (n, n), dtype=jnp.float64)
    key2 = jax.random.fold_in(key, 1)
    b = jax.random.normal(key2, (n, n), dtype=jnp.float64)
    A = a + 1j * b
    return (A + jnp.conj(A).T) / 2.0


# ---------------------------------------------------------------------
# Construction & membership (eager-only checks)
# ---------------------------------------------------------------------
def test_rejects_nonpositive_dimension(ctx_jax):
    """Reject Hermitian spaces with non-positive dimension."""
    with pytest.raises(ValueError):
        DenseHermitianMatrixSpace(ctx=ctx_jax, n=0)


def test_accepts_valid_hermitian(space, jax_jnp):
    """Accept valid Hermitian matrices as members."""
    jax, jnp = jax_jnp
    X = hermitian(jax, jnp, space.n, seed=0)
    space.check_member(X)  # should not raise


def test_rejects_wrong_shape(space, jax_jnp):
    """Reject matrices with wrong shape."""
    _, jnp = jax_jnp
    bad = jnp.zeros((space.n, space.n + 1), dtype=space.ctx.dtype)
    with pytest.raises(TypeError):
        space.check_member(bad)


def test_rejects_non_hermitian(space, jax_jnp):
    """Reject matrices that violate Hermitian symmetry."""
    jax, jnp = jax_jnp
    X = hermitian(jax, jnp, space.n, seed=1)
    X = X.at[0, 1].add(1e-6)
    with pytest.raises(TypeError):
        space.check_member(X)


# ---------------------------------------------------------------------
# Hermitian logic (eager-only checks)
# ---------------------------------------------------------------------
def test_is_hermitian_tolerance(space, jax_jnp):
    """Validate Hermitian tolerance thresholds."""
    jax, jnp = jax_jnp
    X = hermitian(jax, jnp, space.n, seed=2)

    X1 = X.at[0, 1].add(5e-13)
    assert space.is_hermitian(X1)

    X2 = X.at[0, 1].add(1e-6)
    assert not space.is_hermitian(X2)


def test_symmetrize_is_projection(space, jax_jnp):
    """Ensure symmetrize projects to Hermitian space."""
    jax, jnp = jax_jnp
    key = jax.random.PRNGKey(3)
    a = jax.random.normal(key, (space.n, space.n), dtype=jnp.float64)
    key2 = jax.random.fold_in(key, 1)
    b = jax.random.normal(key2, (space.n, space.n), dtype=jnp.float64)
    A = a + 1j * b

    H = space.symmetrize(A)
    assert jnp.allclose(H, jnp.conj(H).T)
    assert jnp.allclose(space.symmetrize(H), H)


# ---------------------------------------------------------------------
# Algebra (eager + check policy)
# ---------------------------------------------------------------------
def test_zeros_identity(space, jax_jnp):
    """Ensure zeros is the additive identity."""
    jax, jnp = jax_jnp
    Z = space.zeros()
    X = hermitian(jax, jnp, space.n, seed=4)

    assert jnp.allclose(space.add(X, Z), X)
    assert jnp.allclose(space.add(Z, X), X)


def test_add_commutes(space, jax_jnp):
    """Check commutativity of addition."""
    jax, jnp = jax_jnp
    X = hermitian(jax, jnp, space.n, seed=5)
    Y = hermitian(jax, jnp, space.n, seed=6)

    assert jnp.allclose(space.add(X, Y), space.add(Y, X))


def test_scale_real_ok_complex_rejected(space, jax_jnp):
    """Allow real scaling and reject complex scaling."""
    jax, jnp = jax_jnp
    X = hermitian(jax, jnp, space.n, seed=7)

    Y = space.scale(2.0, X)
    space.check_member(Y)

    with pytest.raises(TypeError):
        space.scale(1.0 + 1.0j, X)


# ---------------------------------------------------------------------
# Inner product & norm (eager)
# ---------------------------------------------------------------------
def test_inner_conjugate_symmetry(space, jax_jnp):
    """Verify conjugate symmetry of inner product."""
    jax, jnp = jax_jnp
    X = hermitian(jax, jnp, space.n, seed=8)
    Y = hermitian(jax, jnp, space.n, seed=9)

    xy = space.inner(X, Y)
    yx = space.inner(Y, X)
    assert jnp.allclose(xy, jnp.conj(yx))


def test_norm_consistency(space, jax_jnp):
    """Check norm matches inner product consistency."""
    jax, jnp = jax_jnp
    X = hermitian(jax, jnp, space.n, seed=10)

    lhs = space.norm(X) ** 2
    rhs = jnp.real(space.inner(X, X))
    assert jnp.allclose(lhs, rhs)


# ---------------------------------------------------------------------
# Flatten / unflatten (eager)
# ---------------------------------------------------------------------
def test_flatten_shape(space, jax_jnp):
    """Ensure flatten produces the expected vector length."""
    jax, jnp = jax_jnp
    X = hermitian(jax, jnp, space.n, seed=11)

    v = space.flatten(X)
    assert tuple(v.shape) == (space.n * space.n,)


def test_unflatten_returns_hermitian(space, jax_jnp):
    """Ensure unflatten returns a Hermitian matrix."""
    jax, jnp = jax_jnp
    X = hermitian(jax, jnp, space.n, seed=12)

    v = space.flatten(X)
    Y = space.unflatten(v)

    assert tuple(Y.shape) == (space.n, space.n)
    assert jnp.allclose(Y, jnp.conj(Y).T)
    assert jnp.allclose(X, Y)


# ---------------------------------------------------------------------
# Eigen-decomposition (eager)
# ---------------------------------------------------------------------
def test_eigh_reconstruction(space, jax_jnp):
    """Reconstruct a Hermitian matrix from its eigendecomposition."""
    jax, jnp = jax_jnp
    X = hermitian(jax, jnp, space.n, seed=13)

    w, U = space.eigh(X)
    X_rec = U @ jnp.diag(w) @ jnp.conj(U).T
    assert jnp.allclose(X_rec, X)


# ---------------------------------------------------------------------
# JIT compatibility tests (enable_checks=False)
# ---------------------------------------------------------------------
def test_jit_add_matches_eager(ctx_jax_no_checks, jax_jnp):
    """Check JIT add matches eager results."""
    jax, jnp = jax_jnp
    n = 5
    S = DenseHermitianMatrixSpace(ctx=ctx_jax_no_checks, n=n)

    X = hermitian(jax, jnp, n, seed=20)
    Y = hermitian(jax, jnp, n, seed=21)

    eager = S.add(X, Y)

    @jax.jit
    def f(A, B):
        return S.add(A, B)

    out = f(X, Y)
    assert jnp.allclose(out, eager)


def test_jit_scale_inner_norm_flatten_unflatten(ctx_jax_no_checks, jax_jnp):
    """Check JIT compatibility for scale/inner/norm/flatten/unflatten."""
    jax, jnp = jax_jnp
    n = 4
    S = DenseHermitianMatrixSpace(ctx=ctx_jax_no_checks, n=n)

    X = hermitian(jax, jnp, n, seed=22)
    Y = hermitian(jax, jnp, n, seed=23)

    eager_scale = S.scale(2.0, X)
    eager_inner = S.inner(X, Y)
    eager_norm = S.norm(X)
    eager_flat = S.flatten(X)
    eager_unflat = S.unflatten(eager_flat)

    @jax.jit
    def g(A, B):
        return (
            S.scale(2.0, A),
            S.inner(A, B),
            S.norm(A),
            S.flatten(A),
            S.unflatten(S.flatten(A)),
        )

    Z, ip, nx, v, A2 = g(X, Y)

    assert jnp.allclose(Z, eager_scale)
    assert jnp.allclose(ip, eager_inner)
    assert jnp.allclose(nx, eager_norm)
    assert jnp.allclose(v, eager_flat)
    assert jnp.allclose(A2, eager_unflat)
    assert jnp.allclose(A2, jnp.conj(A2).T)  # unflatten symmetrizes


def test_jit_eigh_reconstruction(ctx_jax_no_checks, jax_jnp):
    """Ensure JIT eigendecomposition reconstructs the input."""
    jax, jnp = jax_jnp
    n = 4
    S = DenseHermitianMatrixSpace(ctx=ctx_jax_no_checks, n=n)

    X = hermitian(jax, jnp, n, seed=24)

    # Eager reconstruction
    w, U = S.eigh(X)
    X_rec_eager = U @ jnp.diag(w) @ jnp.conj(U).T

    @jax.jit
    def h(A):
        w_, U_ = S.eigh(A)
        return U_ @ jnp.diag(w_) @ jnp.conj(U_).T

    X_rec_jit = h(X)
    assert jnp.allclose(X_rec_jit, X_rec_eager)
    assert jnp.allclose(X_rec_jit, X)
