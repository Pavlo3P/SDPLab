import numpy as np
import pytest

from qotlib.core import BackendContext, NumpyOps
from qotlib.linalg import stochastic_lanczos


@pytest.fixture
def np_ctx_real64():
    return BackendContext(
        ops=NumpyOps(),
        dtype=np.float64,
        allow_sparse=True,
        enable_checks=True,
    )


@pytest.fixture
def np_ctx_cplx128():
    return BackendContext(
        ops=NumpyOps(),
        dtype=np.complex128,
        allow_sparse=True,
        enable_checks=True,
    )


def random_hermitian(rng: np.random.Generator, n: int, complex: bool):
    if complex:
        A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return (A + A.conj().T) / 2
    else:
        A = rng.normal(size=(n, n))
        return (A + A.T) / 2


def rayleigh(M: np.ndarray, x: np.ndarray) -> float:
    return float(np.vdot(x, M @ x).real / np.vdot(x, x).real)


def test_lanczos_numpy_random_hermitian_real_fixed_seed(np_ctx_real64):
    rng = np.random.default_rng(12345)
    n = 32
    H = random_hermitian(rng, n, complex=False)

    lam_true = float(np.linalg.eigh(H)[0][0])
    v0 = rng.normal(size=(n,))

    def mvp(v):
        return H @ v

    lam, x = stochastic_lanczos(
        np_ctx_real64, mvp, v0,
        max_iter=60,   # a bit more room
        tol=1e-12,
    )

    lam = float(np.asarray(lam))
    x = np.asarray(x)

    assert np.isfinite(lam)
    assert np.all(np.isfinite(x))
    assert np.isclose(np.linalg.norm(x), 1.0, rtol=1e-7, atol=1e-7)

    # Lanczos is approximate: keep a realistic tolerance
    assert np.isclose(lam, lam_true, rtol=1e-4, atol=1e-6)
    assert np.isclose(lam, rayleigh(H, x), rtol=1e-10, atol=1e-10)


def test_lanczos_numpy_random_hermitian_complex_fixed_seed(np_ctx_cplx128):
    rng = np.random.default_rng(54321)
    n = 24
    H = random_hermitian(rng, n, complex=True)

    lam_true = float(np.linalg.eigh(H)[0][0])

    def mvp(v):
        return H @ v

    # Deterministic restarts (THIS is what makes the test robust)
    best_lam = np.inf
    best_x = None

    restarts = 12
    max_iter = 2 * n  # reasonable for n=24; avoid huge max_iter>>n
    tol = 1e-12

    for _ in range(restarts):
        v0 = rng.normal(size=(n,)) + 1j * rng.normal(size=(n,))
        lam, x = stochastic_lanczos(
            np_ctx_cplx128, mvp, v0,
            max_iter=max_iter,
            tol=tol,
        )
        lam = float(np.asarray(lam))
        x = np.asarray(x)

        assert np.isfinite(lam)
        assert np.all(np.isfinite(x))
        assert np.isclose(np.linalg.norm(x), 1.0, rtol=1e-7, atol=1e-7)

        if lam < best_lam:
            best_lam = lam
            best_x = x

    # Best result across restarts should match the true minimum reasonably well
    # (tolerance is deliberately modest; Lanczos is approximate and draw-dependent)
    assert np.isclose(best_lam, lam_true, rtol=1e-3, atol=1e-4)

    # And must be consistent with the Rayleigh quotient of the returned vector
    assert np.isclose(best_lam, rayleigh(H, best_x), rtol=1e-10, atol=1e-10)


def test_lanczos_numpy_orthogonal_start_fixed_seed(np_ctx_real64):
    """
    Deterministic orthogonal-start test: with max_iter=1, the Krylov subspace is span{v0},
    so Lanczos cannot recover the min eigenvector if v0 ⟂ v_min.
    """
    rng = np.random.default_rng(999)
    n = 16
    H = random_hermitian(rng, n, complex=False)

    eigvals, eigvecs = np.linalg.eigh(H)
    lam_true = float(eigvals[0])
    v_min = eigvecs[:, 0]

    # Build v0 orthogonal to v_min (up to numerical precision)
    v0 = rng.normal(size=(n,))
    v0 = v0 - np.dot(v0, v_min) * v_min
    v0 /= np.linalg.norm(v0)

    # Sanity: ensure orthogonality is actually achieved numerically
    assert abs(float(np.dot(v0, v_min))) < 1e-10

    def mvp(v):
        return H @ v

    # Force 1-step Krylov: result must equal Rayleigh quotient of v0
    lam, x = stochastic_lanczos(
        np_ctx_real64, mvp, v0,
        max_iter=1,     # KEY: only span{v0}
        tol=1e-30,
    )

    lam = float(np.asarray(lam))
    x = np.asarray(x)

    rq0 = rayleigh(H, v0)
    assert np.isclose(lam, rq0, rtol=1e-12, atol=1e-12)
    assert lam >= lam_true - 1e-12
    # and typically strictly larger (unless multiplicity / degenerate spectrum)
    assert lam > lam_true + 1e-12
