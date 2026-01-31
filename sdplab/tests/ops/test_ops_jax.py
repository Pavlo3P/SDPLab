import numpy as np
import pytest

@pytest.mark.usefixtures("jax_ops")
def test_jax_ops_basic_correctness(jax_ops):
    """Validate basic reductions and argsort behavior under JAX ops."""
    import jax.numpy as jnp

    a = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    s = jax_ops.sum(a, axis=(1, 2), keepdims=True, dtype=None, out=None, initial=None, where=None)
    s_np = np.sum(np.arange(24, dtype=np.float32).reshape(2, 3, 4), axis=(1, 2), keepdims=True)
    assert np.allclose(np.asarray(s), s_np, rtol=1e-6, atol=1e-6)

    # argsort/sort should accept "unused" params if present in your wrapper signature
    x = jnp.array([3, 1, 2])
    idx = jax_ops.argsort(x, axis=-1, kind=None, order=None, stable=True, descending=False)
    assert np.allclose(np.asarray(idx), np.argsort(np.array([3, 1, 2])))


def test_jax_ops_eigh_stacked(jax_ops):
    """Ensure stacked eigendecomposition works for JAX backend."""
    import jax.numpy as jnp

    x = jnp.zeros((3, 2, 2), dtype=jnp.float64)
    x = x.at[:, 0, 0].set(jnp.array([1.0, 2.0, 3.0]))
    x = x.at[:, 1, 1].set(jnp.array([4.0, 5.0, 6.0]))
    w, v = jax_ops.eigh(x, UPLO="L", symmetrize_input=True)
    assert w.shape == (3, 2)
    assert v.shape == (3, 2, 2)


def test_jax_ops_jit_compatibility(jax_ops):
    """Check JIT compatibility for reshape/sum/ravel in JAX ops."""
    import jax
    import jax.numpy as jnp

    # Keep static args static: axis and keepdims as python literals
    def f(a):
        b = jax_ops.reshape(a, (2, 3, 4), order="C", copy=None, out_sharding=None)
        c = jax_ops.sum(b, axis=(1, 2), keepdims=True, dtype=None, out=None, initial=None, where=None, promote_integers=True)
        d = jax_ops.ravel(c, order="C", out_sharding=None)
        return d

    jf = jax.jit(f)
    a = jnp.arange(24, dtype=jnp.float32)
    out = jf(a)
    assert out.shape == (2,)
    assert np.allclose(np.asarray(out), np.array([66.0, 210.0], dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_jax_sparse_matmul_eager(jax_ops):
    """Validate sparse matmul in eager mode for JAX BCOO."""
    import jax.numpy as jnp
    from jax.experimental import sparse as jsparse

    dense = jnp.array([[1., 0.], [2., 3.]], dtype=jnp.float32)
    A = jsparse.BCOO.fromdense(dense)
    b = jnp.array([4., 5.], dtype=jnp.float32)

    y = jax_ops.sparse_matmul(A, b)
    assert np.allclose(np.asarray(y), np.asarray(dense @ b), rtol=1e-6, atol=1e-6)


def test_jax_sparse_matmul_jit(jax_ops):
    """Validate sparse matmul under JIT for JAX BCOO."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import sparse as jsparse

    dense = jnp.array([[1., 0.], [2., 3.]], dtype=jnp.float32)
    A = jsparse.BCOO.fromdense(dense)
    b = jnp.array([4., 5.], dtype=jnp.float32)

    @jax.jit
    def f(b):
        return jax_ops.sparse_matmul(A, b)

    y = f(b)
    assert np.allclose(np.asarray(y), np.asarray(dense @ b), rtol=1e-6, atol=1e-6)


def test_jax_ops_loops(jax_ops):
    """Validate loop helpers for JAX backend ops."""
    import jax
    from functools import partial
    import jax.numpy as jnp

    init = jnp.array(0, dtype=jnp.int32)

    def body_fun(i, val):
        return val + i

    out = partial(jax.jit, static_argnums=(0, 1, 2))(jax_ops.fori_loop)(0, 5, body_fun, init)
    assert np.allclose(np.asarray(out), np.array(10, dtype=np.int32))

    def cond_fun(val):
        return val < 10

    def while_body(val):
        return val + 4

    out_while = partial(jax.jit, static_argnums=(0, 1))(jax_ops.while_loop)(cond_fun, while_body, init)
    assert np.allclose(np.asarray(out_while), np.array(12, dtype=np.int32))
