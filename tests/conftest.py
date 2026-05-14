from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

if "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
if "JAX_ENABLE_X64" not in os.environ:
    os.environ["JAX_ENABLE_X64"] = "1"


def has_jax() -> bool:
    try:
        import jax  # noqa: F401
    except Exception:
        return False
    return True


def has_scipy() -> bool:
    try:
        import scipy  # noqa: F401
    except Exception:
        return False
    return True


@pytest.fixture(scope="session")
def np_ctx():
    """Provide the NumPy context used by unit tests."""
    import numpy as np
    from spacecore import Context, NumpyOps

    return Context(NumpyOps(), dtype=np.float64)


@pytest.fixture(scope="session")
def jax_ctx():
    """Provide the JAX context used by integration dispatch tests."""
    if not has_jax():
        pytest.skip("jax is not installed")
    import jax.numpy as jnp
    from spacecore import Context, JaxOps

    return Context(JaxOps(), dtype=jnp.float64)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
