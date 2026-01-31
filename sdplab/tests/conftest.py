from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

if "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"


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
def np_ops():
    """Provide a NumPy backend ops instance for tests that need it."""
    from qotlib.core.backend.numpy import NumpyOps

    return NumpyOps()


@pytest.fixture(scope="session")
def jax_ops():
    """Provide a JAX backend ops instance when JAX is installed."""
    if not has_jax():
        pytest.skip("jax is not installed")
    from qotlib.core.backend.jax import JaxOps

    return JaxOps()


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
