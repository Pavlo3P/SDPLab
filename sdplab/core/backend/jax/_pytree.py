from __future__ import annotations
from typing import TypeVar

T = TypeVar("T")

def jax_pytree_class(cls: T) -> T:
    """
    Mark a class as a JAX PyTree node, if JAX is available.

    Safe to import without JAX installed.
    """
    try:
        from jax import tree_util
    except Exception:
        return cls
    try:
        tree_util.register_pytree_node_class(cls)
    except Exception:
        pass
    return cls
