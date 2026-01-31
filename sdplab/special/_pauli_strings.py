import jax.numpy as jnp
from flax.core import FrozenDict
from functools import reduce


pauli_matrices = FrozenDict({
        'I': jnp.array([[1, 0], [0, 1]], dtype=jnp.complex64),
        'X': jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64),
        'Y': jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64),
        'Z': jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64),
    })


def generate_single_pauli_string(s: str) -> jnp.ndarray:
    mats = [pauli_matrices[ch] for ch in s]
    return reduce(jnp.kron, mats)


def generate_pauli_observables(observables: list[str]) -> jnp.ndarray:
    """
    Generate a set of Pauli strings.
    :param observables:
    :return:
    """

    pauli_strings = []
    for s in observables:
        pauli_string = generate_single_pauli_string(s)
        pauli_strings.append(pauli_string)

    return jnp.stack(pauli_strings, axis=0)
