import numpy as np
from functools import reduce
from itertools import product
import jax
import jax.numpy as jnp
from qutip import basis, tensor, qeye, ket2dm

from qotlib.special._pauli_strings import generate_pauli_observables
from qotlib.examples import thermal_gaussian_state, qobj_to_jnp
from qotlib.utils import make_projector, make_herm
from qotlib.sdp import SDPDenseProblem
from ._build import make_qubit_tomography_sdp


def ghz_state(n, theta, phi, p, return_vector=False):
    """
    Returns the n-qubit 'tilted GHZ + white noise' state:
        rho = p * |psi(theta)><psi(theta)| + (1-p) * I/2^n
    where |psi(theta)> = cos(theta)|0...0> + sin(theta)|1...1>.

    Parameters
    ----------
    n : int
        Number of qubits.
    theta : float
        Entanglement angle in radians (0 <= theta <= pi/2).
    p : float
        Purity parameter, 0 <= p <= 1.

    Returns
    -------
    qutip.Qobj
        (2^n × 2^n) density matrix.
    """
    # Single-qubit basis states
    zero = basis(2, 0)
    one = basis(2, 1)

    # Build n-qubit |00…0> and |11…1>
    psi_0 = tensor([zero] * n)
    psi_1 = tensor([one] * n)

    # Tilted GHZ superposition
    psi = np.cos(theta) * psi_0 + np.sin(theta) * np.exp(1j * phi) * psi_1

    if return_vector:
        return jnp.asarray(psi.data.to_array())

    # Pure-state density matrix
    rho_pure = ket2dm(psi)

    # Mix with white noise
    dim = 2 ** n
    rho = p * rho_pure + (1 - p) * qeye(rho_pure.dims[0]) / dim

    return jnp.asarray(rho.data.to_array())


def load_tomography_instance(i: int):
    names = ['I', 'X', 'Y', 'Z']

    # Instance 0
    if i == 0:
        n = 2
        theta = np.pi / 6
        phi = np.pi / 4
        p = .5
        state = ghz_state(n, theta, phi, p)
        # observables = ['X', 'Y', 'I']
        observables = [reduce(lambda x, y: x + y, _) for _ in product(names, repeat=n)]


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Instance 1
    elif i == 1:
        n = 2
        theta = np.pi / 6
        phi = np.pi / 4
        p = 1.

        cnot = jnp.block([
            [jnp.eye(2), jnp.zeros((2, 2))],
            [jnp.zeros((2, 2)), jnp.array([[0., 1.], [1., 0.]], dtype=float)]
        ])
        psi = ghz_state(n, theta, phi, p, return_vector=True)
        psi = cnot @ psi
        state = make_projector(psi)
        observables = [reduce(lambda x, y: x + y, _) for _ in product(names, repeat=n)]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Instance 2
    elif i == 2:
        n = 2
        theta = np.pi / 3
        phi = np.pi / 6
        p = .7
        state = ghz_state(n, theta, phi, p)
        observables = [reduce(lambda x, y: x + y, _) for _ in product(names, repeat=n)]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Instance 3 - exponential matrix, or thermal state
    elif i == 3:
        n = 6
        state = qobj_to_jnp(thermal_gaussian_state(2 ** n, temperature=10.))
        observables = [reduce(lambda x, y: x + y, _) for _ in product(names, repeat=n)]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Instance 4 - exponential matrix, or thermal state 2
    elif i == 4:
        n = 3
        state = qobj_to_jnp(thermal_gaussian_state(2 ** n, temperature=5.))
        observables = [reduce(lambda x, y: x + y, _) for _ in product(names, repeat=n)]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Instance 5 - exponential matrix, in the form exp(\sum_{i = 1}^N alpha_i Q_i), where Q_1, ..., Q_N are observables
    elif i == 5:
        n = 6
        observables = [reduce(lambda x, y: x + y, _) for _ in product(names, repeat=n)]
        M = generate_pauli_observables(observables)
        alpha = jax.random.normal(jax.random.PRNGKey(0), 2 ** (2 * n))
        M = jnp.einsum('i,ijk->jk', alpha, M)
        evals, evecs = jnp.linalg.eigh(M)
        evals = jax.nn.softmax(evals)
        state = (evecs * evals) @ evecs.T.conj()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Instance 6 - exponential matrix, in the form exp(\sum_{i = 1}^N alpha_i Q_i), where Q_1, ..., Q_N are observables; other seed
    elif i == 6:
        n = 6
        observables = [reduce(lambda x, y: x + y, _) for _ in product(names, repeat=n)]
        M = generate_pauli_observables(observables)
        alpha = jax.random.normal(jax.random.PRNGKey(42), 2 ** (2 * n)) * 50.
        M = jnp.einsum('i,ijk->jk', alpha, M)
        evals, evecs = jnp.linalg.eigh(M)
        evals = jax.nn.softmax(evals)
        state = (evecs * evals) @ evecs.T.conj()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Instance 7 - exponential matrix, in the form exp(\sum_{i = 1}^N alpha_i Q_i), where Q_1, ..., Q_N are observables; other seed
    elif i == 7:
        n = 6
        observables = [reduce(lambda x, y: x + y, _) for _ in product(names, repeat=n)]
        M = generate_pauli_observables(observables)
        alpha = jax.random.bernoulli(jax.random.PRNGKey(42), .5, 2 ** (2 * n))
        M = jnp.einsum('i,ijk->jk', alpha, M)
        evals, evecs = jnp.linalg.eigh(M)
        evals = jax.nn.softmax(evals)
        state = (evecs * evals) @ evecs.T.conj()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Instance 8 - exponential matrix, in the form exp(\sum_{i = 1}^N alpha_i Q_i), where Q_1, ..., Q_N are observables; smaller instance
    elif i == 8:
        n = 3
        observables = [reduce(lambda x, y: x + y, _) for _ in product(names, repeat=n)]
        M = generate_pauli_observables(observables)
        alpha = jax.random.bernoulli(jax.random.PRNGKey(42), .5, 2 ** (2 * n))
        M = jnp.einsum('i,ijk->jk', alpha, M)
        evals, evecs = jnp.linalg.eigh(M)
        evals = jax.nn.softmax(evals)
        state = (evecs * evals) @ evecs.T.conj()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Instance 9 - exponential matrix, in the form exp(\sum_{i = 1}^N alpha_i Q_i), where Q_1, ..., Q_N are observables; other seed, smaller instance
    # no diff between the regs
    elif i == 9:
        n = 3
        observables = [reduce(lambda x, y: x + y, _) for _ in product(names, repeat=n)]
        M = generate_pauli_observables(observables)
        alpha = jax.random.normal(jax.random.PRNGKey(42), 2 ** (2 * n)) * .01
        M = jnp.einsum('i,ijk->jk', alpha, M)
        evals, evecs = jnp.linalg.eigh(M)
        evals = jax.nn.softmax(evals)
        state = (evecs * evals) @ evecs.T.conj()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Instance 10 - exponential matrix, in the form exp(\sum_{i = 1}^N alpha_i Q_i), where Q_1, ..., Q_N are observables; other seed, smaller instance
    # entropy is better
    elif i == 10:
        n = 3
        observables = [reduce(lambda x, y: x + y, _) for _ in product(names, repeat=n)]
        M = generate_pauli_observables(observables)
        alpha = jax.random.normal(jax.random.PRNGKey(42), 2 ** (2 * n)) * 50.
        M = jnp.einsum('i,ijk->jk', alpha, M)
        evals, evecs = jnp.linalg.eigh(M)
        evals = jax.nn.softmax(evals)
        state = (evecs * evals) @ evecs.T.conj()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Instance 11
    elif i == 11:
        n = 4
        theta = np.pi / 6
        phi = np.pi / 4
        p = 0.
        state = ghz_state(n, theta, phi, p)
        # observables = ['X', 'Y', 'I']
        observables = [reduce(lambda x, y: x + y, _) for _ in product(names, repeat=n)]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    else:
        raise AssertionError

    A = generate_pauli_observables(observables)

    return A, state


def generate_quantum_tomography_sdp_instance(
        i: int,
        noise_level: float,
        seed: int = 0
) -> tuple[SDPDenseProblem, jnp.ndarray, float]:
    A, state = load_tomography_instance(i)

    noise = jax.random.normal(jax.random.PRNGKey(seed), state.shape) * noise_level
    noise = make_herm(noise)
    noisy_state = (state + noise) / (jnp.trace(state + noise))

    b = jnp.einsum('ij,kji->k', noisy_state, A).real
    sdp = make_qubit_tomography_sdp(A, b)
    ground_truth = 0.

    return sdp, state, ground_truth
