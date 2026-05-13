"""Constants and multiplication tables for single-qubit Pauli operators."""

from typing import Tuple, Dict

pauli_matrices: Dict[str, Tuple[Tuple[float, float | complex], Tuple[float | complex, float]]] = dict(
    I = (
        (1., 0.),
        (0., 1.),
    ),
    X = (
        (0., 1.),
        (1., 0.),
    ),
    Y = (
        (0., -1j),
        (1j, 0.),
    ),
    Z = (
        (1., 0.),
        (0., -1.),
    ),
)

_PAULI_TO_CODE = dict(
    I=0,
    X=1,
    Y=2,
    Z=3,
)

_CODE_TO_PAULI = (
    'I',
    'X',
    'Y',
    'Z',
)

# Single-qubit Pauli multiplication table in encoded form.
#
# Encoding convention:
#   0 -> I
#   1 -> X
#   2 -> Y
#   3 -> Z
#
# Each entry
#   _MUL_TABLE[(a, b)] = (phase, c)
# means that the product of the corresponding single-qubit Pauli operators is
#
#   P_a @ P_b = phase * P_c.
#
# Examples:
#   (1, 2): (1j, 3)   means   X @ Y = +i Z
#   (2, 1): (-1j, 3)  means   Y @ X = -i Z
#   (3, 3): (1.0, 0)  means   Z @ Z = I
#
# This table is used to multiply Pauli strings symbolically, site by site,
# without materializing dense matrices.
#
# If
#   P = P_{a_1} \otimes ... \otimes P_{a_n}
#   Q = P_{b_1} \otimes ... \otimes P_{b_n},
# then for each site k we look up
#
#   P_{a_k} P_{b_k} = phase_k * P_{c_k},
#
# and obtain
#
#   P Q = (prod_k phase_k) * (P_{c_1} \otimes ... \otimes P_{c_n}).
#
# The nontrivial phases ±i appear because Pauli matrices do not commute:
#
#   X Y = +i Z,   Y X = -i Z,
#   Y Z = +i X,   Z Y = -i X,
#   Z X = +i Y,   X Z = -i Y.
#
# Since I is the multiplicative identity, multiplying by code 0 leaves the
# other factor unchanged.
_MUL_TABLE: Dict[Tuple[int, int], Tuple[complex, int]] = {
    (0, 0): (1.0, 0),
    (0, 1): (1.0, 1),
    (0, 2): (1.0, 2),
    (0, 3): (1.0, 3),
    (1, 0): (1.0, 1),
    (1, 1): (1.0, 0),
    (1, 2): (1j, 3),
    (1, 3): (-1j, 2),
    (2, 0): (1.0, 2),
    (2, 1): (-1j, 3),
    (2, 2): (1.0, 0),
    (2, 3): (1j, 1),
    (3, 0): (1.0, 3),
    (3, 1): (1j, 2),
    (3, 2): (-1j, 1),
    (3, 3): (1.0, 0),
}
