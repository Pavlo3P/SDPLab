import numpy as np
import pytest

from sdplab.special.pauli import PauliString, PauliSum


def test_pauli_string_materialize_matches_expected_dense_operator():
    p = PauliString("XZ")
    expected = np.kron(
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[1, 0], [0, -1]], dtype=np.complex128),
    )
    assert np.allclose(np.asarray(p.materialize()), expected)


def test_pauli_string_matvec_matches_materialized_action():
    p = PauliString("YI")
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128)
    y_fast = np.asarray(p.matvec(x))
    y_dense = np.asarray(p.materialize()) @ x
    assert np.allclose(y_fast, y_dense)


def test_pauli_string_multiplication_tracks_phase():
    px = PauliString("X")
    py = PauliString("Y")
    pz = px.multiply(py)
    assert pz.identifier == "Z"
    assert np.allclose(pz.phase, 1j)


def test_pauli_string_support_and_weight():
    p = PauliString("IIXZI")
    assert p.support() == (2, 3)
    assert p.weight() == 2


def test_pauli_string_trace_identity_and_nonidentity():
    p_id = PauliString("II")
    p_xz = PauliString("XZ")
    assert np.allclose(p_id.trace(), 4.0)
    assert np.allclose(p_xz.trace(), 0.0)


def test_pauli_string_commutation():
    assert PauliString("XI").commutes_with(PauliString("ZI")) is False
    assert PauliString("XI").commutes_with(PauliString("IX")) is True
    assert PauliString("XY").commutes_with(PauliString("YY")) is False


def test_scalar_times_pauli_string_returns_pauli_sum():
    s = 2.5 * PauliString("XI")
    assert isinstance(s, PauliSum)
    assert s.n_terms == 1
    assert s.terms[0].identifier == "XI"
    assert np.allclose(np.asarray(s.coeffs)[0], 2.5)


def test_pauli_sum_simplifies_duplicate_terms():
    s = PauliSum(["XI", "XI", "IZ"], coeffs=[1.0, 2.0, -1.0])
    assert s.n_terms == 2
    coeffs = {term.identifier: coeff for term, coeff in zip(s.terms, s.coeffs)}
    assert np.allclose(coeffs["XI"], 3.0)
    assert np.allclose(coeffs["IZ"], -1.0)


def test_pauli_sum_support_is_union_of_term_supports():
    s = PauliSum(["XI", "IZ"], coeffs=[1.0, -1.0])
    assert s.support() == (0, 1)


def test_pauli_sum_matvec_matches_materialized_action():
    s = PauliSum(["XI", "YZ"], coeffs=[2.0, -0.5])
    x = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.complex128)
    y_fast = np.asarray(s.matvec(x))
    y_dense = np.asarray(s.materialize()) @ x
    assert np.allclose(y_fast, y_dense)


def test_pauli_sum_trace():
    s = PauliSum(["II", "XZ"], coeffs=[2.0, -3.0])
    assert np.allclose(s.trace(), 2.0 * 4.0)


def test_pauli_sum_add_term_simplifies():
    s = PauliSum(["XI"], coeffs=[1.0])
    s2 = s.add_term("XI", coeff=2.0)
    assert s2.n_terms == 1
    assert s2.terms[0].identifier == "XI"
    assert np.allclose(np.asarray(s2.coeffs)[0], 3.0)


def test_pauli_sum_zero_cancellation_keeps_zero_identity_term():
    s = PauliSum(["XI", "XI"], coeffs=[1.0, -1.0])
    assert s.n_terms == 1
    assert s.terms[0].identifier == "II"
    assert np.allclose(np.asarray(s.coeffs)[0], 0.0)
    assert np.allclose(np.asarray(s.materialize()), np.zeros((4, 4)))


def test_pauli_sum_scalar_multiplication():
    s = PauliSum(["XI", "IZ"], coeffs=[2.0, -1.0])
    t = -0.5 * s
    coeffs = {term.identifier: coeff for term, coeff in zip(t.terms, t.coeffs)}
    assert np.allclose(coeffs["XI"], -1.0)
    assert np.allclose(coeffs["IZ"], 0.5)


def test_pauli_sum_product_with_pauli_string_matches_dense_product():
    s = PauliSum(["XI", "IZ"], coeffs=[2.0, -1.0])
    p = PauliString("YZ")
    prod_symbolic = s @ p
    prod_dense = np.asarray(s.materialize()) @ np.asarray(p.materialize())
    assert np.allclose(np.asarray(prod_symbolic.materialize()), prod_dense)


def test_pauli_sum_product_with_pauli_sum_matches_dense_product():
    a = PauliSum(["XI", "IZ"], coeffs=[2.0, -1.0])
    b = PauliSum(["YZ", "II"], coeffs=[0.5, 3.0])
    prod_symbolic = a @ b
    prod_dense = np.asarray(a.materialize()) @ np.asarray(b.materialize())
    assert np.allclose(np.asarray(prod_symbolic.materialize()), prod_dense)


def test_from_matrix_roundtrip_single_pauli():
    p = PauliString("YZ")
    a = np.asarray(p.materialize())
    s = PauliSum.from_matrix(a)
    assert s.n_terms == 1
    assert s.terms[0].identifier == "YZ"
    assert np.allclose(np.asarray(s.coeffs)[0], 1.0)
    assert np.allclose(np.asarray(s.materialize()), a)


def test_from_matrix_roundtrip_linear_combination():
    a = (
        1.5 * np.asarray(PauliString("II").materialize())
        - 0.25 * np.asarray(PauliString("XZ").materialize())
        + 2.0 * np.asarray(PauliString("YY").materialize())
    )
    s = PauliSum.from_matrix(a)
    assert np.allclose(np.asarray(s.materialize()), a)

    coeffs = {term.identifier: coeff for term, coeff in zip(s.terms, s.coeffs)}
    assert np.allclose(coeffs["II"], 1.5)
    assert np.allclose(coeffs["XZ"], -0.25)
    assert np.allclose(coeffs["YY"], 2.0)


def test_from_matrix_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        PauliSum.from_matrix(np.zeros((2, 3), dtype=np.complex128))


def test_from_matrix_rejects_dimension_not_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        PauliSum.from_matrix(np.zeros((3, 3), dtype=np.complex128))


def test_pauli_string_matmat_matches_materialized_action():
    p = PauliString("YZ")
    X = np.array(
        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0],
         [7.0, 8.0]],
        dtype=np.complex128,
    )
    y_fast = np.asarray(p.matmat(X))
    y_dense = np.asarray(p.materialize()) @ X
    assert np.allclose(y_fast, y_dense)


def test_pauli_sum_matmat_matches_materialized_action():
    s = PauliSum(["XI", "YZ"], coeffs=[2.0, -0.5])
    X = np.array(
        [[1.0, -1.0, 0.0],
         [2.0,  3.0, 1.0],
         [0.5,  0.0, 4.0],
         [3.0,  2.0, 5.0]],
        dtype=np.complex128,
    )
    y_fast = np.asarray(s.matmat(X))
    y_dense = np.asarray(s.materialize()) @ X
    assert np.allclose(y_fast, y_dense)
