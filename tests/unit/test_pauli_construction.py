"""Regression tests for Pauli construction APIs."""

from __future__ import annotations

from sdplab.special.pauli import PauliSum


def test_pauli_sum_constructs_from_array(np_ctx):
    """Building a Pauli sum from a matrix exercises normalize_context."""
    mat = np_ctx.asarray([[1.0, 0.0], [0.0, 1.0]])

    pauli = PauliSum.from_matrix(mat, ctx=np_ctx)

    assert pauli.n_qubits == 1
