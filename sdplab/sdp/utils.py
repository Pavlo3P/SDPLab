import jax
import jax.numpy as jnp


def convert_complex_sdp_to_real_sdp(
        C: jnp.ndarray,
        A: jnp.ndarray,
        b: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convert a complex-valued semidefinite program (SDP) into its equivalent real-valued form.

    This function implements the transformation described in the following proposition:

        Let C ∈ ℋ_n, A₁,…,A_m ∈ ℋ_n, b₁,…,b_m ∈ ℝ.
        Then the complex-valued SDP

            minimize    ⟨C, X⟩
            subject to  ⟨A_i, X⟩ = b_i,   i = 1,…,m
                        X ⪰ 0,  X ∈ ℂ^{n×n}

        is equivalent to the real-valued SDP:

            minimize    ⟨𝓕(C), Y⟩
            subject to  ⟨𝓕(A_i), Y⟩ = 2 b_i,                i = 1,…,m
                        ⟨[E_ij,  0;  0, -E_ij], Y⟩ = 0,     i<j
                        ⟨[0, E_ij; E_ij,  0], Y⟩ = 0,       i<j
                        Y ⪰ 0,  Y ∈ ℝ^{2n×2n}

        where:
            - 𝓕(M) = [[Re(M), -Im(M)], [Im(M), Re(M)]] is the realification map.
            - E_{ij} ∈ ℝ^{n×n} is the matrix unit with entries:

                    (E_{ij})_{kl} = 1  if (k = i and l = j) or (k = j and l = i),
                                    0  otherwise.

              In other words, E_{ij} has ones at the (i,j) and (j,i) positions and zeros elsewhere.

    Parameters
    ----------
    C : jax.numpy.ndarray, shape (n, n), complex Hermitian
        Cost matrix of the original SDP.
    A : jax.numpy.ndarray, shape (m, n, n), complex Hermitian
        Constraint matrices.
    b : jax.numpy.ndarray, shape (m,), real (or complex with zero imaginary part)
        Right-hand side vector of constraints.

    Returns
    -------
    new_C : jax.numpy.ndarray, shape (2n, 2n), real symmetric
        Realified cost matrix.
    new_A : jax.numpy.ndarray, shape (m + extra, 2n, 2n), real symmetric
        Realified constraint matrices, including additional ones enforcing the Hermitian structure.
    new_b : jax.numpy.ndarray, shape (m + extra,), real
        Realified right-hand side vector.

    Notes
    -----
    - This transformation preserves feasibility and PSD constraints exactly.
    - The factor **2** in `new_b` arises from ⟨𝓕(A_i), Y⟩ = 2 b_i in Proposition 3.5.
    - The **optimal value of the converted real SDP is exactly twice** that of the original
      complex SDP due to the doubling of the trace under realification.
    """

    def F(M):
        """Realification mapping for a complex matrix."""
        return jnp.block([
            [M.real, -M.imag],
            [M.imag,  M.real]
        ])

    n = C.shape[0]
    new_C = F(C)
    new_A = jax.vmap(F)(A)
    new_b = 2 * b.real

    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            E = jnp.zeros((n, n)).at[i, j].set(1.0)
            E = E.at[j, i].set(1.0)
            constraints.append(jnp.block([
                [E, jnp.zeros((n, n))],
                [jnp.zeros((n, n)), -E],
            ]))
            constraints.append(jnp.block([
                [jnp.zeros((n, n)), E],
                [E, jnp.zeros((n, n))],
            ]))

    constraints = jnp.stack(constraints, axis=0)
    new_A = jnp.concatenate([new_A, constraints], axis=0)
    new_b = jnp.concatenate([new_b, jnp.zeros(constraints.shape[0])], axis=0)

    return new_C, new_A, new_b
