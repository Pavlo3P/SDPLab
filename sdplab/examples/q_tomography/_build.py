r"""Builders for qubit tomography SDP feasibility problems.

Quantum tomography estimates an unknown density matrix
:math:`X \in \operatorname{Herm}(d)` from linear measurement constraints.
This module builds the feasibility SDP

.. math::

    \min_{X \in \operatorname{Herm}(d)}\quad 0
    \quad \text{s.t.} \quad
    \mathcal{A}X = b_{\mathrm{obs}},\quad
    \operatorname{Tr}[X] = 1,\quad
    X \succeq 0.

Here :math:`\mathcal{A}: \operatorname{Herm}(d) \to \mathbb{R}^m` is the
measurement map :math:`(\mathcal{A}X)_i = \operatorname{Tr}[M_i X]` and
:math:`b_{\mathrm{obs}}` stores the observed expectation values. The unit-trace
constraint is appended as one extra row measuring the identity, so the returned
object is a plain :class:`~sdplab.problem.SDPProblem`.
"""

from __future__ import annotations

import numpy as np
from spacecore import (
    Context,
    DenseArray,
    DenseVectorSpace,
    HermitianSpace,
    LinOp,
    NumpyOps,
    checked_method,
    jax_pytree_class,
)

from ...problem import SDPProblem


@jax_pytree_class
class TomographyOperator(LinOp[HermitianSpace, DenseVectorSpace]):
    r"""Measurement operator :math:`\mathcal{A}` for state tomography.

    The linear map is
    :math:`\mathcal{A}: \operatorname{Herm}(d) \to \mathbb{R}^m` with

    .. math::

        (\mathcal{A}X)_i = \operatorname{Tr}[M_i X],

    for a stack of Hermitian observables :math:`M_i`. The trace of two
    Hermitian matrices is real, so every value this produces on a domain member
    is a real expectation value (carried in the domain's dtype, which is
    complex whenever the observables are). No real cast is applied: a cast is
    not :math:`\mathbb{C}`-linear, so it would make :meth:`apply` disagree with
    :meth:`to_dense` off the Hermitian subspace. The adjoint is the combination

    .. math::

        \mathcal{A}^\dagger y = \sum_i y_i M_i,

    again Hermitian, which satisfies
    :math:`\langle \mathcal{A}X, y\rangle = \langle X, \mathcal{A}^\dagger y\rangle`.

    Note the placement of the transpose. A :class:`~spacecore.DenseLinOp` built
    straight from the stack ``M`` would instead pair by Frobenius,
    :math:`\sum_{pq} M_{i,pq} X_{pq} = \operatorname{Tr}[M_i^{T} X]`, which for
    a Hermitian observable is the *conjugate* of the intended value -- it flips
    the sign of every measurement with a nonzero imaginary part (the Pauli
    :math:`Y`, say). :meth:`to_dense` therefore reports the transposed stack.
    """

    def __init__(self, M: DenseArray, dom: HermitianSpace, cod: DenseVectorSpace,
                 ctx: Context | str | None = None):
        """Store the ``(m, d, d)`` stack of Hermitian observables."""
        super(TomographyOperator, self).__init__(dom, cod, ctx)
        self.M = self.ctx.asarray(M)

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, X: DenseArray) -> DenseArray:
        r"""Return the expectation values :math:`(\operatorname{Tr}[M_i X])_i`.

        Real for any Hermitian ``X`` (up to round-off), since each
        :math:`M_i` is Hermitian.
        """
        return self.ops.einsum("ipq,qp->i", self.M, X)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: DenseArray) -> DenseArray:
        r"""Return :math:`\mathcal{A}^\dagger y = \sum_i y_i M_i`."""
        return self.ops.einsum("i,ipq->pq", self.ops.asarray(y, dtype=self.dtype), self.M)

    def to_dense(self) -> DenseArray:
        r"""Return the operator tensor of shape ``codomain.shape + domain.shape``.

        ``T[i, p, q] = M[i, q, p]``, so that
        :math:`\sum_{pq} T_{i,pq} X_{pq} = \operatorname{Tr}[M_i X]` -- the
        transpose discussed in the class docstring.
        """
        return self.ops.transpose(self.M, (0, 2, 1))

    def tree_flatten(self):
        """The observable stack is the only array child; spaces are static aux."""
        return (self.M,), (self.dom, self.cod, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild the operator from JAX PyTree data."""
        (M,) = children
        dom, cod, ctx = aux
        return cls(M, dom, cod, ctx)

    def _convert(self, new_ctx: Context) -> TomographyOperator:
        """Return an equivalent operator in ``new_ctx``."""
        return TomographyOperator(
            self.M, self.dom.convert(new_ctx), self.cod.convert(new_ctx), new_ctx
        )


def generate_qubit_tomography(
    M: DenseArray,
    b_obs: DenseArray,
    atol: float = 0.0,
    rtol: float = 0.0,
    enforce_herm: bool = True,
    ctx: Context | str | None = None,
) -> SDPProblem:
    r"""Build the tomography feasibility SDP for observables ``M``.

    The unknown is a density matrix :math:`X \in \operatorname{Herm}(d)` and
    the returned problem is

    .. math::

        \min_{X \in \operatorname{Herm}(d)}\quad 0
        \quad \text{s.t.} \quad
        \operatorname{Tr}[M_i X] = b_i,\quad
        \operatorname{Tr}[X] = 1,\quad
        X \succeq 0.

    The trace-one constraint is appended as one extra row measuring the
    identity, so the constraint operator has ``m + 1`` rows.

    Args:
        M: Stack of Hermitian observables, shape ``(m, d, d)``.
        b_obs: Observed expectation values :math:`b_i = \operatorname{Tr}[M_i X]`,
            length ``m``. Taken as real -- an imaginary part would not be
            attainable by a Hermitian observable on a Hermitian state.
        atol: Absolute Hermitian membership tolerance.
        rtol: Relative Hermitian membership tolerance.
        enforce_herm: Whether the primal domain enforces Hermitian matrices.
        ctx: Optional backend context for the returned problem. The problem is
            assembled on NumPy (in a complex dtype when ``M`` is complex) and
            converted, matching :func:`~sdplab.examples.generate_max_cut`.

    Returns:
        Feasibility :class:`~sdplab.problem.SDPProblem` with a zero cost.

    Raises:
        ValueError: If ``M`` is not a stack of square matrices, if ``b_obs``
            does not match its length, or if any observable is not Hermitian.
    """
    M = np.asarray(M)
    b_obs = np.asarray(b_obs).reshape(-1)

    if M.ndim != 3 or M.shape[-1] != M.shape[-2]:
        raise ValueError(f"M must have shape (m, d, d); got {M.shape}.")
    if b_obs.shape[0] != M.shape[0]:
        raise ValueError(
            f"b_obs has length {b_obs.shape[0]}, but M has {M.shape[0]} observables."
        )
    if not np.allclose(M, np.conj(np.swapaxes(M, -1, -2)), atol=1e-10):
        raise ValueError("every observable M[i] must be Hermitian.")

    m, d = M.shape[0], M.shape[-1]
    is_complex = np.iscomplexobj(M)
    dtype = "complex128" if is_complex else "float64"
    np_ctx = Context(NumpyOps(), dtype=dtype, check_level="none")

    # Append Tr[X] = 1 as one extra measurement of the identity.
    M_full = np.concatenate([M.astype(dtype), np.eye(d, dtype=dtype)[None]], axis=0)
    b_full = np.concatenate([b_obs.real.astype("float64"), [1.0]])

    dom = HermitianSpace(d, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=np_ctx)
    cod = DenseVectorSpace((m + 1,), ctx=np_ctx)
    A = TomographyOperator(M_full, dom, cod, np_ctx)

    sdp = SDPProblem(dom.zeros(), A, b_full, ctx=np_ctx)   # zero cost: pure feasibility

    if ctx is not None:
        sdp = sdp.convert(ctx)
    return sdp
