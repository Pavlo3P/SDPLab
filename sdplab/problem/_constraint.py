r"""Constraint operators with an explicit cvxpy encoding.

The generic solver in :mod:`sdplab.solvers._cvxpy` never touches the operator
tensor directly; it asks the constraint operator for a *list of per-constraint
matrices*. The contract is:

    :meth:`ConstraintOp.to_cvxpy` returns ``[A_0, ..., A_{m-1}]`` such that the
    ``i``-th scalar equality of the SDP is

    .. math::

        \operatorname{Re}\operatorname{Tr}[A_i\, X] = b_i,

    where ``b = rhs_to_cvxpy(problem.b)`` is the matching real right-hand side
    and ``X`` is the primal cone variable in :attr:`dom`.

This ``trace`` convention is what cvxpy consumes natively (it accepts a sparse
``A_i`` in ``cp.trace(A_i @ X)``), and it is the convention the QOT partial-trace
operator already builds its Hermitian generators in.

Reverse conversion of the *dual* is :meth:`ConstraintOp.dual_from_cvxpy`, which
reassembles a per-constraint value vector back into a :attr:`cod` element. Sign
conventions (cvxpy returns ``-`` the standard-form equality dual) are handled by
the solver, which passes ``-lambda`` to :meth:`dual_from_cvxpy`; the method only
inverts the row layout, not the sign. The primal ``X`` needs no reverse hook
here -- it is read back from the cone variable by the solver, driven by the
domain cone type, independent of the constraint operator.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from spacecore import (
    Context,
    LinOp,
    DenseLinOp,
    SparseLinOp,
    EuclideanJordanAlgebraSpace,
    InnerProductSpace,
    DenseArray,
    SparseArray,
    jax_pytree_class,
)


class ConstraintOp(LinOp[EuclideanJordanAlgebraSpace, InnerProductSpace]):
    r"""Linear constraint operator :math:`\mathcal{A}: \operatorname{dom} \to \operatorname{cod}` with a cvxpy encoding.

    Beyond the :class:`~spacecore.LinOp` action, a constraint operator knows how
    to present itself to cvxpy as a list of per-constraint matrices (see the
    module docstring). Concrete backends:

    - :class:`DenseConstraintOp` / :class:`SparseConstraintOp` wrap a stored
      operator tensor and derive the ``A_i`` from it.
    - :class:`MatrixFreeConstraintOp` is defined only by ``apply``/``rapply``
      and materializes the ``A_i`` through the adjoint; the QOT operator extends
      it and overrides :meth:`to_cvxpy` with a cheaper Hermitian-generator form.
    - :class:`WrappedConstraintOp` adapts an arbitrary :class:`~spacecore.LinOp`
      by delegation, keeping it matrix-free.
    """

    @abstractmethod
    def to_cvxpy(self) -> list[DenseArray | SparseArray]:
        r"""Return the per-constraint matrices ``[A_0, ..., A_{m-1}]``.

        Each ``A_i`` lives in the domain matrix space and is oriented so that
        the ``i``-th equality reads :math:`\operatorname{Re}\operatorname{Tr}[A_i X] = b_i`.
        """

    def rhs_to_cvxpy(self, b: Any) -> DenseArray:
        r"""Return the real right-hand side matching the rows of :meth:`to_cvxpy`.

        The default flattens ``b`` into ``cod`` coordinates and takes the real
        part -- correct when the codomain is a real coordinate space of ``m``
        scalar constraints. Operators with a structured codomain (e.g. QOT's
        Hermitian marginals reduced to real generators) override this.
        """
        return self.ops.real(self.cod.flatten(b))

    def dual_from_cvxpy(self, y: DenseArray) -> Any:
        r"""Reassemble a per-constraint value vector into a :attr:`cod` element.

        Inverse of the row layout produced by :meth:`rhs_to_cvxpy`. The default
        unflattens ``y`` back into ``cod``; structured codomains override. The
        caller (solver) is responsible for the dual sign.
        """
        return self.cod.unflatten(y)

    @classmethod
    def from_linop(cls, op: LinOp) -> ConstraintOp:
        raise NotImplementedError()


class MatrixFreeConstraintOp(ConstraintOp):
    r"""Constraint operator defined only by ``apply``/``rapply`` (no stored matrix).

    The default :meth:`to_cvxpy` materializes each constraint matrix as
    :math:`A_i = \mathcal{A}^\dagger e_i`, the adjoint applied to the ``i``-th
    codomain basis vector. For a Hermitian domain this is exactly the matrix for
    which :math:`\operatorname{Tr}[A_i X] = (\mathcal{A}X)_i`. Subclasses with a
    cheaper or reduced encoding (e.g. :class:`~sdplab.special.qot.QOTConstraintOp`)
    override :meth:`to_cvxpy`, :meth:`rhs_to_cvxpy`, and :meth:`dual_from_cvxpy`.

    The default assumes a real coordinate codomain, so that the unit vectors
    ``e_i`` are valid codomain members.
    """

    def to_cvxpy(self) -> list[DenseArray]:
        r"""Materialize ``[A_i = adjoint(e_i)]`` over the codomain coordinate basis."""
        m = self.cod.size
        eye = self.ops.eye(m, dtype=self.dtype)
        return [self.rapply(self.cod.unflatten(eye[i])) for i in range(m)]


@jax_pytree_class
class WrappedConstraintOp(MatrixFreeConstraintOp):
    r"""Adapt an arbitrary :class:`~spacecore.LinOp` into a constraint operator.

    This is the fallback used by :func:`sdplab.problem._base._dispatch_constraint`
    for a user-supplied operator that is neither a
    :class:`~spacecore.DenseLinOp` nor a :class:`~spacecore.SparseLinOp` -- a
    hand-written matrix-free operator, an algebra expression, and so on. The
    wrapper only forwards ``apply``/``rapply``, so the operator stays
    matrix-free for the first-order solvers; the per-constraint matrices are
    built lazily by the inherited :meth:`MatrixFreeConstraintOp.to_cvxpy` and
    only if the cvxpy backend is actually called.

    As for every :class:`MatrixFreeConstraintOp`, the default cvxpy encoding
    assumes a real coordinate codomain. Wrap a structured-codomain operator in a
    subclass that overrides :meth:`~ConstraintOp.to_cvxpy`,
    :meth:`~ConstraintOp.rhs_to_cvxpy`, and :meth:`~ConstraintOp.dual_from_cvxpy`.
    """

    def __init__(self, op: LinOp, ctx: Context | str | None = None):
        """Wrap ``op``, converting it (and its spaces) onto the resolved context."""
        if not isinstance(op, LinOp):
            raise TypeError(
                f"WrappedConstraintOp requires a LinOp; got {type(op).__name__}."
            )
        super().__init__(op.dom, op.cod, ctx)
        self.op = op.convert(self.ctx)

    def apply(self, x: Any) -> Any:
        """Apply the wrapped operator (its own membership checks still run)."""
        return self.op.apply(x)

    def rapply(self, y: Any) -> Any:
        """Apply the adjoint of the wrapped operator."""
        return self.op.rapply(y)

    def to_dense(self) -> DenseArray:
        """Delegate dense materialization, keeping any efficient override."""
        return self.op.to_dense()

    def to_matrix(self) -> DenseArray:
        """Delegate flat-matrix materialization, keeping any efficient override."""
        return self.op.to_matrix()

    def tree_flatten(self):
        """Return children and auxiliary data for JAX PyTree flattening."""
        return (self.op,), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild the wrapper from JAX PyTree data."""
        (op,) = children
        return cls(op)

    def _convert(self, new_ctx) -> "WrappedConstraintOp":
        return type(self)(self.op.convert(new_ctx), new_ctx)

    @classmethod
    def from_linop(cls, op: LinOp) -> "WrappedConstraintOp":
        """Wrap any :class:`~spacecore.LinOp` without materializing it."""
        return cls(op)


@jax_pytree_class
class DenseConstraintOp(ConstraintOp, DenseLinOp):
    r"""Constraint operator backed by a dense coordinate tensor.

    The stored tensor ``T`` of shape ``cod.shape + dom.shape`` acts by
    :math:`(\mathcal{A}X)_i = \sum_{jk} T_{i,jk} X_{jk}` (a Frobenius pairing,
    no conjugation). To present the ``trace`` convention
    :math:`\operatorname{Tr}[A_i X] = (\mathcal{A}X)_i`, the ``i``-th constraint
    matrix is the **transpose** of the ``i``-th tensor slice, ``A_i = T_i^{T}``.

    Note the conjugation carefully: for a genuinely complex Hermitian slice the
    transpose already equals the conjugate, ``T_i^{T} = \overline{T_i}``, and
    that is the correct Hermitian matrix for which ``Tr[A_i X]`` reproduces the
    operator. A conjugate-*transpose* ``T_i^{H}`` would instead return ``T_i``
    itself and break the identity. The transpose is a no-op only for the real
    symmetric constraint matrices of a real SDP.
    """

    def to_cvxpy(self) -> list[DenseArray]:
        T = self.to_dense()
        return [self.ops.transpose(T[i]) for i in range(self.cod.size)]

    def _convert(self, new_ctx) -> "DenseConstraintOp":
        return type(self).from_linop(DenseLinOp._convert(self, new_ctx))

    @classmethod
    def from_linop(cls, op: DenseLinOp) -> "DenseConstraintOp":
        """Wrap an existing :class:`~spacecore.DenseLinOp` as a constraint operator."""
        return cls(op.A, op.dom, op.cod, op.ctx)


@jax_pytree_class
class SparseConstraintOp(ConstraintOp, SparseLinOp):
    r"""Constraint operator backed by a sparse coordinate matrix.

    :attr:`~spacecore.SparseLinOp.A` stores the flattened ``(m, dom.size)``
    matrix whose ``i``-th row, reshaped to ``dom.shape``, is the operator tensor
    slice ``T_i``. :meth:`to_cvxpy` returns each ``A_i = T_i^{T}`` as a sparse
    ``(n, n)`` matrix (the ``trace`` convention). As for the dense case the map
    is a plain transpose, which for a complex Hermitian slice equals its
    conjugate ``\overline{T_i}`` -- the correct Hermitian matrix -- and is a
    no-op only for a real symmetric SDP; it keeps large constraints sparse.

    :meth:`to_cvxpy` uses ``scipy.sparse`` row-slice/reshape/transpose
    semantics -- the only sparse backend in use -- and feeds the cvxpy backend,
    which operates on numpy/scipy data.
    """

    def to_cvxpy(self) -> list[SparseArray]:
        shape = self.dom.shape
        A = self.A
        return [A[i, :].reshape(shape).T for i in range(self.cod.size)]

    def _convert(self, new_ctx) -> "SparseConstraintOp":
        return type(self).from_linop(SparseLinOp._convert(self, new_ctx))

    @classmethod
    def from_linop(cls, op: SparseLinOp) -> "SparseConstraintOp":
        """Wrap an existing :class:`~spacecore.SparseLinOp` as a constraint operator."""
        return cls(op.A, op.dom, op.cod, op.ctx)
