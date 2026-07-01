from spacecore import (
    TreeSpace,
    StackedSpace,
    CoordinateSpace,
    EuclideanJordanAlgebraSpace,
    ElementwiseJordanSpace,
    EuclideanElementwiseJordanSpace,
    HermitianSpace,
)
from spacecore.backend import Context


def _real_context(ctx: Context) -> Context:
    """Return ``ctx`` with a real representation dtype.

    Spectra of Euclidean-Jordan elements (Hermitian matrices, elementwise
    algebras) are always real, so eigenvalue vectors live in a real context
    even when the algebra elements are complex. Backend family and validation
    policy are preserved.
    """
    real_dtype = ctx.ops.real_dtype(ctx.dtype)
    if real_dtype == ctx.dtype:
        return ctx
    return Context(ctx.ops, dtype=real_dtype, check_level=ctx.check_level)


def create_eigval_space(
    obj_space: EuclideanJordanAlgebraSpace,
) -> EuclideanElementwiseJordanSpace:
    """Build the coordinate space in which spectra of ``obj_space`` live.

    Handles nested ``TreeSpace`` / ``StackedSpace`` structure (stacks at the
    top level and on tree leaves) and heterogeneous leaf shapes. The result is
    always real-valued, matching the real spectra of Euclidean-Jordan algebras.
    """
    if isinstance(obj_space, StackedSpace):
        base_eig_space = create_eigval_space(obj_space.base)
        # No ctx=: resolve from the already-real eigval base, so a complex
        # parent context cannot re-complexify the leaf.
        return StackedSpace(base_eig_space, obj_space.count)

    if isinstance(obj_space, TreeSpace):
        leaf_spaces = tuple(
            create_eigval_space(leaf) for leaf in obj_space.leaf_spaces
        )
        # No ctx=: context resolves from the uniformly-real eigval leaves.
        return TreeSpace(obj_space.treedef, leaf_spaces)

    if isinstance(obj_space, HermitianSpace):
        # Eigenvalues of an (n, n) Hermitian matrix form a real length-n vector.
        return EuclideanElementwiseJordanSpace(
            (obj_space.n,), ctx=_real_context(obj_space.ctx)
        )

    if isinstance(obj_space, ElementwiseJordanSpace):
        # Elementwise Jordan spectrum is the identity: same shape, real field.
        return EuclideanElementwiseJordanSpace(
            obj_space.shape, ctx=_real_context(obj_space.ctx)
        )

    raise TypeError(
        f"Cannot infer eigval space from {type(obj_space).__name__}."
    )