r"""Tools for constructing and solving semidefinite programming problems.

Beginner mental model
---------------------
An SDP is a linear optimization problem where the unknown is a matrix. It is
like linear programming, except the variable ``X`` is required to be positive
semidefinite instead of just componentwise nonnegative.

The central mathematical object in SDPLab is

.. math::

    \min_X \quad \operatorname{Tr}[C X]
    \quad \text{s.t.} \quad \mathcal{A}X = b,\quad X \succeq 0.

The pieces mean:

    X:
        The unknown matrix. In dense SDPs it is symmetric in the real case or
        Hermitian in the complex case.
    :math:`X \succeq 0`:
        Positive semidefinite. Every eigenvalue of ``X`` is nonnegative.
    C:
        The cost matrix. The optimizer tries to make ``Tr[C X]`` small.
    :math:`\operatorname{Tr}[C X]`:
        The trace objective. For dense matrices this is the real part of
        ``trace(C @ X)``.
    :math:`A` or :math:`\mathcal{A}`:
        A linear constraint operator ``A : dom -> cod``. It converts a matrix
        ``X`` into the list or structured collection of constraint values.
    b:
        The right-hand side of the equality constraints. It lies in ``cod``.

How to build an SDP in this library:

    1. Choose ``dom``, the space where ``X`` and ``C`` live.
    2. Choose ``cod``, the space where ``A X`` and ``b`` live.
    3. Build a linear operator ``A : dom -> cod``.
    4. Pick ``C in dom`` and ``b in cod``.
    5. Wrap them as ``SDPDenseProblem(C, A, b, ...)``.
    6. Send the problem to a solver or wrap it with a regularizer.

Dual variables use the same constraint space as ``b``. If ``y in cod``, then
:math:`A^\dagger y \in \mathrm{dom}` is the adjoint constraint expression.
Many algorithms inspect or diagonalize :math:`A^\dagger y - C`. In formulas below, ``A`` and
:math:`\mathcal{A}` mean the same constraint operator.
"""
