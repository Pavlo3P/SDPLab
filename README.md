# SDPLab

SDPLab is a small library for building and experimenting with semidefinite
programming (SDP) problems.

## What Is an SDP?

An SDP is an optimization problem where the unknown is a matrix-like variable:

$$
\begin{aligned}
\min_{X \in \operatorname{dom}(\mathcal{A})}\quad
    & \operatorname{Tr}[C X] \\
\text{s.t.}\quad
    & \mathcal{A}X = b, \\
    & X \succeq 0.
\end{aligned}
$$

Here

- $X \in \operatorname{dom}(\mathcal{A})$ is the primal variable.
- $X \succeq 0$ means that $X$ is positive semidefinite, so all eigenvalues of
  $X$ are nonnegative.
- $C \in \operatorname{dom}(\mathcal{A})$ is the cost matrix or cost element.
- $\operatorname{Tr}[C X]$ is the trace objective. For dense matrices, think
  `trace(C @ X)`.
- $\mathcal{A} : \operatorname{dom}(\mathcal{A})
  \to \operatorname{cod}(\mathcal{A})$ is the linear constraint operator.
- $b \in \operatorname{cod}(\mathcal{A})$ is the constraint right-hand side.
- $\mathcal{A}X = b$ means that $X$ must satisfy a list or structured
  collection of linear equalities.

The library names match this notation:

- `dom` is $\operatorname{dom}(\mathcal{A})$, the space containing $C$ and $X$.
- `cod` is $\operatorname{cod}(\mathcal{A})$, the space containing
  $\mathcal{A}X$, $b$, and dual variables $y$.
- `SDPProblem` stores
  $C \in \operatorname{dom}(\mathcal{A})$,
  $\mathcal{A} : \operatorname{dom}(\mathcal{A})
  \to \operatorname{cod}(\mathcal{A})$, and
  $b \in \operatorname{cod}(\mathcal{A})$.
- `SDPPrimal` wraps a candidate primal variable
  $X \in \operatorname{dom}(\mathcal{A})$.
- `SDPDual` wraps a candidate dual variable
  $y \in \operatorname{cod}(\mathcal{A})$.

## How To Build a Problem

1. Decide what shape and type the unknown variable $X$ has.
   This determines $\operatorname{dom}(\mathcal{A})$. For a dense complex SDP,
   this is usually a Hermitian matrix space.

2. Decide what linear constraints are needed.
   This determines $\operatorname{cod}(\mathcal{A})$ and the linear map
   $\mathcal{A} : \operatorname{dom}(\mathcal{A})
   \to \operatorname{cod}(\mathcal{A})$.

3. Pick the right-hand side
   $b \in \operatorname{cod}(\mathcal{A})$.
   Feasible primal variables satisfy $\mathcal{A}X = b$.

4. Pick the cost element
   $C \in \operatorname{dom}(\mathcal{A})$.
   The solver minimizes $\operatorname{Tr}[C X]$.

5. Wrap the data as an SDP problem.

Example skeleton:

```python
from spacecore import Context, NumpyOps, DenseLinOp, HermitianSpace, VectorSpace
from sdplab.sdp import SDPDenseProblem
from sdplab.solvers import run_cvxpy_solver

# Passing ctx is optional, see spacecore lib 
# documentation for more details
ctx = Context(NumpyOps())

dom = HermitianSpace(n, ctx=ctx)      # X and C are n x n Hermitian matrices
cod = VectorSpace((m,), ctx=ctx)      # A X and b are length-m vectors

A_op = DenseLinOp(A_mats, dom, cod, ctx=ctx)
problem = SDPDenseProblem(C, A_op, b, tau=None, ctx=ctx)

primal, dual = run_cvxpy_solver(problem)
```

If `tau` is set, the dense problem also includes the affine constraint

$$
\operatorname{Tr}[X] = \tau.
$$

For density matrices, use `tau=1.0`.

## Dual Objects

The dual variable satisfies

$$
y \in \operatorname{cod}(\mathcal{A}).
$$

The adjoint operator

$$
\mathcal{A}^\dagger :
\operatorname{cod}(\mathcal{A})
\to
\operatorname{dom}(\mathcal{A})
$$

moves dual variables back into the primal space. The expression

$$
\mathcal{A}^\dagger y - C
$$

is the dual slack expression used by dual solvers and regularized primal
recovery.

## Regularized SDPs

Superlinear convex function $\varphi$ could be used for the penalization of primal problem
depending on strength $\varepsilon > 0$

$$
\begin{aligned}
\min_{X \in \operatorname{dom}(\mathcal{A})}\quad
    & \operatorname{Tr}[C X] +\varepsilon \operatorname{Tr}[\varphi(X)] \\
\text{s.t.}\quad
    & \mathcal{A}X = b, \\
    & X \succeq 0.
\end{aligned}
$$

The respective unconstrained dual problem is

$$\max_{y\in\operatorname{cod}(\mathcal{A})} D_\varepsilon(y)=\operatorname{Tr}[b y]-\varepsilon\operatorname{Tr}\left[\psi\left(\frac{\mathcal{A}^\dagger y-C}{\varepsilon}\right)\right]$$

where $\psi$ is the Legendre transform of $\varphi$.

This makes it possible to optimize in the dual space using gradient methods and recover a primal
matrix from the eigendecomposition of $\mathcal{A}^\dagger y - C$. If
$s_i$ are the eigenvalues of $\mathcal{A}^\dagger y - C$, the recovery uses the
spectral first-order relation

$$
\lambda_i(X) = \psi'\left(\frac{s_i}{\varepsilon}\right).
$$

Built-in regularizers:

- `EntropyReg`: entropy-style spectral regularization for the primal SDP.
- `EntropyRegLog`: entropy-style spectral regularization for trace-normalized
  problems, such as problems with $\operatorname{Tr}[X] = 1$.
- `QuadraticReg`: quadratic spectral regularization for the primal SDP.
