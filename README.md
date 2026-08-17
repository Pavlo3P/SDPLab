# SDPLab

A library for building semidefinite and conic programs and solving them
through their **smoothed dual**. Problem data, spectral regularizers, and the
dual functional live here; every first-order optimization loop is delegated to
[spacecore](https://pypi.org/project/spacecore/).

## Install

```bash
pip install sdplab
```

The JAX backend and the optax optimizer are optional:

```bash
pip install "sdplab[jax]"
```

## The problem

$$
\begin{aligned}
\min_{X \in \mathrm{dom}(\mathcal{A})}\quad
    & \langle C, X\rangle \\
\text{s.t.}\quad
    & \mathcal{A}X = b, \\
    & X \succeq 0.
\end{aligned}
$$

| symbol | code | meaning |
| --- | --- | --- |
| $\mathrm{dom}(\mathcal{A})$ | `problem.dom` | space of $X$ and $C$ |
| $\mathrm{cod}(\mathcal{A})$ | `problem.cod` | space of $\mathcal{A}X$, $b$, and the dual $y$ |
| $\mathcal{A}$ | `problem.A` | linear constraint operator, a spacecore `LinOp` |
| $\mathcal{A}^\dagger y - C$ | `problem.dual_slack(y)` | dual slack |

$X$ is a plain element of a spacecore Euclidean Jordan algebra space — a
Hermitian matrix (classic SDP), a nonnegative vector (LP), or a tree of such
blocks — and $X \succeq 0$ means a nonnegative Jordan spectrum. Solvers accept
and return raw arrays and trees; there are no wrapper objects.

```python
import numpy as np
from spacecore import Context, DenseLinOp, DenseVectorSpace, HermitianSpace, NumpyOps
from sdplab import SDPProblem

ctx = Context(NumpyOps())                   # optional; see the spacecore docs
dom = HermitianSpace(n, ctx=ctx)            # X and C live here
cod = DenseVectorSpace((m,), ctx=ctx)       # A X, b, and the dual y live here

A = DenseLinOp(A_mats, dom, cod, ctx=ctx)
problem = SDPProblem(C, A, b, ctx=ctx)
```

Solve it directly with the CVXPY reference backend:

```python
from sdplab.solvers import run_cvxpy_solver

X, y = run_cvxpy_solver(problem, solver="CLARABEL")
```

## The smoothed dual

Penalize the primal with a superlinear convex $\varphi$ at strength
$\varepsilon > 0$:

$$
\min_X\ \langle C, X\rangle + \varepsilon\,\mathrm{Tr}[\varphi(X)]
\quad\text{s.t.}\quad \mathcal{A}X = b,\ X \succeq 0.
$$

Its dual is unconstrained and differentiable:

$$
\max_{y}\ D_\varepsilon(y) = \langle b, y\rangle - \varepsilon\,
\mathrm{Tr}\!\left[\psi\!\left(\frac{\mathcal{A}^\dagger y - C}{\varepsilon}\right)\right],
$$

with $\psi$ the Legendre transform of $\varphi$. Gradient methods apply, and the
primal is read back off the eigenvalues $s_i$ of the dual slack:
$\lambda_i(X) = \psi'(s_i/\varepsilon)$.

```python
from sdplab import EntropyReg, RegularizedSDPDualFunctional, run_regularized_solver
from sdplab.examples import generate_max_cut

problem = generate_max_cut(8, seed=0, unit_trace=True)
dual = RegularizedSDPDualFunctional(problem, EntropyReg(problem.dom))

result = run_regularized_solver(dual.bind(0.1), verbose=0)
X = dual.primal_from_dual(result.dual, 0.1)

problem.primal_objective(X)     # -5.0859, against -5.0990 from CVXPY
```

$\varepsilon$ is a per-call argument, so a continuation schedule can lower it
without rebuilding anything; `bind(eps)` fixes it and yields a standard
single-argument `spacecore.Functional`.

## Regularizers

| class | $\varphi(t)$ | recovered primal |
| --- | --- | --- |
| `EntropyReg` | $t(\log t - 1)$ | full rank, Gibbs state |
| `QuadraticReg` | $t^2/2$ | clipped, $\max\\{s/\varepsilon, 0\\}$ |
| `TsallisReg` | $(t^q - t)/(q-1)$, $q > 1$ | **exactly** low rank |

`TsallisReg` is the q-exponential (α-entmax) family: softmax at $q \to 1$,
sparsemax at $q = 2$.

## Examples

`sdplab.examples` ships three instances, chosen to differ in the structure of
$\mathcal{A}$:

- `generate_max_cut` — real, diagonal extraction. `unit_trace=True` rescales the
  variable so $\mathrm{Tr}\,X = 1$.
- `generate_random_qot` — quantum optimal transport: complex, with a stacked
  Hermitian-block codomain, so the dual is a tuple of matrices.
- `generate_qubit_tomography` — zero cost, so pure feasibility.

## Backends

Everything is written against spacecore's backend contract, so NumPy, JAX, and
torch contexts all work. `run_regularized_solver` picks
`spacecore.minimize_optax` on a JAX backend and `spacecore.minimize_scipy`
otherwise. 
