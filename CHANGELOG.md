# Changelog

## [0.0.1] - 2026-08-16

Initial release, built on `spacecore >= 0.4.2, < 0.5`. Problem data, spectral
regularizers, and the smoothed dual are the package; every first-order
optimization loop is delegated to `spacecore.optimize`.

### Added

- **Problem data** (`sdplab.problem`) — `SDPProblem` holding `(C, A, b)` over any
  Euclidean Jordan algebra domain, including `TreeSpace`. Costs are operators
  (`Cost`, `ElementCost`, `HermitianCost` and dense/sparse variants), not just
  stored elements, so a Hermitian cost also acts on vectors. Constraint operators
  (`ConstraintOp`) carry an explicit cvxpy encoding: `to_cvxpy` returns
  per-constraint matrices in the `Re Tr[A_i X] = b_i` convention, with
  `rhs_to_cvxpy`/`dual_from_cvxpy` inverting the row layout. `DenseConstraintOp`
  and `SparseConstraintOp` read the `A_i` from storage; `MatrixFreeConstraintOp`
  materializes them through the adjoint; `WrappedConstraintOp` adapts an arbitrary
  `LinOp` without densifying it.

- **Spectral regularizers** (`sdplab.regularization`) — `Regularizer` applies a
  scalar convex `φ` spectrally over the domain, with `EntropyReg`, `QuadraticReg`,
  and `TsallisReg` (the q-exponential / α-entmax family, compactly supported for
  `q > 1`, so the recovered primal is exactly low rank). Every `phi` carries the
  domain indicator: `+inf` below a small negative tolerance, the limit value
  inside it, so `phi` is the Fenchel partner of the `phi_star` beside it.

- **Unit-trace primal recovery** — the trace constraint's multiplier enters
  additively in the argument of `ψ'`, not as a division, so the softmax
  `g_i / Σ g_j` is the fixed-trace primal only where `log ψ'` is affine (the
  entropy family). `TsallisReg` therefore takes `normalization="theta"` (default)
  solving `Σ ψ'((s_i − θ)/ε) = 1` for a chemical potential, or `"softmax"` for the
  cheaper base-class form. The root find is a fixed-trip `fori_loop` with a
  closed-form bracket, so it traces under `jax.jit`.

- **Smoothed dual** (`sdplab.regularization`) — `RegularizedSDPDualFunctional`
  evaluates `D_ε(y) = ⟨b,y⟩ − ε Tr[ψ((A†y − C)/ε)]` and recovers the primal from
  the dual slack. ε is a per-call argument, so a continuation schedule varies it
  without rebuilding; `bind(eps)` fixes it and yields a standard single-argument
  `Functional`.

- **Solvers** (`sdplab.solvers`) — `run_regularized_solver` takes a
  `BoundDualFunctional` and dispatches to `spacecore.minimize_scipy` or
  `spacecore.minimize_optax`, translating results into `OptimizeResult`.
  `run_cvxpy_solver` is the reference backend.

- **Examples** (`sdplab.examples`) — `generate_max_cut` (with `unit_trace` for the
  rescaled variable), `generate_random_qot`, `generate_qubit_tomography`, plus the
  operators and the Erdős–Rényi sampler behind them.

- **Domain-specific** (`sdplab.special`) — QOT partial-trace operator with a sparse
  materialization and a reduced Hermitian-generator cvxpy encoding, a dedicated QOT
  dual solver, and Pauli string/sum algebra. `sdplab.linalg` carries the Kronecker
  and partial-trace kernels.

### Notes

- Top-level solver entry points resolve lazily (PEP 562), so `import sdplab` does
  not pull in CVXPY.
- `RegularizedSDPDualFunctional` always reports the *maximization* objective;
  negate through the spacecore functional algebra (`-bound`) for a minimizer.
