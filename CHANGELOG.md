# Changelog

## [Unreleased] - 2026-05-27

### Changed

- Requires `spacecore >= 0.2.0, < 0.3`.
- Replaced private `ctx_manager` usage with the public `normalize_context`
  function exported from `spacecore._contextual`
  (`sdplab/special/pauli/_operators.py`).

### Removed

- Removed `sdplab.linalg.stochastic_lanczos`. Use `spacecore.lanczos_smallest`
  instead. The spacecore version takes a `LinOp` and returns a
  `LanczosResult` named tuple including residual norm, Krylov dimension, and
  convergence flag, rather than a plain `(eigenvalue, eigenvector)` pair.
- Removed `sdplab.linalg.power_method`. Use `spacecore.power_iteration`
  instead. The new version takes a `LinOp` and returns a
  `PowerIterationResult` named tuple including the Rayleigh-quotient eigenvalue
  estimate and a convergence flag.

### Added

- Regression test for `sdplab.special.pauli` construction to lock in the
  `normalize_context` API contract.
