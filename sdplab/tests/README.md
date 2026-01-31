# Test layout

This directory groups tests by functional area.

- `core/`: core API unit tests for backend context, ops, spaces, linops, low-rank, and type protocols.
- `ops/`: backend ops coverage (NumPy/JAX) beyond the core API tests.
- `space/`: vector and Hermitian matrix spaces for NumPy/JAX beyond core coverage.
- `linop/`: dense and sparse linear operators for NumPy/JAX beyond core coverage.
- `low_rank/`: low-rank matrix types and low-rank Hermitian spaces beyond core coverage.
- `conftest.py`: shared fixtures and optional backend availability helpers.
