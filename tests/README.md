# Test layout

This directory groups tests by test role.

- `unit/`: small deterministic checks for individual SDP, linalg, regularizer, and solver helper functions.
- `integration/`: cross-module checks that verify backend dispatch and solver launch behavior.
- `conftest.py`: shared backend context fixtures and optional dependency checks.
