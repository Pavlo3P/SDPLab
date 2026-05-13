"""Example generators for quantum optimal transport problems.

The exported generator returns both an ``SDPDenseProblem`` and a known
feasible primal state, which makes it useful for solver tests and demos.
"""

from ._random import generate_random_qot
