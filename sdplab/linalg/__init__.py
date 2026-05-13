"""Linear algebra helpers used by SDP models and solvers.

The helpers here support common matrix operations behind SDP modeling:
Kronecker products for tensor-product systems, log-trace-exp for spectral
regularizers, and iterative eigenvalue routines for large operators.
"""

from ._eigval import power_method, stochastic_lanczos
from ._matrix import kron_all, log_trace_exp

__all__ = [
    "power_method",
    "stochastic_lanczos",
    "kron_all",
    "log_trace_exp",
]
