# from scipy import sparse
# from jax.experimental import sparse as jsparse
from typing import Any, TypeAlias


# ScipySparseArray = Union[
#     sparse.bsr_matrix,
#     sparse.coo_matrix,
#     sparse.csc_matrix,
#     sparse.csr_matrix,
#     sparse.dia_matrix,
#     sparse.dok_matrix,
#     sparse.lil_matrix,
# ]
# JaxSparseArray = jsparse.BCOO
# SparseArray = Union[ScipySparseArray, JaxSparseArray]

SparseArray: TypeAlias = Any
