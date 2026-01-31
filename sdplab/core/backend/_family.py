from enum import Enum, auto


class BackendFamily(Enum):
    NUMPY = auto()
    JAX = auto()
