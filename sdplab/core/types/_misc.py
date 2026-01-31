from typing import Any, Tuple, Union, TypeVar

Index = Union[
    int,
    slice,
    Any,
    Tuple[Any, ...],
]

T = TypeVar("T")
Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")
R = TypeVar("R")