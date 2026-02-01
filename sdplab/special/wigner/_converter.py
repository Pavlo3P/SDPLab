from __future__ import annotations

from abc import abstractmethod

from qutip import Qobj
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Union

from ...qot import QOTPrimal
from ...sdp import SDPPrimal

Primal = Union[QOTPrimal, SDPPrimal]
Array = jnp.ndarray

class QutipConverter:
    primal: Primal
    qobj: Qobj

    @classmethod
    def from_qobj(cls, qobj: Qobj) -> QutipConverter:
        converter = cls.__new__(cls)
        converter.qobj = qobj

        if len(qobj.dims) == 2:
            X = jnp.asarray(qobj.full())
            primal = SDPPrimal(X)
        else:
            # Assuming dims = [[primal.d] * primal.N] * 2
            d = qobj.dims[0][0]
            N = len(qobj.dims[0])
            X = jnp.asarray(qobj.full())
            primal = QOTPrimal(X, d=d, N=N)
        converter.primal = primal
        return converter


    @classmethod
    def from_qotprimal(cls, primal: QOTPrimal) -> QutipConverter:
        converter = cls.__new__(cls)
        converter.primal = primal
        dims = [[primal.d] * primal.N] * 2
        converter.qobj = Qobj(primal.X, dims=dims)
        return converter

    @classmethod
    def from_sdpprimal(cls, primal: SDPPrimal) -> QutipConverter:
        converter = cls.__new__(cls)
        converter.primal = primal
        dims = [[primal.shape[0]]] * 2
        converter.qobj = Qobj(primal.X, dims=dims)
        return converter

    @classmethod
    def from_dense(cls, dense: Array) -> QutipConverter:
        qobj = Qobj(dense)
        return cls.from_qobj(qobj)

    @staticmethod
    def convert(obj: Union[Primal, Qobj, Array]) -> QutipConverter:
        if isinstance(obj, QOTPrimal):
            converter = QutipConverter.from_qotprimal(obj)
        elif isinstance(obj, Qobj):
            converter = QutipConverter.from_qobj(obj)
        elif isinstance(obj, SDPPrimal):
            converter = QutipConverter.from_sdpprimal(obj)
        elif isinstance(obj, Array):
            converter = QutipConverter.from_dense(obj)
        else:
            raise TypeError(f"Cannot convert {type(obj)} to QutipConverter")

        return converter

    @property
    def dense(self) -> Array:
        return self.primal.X
