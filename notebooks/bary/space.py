from __future__ import annotations

from numbers import Number
from typing import Any, Tuple

import spacecore as sc
from spacecore import (
    Space,
    jax_pytree_class,
    DenseArray,
    Context,
    HermitianSpace,
    normalize_context,
    SpaceCheck,
    HermitianCheck,
    checked_method
)
from dataclasses import dataclass
from sdplab.special.qot import BlockMatrixSpace

@jax_pytree_class
@dataclass
class BaryPrimalEl:
    rho: DenseArray  # (d0, d0)
    couplings: DenseArray   # (s, d0 * d, d0 * d)

    @property
    def shape(self):
        return (self.rho.shape, self.couplings.shape)

    @property
    def dtype(self):
        return self.rho.dtype

    def tree_flatten(self):
        return (self.rho, self.couplings), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        rho, couplings = children
        return cls(rho=rho, couplings=couplings)


@jax_pytree_class
@dataclass
class BaryDualEl:
    U: DenseArray  # (s, d0, d0)
    V: DenseArray  # (s, d, d)

    @property
    def shape(self):
        return (self.U.shape, self.V.shape)

    @property
    def dtype(self):
        return self.U.dtype

    def tree_flatten(self):
        return (self.U, self.V), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        U, V = children
        return cls(U=U, V=V)


@dataclass(frozen=True)
class PrimalBaryTypeCheck(SpaceCheck):
    name: str = "PrimalBaryTypeCheck"

    def is_valid(self, space: Any, x: Any):
        return isinstance(x, BaryPrimalEl)

    def error_message(self, space: Any, x: Any):
        ...


@dataclass(frozen=True)
class DualBaryTypeCheck(SpaceCheck):
    name: str = "DualBaryTypeCheck"

    def is_valid(self, space: Any, x: Any):
        return isinstance(x, BaryDualEl)

    def error_message(self, space: Any, x: Any):
        ...


@dataclass(frozen=True)
class RhoHermCheck(HermitianCheck):
    name: str = "RhoHermCheck"

    def is_valid(self, space: Any, x: BaryPrimalEl):
        return super().is_valid(space, x.rho)


@dataclass(frozen=True)
class BlockHermCheck(HermitianCheck):
    name: str = "BlockHermCheck"

    def is_valid(self, space: Any, x: Any):
        ...

    def error_message(self, space: Any, x: Any):
        ...

@dataclass(frozen=True)
class CouplingHermCheck(BlockHermCheck):
    name: str = "CouplingHermCheck"

    def is_valid(self, space: Any, x: BaryPrimalEl):
        return super().is_valid(space, x.couplings)


@dataclass(frozen=True)
class UBlockHermCheck(BlockHermCheck):
    name: str = "UBlockHermCheck"

    def is_valid(self, space: Any, x: BaryDualEl):
        return super().is_valid(space, x.U)


@dataclass(frozen=True)
class VBlockHermCheck(BlockHermCheck):
    name: str = "VBlockHermCheck"

    def is_valid(self, space: Any, x: BaryDualEl):
        return super().is_valid(space, x.V)


class BaryPrimalSpace(Space):
    def _local_checks(self):
        return PrimalBaryTypeCheck(), RhoHermCheck(), CouplingHermCheck()

    def __init__(self,
                 *,
                 d0: int, d: int, s: int,
                 atol: float = 0.,
                 rtol: float = 0.,
                 enforce_herm: bool = True,
                 ctx: Context | str | None = None):
        ctx = normalize_context(ctx)
        self._rho_space = HermitianSpace(d0, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=ctx)
        self._coupling_space = BlockMatrixSpace(d=d, N=s, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=ctx)
        shape = self._rho_space.shape + self._coupling_space.shape

        super(BaryPrimalSpace, self).__init__(shape=shape, ctx=ctx)

    @property
    def rho_space(self) -> HermitianSpace:
        return self._rho_space

    @property
    def coupling_space(self) -> BlockMatrixSpace:
        return self._coupling_space

    @property
    def rho_shape(self) -> Tuple[int, int]:
        return self._rho_space.shape

    @property
    def coupling_shape(self) -> Tuple[int, int, int]:
        return self._coupling_space.shape

    @property
    def s(self) -> int:
        return self.coupling_shape[0]

    @property
    def d(self) -> int:
        return self.coupling_shape[1]

    @property
    def d0(self) -> int:
        return self.rho_shape[0]

    def zeros(self):
        return BaryPrimalEl(
            rho=self.rho_space.zeros(),
            couplings=self.coupling_space.zeros(),
        )

    @checked_method(in_space="self", arg_positions=(0, 1))
    def add(self, x: BaryPrimalEl, y: BaryPrimalEl) -> BaryPrimalEl:
        return BaryPrimalEl(
            rho=self.rho_space.add(x, y),
            couplings=self.coupling_space.add(x, y),
        )

    @checked_method(in_space="self", arg_positions=(1,))
    def scale(self, a: Any, x: BaryPrimalEl) -> BaryPrimalEl:
        return BaryPrimalEl(
            rho=self.rho_space.scale(a, x.rho),
            couplings=self.coupling_space.scale(a, x.couplings),
        )

    @checked_method(in_space="self", arg_positions=(0, 1))
    def inner(self, x: BaryPrimalEl, y: BaryPrimalEl) -> Number:
        return self.rho_space.inner(x.rho, y.rho) + self.coupling_space.inner(x.couplings, y.couplings)

    @checked_method(in_space="self", arg_positions=(0, ))
    def eigh(self, x: BaryPrimalEl, k: int = None) -> Tuple[Tuple[DenseArray, DenseArray], Tuple[DenseArray, DenseArray]]:
        return self.rho_space.eigh(x.rho, k), self.coupling_space.eigh(x.couplings, k)

    @checked_method(in_space="self", arg_positions=(0, ))
    def flatten(self, x: BaryPrimalEl) -> DenseArray:
        pass

    def unflatten(self, v: DenseArray) -> BaryPrimalEl:
        pass


class BaryDualSpace(Space):
    def _local_checks(self):
        return DualBaryTypeCheck(), UBlockHermCheck(), VBlockHermCheck()

    def __init__(self,
                 *,
                 d0: int, d: int, s: int,
                 atol: float = 0.,
                 rtol: float = 0.,
                 enforce_herm: bool = True,
                 ctx: Context | str | None = None):
        ctx = normalize_context(ctx)
        self._u_space = BlockMatrixSpace(d=d0, N=s, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=ctx)
        self._v_space = BlockMatrixSpace(d=d, N=s, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=ctx)
        shape = self._u_space.shape + self._v_space.shape

        super(BaryDualSpace, self).__init__(shape=shape, ctx=ctx)

    @property
    def u_space(self) -> BlockMatrixSpace:
        return self._u_space

    @property
    def v_space(self) -> BlockMatrixSpace:
        return self._v_space

    @property
    def u_shape(self) -> Tuple[int, int, int]:
        return self._u_space.shape

    @property
    def v_shape(self) -> Tuple[int, int, int]:
        return self._v_space.shape

    @property
    def s(self) -> int:
        return self.u_shape[0]

    @property
    def d(self) -> int:
        return self.v_shape[1]

    @property
    def d0(self) -> int:
        return self.u_shape[1]

    def zeros(self) -> BaryDualEl:
        return BaryDualEl(
            U = self.u_space.zeros(),
            V = self.v_space.zeros(),
        )

    @checked_method(in_space="self", arg_positions=(0, 1))
    def add(self, x: BaryDualEl, y: BaryDualEl) -> BaryDualEl:
        return BaryDualEl(
            U = self.u_space.add(x.U, y.U),
            V = self.v_space.add(x.V, y.V),
        )

    @checked_method(in_space="self", arg_positions=(1,))
    def scale(self, a: Any, x: BaryDualEl) -> BaryDualEl:
        return BaryDualEl(
            U=self.u_space.scale(a, x.U),
            V=self.v_space.scale(a, x.V),
        )

    @checked_method(in_space="self", arg_positions=(0, 1))
    def inner(self, x: BaryDualEl, y: BaryDualEl) -> Number:
        return self.u_space.inner(x.U, y.U) + self.v_space.inner(x.V, y.V)

    @checked_method(in_space="self", arg_positions=(0, ))
    def eigh(self, x: BaryDualEl, k: int = None) -> Tuple[Tuple[DenseArray, DenseArray], Tuple[DenseArray, DenseArray]]:
        return self.u_space.eigh(x.U, k), self.v_space.eigh(x.V, k)

    @checked_method(in_space="self", arg_positions=(0, ))
    def flatten(self, x: BaryPrimalEl) -> DenseArray:
        pass

    def unflatten(self, v: DenseArray) -> BaryPrimalEl:
        pass
