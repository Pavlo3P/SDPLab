from __future__ import annotations

from typing import Tuple
from dataclasses import dataclass
import spacecore as sc
from spacecore import ContextBound, DenseArray, Context, resolve_context_priority, HermitianSpace

from sdplab.sdp import SDPPrimal, SDPDual
from sdplab.regularization import AbstractRegularizer
from sdplab.special.qot import BlockMatrixSpace
from .space import BaryPrimalEl, BaryDualEl, BaryDualSpace, BaryPrimalSpace

def _infer_barycenter_problem_dimensions(
        cost_vector: DenseArray,
        sigma: DenseArray,
) -> Tuple[int, int, int]:
    cost_shape = cost_vector.shape
    sigma_shape = sigma.shape

    if len(cost_shape) != 3:
        raise ValueError
    if len(sigma_shape) != 3:
        raise ValueError

    s1 = cost_shape[0]
    s2 = sigma_shape[0]
    if s1 != s2:
        raise ValueError
    s = s1

    d = sigma.shape[1]
    if d != sigma_shape[2]:
        raise ValueError

    dd0 = cost_shape[1]
    if dd0 != cost_shape[2]:
        raise ValueError

    d0 = int(dd0 / d)
    if float(d0) != dd0 / d:
        raise ValueError

    return s, d, d0

def _validate_alpha(alpha: DenseArray, s: int) -> DenseArray:
    s1 = alpha.shape[0]
    if s1 != s:
        raise ValueError
    all_positive = all(alpha > 0)
    if not all_positive:
        raise ValueError
    return alpha


@dataclass(init=False)
class IOBarycenter(ContextBound):
    def __init__(
            self,
            cost_vector: DenseArray,  # (s, d0 * d, d0 * d)
            alpha: DenseArray,  # (s,)
            sigma: DenseArray,  # (s, d, d)
            inner_reg: AbstractRegularizer,  # QOT regularization
            outer_reg: AbstractRegularizer,  # Barycenter regularization
            atol: float = 0.,
            rtol: float = 0.,
            enforce_herm: bool = True,
            ctx: Context | str | None = None
    ):
        ctx = resolve_context_priority(ctx, cost_vector, inner_reg, outer_reg)
        super(IOBarycenter, self).__init__(ctx)
        self.ctx.assert_dense(cost_vector)
        alpha = self.ctx.asarray(alpha)
        sigma = self.ctx.asarray(sigma)

        s, d, d0 = _infer_barycenter_problem_dimensions(cost_vector, sigma)
        alpha = _validate_alpha(alpha, s)

        self.s = s
        self.d = d
        self.d0 = d0
        self.cost_vector = cost_vector  # No asarray conversion to prevent conversion of dtype
        self.weighted_cost_vector = self.ops.einsum('i,ijk->ijk', alpha, cost_vector)
        self.alpha = alpha
        self.sigma = sigma
        self.inner_reg = inner_reg.convert(ctx)
        self.outer_reg = outer_reg.convert(ctx)

        self.primal_space = BaryPrimalSpace(d0=d0, s=s, d=d, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=ctx)
        self.dual_space = BaryDualSpace(d0=d0, s=s, d=d, atol=atol, rtol=rtol, enforce_herm=enforce_herm, ctx=ctx)

    @property
    def rho_space(self) -> HermitianSpace:
        return self.primal_space.rho_space

    @property
    def coupling_space(self) -> BlockMatrixSpace:
        return self.primal_space.coupling_space

    @property
    def u_space(self) -> BlockMatrixSpace:
        return self.dual_space.u_space

    @property
    def v_space(self) -> BlockMatrixSpace:
        return self.dual_space.v_space

    def primal_objective(self, primal: SDPPrimal, inner_reg: bool = True, outer_reg: bool = True) -> float:
        X: BaryPrimalEl = primal.X
        cost = self.coupling_space.inner(X.couplings, self.weighted_cost_vector)

        if inner_reg:
            ...

        if outer_reg:
            outer_reg = self.outer_reg(X.rho)

        return cost


    def dual_objective(self, dual: SDPDual, inner_reg: bool = True, outer_reg: bool = True):
        y: BaryDualEl = dual.y
        cost = self.v_space.inner(y.V, self.sigma)

        if inner_reg:
            ...

        if outer_reg:
            ...

        return cost

    def dual_grad(self, ):
        ...
