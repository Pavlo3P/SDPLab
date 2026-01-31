import jax.numpy as jnp
import numpy as np
import jax
from matplotlib import pyplot as plt
from matplotlib import colors
from qutip import Qobj, wigner, plot_wigner
from dataclasses import dataclass, field
from typing import Union

from ._converter import QutipConverter, Primal


def _qutip_like_norm(W, pos_scale: float = 0.15):
    """
    QuTiP-style normalization:
      - strong emphasis on negatives,
      - positives are washed out (small vmax),
      - white at zero.
    pos_scale is the fraction of |min(W)| used as vmax.
    """
    Wn = np.asarray(W)
    wmin = float(Wn.min())
    wmax_pos = float(Wn[Wn > 0].max()) if (Wn > 0).any() else 0.0
    vmax = min(wmax_pos, pos_scale * abs(wmin))
    vmin = wmin
    return colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


@dataclass
class WignerTransform:
    x: jnp.ndarray
    p: jnp.ndarray
    hbar: float = 1.

    def compute(self, primal: Union[Primal, Qobj]) -> jnp.ndarray:
        converter = QutipConverter.convert(primal)
        wt = wigner(converter.qobj, self.x, self.p, g=(2 / self.hbar) ** .5)
        return wt

    def plot(self, primal: Union[Primal, Qobj]):
        converter = QutipConverter.convert(primal)
        fig, _ = plot_wigner(converter.qobj, self.x, self.p, g=(2 / self.hbar) ** .5)
        return fig
