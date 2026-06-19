"""Boltzmann-inverted 2-D Free-Energy Landscape from PC1/PC2 projections."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .types import FELResult


def compute_fel(
    projections: np.ndarray,
    temperature: float = 300.0,
    n_bins: int = 50,
    sigma: float = 1.0,
) -> FELResult:
    """Compute the Free-Energy Landscape over the first two principal components.

    ``F = -kBT ln(rho)`` where ``rho`` is a Gaussian-smoothed 2-D density of the
    PC1/PC2 projections, shifted so that ``min(F) == 0``.
    """
    projections = np.asarray(projections, dtype=np.float64)
    kBT = 0.008314462 * temperature  # kJ/mol
    pc1, pc2 = projections[:, 0], projections[:, 1]

    pad_x = (pc1.max() - pc1.min()) * 0.20
    pad_y = (pc2.max() - pc2.min()) * 0.20
    rng = [
        [pc1.min() - pad_x, pc1.max() + pad_x],
        [pc2.min() - pad_y, pc2.max() + pad_y],
    ]

    hist, xe, ye = np.histogram2d(pc1, pc2, bins=n_bins, range=rng, density=True)
    hist_s = gaussian_filter(hist, sigma=sigma)

    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(hist_s > 0, -kBT * np.log(hist_s), np.nan)
    F -= np.nanmin(F)

    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])

    return FELResult(
        F=F,
        xcenters=xc,
        ycenters=yc,
        xedges=xe,
        yedges=ye,
        pc1=pc1,
        pc2=pc2,
        kBT=kBT,
        temperature=temperature,
    )
