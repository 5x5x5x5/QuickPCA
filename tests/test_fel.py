"""Tests for the Free-Energy Landscape computation."""

from __future__ import annotations

import numpy as np

from quickpca.fel import compute_fel
from quickpca.pca import compute_pca
from quickpca.types import FELResult


def test_fel_min_zero_and_shapes(synthetic_coords):
    pca = compute_pca(synthetic_coords, n_components=5, backend="numpy")
    n_bins = 40
    fel = compute_fel(pca.projections, temperature=300.0, n_bins=n_bins, sigma=1.0)

    assert isinstance(fel, FELResult)
    assert fel.F.shape == (n_bins, n_bins)
    assert fel.xcenters.shape == (n_bins,)
    assert fel.ycenters.shape == (n_bins,)
    assert fel.xedges.shape == (n_bins + 1,)
    assert fel.yedges.shape == (n_bins + 1,)

    assert np.isclose(np.nanmin(fel.F), 0.0, atol=1e-9)
    assert np.isclose(fel.kBT, 0.008314462 * 300.0)
    assert fel.temperature == 300.0


def test_fel_centers_consistent(synthetic_coords):
    pca = compute_pca(synthetic_coords, n_components=5, backend="numpy")
    fel = compute_fel(pca.projections, n_bins=50)
    assert np.allclose(fel.xcenters, 0.5 * (fel.xedges[:-1] + fel.xedges[1:]))
    assert np.allclose(fel.ycenters, 0.5 * (fel.yedges[:-1] + fel.yedges[1:]))
