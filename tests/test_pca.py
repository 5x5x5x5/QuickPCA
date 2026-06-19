"""Tests for the high-level PCA pipeline."""

from __future__ import annotations

import numpy as np

from quickpca.pca import compute_pca
from quickpca.types import PCAResult


def test_compute_pca_shapes(synthetic_coords):
    res = compute_pca(synthetic_coords, n_components=10, backend="numpy")
    assert isinstance(res, PCAResult)

    F, N, _ = synthetic_coords.shape
    nc = res.n_components
    assert res.n_frames == F
    assert res.n_atoms == N
    assert res.backend == "numpy"

    assert res.projections.shape == (F, nc)
    assert res.components.shape == (nc, 3 * N)
    assert res.explained_variance.shape == (nc,)
    assert res.explained_variance_ratio.shape == (nc,)
    assert res.cumulative_variance.shape == (nc,)
    assert res.mean.shape == (3 * N,)


def test_cross_correlation_properties(synthetic_coords):
    res = compute_pca(synthetic_coords, n_components=10, backend="numpy")
    cc = res.cross_correlation
    N = synthetic_coords.shape[1]

    assert cc.shape == (N, N)
    assert np.allclose(cc, cc.T, atol=1e-5)
    assert np.allclose(np.diag(cc), 1.0, atol=1e-4)
    assert cc.min() >= -1.0001 and cc.max() <= 1.0001


def test_first_pcs_capture_most_variance(synthetic_coords):
    res = compute_pca(synthetic_coords, n_components=10, backend="numpy")
    # The synthetic data has ~3 collective modes, so the first few PCs dominate.
    assert res.cumulative_variance[2] > 0.9
    assert res.explained_variance_ratio[0] >= res.explained_variance_ratio[1]


def test_no_align_path(synthetic_coords):
    res = compute_pca(synthetic_coords, n_components=5, backend="numpy", align=False)
    assert res.projections.shape[0] == synthetic_coords.shape[0]
