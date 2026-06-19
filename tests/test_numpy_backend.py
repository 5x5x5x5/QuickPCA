"""Tests for the NumPy compute backend."""

from __future__ import annotations

import numpy as np

from quickpca.backends import get_backend
from quickpca.backends.numpy_backend import NumpyBackend
from quickpca.types import PCADecomposition


def _rmsd_to_ref(frames: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(((frames - ref[None]) ** 2).sum(axis=2).mean()))


def test_align_frames_shape_and_reduces_rmsd(synthetic_coords):
    be = get_backend("numpy")
    assert isinstance(be, NumpyBackend)

    ref = synthetic_coords[0]
    aligned = be.align_frames(synthetic_coords, ref)

    assert aligned.shape == synthetic_coords.shape
    assert aligned.dtype == np.float64

    # Apply a random rotation+translation to each frame, then realign.
    rng = np.random.default_rng(0)
    perturbed = np.empty_like(synthetic_coords)
    for i in range(synthetic_coords.shape[0]):
        a, b, c = rng.uniform(0, 2 * np.pi, size=3)
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
        R = Rz @ Ry @ Rx
        perturbed[i] = synthetic_coords[i] @ R.T + rng.uniform(-10, 10, size=3)

    realigned = be.align_frames(perturbed, ref)
    assert _rmsd_to_ref(realigned, ref) < _rmsd_to_ref(perturbed, ref)


def test_pca_shapes_and_reconstruction(synthetic_coords):
    be = get_backend("numpy")
    F, N, _ = synthetic_coords.shape
    X = synthetic_coords.reshape(F, 3 * N)
    n_comp = 8

    decomp = be.pca(X, n_comp)
    assert isinstance(decomp, PCADecomposition)
    assert decomp.projections.shape == (F, n_comp)
    assert decomp.components.shape == (n_comp, 3 * N)
    assert decomp.explained_variance.shape == (n_comp,)
    assert decomp.explained_variance_ratio.shape == (n_comp,)
    assert decomp.mean.shape == (3 * N,)

    # explained_variance_ratio sorted descending and sums to <= 1 + eps.
    evr = decomp.explained_variance_ratio
    assert np.all(np.diff(evr) <= 1e-9)
    assert evr.sum() <= 1.0 + 1e-8

    # Reconstruction with the full-rank decomposition reproduces X exactly.
    full = be.pca(X, min(X.shape))
    recon = full.projections @ full.components + full.mean
    assert np.allclose(recon, X, atol=1e-6)


def test_pca_matches_sklearn(synthetic_coords):
    try:
        from sklearn.decomposition import PCA
    except Exception:  # pragma: no cover
        import pytest

        pytest.skip("scikit-learn not installed")

    be = get_backend("numpy")
    F, N, _ = synthetic_coords.shape
    X = synthetic_coords.reshape(F, 3 * N)
    n_comp = 6

    decomp = be.pca(X, n_comp)
    sk = PCA(n_components=n_comp, svd_solver="full").fit(X)

    assert np.allclose(decomp.explained_variance, sk.explained_variance_, atol=1e-6)
    assert np.allclose(
        decomp.explained_variance_ratio, sk.explained_variance_ratio_, atol=1e-8
    )
    # Components match up to sign; both use sklearn-style svd_flip so they agree.
    assert np.allclose(np.abs(decomp.components), np.abs(sk.components_), atol=1e-6)
