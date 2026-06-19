"""Tests for the optional JAX compute backend.

The JAX backend is numerically equivalent to the NumPy reference backend: both
use Kabsch alignment and a full-SVD PCA with the sklearn ``svd_flip`` sign
convention. These tests assert agreement within floating-point tolerance and
correct output shapes. They are skipped cleanly when JAX is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from quickpca.backends import available_backends, get_backend  # noqa: E402
from quickpca.backends.jax_backend import JaxBackend  # noqa: E402
from quickpca.backends.numpy_backend import NumpyBackend  # noqa: E402
from quickpca.types import PCADecomposition  # noqa: E402


def test_jax_backend_is_registered():
    """The JAX backend self-registers and is constructible by name."""
    assert "jax" in available_backends()
    be = get_backend("jax")
    assert isinstance(be, JaxBackend)
    assert be.name == "jax"


def test_align_frames_matches_numpy(synthetic_coords):
    """JAX alignment matches NumPy alignment within tolerance."""
    jax_be = get_backend("jax")
    np_be = NumpyBackend()
    ref = synthetic_coords[0]

    jax_aligned = jax_be.align_frames(synthetic_coords, ref)
    np_aligned = np_be.align_frames(synthetic_coords, ref)

    assert jax_aligned.shape == synthetic_coords.shape
    assert jax_aligned.dtype == np.float64
    assert isinstance(jax_aligned, np.ndarray)
    assert np.allclose(jax_aligned, np_aligned, atol=1e-8)


def test_align_frames_matches_numpy_after_perturbation(synthetic_coords):
    """Alignment agrees even after random rigid-body perturbations per frame."""
    jax_be = get_backend("jax")
    np_be = NumpyBackend()
    ref = synthetic_coords[0]

    rng = np.random.default_rng(7)
    perturbed = np.empty_like(synthetic_coords)
    for i in range(synthetic_coords.shape[0]):
        a, b, c = rng.uniform(0, 2 * np.pi, size=3)
        rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
        rot = rz @ ry @ rx
        perturbed[i] = synthetic_coords[i] @ rot.T + rng.uniform(-10, 10, size=3)

    jax_aligned = jax_be.align_frames(perturbed, ref)
    np_aligned = np_be.align_frames(perturbed, ref)
    assert np.allclose(jax_aligned, np_aligned, atol=1e-7)


def test_pca_shapes_and_types(synthetic_coords):
    """The JAX PCA returns a PCADecomposition with the contracted shapes."""
    be = get_backend("jax")
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
    for arr in (
        decomp.projections,
        decomp.components,
        decomp.explained_variance,
        decomp.explained_variance_ratio,
        decomp.mean,
    ):
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64


def test_pca_matches_numpy(synthetic_coords):
    """JAX PCA matches NumPy PCA within tolerance (components up to sign)."""
    jax_be = get_backend("jax")
    np_be = NumpyBackend()
    F, N, _ = synthetic_coords.shape
    X = synthetic_coords.reshape(F, 3 * N)
    n_comp = 6

    jx = jax_be.pca(X, n_comp)
    npd = np_be.pca(X, n_comp)

    assert np.allclose(jx.mean, npd.mean, atol=1e-8)
    assert np.allclose(jx.explained_variance, npd.explained_variance, atol=1e-6)
    assert np.allclose(
        jx.explained_variance_ratio, npd.explained_variance_ratio, atol=1e-8
    )

    # Both apply the sklearn svd_flip convention, so signs should agree; fall
    # back to absolute values to be robust to any residual degenerate flip.
    if np.allclose(jx.components, npd.components, atol=1e-6):
        assert np.allclose(jx.projections, npd.projections, atol=1e-6)
    else:
        assert np.allclose(np.abs(jx.components), np.abs(npd.components), atol=1e-6)
        assert np.allclose(np.abs(jx.projections), np.abs(npd.projections), atol=1e-6)


def test_pca_reconstruction(synthetic_coords):
    """A full-rank JAX decomposition reconstructs X exactly."""
    be = get_backend("jax")
    F, N, _ = synthetic_coords.shape
    X = synthetic_coords.reshape(F, 3 * N)

    full = be.pca(X, min(X.shape))
    recon = full.projections @ full.components + full.mean
    assert np.allclose(recon, X, atol=1e-6)
