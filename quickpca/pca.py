"""High-level PCA pipeline orchestrating alignment, decomposition and crosscorr."""

from __future__ import annotations

import numpy as np

from .backends import get_backend
from .crosscorr import cross_correlation
from .types import PCAResult


def compute_pca(
    coords: np.ndarray,
    n_components: int = 10,
    backend: str = "numpy",
    align: bool = True,
    ref_index: int = 0,
) -> PCAResult:
    """Run the full PCA pipeline on a coordinate array.

    Parameters
    ----------
    coords:
        Array of shape ``(F, N, 3)``.
    n_components:
        Maximum number of principal components to retain.
    backend:
        Name of a registered compute backend.
    align:
        If True, Kabsch-align every frame onto ``coords[ref_index]``.
    ref_index:
        Index of the reference frame for alignment.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n_frames, n_atoms, _ = coords.shape

    be = get_backend(backend)

    ref = coords[ref_index]
    frames = be.align_frames(coords, ref) if align else coords

    X = frames.reshape(n_frames, 3 * n_atoms)
    n_comp = min(n_components, min(X.shape) - 1)

    decomp = be.pca(X, n_comp)

    cc = cross_correlation(decomp.components, decomp.explained_variance, n_atoms)
    cumulative = np.cumsum(decomp.explained_variance_ratio)

    return PCAResult(
        projections=decomp.projections,
        explained_variance=decomp.explained_variance,
        explained_variance_ratio=decomp.explained_variance_ratio,
        cumulative_variance=cumulative,
        components=decomp.components,
        cross_correlation=cc,
        n_components=n_comp,
        n_atoms=n_atoms,
        n_frames=n_frames,
        mean=decomp.mean,
        backend=be.name,
    )


def compute_pca_from_files(
    topology: str,
    trajectory: str | None = None,
    selection: str = "name CA",
    interval: int = 1,
    n_components: int = 10,
    backend: str = "numpy",
    align: bool = True,
) -> PCAResult:
    """Load a trajectory from files and run :func:`compute_pca`."""
    from .io.loader import load_trajectory

    traj = load_trajectory(topology, trajectory, selection=selection, interval=interval)
    return compute_pca(
        traj.coords,
        n_components=n_components,
        backend=backend,
        align=align,
    )
