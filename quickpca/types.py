"""Core dataclasses for QuickPCA.

All numerical arrays use NumPy. These types form part of the frozen public
contract that downstream backends, loaders and reporters rely on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    """A trajectory of Cartesian coordinates plus per-atom metadata.

    Attributes
    ----------
    coords:
        Array of shape ``(n_frames, n_atoms, 3)`` and dtype ``float64``.
    atom_names, resids, resnames:
        Per-atom metadata arrays of length ``n_atoms``.
    n_frames, n_atoms:
        Convenience integer dimensions.
    """

    coords: np.ndarray
    atom_names: np.ndarray
    resids: np.ndarray
    resnames: np.ndarray
    n_frames: int
    n_atoms: int


@dataclass
class PCADecomposition:
    """Raw output of :meth:`Backend.pca`.

    Mirrors the relevant pieces of an sklearn ``PCA`` fit.
    """

    projections: np.ndarray            # (n_frames, n_components)
    components: np.ndarray             # (n_components, n_features)
    explained_variance: np.ndarray     # (n_components,)
    explained_variance_ratio: np.ndarray  # (n_components,)
    mean: np.ndarray                   # (n_features,)


@dataclass
class PCAResult:
    """High-level PCA result produced by :func:`quickpca.pca.compute_pca`."""

    projections: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    components: np.ndarray
    cross_correlation: np.ndarray      # (n_atoms, n_atoms)
    n_components: int
    n_atoms: int
    n_frames: int
    mean: np.ndarray
    backend: str


@dataclass
class FELResult:
    """Boltzmann-inverted 2-D Free-Energy Landscape over PC1/PC2."""

    F: np.ndarray
    xcenters: np.ndarray
    ycenters: np.ndarray
    xedges: np.ndarray
    yedges: np.ndarray
    pc1: np.ndarray
    pc2: np.ndarray
    kBT: float
    temperature: float
