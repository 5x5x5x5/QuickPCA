"""NumPy reference backend.

Implements Kabsch alignment and full-SVD PCA producing results that match the
original sklearn ``PCA(svd_solver="full")`` pipeline.
"""

from __future__ import annotations

import numpy as np

from ..types import PCADecomposition
from . import register_backend
from .base import Backend


@register_backend
class NumpyBackend(Backend):
    """Pure-NumPy compute backend."""

    name = "numpy"

    def align_frames(self, frames: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Kabsch-align each frame onto ``ref``.

        Each frame is centred on its own centroid, rotated onto ``ref`` (which
        is centred on the reference centroid), then translated back to the
        reference centroid. Reflections are corrected via the sign of the
        determinant.
        """
        frames = np.asarray(frames, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)

        ref_com = ref.mean(axis=0)
        ref_centered = ref - ref_com

        aligned = np.empty_like(frames)
        for i in range(frames.shape[0]):
            coords = frames[i]
            coords_centered = coords - coords.mean(axis=0)
            H = coords_centered.T @ ref_centered
            U, _, Vt = np.linalg.svd(H)
            d = np.sign(np.linalg.det(Vt.T @ U.T))
            R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
            aligned[i] = coords_centered @ R.T + ref_com
        return aligned

    def pca(self, X: np.ndarray, n_components: int) -> PCADecomposition:
        """Full-SVD PCA equivalent to sklearn ``PCA(svd_solver="full")``."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        mean = X.mean(axis=0)
        X_centered = X - mean

        # Full SVD: X_centered = U @ diag(S) @ Vt
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Sign convention matching sklearn (svd_flip on u/v columns).
        max_abs_cols = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[max_abs_cols, range(U.shape[1])])
        U *= signs
        Vt *= signs[:, np.newaxis]

        components = Vt[:n_components]
        explained_variance = (S**2) / (n_samples - 1)
        total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var

        projections = X_centered @ components.T

        return PCADecomposition(
            projections=projections[:, :n_components].astype(np.float64),
            components=components.astype(np.float64),
            explained_variance=explained_variance[:n_components].astype(np.float64),
            explained_variance_ratio=explained_variance_ratio[:n_components].astype(np.float64),
            mean=mean.astype(np.float64),
        )
