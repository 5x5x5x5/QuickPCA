"""Abstract backend protocol for QuickPCA compute kernels."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import PCADecomposition


class Backend(ABC):
    """Pluggable compute backend.

    Implementations provide the two numerically heavy primitives used by the
    PCA pipeline: Kabsch frame alignment and full-SVD PCA.
    """

    name: str  # class attribute, e.g. "numpy"

    @abstractmethod
    def align_frames(self, frames: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Kabsch-align every frame onto ``ref``.

        Parameters
        ----------
        frames:
            Array of shape ``(F, N, 3)``.
        ref:
            Reference coordinates of shape ``(N, 3)``.

        Returns
        -------
        np.ndarray
            Aligned frames of shape ``(F, N, 3)`` as ``float64``.
        """

    @abstractmethod
    def pca(self, X: np.ndarray, n_components: int) -> PCADecomposition:
        """Run full-SVD PCA on raw (not pre-centered) data ``X`` of shape ``(F, D)``."""
