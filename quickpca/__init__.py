"""QuickPCA — headless essential-dynamics PCA for MD trajectories.

Author: Gleb Novikov — The Visual Hub. MIT licensed.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .backends import available_backends, get_backend
from .config import ReportConfig
from .fel import compute_fel
from .io.loader import load_trajectory
from .pca import compute_pca, compute_pca_from_files
from .report import plot_report
from .types import FELResult, PCADecomposition, PCAResult, Trajectory

__all__ = [
    "__version__",
    "compute_pca",
    "compute_pca_from_files",
    "compute_fel",
    "plot_report",
    "load_trajectory",
    "get_backend",
    "available_backends",
    "PCAResult",
    "FELResult",
    "PCADecomposition",
    "Trajectory",
    "ReportConfig",
]
