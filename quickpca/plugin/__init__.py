"""PyMOL integration for QuickPCA.

A thin adapter that extracts coordinates from PyMOL and delegates all analysis
to the headless ``quickpca`` package.

Author: Gleb Novikov — The Visual Hub. MIT licensed.
"""

from __future__ import annotations

from .pymol_plugin import coords_from_pymol, run_from_pymol

__all__ = ["coords_from_pymol", "run_from_pymol"]
