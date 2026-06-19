# quick_pca.py (ver 1.00)
# Essential Dynamics Analysis for MD trajecotories in PyMOL.
# Author: Gleb Novikov
# © The Visual Hub 2026
# For educational use only.
# If you use QuickPCA in your research, please cite this tool.
# Any contributions toward its development are also appreciated.
#
# INSTRUCTIONS:
# 1. Place this script in the same folder as your topology (pdb) and trajectory files.
# 2. Drag-and-drop the script into the PyMOL window — it runs automatically.
#    Supported trajectory formats: .nc  .xtc  .trr  .dcd
#
#
# Output: PCA_Report.png  (Free-Energy Landscape + explained-variance chart
#                          + cross-correlation matrix + PC1/PC2 projections)
#
# This script is now a thin drag-and-drop shim: all PCA / FEL / plotting math
# lives in the pip-installable ``quickpca`` package. Install it with:
#   pip install quickpca
"""QuickPCA drag-and-drop shim for PyMOL.

The heavy lifting (PCA, Free-Energy Landscape, plotting) lives in the
``quickpca`` package; this file only wires PyMOL's drag-and-drop entry point to
:func:`quickpca.plugin.run_from_pymol`. It can be imported headlessly because
``pymol`` is only touched inside :func:`main`.
"""

from __future__ import annotations

import glob
import os

# =============================================================================
# ⚙️  USER SETTINGS  — edit these before running
# =============================================================================

# SVD performed directly on (n_frames × 3N) Cα coordinate matrix
PCA_SEL = "polymer and name CA"

# Number of principal components to compute (≥2 required)
PCA_NCOMP = 10

# Free-Energy Landscape histogram resolution and smoothing
PCA_NBINS = 50      # bins per axis
PCA_SIGMA = 1.0     # Gaussian σ in bin units

# Temperature (Kelvin) for Boltzmann inversion
PCA_TEMP = 300.0

# Input trajectory options
MD_INTERVAL = 5  # takes every 5 snapshots from initial data

# Compute backend ("numpy" or "jax" if installed)
PCA_BACKEND = "numpy"

# Output filename
OUTPUT_PNG = "PCA_Report.png"

# Trajectory formats searched for, in priority order.
TRAJ_GLOBS = ("*.nc", "*.xtc", "*.trr", "*.dcd")


def _find_trajectory() -> str | None:
    """Return the first trajectory file in the current directory, if any."""
    return next(
        (f for pattern in TRAJ_GLOBS for f in sorted(glob.glob(pattern))),
        None,
    )


def _open_report(path: str) -> None:
    """Open the saved report with the OS default viewer (best-effort).

    Mirrors the original drag-drop script, which ran ``open`` on macOS. Falls
    back to the platform-appropriate opener and silently ignores failures so a
    headless or unsupported environment never crashes the run.
    """
    import subprocess
    import sys

    try:
        if sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        elif sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]  # noqa: SIM115
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception:  # noqa: BLE001 - opening the viewer is best-effort
        pass


# =============================================================================
# 🚀  MAIN PIPELINE  — runs automatically on drag-and-drop in PyMOL
# =============================================================================

def main() -> None:
    """Auto-detect a trajectory, load it, and run the QuickPCA report.

    PyMOL is imported here (not at module import time) so this file stays
    importable in headless environments.
    """
    import time

    from pymol import cmd

    from quickpca.plugin import run_from_pymol

    start_all = time.time()

    all_objects = cmd.get_names("objects")
    if not all_objects:
        print("❌  No objects loaded in PyMOL. Load topology + trajectory first.")
        return

    target = all_objects[0]
    print(f"✨  Target object: {target}")

    traj = _find_trajectory()
    if traj:
        print(f"💫  Loading trajectory: {traj}")
        cmd.load_traj(traj, target, interval=MD_INTERVAL)
    else:
        print("ℹ️   No trajectory file found — using states already in PyMOL.")

    output = run_from_pymol(
        target,
        cmd=cmd,
        selection=PCA_SEL,
        n_components=PCA_NCOMP,
        temperature=PCA_TEMP,
        n_bins=PCA_NBINS,
        sigma=PCA_SIGMA,
        backend=PCA_BACKEND,
        output=OUTPUT_PNG,
    )
    print(f"👑  PCA report saved → {output}")

    # Pop the finished report open, as the original drag-drop script did.
    _open_report(output)

    total_elapsed = time.time() - start_all
    hours, rem = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print()
    print(f"🕰️ Total Execution Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")


# run the workflow in PyMOL on drag-and-drop
if __name__ == "__main__" or __name__ == "pymol":
    main()
