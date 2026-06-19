"""MDAnalysis-based headless trajectory loader."""

from __future__ import annotations

import numpy as np

from ..types import Trajectory


def load_trajectory(
    topology: str,
    trajectory: str | None = None,
    selection: str = "name CA",
    interval: int = 1,
) -> Trajectory:
    """Load a trajectory into a :class:`Trajectory` using MDAnalysis.

    Parameters
    ----------
    topology:
        Path to the topology/structure file (e.g. PDB).
    trajectory:
        Optional path to a trajectory file (e.g. NetCDF/XTC/DCD). If omitted,
        only the topology is used (single frame / its own coordinates).
    selection:
        MDAnalysis atom-selection string.
    interval:
        Stride for frame sampling.

    Returns
    -------
    Trajectory
    """
    try:
        import MDAnalysis as mda
    except ImportError as exc:  # pragma: no cover - exercised only without MDA
        raise ImportError(
            "MDAnalysis is required for headless trajectory loading: "
            "pip install quickpca[all] or pip install MDAnalysis"
        ) from exc

    if trajectory is not None:
        u = mda.Universe(topology, trajectory)
    else:
        u = mda.Universe(topology)

    ag = u.select_atoms(selection)
    if len(ag) == 0:
        raise ValueError(
            f"Selection {selection!r} matched 0 atoms in {topology!r}."
        )

    frames = []
    for _ in u.trajectory[::interval]:
        frames.append(ag.positions.astype(np.float64))

    if len(frames) < 3:
        raise ValueError(
            f"Need at least 3 frames for PCA, found {len(frames)} "
            f"(interval={interval})."
        )

    coords = np.asarray(frames, dtype=np.float64)
    n_frames, n_atoms, _ = coords.shape

    return Trajectory(
        coords=coords,
        atom_names=np.asarray(ag.names),
        resids=np.asarray(ag.resids),
        resnames=np.asarray(ag.resnames),
        n_frames=n_frames,
        n_atoms=n_atoms,
    )
