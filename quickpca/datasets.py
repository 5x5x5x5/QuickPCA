"""Tiny, deterministic sample datasets for docs, tests and demos.

These helpers let examples and the test-suite exercise QuickPCA without the
bundled multi-megabyte production trajectory. They provide:

* :func:`sample_coords` — a purely in-memory synthetic trajectory with a known
  low-dimensional collective motion, ideal for showcasing PCA.
* :func:`sample_pdb_path` — the filesystem path to a small multi-model PDB that
  ships inside the wheel.
* :func:`load_sample` — that PDB parsed into a :class:`~quickpca.types.Trajectory`.

Author: Gleb Novikov — The Visual Hub. MIT licensed.
"""

from __future__ import annotations

from importlib import resources

import numpy as np

from .io.loader import load_trajectory
from .types import Trajectory

__all__ = ["sample_coords", "sample_pdb_path", "load_sample"]

#: Name of the bundled sample structure inside ``quickpca/data``.
_SAMPLE_PDB = "sample.pdb"


def sample_coords(
    n_frames: int = 200,
    n_atoms: int = 24,
    seed: int = 0,
) -> np.ndarray:
    """Return a deterministic synthetic trajectory suitable for PCA demos.

    A static base structure (a gently coiled chain) is modulated in time by a
    few collective sinusoidal modes and sprinkled with small Gaussian noise.
    Because the motion lives in a low-dimensional subspace, PCA recovers a
    handful of dominant components with sharply decaying explained variance.

    The result is fully determined by ``(n_frames, n_atoms, seed)`` and involves
    no file I/O, so it is safe to call from tests and documentation builds.

    Parameters
    ----------
    n_frames:
        Number of frames (must be ``>= 1``).
    n_atoms:
        Number of pseudo-atoms per frame (must be ``>= 1``).
    seed:
        Seed for the noise generator, fixing the output exactly.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames, n_atoms, 3)`` and dtype ``float64``.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}.")
    if n_atoms < 1:
        raise ValueError(f"n_atoms must be >= 1, got {n_atoms}.")

    rng = np.random.default_rng(seed)

    # Static base structure (roughly a stretched, gently twisting coil).
    base = np.zeros((n_atoms, 3))
    base[:, 0] = np.linspace(0.0, 38.0, n_atoms)
    base[:, 1] = 3.0 * np.sin(np.linspace(0.0, 6.0, n_atoms))
    base[:, 2] = 3.0 * np.cos(np.linspace(0.0, 6.0, n_atoms))

    t = np.linspace(0.0, 4.0 * np.pi, n_frames)

    # A few collective modes: each a fixed spatial pattern modulated in time.
    mode1 = np.zeros((n_atoms, 3))
    mode1[:, 1] = np.linspace(-1.0, 1.0, n_atoms)
    mode2 = np.zeros((n_atoms, 3))
    mode2[:, 2] = np.sin(np.linspace(0.0, 2.0 * np.pi, n_atoms))
    mode3 = np.zeros((n_atoms, 3))
    mode3[:, 0] = np.cos(np.linspace(0.0, np.pi, n_atoms))

    amp1 = 5.0 * np.sin(t)
    amp2 = 3.0 * np.sin(0.5 * t + 0.7)
    amp3 = 1.5 * np.sin(1.3 * t + 1.9)

    coords = (
        base[None, :, :]
        + amp1[:, None, None] * mode1[None, :, :]
        + amp2[:, None, None] * mode2[None, :, :]
        + amp3[:, None, None] * mode3[None, :, :]
    )
    coords = coords + rng.normal(scale=0.05, size=coords.shape)
    return coords.astype(np.float64)


def sample_pdb_path() -> str:
    """Return the filesystem path to the bundled multi-model sample PDB.

    The file ships inside the installed package (``quickpca/data/sample.pdb``)
    and is located via :mod:`importlib.resources`, so it works both from a
    source checkout and from an installed wheel.

    Returns
    -------
    str
        Absolute path to the bundled ``sample.pdb``.
    """
    resource = resources.files("quickpca.data").joinpath(_SAMPLE_PDB)
    # The bundled data is a real on-disk file, so this is effectively a no-op
    # context manager that simply yields its path.
    with resources.as_file(resource) as path:
        return str(path)


def load_sample(selection: str = "name CA", interval: int = 1) -> Trajectory:
    """Load the bundled sample PDB into a :class:`~quickpca.types.Trajectory`.

    This is a thin convenience wrapper around
    :func:`quickpca.load_trajectory` pointed at :func:`sample_pdb_path`. It
    requires MDAnalysis (a core QuickPCA dependency).

    Parameters
    ----------
    selection:
        MDAnalysis atom-selection string.
    interval:
        Stride for frame sampling.

    Returns
    -------
    Trajectory
    """
    return load_trajectory(sample_pdb_path(), selection=selection, interval=interval)
