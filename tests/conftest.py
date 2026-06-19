"""Shared pytest fixtures for QuickPCA tests."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from quickpca.types import Trajectory  # noqa: E402


@pytest.fixture
def synthetic_coords() -> np.ndarray:
    """Deterministic (n_frames, n_atoms, 3) coordinates with embedded signal.

    A handful of low-dimensional collective sinusoidal motions are superimposed
    on a static base structure plus small Gaussian noise, so PCA recovers real
    low-rank signal.
    """
    rng = np.random.default_rng(1234)
    n_frames, n_atoms = 200, 24

    # Static base structure (roughly a stretched coil).
    base = np.zeros((n_atoms, 3))
    base[:, 0] = np.linspace(0.0, 38.0, n_atoms)
    base[:, 1] = 3.0 * np.sin(np.linspace(0.0, 6.0, n_atoms))
    base[:, 2] = 3.0 * np.cos(np.linspace(0.0, 6.0, n_atoms))

    t = np.linspace(0.0, 4.0 * np.pi, n_frames)

    # A few collective modes (each a fixed spatial pattern modulated in time).
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


@pytest.fixture
def synthetic_traj(synthetic_coords: np.ndarray) -> Trajectory:
    """A Trajectory wrapping ``synthetic_coords`` with fake metadata arrays."""
    n_frames, n_atoms, _ = synthetic_coords.shape
    return Trajectory(
        coords=synthetic_coords,
        atom_names=np.array(["CA"] * n_atoms),
        resids=np.arange(1, n_atoms + 1),
        resnames=np.array(["ALA"] * n_atoms),
        n_frames=n_frames,
        n_atoms=n_atoms,
    )
