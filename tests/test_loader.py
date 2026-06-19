"""Tests for the MDAnalysis-based trajectory loader (data-dependent)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("MDAnalysis")

from quickpca.io.loader import load_trajectory  # noqa: E402
from quickpca.types import Trajectory  # noqa: E402

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOP = os.path.join(_HERE, "data", "reference.pdb")
_TRAJ = os.path.join(_HERE, "data", "trajectory.nc")


@pytest.mark.skipif(
    not (os.path.exists(_TOP) and os.path.exists(_TRAJ)),
    reason="sample data files not available",
)
def test_load_trajectory():
    traj = load_trajectory(_TOP, _TRAJ, interval=10)
    assert isinstance(traj, Trajectory)
    assert traj.n_frames >= 3
    assert traj.n_atoms > 0
    assert traj.coords.shape == (traj.n_frames, traj.n_atoms, 3)
    assert traj.coords.dtype.kind == "f"
    assert len(traj.atom_names) == traj.n_atoms
    assert len(traj.resids) == traj.n_atoms
    assert len(traj.resnames) == traj.n_atoms
