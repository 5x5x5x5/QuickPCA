"""Tests for the bundled sample-dataset helpers."""

from __future__ import annotations

import os

import numpy as np
import pytest

from quickpca.datasets import sample_coords, sample_pdb_path


def test_sample_coords_default_shape():
    coords = sample_coords()
    assert coords.shape == (200, 24, 3)
    assert coords.dtype == np.float64


def test_sample_coords_custom_shape():
    coords = sample_coords(n_frames=37, n_atoms=11, seed=5)
    assert coords.shape == (37, 11, 3)


def test_sample_coords_is_deterministic():
    a = sample_coords()
    b = sample_coords()
    assert np.array_equal(a, b)
    # A different seed yields different coordinates.
    c = sample_coords(seed=1)
    assert not np.array_equal(a, c)


def test_sample_coords_has_low_rank_signal():
    # The embedded collective motion should dominate the variance, so the
    # spread along the first few axes far exceeds the residual noise floor.
    coords = sample_coords()
    flat = coords.reshape(coords.shape[0], -1)
    variances = np.sort(flat.var(axis=0))[::-1]
    assert variances[0] > 10.0 * variances[-1]


@pytest.mark.parametrize("bad_kwargs", [{"n_frames": 0}, {"n_atoms": 0}])
def test_sample_coords_rejects_empty(bad_kwargs):
    with pytest.raises(ValueError):
        sample_coords(**bad_kwargs)


def test_sample_pdb_path_exists():
    path = sample_pdb_path()
    assert os.path.isfile(path)
    assert path.endswith("sample.pdb")
    assert os.path.getsize(path) > 0


def test_load_sample_returns_trajectory():
    pytest.importorskip("MDAnalysis")
    from quickpca.datasets import load_sample
    from quickpca.types import Trajectory

    traj = load_sample()
    assert isinstance(traj, Trajectory)
    assert traj.n_frames >= 3
    assert traj.n_atoms > 0
    assert traj.coords.shape == (traj.n_frames, traj.n_atoms, 3)
    assert traj.coords.dtype.kind == "f"
