"""Tests for the ``cluster`` subcommand and its ``cluster_pcs`` library fn."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from quickpca.commands.cluster import (  # noqa: E402
    _cmd_cluster,
    cluster_pcs,
    plot_clusters,
    register,
)


def _three_blobs(seed: int = 0) -> np.ndarray:
    """Three well-separated 2-D Gaussian blobs as fake PC projections."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-10.0, -10.0], [0.0, 10.0], [10.0, -10.0]])
    per = 80
    blobs = [c + rng.normal(scale=0.5, size=(per, 2)) for c in centers]
    # Pad with extra (noise) PC columns to mimic a real projection array.
    proj2d = np.vstack(blobs)
    extra = rng.normal(scale=0.1, size=(proj2d.shape[0], 3))
    return np.hstack([proj2d, extra])


def test_cluster_pcs_recovers_three_blobs():
    proj = _three_blobs()
    n_frames = proj.shape[0]

    labels, centers = cluster_pcs(proj, n_clusters=3)

    # Labels cover every frame.
    assert labels.shape == (n_frames,)
    assert len(labels) == n_frames

    # Exactly three clusters, all populated.
    unique = np.unique(labels)
    assert len(unique) == 3
    counts = np.bincount(labels, minlength=3)
    assert np.all(counts > 0)
    # Balanced blobs → each ~one third of the frames.
    assert np.all(counts >= n_frames // 4)

    # Centroids returned in the clustered subspace (n_clusters, n_dims).
    assert centers.shape == (3, 2)


def test_cluster_pcs_default_n_dims_uses_two_pcs():
    proj = _three_blobs()
    _labels, centers = cluster_pcs(proj, n_clusters=3)
    assert centers.shape[1] == 2


def test_cluster_pcs_is_deterministic():
    proj = _three_blobs()
    labels_a, _ = cluster_pcs(proj, n_clusters=3, random_state=0)
    labels_b, _ = cluster_pcs(proj, n_clusters=3, random_state=0)
    np.testing.assert_array_equal(labels_a, labels_b)


def test_cluster_pcs_rejects_bad_input():
    with pytest.raises(ValueError):
        cluster_pcs(np.zeros((5,)), n_clusters=2)  # not 2-D
    with pytest.raises(ValueError):
        cluster_pcs(np.zeros((3, 2)), n_clusters=10)  # k > n_frames


def test_plot_clusters_writes_png(tmp_path):
    proj = _three_blobs()
    labels, centers = cluster_pcs(proj, n_clusters=3)
    out = tmp_path / "clusters.png"
    path = plot_clusters(proj, labels, centers, output=str(out))
    assert os.path.exists(path)
    assert os.path.getsize(path) > 5_000


def test_cmd_cluster_with_synthetic_traj(monkeypatch, tmp_path, synthetic_traj):
    """Drive the full CLI func with a patched loader (no MDAnalysis needed)."""
    import quickpca.commands.cluster as cluster_mod

    monkeypatch.setattr(
        "quickpca.io.loader.load_trajectory",
        lambda *a, **k: synthetic_traj,
    )

    out = tmp_path / "out.png"
    args = type(
        "Args",
        (),
        {
            "topology": "ignored.pdb",
            "trajectory": None,
            "selection": "name CA",
            "interval": 1,
            "ncomp": 10,
            "clusters": 3,
            "output": str(out),
        },
    )()

    rc = cluster_mod._cmd_cluster(args)
    assert rc == 0
    assert os.path.exists(out)
    assert os.path.getsize(out) > 5_000


def test_register_adds_cluster_subcommand():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register(subparsers)

    args = parser.parse_args(["cluster", "top.pdb", "traj.nc", "-k", "5"])
    assert args.command == "cluster"
    assert args.topology == "top.pdb"
    assert args.trajectory == "traj.nc"
    assert args.clusters == 5
    assert args.func is _cmd_cluster
