"""Tests for the convergence-diagnostics subcommand and library functions."""

from __future__ import annotations

import numpy as np
import pytest

from quickpca.commands.convergence import (
    block_boundaries,
    block_drift,
    cosine_content,
    register,
)


def test_cosine_content_pure_half_cosine():
    """A pure half-cosine signal has cosine content ~1 for i = 0."""
    n = 2000
    t = (np.arange(n) + 0.5) / n
    signal = np.cos(np.pi * t)
    assert cosine_content(signal, 0) == pytest.approx(1.0, abs=1e-3)


def test_cosine_content_higher_modes():
    """A pure i-th cosine mode has cosine content ~1 for that i and ~0 for others."""
    n = 2000
    t = (np.arange(n) + 0.5) / n
    signal = np.cos(2 * np.pi * t)  # second mode → i = 1
    assert cosine_content(signal, 1) == pytest.approx(1.0, abs=1e-3)
    assert cosine_content(signal, 0) < 0.05


def test_cosine_content_white_noise_is_low():
    """I.i.d. random noise has low cosine content (<0.3)."""
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(5000)
    assert cosine_content(noise, 0) < 0.3


def test_cosine_content_constant_is_zero():
    """A constant (zero-variance) projection returns 0.0 rather than NaN."""
    assert cosine_content(np.ones(100), 0) == 0.0


def test_cosine_content_short_input():
    """Degenerate inputs of length < 2 return 0.0."""
    assert cosine_content(np.array([1.0]), 0) == 0.0
    assert cosine_content(np.array([]), 0) == 0.0


def test_block_drift_shape_and_values():
    """block_drift returns one cumulative mean per block boundary."""
    p = np.arange(100, dtype=np.float64)
    drift = block_drift(p, n_blocks=10)
    assert drift.shape == (10,)
    # Final cumulative mean equals the mean of the whole series.
    assert drift[-1] == pytest.approx(p.mean())


def test_block_drift_clamps_blocks():
    """Requesting more blocks than frames clamps to the number of frames."""
    drift = block_drift(np.arange(4, dtype=np.float64), n_blocks=10)
    assert drift.shape == (4,)


def test_block_boundaries_increasing_unique():
    """block_boundaries returns sorted, unique, strictly-positive frame counts."""
    b = block_boundaries(100, 10)
    assert np.array_equal(b, np.arange(10, 101, 10))
    # Clamps and de-duplicates when more blocks than frames are requested.
    b_small = block_boundaries(4, 10)
    assert np.array_equal(b_small, np.array([1, 2, 3, 4]))
    assert b_small[-1] == 4


def test_block_boundaries_drift_alignment():
    """Variance boundaries and per-PC drift share the same number of points."""
    n_frames = 37  # not a multiple of n_blocks → exercises the alignment fix
    boundaries = block_boundaries(n_frames, 10)
    drift = block_drift(np.arange(n_frames, dtype=np.float64), 10)
    assert boundaries.size == drift.size


def test_convergence_cli_end_to_end(tmp_path, synthetic_traj, monkeypatch):
    """The convergence subcommand runs end-to-end and writes a non-trivial PNG."""
    import quickpca.io.loader as loader
    from quickpca.cli import main

    # Avoid depending on MDAnalysis/sample files: feed synthetic coordinates.
    # The command imports load_trajectory locally, so patch it at the source.
    monkeypatch.setattr(loader, "load_trajectory", lambda *a, **k: synthetic_traj)

    out = tmp_path / "conv.png"
    rc = main(
        [
            "convergence",
            "ignored.pdb",
            "--ncomp",
            "4",
            "-o",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 5 * 1024


def test_register_adds_parser():
    """register() wires up a 'convergence' parser with a func default."""
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    register(sub)
    args = parser.parse_args(["convergence", "top.pdb", "--ncomp", "3"])
    assert args.ncomp == 3
    assert args.output == "convergence.png"
    assert callable(args.func)
