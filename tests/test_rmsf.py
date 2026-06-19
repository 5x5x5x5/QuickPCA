"""Tests for the per-residue RMSF subcommand."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from quickpca.commands import rmsf as rmsf_cmd
from quickpca.commands.rmsf import compute_rmsf, register


def test_rmsf_shape_and_nonnegative(synthetic_coords: np.ndarray) -> None:
    """RMSF has one value per atom and is everywhere non-negative."""
    n_atoms = synthetic_coords.shape[1]
    rmsf = compute_rmsf(synthetic_coords)

    assert rmsf.shape == (n_atoms,)
    assert rmsf.dtype == np.float64
    assert np.all(rmsf >= 0.0)
    assert np.all(np.isfinite(rmsf))


def test_rmsf_static_structure_is_zero() -> None:
    """A structure with no motion has (essentially) zero RMSF everywhere."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(10, 3))
    # 50 identical frames → no fluctuation.
    coords = np.repeat(base[None, :, :], 50, axis=0)

    rmsf = compute_rmsf(coords, align=True)

    assert rmsf.shape == (10,)
    assert np.allclose(rmsf, 0.0, atol=1e-8)


def test_rmsf_static_structure_zero_without_align() -> None:
    """Static structure is also ~0 RMSF when alignment is disabled."""
    base = np.linspace(0.0, 9.0, 30).reshape(10, 3)
    coords = np.repeat(base[None, :, :], 20, axis=0)

    rmsf = compute_rmsf(coords, align=False)

    assert np.allclose(rmsf, 0.0, atol=1e-12)


def test_rmsf_detects_single_mobile_atom() -> None:
    """An isolated wobbling atom registers larger RMSF than its still peers."""
    n_frames, n_atoms = 40, 6
    coords = np.zeros((n_frames, n_atoms, 3))
    # Spread atoms out so alignment has a stable, non-degenerate reference.
    coords[:, :, 0] = np.linspace(0.0, 50.0, n_atoms)[None, :]
    # Make atom 3 oscillate along z.
    coords[:, 3, 2] += 4.0 * np.sin(np.linspace(0.0, 6.0, n_frames))

    rmsf = compute_rmsf(coords, align=True)

    assert rmsf[3] == pytest.approx(np.max(rmsf))
    assert rmsf[3] > rmsf[0]


def test_rmsf_rejects_bad_shape() -> None:
    """Non ``(F, N, 3)`` input is rejected with a clear error."""
    with pytest.raises(ValueError):
        compute_rmsf(np.zeros((5, 4)))


def test_register_adds_rmsf_subcommand() -> None:
    """``register`` wires up a parser whose ``func`` returns an int exit code."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    register(subparsers)

    args = parser.parse_args(["rmsf", "top.pdb", "traj.nc", "-i", "5"])
    assert args.topology == "top.pdb"
    assert args.trajectory == "traj.nc"
    assert args.interval == 5
    assert args.output == "RMSF.png"
    assert callable(args.func)


def test_cmd_rmsf_writes_png(tmp_path, monkeypatch, synthetic_traj) -> None:
    """End-to-end command path produces a PNG and returns exit code 0."""
    # The command imports load_trajectory lazily from the loader module, so
    # patch it at the source.
    import quickpca.io.loader as loader

    monkeypatch.setattr(loader, "load_trajectory", lambda *a, **k: synthetic_traj)

    out = tmp_path / "rmsf.png"
    args = argparse.Namespace(
        topology="ignored.pdb",
        trajectory=None,
        selection="name CA",
        interval=1,
        output=str(out),
        no_align=False,
    )
    rc = rmsf_cmd._cmd_rmsf(args)

    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 5_000


def test_cmd_rmsf_handles_loader_error(monkeypatch, capsys) -> None:
    """A loader failure is reported cleanly and returns exit code 1."""
    import quickpca.io.loader as loader

    def _boom(*_a, **_k):
        raise ValueError("no such file")

    monkeypatch.setattr(loader, "load_trajectory", _boom)

    args = argparse.Namespace(
        topology="missing.pdb",
        trajectory=None,
        selection="name CA",
        interval=1,
        output="unused.png",
        no_align=False,
    )
    rc = rmsf_cmd._cmd_rmsf(args)

    assert rc == 1
    assert "error:" in capsys.readouterr().err
