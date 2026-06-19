"""Tests for the PyMOL integration layer using a fake ``cmd`` object.

Real PyMOL is not pip-installable, so these tests drive
``quickpca.plugin`` with a hand-rolled fake ``cmd`` that mimics the small slice
of the PyMOL API the adapter relies on (``count_states``, ``get_model``,
``get_names``).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from quickpca.plugin import coords_from_pymol, run_from_pymol  # noqa: E402

N_STATES = 50
N_ATOMS = 24


class _FakeAtom:
    """Minimal stand-in for a PyMOL model atom."""

    def __init__(self, coord: list[float]) -> None:
        self.coord = coord


class _FakeModel:
    """Minimal stand-in for a PyMOL ``ChemPyModel`` (exposes ``.atom``)."""

    def __init__(self, coords: np.ndarray) -> None:
        self.atom = [_FakeAtom(list(map(float, row))) for row in coords]


class _FakeCmd:
    """Fake PyMOL ``cmd`` backed by a synthetic ``(F, N, 3)`` coordinate array."""

    def __init__(self, coords: np.ndarray, name: str = "x") -> None:
        self._coords = coords
        self._name = name

    def get_names(self, kind: str = "objects") -> list[str]:
        return [self._name]

    def count_states(self, obj_name: str) -> int:
        return self._coords.shape[0]

    def get_model(self, selection: str, state: int = 1) -> _FakeModel:
        # PyMOL states are 1-indexed.
        return _FakeModel(self._coords[state - 1])


@pytest.fixture
def fake_coords() -> np.ndarray:
    """Deterministic synthetic trajectory with recoverable low-rank signal."""
    rng = np.random.default_rng(7)
    base = np.zeros((N_ATOMS, 3))
    base[:, 0] = np.linspace(0.0, 30.0, N_ATOMS)

    t = np.linspace(0.0, 4.0 * np.pi, N_STATES)
    mode = np.zeros((N_ATOMS, 3))
    mode[:, 1] = np.linspace(-1.0, 1.0, N_ATOMS)
    mode2 = np.zeros((N_ATOMS, 3))
    mode2[:, 2] = np.sin(np.linspace(0.0, 2.0 * np.pi, N_ATOMS))

    coords = (
        base[None, :, :]
        + (5.0 * np.sin(t))[:, None, None] * mode[None, :, :]
        + (3.0 * np.sin(0.5 * t + 0.7))[:, None, None] * mode2[None, :, :]
    )
    coords = coords + rng.normal(scale=0.05, size=coords.shape)
    return coords.astype(np.float64)


@pytest.fixture
def fake_cmd(fake_coords: np.ndarray) -> _FakeCmd:
    return _FakeCmd(fake_coords)


def test_coords_from_pymol_shape(fake_cmd: _FakeCmd) -> None:
    coords = coords_from_pymol(fake_cmd, "x")
    assert coords.shape == (N_STATES, N_ATOMS, 3)
    assert coords.dtype == np.float64


def test_coords_from_pymol_matches_source(fake_cmd: _FakeCmd, fake_coords: np.ndarray) -> None:
    coords = coords_from_pymol(fake_cmd, "x")
    np.testing.assert_allclose(coords, fake_coords)


def test_coords_from_pymol_raises_when_no_states() -> None:
    empty = _FakeCmd(np.empty((0, N_ATOMS, 3)))
    with pytest.raises(ValueError):
        coords_from_pymol(empty, "x")


def test_coords_from_pymol_skips_empty_states(fake_coords: np.ndarray) -> None:
    """Empty states are skipped (as in the original script), not fatal."""

    class _GappyCmd(_FakeCmd):
        def get_model(self, selection: str, state: int = 1) -> _FakeModel:
            # State 2 has no matching atoms; it must be skipped, not crash.
            if state == 2:
                return _FakeModel(np.empty((0, 3)))
            return super().get_model(selection, state=state)

    gappy = _GappyCmd(fake_coords)
    coords = coords_from_pymol(gappy, "x")
    assert coords.shape == (N_STATES - 1, N_ATOMS, 3)


def test_coords_from_pymol_raises_when_all_states_empty() -> None:
    class _AllEmpty(_FakeCmd):
        def count_states(self, obj_name: str) -> int:
            return 5

        def get_model(self, selection: str, state: int = 1) -> _FakeModel:
            return _FakeModel(np.empty((0, 3)))

    allempty = _AllEmpty(np.empty((0, N_ATOMS, 3)))
    with pytest.raises(ValueError):
        coords_from_pymol(allempty, "x")


def test_run_from_pymol_raises_below_min_frames(tmp_path) -> None:
    """Fewer than three usable frames yields a clear error, not an IndexError."""
    coords = np.zeros((2, N_ATOMS, 3))
    coords[1, :, 0] = 1.0  # make the two frames distinct
    two_frame = _FakeCmd(coords)
    with pytest.raises(ValueError, match="at least 3 frames"):
        run_from_pymol(obj_name="x", cmd=two_frame, output=str(tmp_path / "x.png"))


def test_run_from_pymol_writes_png(fake_cmd: _FakeCmd, tmp_path) -> None:
    out = tmp_path / "report.png"
    result = run_from_pymol(obj_name="x", cmd=fake_cmd, output=str(out))
    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 5_000


def test_run_from_pymol_autoselects_first_object(fake_cmd: _FakeCmd, tmp_path) -> None:
    out = tmp_path / "auto.png"
    result = run_from_pymol(cmd=fake_cmd, output=str(out))
    assert result == str(out)
    assert out.stat().st_size > 5_000


def test_run_from_pymol_no_objects_raises(tmp_path) -> None:
    class _Empty(_FakeCmd):
        def get_names(self, kind: str = "objects") -> list[str]:
            return []

    empty = _Empty(np.empty((0, N_ATOMS, 3)))
    with pytest.raises(ValueError):
        run_from_pymol(cmd=empty, output=str(tmp_path / "x.png"))


def test_run_from_pymol_lazy_import_without_pymol(tmp_path) -> None:
    """With cmd=None and no PyMOL installed, the lazy import should fail loudly."""
    try:
        import pymol  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            run_from_pymol(obj_name="x", output=str(tmp_path / "x.png"))
    else:
        pytest.skip("PyMOL is installed; lazy-import failure path not exercised.")
