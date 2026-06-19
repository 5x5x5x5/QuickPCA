"""Tests for the Streamlit web demo and the ``quickpca web`` subcommand.

These tests avoid launching a Streamlit server. They exercise the pure helper
functions in ``app.py`` and verify the CLI wiring of ``quickpca.commands.web``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import pytest

import quickpca.commands.web as web

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "app.py"
_APP_MODULE_NAME = "quickpca_demo_app"


@pytest.fixture
def app_module():
    """Import ``app.py`` (repo root) as a module without running Streamlit.

    Registers the module in ``sys.modules`` before exec (so dataclasses can
    resolve it by ``__module__``) and removes it afterwards to avoid leaking
    state between tests.
    """
    spec = importlib.util.spec_from_file_location(_APP_MODULE_NAME, APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_APP_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(_APP_MODULE_NAME, None)


# ── CLI subcommand wiring ────────────────────────────────────────────────────


def test_register_is_callable():
    assert callable(web.register)


def test_register_wires_web_subparser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    web.register(subparsers)

    args = parser.parse_args(["web"])
    assert args.command == "web"
    assert callable(args.func)
    # --port is forwarded; default is None.
    assert args.port is None
    args = parser.parse_args(["web", "--port", "1234"])
    assert args.port == 1234


def test_web_discovered_by_cli():
    """The full CLI parser should discover and expose the ``web`` subcommand."""
    from quickpca.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["web", "--port", "9000"])
    assert args.func is web._run_web
    assert args.port == 9000


def test_app_path_points_to_repo_root_app():
    assert web._app_path() == APP_PATH
    assert APP_PATH.is_file()


def test_web_errors_without_streamlit(monkeypatch, capsys):
    """When streamlit is missing, the func returns 1 with a friendly message."""
    monkeypatch.setattr(web, "_streamlit_available", lambda: False)
    rc = web._run_web(argparse.Namespace(port=None))
    assert rc == 1
    err = capsys.readouterr().err
    assert "streamlit" in err.lower()


def test_web_invokes_streamlit_run(monkeypatch):
    """With streamlit present, it shells out to ``streamlit run app.py``."""
    monkeypatch.setattr(web, "_streamlit_available", lambda: True)
    captured: dict[str, list[str]] = {}

    def fake_call(cmd):
        captured["cmd"] = cmd
        return 0

    monkeypatch.setattr(web.subprocess, "call", fake_call)
    rc = web._run_web(argparse.Namespace(port=8502))
    assert rc == 0
    cmd = captured["cmd"]
    assert "streamlit" in cmd
    assert "run" in cmd
    assert str(web._app_path()) in cmd
    assert "--server.port" in cmd
    assert "8502" in cmd


# ── app.py validity and pure helpers ─────────────────────────────────────────


def test_app_py_is_valid_syntax():
    import ast

    ast.parse(APP_PATH.read_text())


def test_explained_variance_table(app_module):
    app = app_module
    import numpy as np

    rng = np.random.default_rng(0)
    coords = rng.normal(size=(40, 10, 3))
    pca = app.compute_pca(coords, n_components=5)
    table = app.explained_variance_table(pca, n=5)
    assert len(table) == 5
    assert table[0]["PC"] == 1

    evs = [row["Explained variance (%)"] for row in table]
    cums = [row["Cumulative (%)"] for row in table]
    # Cumulative must equal the running sum of the per-PC values (rounding
    # tolerated), and the first row's cumulative must equal its own EV.
    assert cums[0] == pytest.approx(evs[0], abs=0.05)
    for i in range(1, len(cums)):
        assert cums[i] == pytest.approx(cums[i - 1] + evs[i], abs=0.05)
        # And the values must track the underlying PCAResult arrays.
        assert evs[i] == pytest.approx(float(pca.explained_variance_ratio[i]) * 100, abs=0.01)


def test_run_pipeline_on_coords_writes_png(tmp_path, synthetic_coords, app_module):
    """The pure pipeline helper renders a non-trivial PNG on synthetic input."""
    app = app_module
    out = tmp_path / "report.png"
    result = app.run_pipeline_on_coords(synthetic_coords, output_png=str(out), n_components=8)
    assert result.png_path == str(out)
    assert out.is_file()
    assert os.path.getsize(out) > 5 * 1024
    assert result.pca.n_frames == synthetic_coords.shape[0]


def test_run_pipeline_from_files(tmp_path, app_module):
    """End-to-end file pipeline using the bundled sample data, if present."""
    pytest.importorskip("MDAnalysis")
    app = app_module
    if not app.sample_data_available():
        pytest.skip("bundled sample data not available")
    out = tmp_path / "report.png"
    result = app.run_pipeline(
        str(app.SAMPLE_TOPOLOGY),
        str(app.SAMPLE_TRAJECTORY),
        output_png=str(out),
        n_components=6,
    )
    assert out.is_file()
    assert os.path.getsize(out) > 5 * 1024
    assert result.pca.n_atoms > 0


def test_streamlit_main_importable_if_installed(app_module):
    """If streamlit is installed, ``main`` exists and is callable (not run)."""
    pytest.importorskip("streamlit")
    assert callable(app_module.main)
