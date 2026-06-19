"""Tests for the CLI entry point."""

from __future__ import annotations

import pytest

from quickpca import __version__
from quickpca.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_backends_command(capsys):
    rc = main(["backends"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "numpy" in out


def test_no_command_prints_help():
    rc = main([])
    assert rc == 1
