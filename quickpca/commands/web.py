"""``quickpca web`` subcommand: launch the Streamlit demo app.

Discovered automatically by :func:`quickpca.cli._discover_commands`. The
subcommand shells out to ``streamlit run <app.py>`` so the demo behaves exactly
as if launched manually.

Author: Gleb Novikov — The Visual Hub. MIT licensed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


def _app_path() -> Path:
    """Return the absolute path to the repo-root ``app.py`` Streamlit script."""
    # app.py lives at the repository root: <root>/quickpca/commands/web.py
    return Path(__file__).resolve().parents[2] / "app.py"


def _streamlit_available() -> bool:
    """Return True if the ``streamlit`` package is importable."""
    return find_spec("streamlit") is not None


def _run_web(args: argparse.Namespace) -> int:
    """Launch the Streamlit demo, optionally forwarding ``--server.port``."""
    if not _streamlit_available():
        print(
            "error: streamlit is not installed. Install the web extra with "
            "'pip install quickpca[web]' (or 'pip install streamlit').",
            file=sys.stderr,
        )
        return 1

    app = _app_path()
    if not app.is_file():
        print(f"error: demo app not found at {app}", file=sys.stderr)
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", str(app)]
    if args.port is not None:
        cmd += ["--server.port", str(args.port)]

    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:  # pragma: no cover - interactive Ctrl-C
        return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``web`` subcommand on the QuickPCA CLI."""
    p = subparsers.add_parser(
        "web",
        help="Launch the QuickPCA Streamlit web demo.",
        description="Launch the QuickPCA Streamlit web demo via 'streamlit run'.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for the Streamlit server (forwarded as --server.port).",
    )
    p.set_defaults(func=_run_web)
