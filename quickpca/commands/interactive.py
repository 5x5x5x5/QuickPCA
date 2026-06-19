"""``quickpca interactive`` — build a shareable interactive HTML report.

Auto-discovered CLI subcommand: loads a trajectory, runs PCA and the FEL, then
writes a standalone Plotly HTML report.

Author: Gleb Novikov — The Visual Hub. MIT licensed.
"""

from __future__ import annotations

import argparse
import sys


def _cmd_interactive(args: argparse.Namespace) -> int:
    """Load → PCA → FEL → interactive HTML report."""
    from ..fel import compute_fel
    from ..io.loader import load_trajectory
    from ..pca import compute_pca
    from ..report_html import build_html_report

    try:
        traj = load_trajectory(
            args.topology,
            args.trajectory,
            selection=args.selection,
            interval=args.interval,
        )
        pca = compute_pca(
            traj.coords,
            n_components=args.ncomp,
        )
        fel = compute_fel(
            pca.projections,
            temperature=args.temp,
            n_bins=args.nbins,
            sigma=args.sigma,
        )
        path = build_html_report(pca, fel, output=args.output)
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved interactive PCA report → {path}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the ``interactive`` subcommand to the CLI parser."""
    p = subparsers.add_parser(
        "interactive",
        help="Build a shareable interactive HTML (Plotly) PCA report.",
    )
    p.add_argument("topology", help="Topology/structure file (e.g. PDB).")
    p.add_argument("trajectory", nargs="?", default=None, help="Trajectory file.")
    p.add_argument("--selection", "-s", default="name CA", help="Atom selection.")
    p.add_argument("--interval", "-i", type=int, default=1, help="Frame stride.")
    p.add_argument("--ncomp", type=int, default=10, help="Number of PCs.")
    p.add_argument("--temp", type=float, default=300.0, help="Temperature (K).")
    p.add_argument("--nbins", type=int, default=50, help="FEL histogram bins.")
    p.add_argument("--sigma", type=float, default=1.0, help="FEL Gaussian sigma.")
    p.add_argument("--output", "-o", default="PCA_Report.html", help="Output HTML path.")
    p.set_defaults(func=_cmd_interactive)
