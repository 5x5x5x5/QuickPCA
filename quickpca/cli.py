"""argparse-based CLI with auto-discovered subcommands."""

from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
import sys

from . import __version__
from .backends import available_backends


def _cmd_run(args: argparse.Namespace) -> int:
    """Built-in ``run`` subcommand: full load → PCA → FEL → report pipeline."""
    from .fel import compute_fel
    from .io.loader import load_trajectory
    from .pca import compute_pca
    from .report import plot_report

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
            backend=args.backend,
            align=not args.no_align,
        )
        fel = compute_fel(
            pca.projections,
            temperature=args.temp,
            n_bins=args.nbins,
            sigma=args.sigma,
        )
        path = plot_report(
            pca, fel, output=args.output, temperature=args.temp
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved PCA report → {path}")
    return 0


def _cmd_backends(args: argparse.Namespace) -> int:
    """Built-in ``backends`` subcommand: list registered backends."""
    for name in available_backends():
        print(name)
    return 0


def _build_run_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("run", help="Run the full PCA/FEL report pipeline.")
    p.add_argument("topology", help="Topology/structure file (e.g. PDB).")
    p.add_argument("trajectory", nargs="?", default=None, help="Trajectory file.")
    p.add_argument("--selection", "-s", default="name CA", help="Atom selection.")
    p.add_argument("--ncomp", type=int, default=10, help="Number of PCs.")
    p.add_argument("--interval", "-i", type=int, default=1, help="Frame stride.")
    p.add_argument("--temp", type=float, default=300.0, help="Temperature (K).")
    p.add_argument("--nbins", type=int, default=50, help="FEL histogram bins.")
    p.add_argument("--sigma", type=float, default=1.0, help="FEL Gaussian sigma.")
    p.add_argument(
        "--backend", "-b", default="numpy",
        help=f"Compute backend (available: {', '.join(available_backends())}).",
    )
    p.add_argument(
        "--output", "-o", default="PCA_Report.png", help="Output PNG path."
    )
    p.add_argument(
        "--no-align", action="store_true", help="Skip Kabsch alignment."
    )
    p.set_defaults(func=_cmd_run)


def _build_backends_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("backends", help="List available compute backends.")
    p.set_defaults(func=_cmd_backends)


def _discover_commands(subparsers: argparse._SubParsersAction) -> None:
    """Import every module in ``quickpca.commands`` and call its ``register``."""
    from . import commands

    for _finder, name, _ispkg in pkgutil.iter_modules(
        commands.__path__, "quickpca.commands."
    ):
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not crash CLI
            print(f"warning: failed to import {name}: {exc}", file=sys.stderr)
            continue
        register = getattr(module, "register", None)
        if callable(register):
            try:
                register(subparsers)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"warning: failed to register {name}: {exc}", file=sys.stderr
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quickpca",
        description="QuickPCA — headless essential-dynamics PCA for MD trajectories.",
    )
    parser.add_argument(
        "--version", action="version", version=f"quickpca {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    _build_run_parser(subparsers)
    _build_backends_parser(subparsers)
    _discover_commands(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")

    parser = build_parser()
    args = parser.parse_args(argv)

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1
    return func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
