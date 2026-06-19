"""Per-residue (per-atom) RMSF subcommand.

Computes the root-mean-square fluctuation (RMSF) of each selected atom about
its time-averaged position, optionally Kabsch-aligning every frame onto a
reference frame first (via the NumPy backend), and renders a headless
matplotlib plot of RMSF versus residue/atom index.

Author: Gleb Novikov — The Visual Hub. MIT licensed.
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import for headless safety

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..backends import get_backend  # noqa: E402


def compute_rmsf(
    coords: np.ndarray,
    align: bool = True,
    ref_index: int = 0,
) -> np.ndarray:
    """Compute per-atom RMSF from a coordinate trajectory.

    The RMSF of atom ``i`` is the root of the mean (over frames) squared
    displacement of its position about its own time-averaged position::

        RMSF_i = sqrt( mean_t || x_i(t) - <x_i> ||^2 )

    Parameters
    ----------
    coords:
        Array of shape ``(F, N, 3)`` of Cartesian coordinates.
    align:
        If True, Kabsch-align every frame onto ``coords[ref_index]`` using the
        NumPy backend before computing fluctuations. This removes rigid-body
        translation/rotation so the RMSF reflects internal motion only.
    ref_index:
        Index of the reference frame used for alignment.

    Returns
    -------
    np.ndarray
        Non-negative RMSF values of shape ``(N,)`` and dtype ``float64``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError(
            f"coords must have shape (F, N, 3); got {coords.shape!r}."
        )

    if align:
        backend = get_backend("numpy")
        ref = coords[ref_index]
        frames = backend.align_frames(coords, ref)
    else:
        frames = coords

    mean_pos = frames.mean(axis=0)                       # (N, 3)
    disp = frames - mean_pos[None, :, :]                 # (F, N, 3)
    msf = np.mean(np.sum(disp**2, axis=2), axis=0)       # (N,)
    return np.sqrt(msf).astype(np.float64)


def _plot_rmsf(
    rmsf: np.ndarray,
    resids: np.ndarray | None,
    output: str,
) -> str:
    """Render an RMSF-vs-index line plot and save it as a PNG.

    Parameters
    ----------
    rmsf:
        Per-atom RMSF values of shape ``(N,)``.
    resids:
        Optional per-atom residue ids used for the x-axis; falls back to a
        1-based atom index when ``None`` or mismatched in length.
    output:
        Output PNG path.

    Returns
    -------
    str
        The ``output`` path.
    """
    if resids is not None and len(resids) == len(rmsf):
        x = np.asarray(resids)
        xlabel = "Residue index"
    else:
        x = np.arange(1, len(rmsf) + 1)
        xlabel = "Atom index"

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(x, rmsf, color="steelblue", lw=1.6, marker="o", ms=3.5)
    ax.fill_between(x, rmsf, color="steelblue", alpha=0.18)
    ax.axhline(
        float(np.mean(rmsf)),
        color="coral",
        ls="--",
        lw=1.2,
        label=f"Mean = {np.mean(rmsf):.2f} Å",
    )

    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel("RMSF (Å)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Per-Residue Root-Mean-Square Fluctuation",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(bottom=0.0)
    ax.margins(x=0.01)
    ax.grid(True, color="lightgray", alpha=0.6, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=True)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output


def _cmd_rmsf(args: argparse.Namespace) -> int:
    """``rmsf`` subcommand: load a trajectory, compute RMSF and plot it."""
    from ..io.loader import load_trajectory

    try:
        traj = load_trajectory(
            args.topology,
            args.trajectory,
            selection=args.selection,
            interval=args.interval,
        )
        rmsf = compute_rmsf(traj.coords, align=not args.no_align)
        path = _plot_rmsf(rmsf, traj.resids, args.output)
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved RMSF plot → {path}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``rmsf`` subcommand on the shared argparse subparsers."""
    p = subparsers.add_parser(
        "rmsf",
        help="Compute per-residue RMSF and save a plot.",
        description=(
            "Compute the per-atom root-mean-square fluctuation (RMSF) of a "
            "trajectory after Kabsch alignment and save an RMSF-vs-index plot."
        ),
    )
    p.add_argument("topology", help="Topology/structure file (e.g. PDB).")
    p.add_argument(
        "trajectory", nargs="?", default=None, help="Trajectory file."
    )
    p.add_argument(
        "--selection", "-s", default="name CA", help="Atom selection."
    )
    p.add_argument(
        "--interval", "-i", type=int, default=1, help="Frame stride."
    )
    p.add_argument(
        "--output", "-o", default="RMSF.png", help="Output PNG path."
    )
    p.add_argument(
        "--no-align", action="store_true", help="Skip Kabsch alignment."
    )
    p.set_defaults(func=_cmd_rmsf)
