"""Essential-dynamics convergence diagnostics subcommand.

Provides the Hess cosine-content metric for principal-component projection
time series together with a block-wise cumulative-variance/drift curve, and a
``convergence`` CLI subcommand that renders both as a 2-panel figure.

The cosine content of a PC projection quantifies how closely the projected
motion resembles free diffusion (a random walk). A value near 1 indicates that
the apparent collective motion is dominated by random diffusion rather than
genuine, sampled dynamics — a classic warning sign of an under-converged
trajectory. A low value indicates real, structured dynamics.

References
----------
B. Hess, "Convergence of sampling in protein simulations",
Phys. Rev. E 65, 031910 (2002).

Author: Gleb Novikov — The Visual Hub. MIT licensed.
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import for headless safety

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def cosine_content(projection: np.ndarray, i: int) -> float:
    """Return the Hess cosine content of a PC projection time series.

    The cosine content of the ``i``-th principal component projection ``p(t)``
    is defined (Hess, 2002) as

    .. math::

        c_i = \\frac{2}{T}
              \\frac{\\left(\\int_0^T \\cos\\!\\frac{(i+1)\\pi t}{T}\\,
                            p(t)\\, dt\\right)^2}
                   {\\int_0^T p(t)^2\\, dt}

    evaluated here in discrete form over the sampled frames. A value near 1
    signals random-diffusion-like sampling (poor convergence); a low value
    signals real, structured dynamics.

    Parameters
    ----------
    projection:
        1-D array of the projection of every frame onto a single principal
        component, i.e. one column of ``PCAResult.projections``.
    i:
        Zero-based principal-component index. The cosine basis frequency is
        ``(i + 1) * pi``, so ``i = 0`` corresponds to a half-cosine over the
        full time series.

    Returns
    -------
    float
        The cosine content ``c_i`` in ``[0, 1]``. Returns ``0.0`` for a
        constant (zero-variance) projection.
    """
    p = np.asarray(projection, dtype=np.float64).ravel()
    n = p.size
    if n < 2:
        return 0.0

    denom = np.sum(p * p)
    # Guard against constant / zero-variance projections (no real motion):
    # the metric is undefined, so report zero cosine content.
    if denom <= np.finfo(np.float64).tiny or np.allclose(p, p[0]):
        return 0.0

    # Sample the cosine basis at the frame midpoints t_k = (k + 0.5),
    # with total length T = n; this is the standard discretisation used by
    # GROMACS' g_analyze and gives c_0 = 1 for a pure half-cosine signal.
    t = (np.arange(n) + 0.5) / n
    basis = np.cos((i + 1) * np.pi * t)

    numer = np.sum(basis * p) ** 2
    return float(2.0 / n * numer / denom)


def block_boundaries(n_frames: int, n_blocks: int = 10) -> np.ndarray:
    """Return the increasing frame counts at each contiguous-block boundary.

    The trajectory of ``n_frames`` frames is divided into ``n_blocks`` blocks of
    (near-)equal length and the cumulative frame count at the end of each block
    is returned. Duplicate boundaries (which arise when ``n_blocks`` exceeds
    ``n_frames``) are removed so every boundary is unique and strictly positive.

    Returns
    -------
    numpy.ndarray
        Sorted, unique array of frame counts in ``(0, n_frames]``.
    """
    n_blocks = max(1, min(n_blocks, max(n_frames, 1)))
    edges = np.linspace(0, n_frames, n_blocks + 1, dtype=int)[1:]
    return np.unique(edges[edges >= 1])


def block_drift(projection: np.ndarray, n_blocks: int = 10) -> np.ndarray:
    """Return the cumulative running mean of a PC projection.

    The trajectory is split into ``n_blocks`` contiguous time blocks and the
    cumulative mean of ``projection`` is reported at each block boundary. A
    cumulative mean that keeps drifting (rather than settling onto a plateau)
    is another indicator of incomplete sampling.

    Parameters
    ----------
    projection:
        1-D array of a single PC projection over all frames.
    n_blocks:
        Number of contiguous blocks to evaluate the cumulative mean at.

    Returns
    -------
    numpy.ndarray
        Array of cumulative means, one per unique block boundary (at most
        ``min(n_blocks, n_frames)`` entries).
    """
    p = np.asarray(projection, dtype=np.float64).ravel()
    boundaries = block_boundaries(p.size, n_blocks)
    if boundaries.size == 0:
        return np.array([])
    cumsum = np.cumsum(p)
    return cumsum[boundaries - 1] / boundaries


def _plot_convergence(
    cosines: np.ndarray,
    fractions: np.ndarray,
    cumulative_variance: np.ndarray,
    drifts: list[np.ndarray],
    output: str,
) -> str:
    """Render the 2-panel convergence figure and save it as a PNG.

    Panel layout::

        Left   Cosine content per principal component (bar chart)
        Right  Block-wise cumulative variance and per-PC cumulative-mean drift

    Parameters
    ----------
    cosines:
        Cosine content per principal component.
    fractions:
        Fraction (in percent) of the trajectory used at each block boundary;
        shared x positions for both the variance curve and the drift curves.
    cumulative_variance:
        Cumulative total variance evaluated at each ``fractions`` boundary.
    drifts:
        Per-PC cumulative-mean drift evaluated at each ``fractions`` boundary.
    output:
        Destination PNG path.

    Returns
    -------
    str
        The ``output`` path.
    """
    fig, (ax_cos, ax_drift) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Essential-Dynamics Convergence Diagnostics",
        fontsize=15,
        fontweight="bold",
    )

    # ── Panel 1: Cosine content per PC ───────────────────────────────────────
    ncomp = cosines.size
    x = np.arange(1, ncomp + 1)
    colors = ["crimson" if c >= 0.5 else "steelblue" for c in cosines]
    bars = ax_cos.bar(x, cosines, color=colors, alpha=0.85, edgecolor="navy", linewidth=0.6)
    for bar, c in zip(bars, cosines, strict=False):
        ax_cos.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{c:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax_cos.axhline(0.5, ls="--", color="gray", lw=1.0, alpha=0.7, label="diffusion threshold")
    ax_cos.set_xlabel("Principal Component", fontsize=11, fontweight="bold")
    ax_cos.set_ylabel("Cosine content", fontsize=11, fontweight="bold")
    ax_cos.set_title(
        "Hess Cosine Content (→1 = random diffusion)",
        fontsize=12,
        fontweight="bold",
    )
    ax_cos.set_xticks(list(x))
    ax_cos.set_ylim(0, 1.05)
    ax_cos.legend(fontsize=9, loc="upper right")
    ax_cos.grid(True, axis="y", color="skyblue", alpha=0.4, linestyle="--")
    ax_cos.set_axisbelow(True)

    # ── Panel 2: Block-wise cumulative variance + per-PC drift ───────────────
    # Both curves share the same block boundaries (``fractions``), so they line
    # up on the common x-axis created by ``twinx``.
    ax_drift.plot(
        fractions,
        cumulative_variance,
        "o-",
        color="darkgreen",
        lw=2.0,
        ms=5,
        label="Cumulative total variance",
    )
    ax_drift.set_xlabel("Trajectory used (%)", fontsize=11, fontweight="bold")
    ax_drift.set_ylabel(
        "Cumulative total variance", fontsize=11, fontweight="bold", color="darkgreen"
    )
    ax_drift.tick_params(axis="y", labelcolor="darkgreen")
    ax_drift.set_title(
        "Block-wise Variance Growth & PC Mean Drift",
        fontsize=12,
        fontweight="bold",
    )
    ax_drift.grid(True, color="lightgray", alpha=0.5, linestyle="--")

    ax_twin = ax_drift.twinx()
    cmap = plt.get_cmap("viridis")
    for idx, drift in enumerate(drifts):
        ax_twin.plot(
            fractions,
            drift,
            "--",
            color=cmap(idx / max(1, len(drifts) - 1)),
            lw=1.4,
            label=f"PC{idx + 1} mean",
        )
    ax_twin.set_ylabel("Cumulative PC projection mean", fontsize=10, color="gray")
    ax_twin.tick_params(axis="y", labelcolor="gray")
    ax_twin.legend(fontsize=8, loc="lower right", ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output


def _cmd_convergence(args: argparse.Namespace) -> int:
    """``convergence`` subcommand: load → PCA → cosine content + drift figure."""
    from ..io.loader import load_trajectory
    from ..pca import compute_pca

    try:
        traj = load_trajectory(
            args.topology,
            args.trajectory,
            selection=args.selection,
            interval=args.interval,
        )
        pca = compute_pca(traj.coords, n_components=args.ncomp)

        ncomp = min(args.ncomp, pca.projections.shape[1])
        if ncomp < 1:
            raise ValueError("PCA produced no components to analyse.")
        cosines = np.array([cosine_content(pca.projections[:, k], k) for k in range(ncomp)])

        # Shared block boundaries for the variance and drift curves. Each
        # boundary is a cumulative frame count; ``fractions`` maps it onto the
        # percentage of the trajectory used, so both curves align on one x-axis.
        n_blocks = 10
        n_frames = pca.projections.shape[0]
        boundaries = block_boundaries(n_frames, n_blocks)
        fractions = boundaries / n_frames * 100.0

        # Cumulative total variance: the sum of the per-component variances
        # captured as progressively more of the trajectory is included. Using
        # the summed per-PC variance (not the pooled flattened variance) keeps
        # the curve on the same scale as the trajectory's true total variance.
        cumulative_variance = np.array(
            [float(pca.projections[:b, :ncomp].var(axis=0).sum()) for b in boundaries]
        )

        drifts = [block_drift(pca.projections[:, k], n_blocks) for k in range(ncomp)]

        path = _plot_convergence(cosines, fractions, cumulative_variance, drifts, args.output)
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("Cosine content per principal component:")
    for k, c in enumerate(cosines):
        flag = "  (diffusion-like)" if c >= 0.5 else ""
        print(f"  PC{k + 1}: {c:.4f}{flag}")
    print(f"Saved convergence diagnostics → {path}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``convergence`` subcommand on ``subparsers``."""
    p = subparsers.add_parser(
        "convergence",
        help="Essential-dynamics convergence diagnostics (cosine content + drift).",
    )
    p.add_argument("topology", help="Topology/structure file (e.g. PDB).")
    p.add_argument("trajectory", nargs="?", default=None, help="Trajectory file.")
    p.add_argument("--selection", "-s", default="name CA", help="Atom selection.")
    p.add_argument("--interval", "-i", type=int, default=1, help="Frame stride.")
    p.add_argument("--ncomp", type=int, default=5, help="Number of PCs to analyse.")
    p.add_argument("--output", "-o", default="convergence.png", help="Output PNG path.")
    p.set_defaults(func=_cmd_convergence)
