#!/usr/bin/env python3
"""Regenerate the QuickPCA visual-brand assets.

This script is fully reproducible and has no network dependencies. It produces:

* ``assets/logo.png``           -- raster render of ``assets/logo.svg``
* ``assets/social-preview.png`` -- 1280x640 GitHub social-preview banner
* ``assets/example_report.png`` -- a real PCA/FEL report figure

The logo is rendered from the hand-authored ``logo.svg`` via ``cairosvg`` when
available; otherwise an equivalent logo is drawn directly with matplotlib so the
script always succeeds. The example report is produced by the actual QuickPCA
pipeline (preferring the bundled ``data/`` files, falling back to synthetic
coordinates when MDAnalysis or the trajectory is unavailable).

Author: Gleb Novikov -- The Visual Hub. MIT licensed.

Usage
-----
    python assets/generate_assets.py [--no-report]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch  # noqa: E402

# --- Brand palette ----------------------------------------------------------
INDIGO = "#3b2f8f"
BLUE = "#3457b0"
TEAL = "#16a3a3"
CORAL = "#e8543c"
PC2_BLUE = "#2f6fd0"
INK = "#2b2b3a"
SUBTLE = "#5a5a6a"

ASSETS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ASSETS_DIR.parent

LOGO_SVG = ASSETS_DIR / "logo.svg"
LOGO_PNG = ASSETS_DIR / "logo.png"
SOCIAL_PNG = ASSETS_DIR / "social-preview.png"
REPORT_PNG = ASSETS_DIR / "example_report.png"

TAGLINE = "Essential Dynamics Analysis for MD trajectories"


# ---------------------------------------------------------------------------
# Logo
# ---------------------------------------------------------------------------
def render_logo(width: int = 800) -> Path:
    """Render ``logo.svg`` to ``logo.png`` (cairosvg), else draw with matplotlib."""
    try:
        import cairosvg

        cairosvg.svg2png(
            url=str(LOGO_SVG),
            write_to=str(LOGO_PNG),
            output_width=width,
            background_color="white",
        )
        print(f"logo.png  <- cairosvg render of {LOGO_SVG.name} ({width}px)")
        return LOGO_PNG
    except Exception as exc:  # noqa: BLE001 - cairosvg/cairo may be missing
        print(f"cairosvg unavailable ({exc}); drawing logo with matplotlib.")
        return _draw_logo_matplotlib(width)


def _draw_logo_matplotlib(width: int) -> Path:
    """Vector-faithful matplotlib fallback for the logo glyph + wordmark."""
    dpi = 200
    fig_w = width / dpi
    fig_h = fig_w * (260 / 800)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")
    # Coordinates mirror the SVG's 800x260 canvas (y inverted for display).
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 800)
    ax.set_ylim(260, 0)
    ax.axis("off")

    cx, cy = 130, 130
    ax.add_patch(plt.Circle((cx, cy), 104, facecolor="white",
                            edgecolor=BLUE, linewidth=4.5, zorder=1))

    angle = -30.0  # degrees, PC1 along major axis
    ax.add_patch(Ellipse((cx, cy), 160, 80, angle=angle, facecolor=TEAL,
                         alpha=0.16, edgecolor=BLUE, linewidth=2.2,
                         linestyle=(0, (5, 4)), zorder=2))

    th = np.radians(angle)
    cos, sin = np.cos(th), np.sin(th)

    def rot(px, py):
        return cx + px * cos - py * sin, cy + px * sin + py * cos

    # PC1 (major) and PC2 (minor) eigenvector arrows.
    x0, y0 = rot(-86, 0)
    x1, y1 = rot(86, 0)
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=18, color=CORAL, lw=3.2, zorder=4))
    x0, y0 = rot(0, 50)
    x1, y1 = rot(0, -50)
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=18, color=PC2_BLUE, lw=3.2, zorder=4))

    pts = [(-58, -9), (-40, 10), (-25, -14), (-14, 8), (2, -6), (14, 12),
           (26, -11), (42, 6), (56, -8), (-30, -2), (34, 2)]
    for px, py in pts:
        rx, ry = rot(px, py)
        ax.add_patch(plt.Circle((rx, ry), 5, facecolor=INDIGO, zorder=5))

    # Wordmark.
    ax.text(262, 118, "Quick", fontsize=42, fontweight="bold", color=INK,
            ha="left", va="center", family="DejaVu Sans")
    quick_w = 152  # approx visual width of "Quick" at this size
    ax.text(262 + quick_w, 118, "PCA", fontsize=42, fontweight="bold",
            color=BLUE, ha="left", va="center", family="DejaVu Sans")
    ax.text(265, 160, "Essential Dynamics Analysis", fontsize=14,
            color=SUBTLE, ha="left", va="center", family="DejaVu Sans")

    fig.savefig(LOGO_PNG, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"logo.png  <- matplotlib fallback ({width}px)")
    return LOGO_PNG


# ---------------------------------------------------------------------------
# Social preview banner
# ---------------------------------------------------------------------------
def render_social_preview() -> Path:
    """Draw a 1280x640 GitHub social-preview banner with a PCA/FEL motif."""
    dpi = 100
    fig = plt.figure(figsize=(12.8, 6.4), dpi=dpi)
    fig.patch.set_facecolor("#0f1226")

    # Full-bleed background gradient (indigo -> deep teal).
    bg = fig.add_axes([0, 0, 1, 1])
    bg.axis("off")
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    bg.imshow(grad, aspect="auto", extent=[0, 1, 0, 1], origin="lower",
              cmap=_brand_cmap(), alpha=1.0, zorder=0)

    # --- Right-hand FEL motif --------------------------------------------
    fel = fig.add_axes([0.655, 0.17, 0.30, 0.66])
    _draw_fel_motif(fel)

    # --- Wordmark + tagline ----------------------------------------------
    txt = fig.add_axes([0.05, 0, 0.56, 1])
    txt.axis("off")
    txt.set_xlim(0, 1)
    txt.set_ylim(0, 1)
    txt.text(0.0, 0.62, "QuickPCA", fontsize=70, fontweight="bold",
             color="white", ha="left", va="center", family="DejaVu Sans")
    txt.text(0.01, 0.45, TAGLINE, fontsize=19, color="#cdd6f4",
             ha="left", va="center", family="DejaVu Sans")
    txt.plot([0.01, 0.40], [0.55, 0.55], color=TEAL, lw=3, solid_capstyle="round")
    txt.text(0.01, 0.30, "PCA  •  Free-Energy Landscapes  •  Cross-correlation",
             fontsize=14, color="#9aa3c4", ha="left", va="center",
             family="DejaVu Sans")
    txt.text(0.01, 0.12, "Gleb Novikov · The Visual Hub", fontsize=12,
             color="#7c84a8", ha="left", va="center", family="DejaVu Sans")

    fig.savefig(SOCIAL_PNG, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("social-preview.png <- 1280x640 banner")
    return SOCIAL_PNG


def _brand_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "qpca_brand", ["#1a1140", "#241a5c", "#143a6b", "#0e5d68"]
    )


def _draw_fel_motif(ax) -> None:
    """A compact, smooth synthetic free-energy landscape with PC axes."""
    rng = np.random.default_rng(7)
    n = 1600
    # Two-basin landscape so the FEL has visible minima.
    a = rng.normal([-1.0, 0.3], 0.55, size=(n // 2, 2))
    b = rng.normal([1.1, -0.4], 0.65, size=(n - n // 2, 2))
    pts = np.vstack([a, b])
    h, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=40, density=True)
    from scipy.ndimage import gaussian_filter

    h = gaussian_filter(h, sigma=1.2)
    with np.errstate(divide="ignore"):
        f = np.where(h > 0, -np.log(h), np.nan)
    f -= np.nanmin(f)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    xx, yy = np.meshgrid(xc, yc)
    levels = np.linspace(0, np.nanpercentile(f, 96), 18)
    ax.contourf(xx, yy, f.T, levels=levels, cmap="RdYlBu_r", extend="max")
    ax.contour(xx, yy, f.T, levels=levels[::3], colors="white",
               linewidths=0.4, alpha=0.35)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(1.2)
    ax.set_xlabel("PC1", color="white", fontsize=12, fontweight="bold")
    ax.set_ylabel("PC2", color="white", fontsize=12, fontweight="bold")


# ---------------------------------------------------------------------------
# Example report (real pipeline; synthetic fallback)
# ---------------------------------------------------------------------------
def render_example_report() -> Path:
    """Run the QuickPCA pipeline to produce the example report figure."""
    from quickpca import compute_fel, compute_pca, plot_report

    ref = REPO_ROOT / "data" / "reference.pdb"
    trj = REPO_ROOT / "data" / "trajectory.nc"

    coords = None
    if ref.exists() and trj.exists():
        try:
            from quickpca import load_trajectory

            traj = load_trajectory(str(ref), str(trj), selection="name CA",
                                   interval=5)
            coords = traj.coords
            print(f"example_report.png <- real data ({traj.n_frames} frames, "
                  f"{traj.n_atoms} atoms)")
        except Exception as exc:  # noqa: BLE001 - MDAnalysis/reader may fail
            print(f"trajectory load failed ({exc}); using synthetic coords.")

    if coords is None:
        coords = _synthetic_coords()
        print(f"example_report.png <- synthetic coords ({coords.shape[0]} frames, "
              f"{coords.shape[1]} atoms)")

    pca = compute_pca(coords, n_components=10, backend="numpy")
    fel = compute_fel(pca.projections, temperature=300.0)
    plot_report(pca, fel, output=str(REPORT_PNG), temperature=300.0)
    print("example_report.png written")
    return REPORT_PNG


def _synthetic_coords(n_frames: int = 600, n_atoms: int = 60) -> np.ndarray:
    """Generate a smooth, collective-motion trajectory for a fallback report.

    A static backbone undergoes two slow correlated modes plus thermal noise,
    so PCA recovers a clear PC1/PC2 with a multi-basin landscape.
    """
    rng = np.random.default_rng(42)
    t = np.linspace(0, 6 * np.pi, n_frames)

    # Reference backbone roughly along a helix.
    idx = np.arange(n_atoms)
    ref = np.column_stack([
        3.0 * np.cos(idx * 0.6),
        3.0 * np.sin(idx * 0.6),
        idx * 1.5,
    ]).astype(np.float64)

    # Two collective eigen-modes spanning the chain.
    mode1 = np.column_stack([
        np.sin(idx / n_atoms * np.pi),
        np.zeros(n_atoms),
        np.cos(idx / n_atoms * np.pi) * 0.5,
    ])
    mode2 = np.column_stack([
        np.zeros(n_atoms),
        np.sin(2 * idx / n_atoms * np.pi),
        np.zeros(n_atoms),
    ])

    # Slow, multi-basin amplitudes (mix of two periods).
    amp1 = 4.0 * (np.sin(t) + 0.4 * np.sin(2.3 * t))
    amp2 = 3.0 * np.cos(0.7 * t)

    coords = (ref[None, :, :]
              + amp1[:, None, None] * mode1[None, :, :]
              + amp2[:, None, None] * mode2[None, :, :])
    coords += rng.normal(0, 0.25, size=coords.shape)
    return coords


# ---------------------------------------------------------------------------
def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser(description="Regenerate QuickPCA brand assets.")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip the example_report.png pipeline step.")
    args = parser.parse_args()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    render_logo()
    render_social_preview()
    if not args.no_report:
        render_example_report()

    print("\nAssets written to", ASSETS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
