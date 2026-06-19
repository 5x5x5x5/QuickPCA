"""Headless matplotlib report figure (2x2 layout) for a PCA/FEL run."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import for headless safety

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from .types import FELResult, PCAResult  # noqa: E402


def plot_report(
    pca: PCAResult,
    fel: FELResult,
    output: str = "PCA_Report.png",
    temperature: float = 300.0,
    title: str = "Essential Dynamics — PCA Report",
) -> str:
    """Render the full PCA/FEL report and save it as a PNG.

    Panel layout::

        Top-left     Free-Energy Landscape (PC1 vs PC2)
        Top-right    Residue cross-correlation matrix
        Bottom-left  Explained-variance bar chart (first 10 PCs)
        Bottom-right PC1 & PC2 projection histograms + KDE

    Returns
    -------
    str
        The ``output`` path.
    """
    evr = pca.explained_variance_ratio
    nc = pca.n_components
    F = fel.F
    xc, yc = fel.xcenters, fel.ycenters
    pc1, pc2 = fel.pc1, fel.pc2

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    gs = GridSpec(
        2, 2, figure=fig,
        hspace=0.32, wspace=0.35,
        top=0.94, bottom=0.06, left=0.07, right=0.97,
    )

    ax_fel = fig.add_subplot(gs[0, 0])
    ax_cc = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_kde = fig.add_subplot(gs[1, 1])

    # ── Panel 1: Free-Energy Landscape ───────────────────────────────────────
    F_plot = np.where(np.isnan(F), np.nanmax(F), F)
    XX, YY = np.meshgrid(xc, yc)
    levels = np.linspace(0, np.nanpercentile(F, 97), 30)

    cf = ax_fel.contourf(
        XX, YY, F_plot.T, levels=levels, cmap="RdYlBu_r", extend="max"
    )
    ax_fel.contour(
        XX, YY, F_plot.T, levels=levels[::5], colors="k", linewidths=0.4, alpha=0.5
    )

    cbar = fig.colorbar(cf, ax=ax_fel, fraction=0.046, pad=0.04)
    cbar.set_label("Free Energy (kJ mol⁻¹)", fontsize=10)
    F_max = np.nanpercentile(F, 97)
    _step = max(1, int(round(F_max / 6)))
    cbar.set_ticks(range(0, int(F_max) + _step, _step))

    ax_fel.plot(pc1, pc2, color="white", lw=0.25, alpha=0.3, rasterized=True)
    ax_fel.scatter(
        pc1[0], pc2[0], c="lime", s=130, marker="*",
        zorder=5, edgecolors="k", lw=0.7, label="Start",
    )
    ax_fel.scatter(
        pc1[-1], pc2[-1], c="red", s=130, marker="*",
        zorder=5, edgecolors="k", lw=0.7, label="End",
    )

    ax_fel.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", fontsize=11, fontweight="bold")
    ax_fel.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", fontsize=11, fontweight="bold")
    ax_fel.set_title(
        f"Free-Energy Landscape  (T = {temperature:.0f} K)",
        fontsize=12, fontweight="bold",
    )
    ax_fel.set_xlim(fel.xedges[0], fel.xedges[-1])
    ax_fel.set_ylim(fel.yedges[0], fel.yedges[-1])
    ax_fel.legend(fontsize=10, loc="upper right", frameon=True)
    ax_fel.grid(True, color="white", alpha=0.15, linestyle="--", linewidth=0.5)

    # ── Panel 2: Cross-Correlation Matrix ────────────────────────────────────
    cc = pca.cross_correlation
    im = ax_cc.imshow(
        cc, cmap="RdBu_r", vmin=-1, vmax=1,
        aspect="auto", origin="lower", interpolation="nearest",
    )
    fig.colorbar(im, ax=ax_cc, fraction=0.046, pad=0.04, label="Cross-correlation")
    ax_cc.set_xlabel("Residue index", fontsize=11, fontweight="bold")
    ax_cc.set_ylabel("Residue index", fontsize=11, fontweight="bold")
    ax_cc.set_title("Residue Cross-Correlation Matrix", fontsize=12, fontweight="bold")

    # ── Panel 3: Explained-Variance Bar Chart ────────────────────────────────
    n_show = min(nc, 10)
    x_ticks = range(1, n_show + 1)

    bars = ax_bar.bar(
        x_ticks, evr[:n_show] * 100,
        color="steelblue", alpha=0.85, edgecolor="navy", linewidth=0.6,
    )

    for bar, pct in zip(bars, evr[:n_show] * 100, strict=False):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    ax2 = ax_bar.twinx()
    ax2.plot(
        x_ticks, np.cumsum(evr[:n_show]) * 100,
        "o--", color="coral", lw=1.8, ms=5, label="Cumulative",
    )
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=10, color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")
    ax2.set_ylim(0, 105)
    ax2.axhline(80, ls=":", color="gray", alpha=0.6, lw=1.0)
    ax2.axhline(90, ls=":", color="gray", alpha=0.6, lw=1.0)
    ax2.legend(loc="center right", fontsize=9)

    ax_bar.set_xlabel("Principal Component", fontsize=11, fontweight="bold")
    ax_bar.set_ylabel("Explained Variance (%)", fontsize=11, fontweight="bold")
    ax_bar.set_title(
        f"First {n_show} PCs — Explained Variance", fontsize=12, fontweight="bold"
    )
    ax_bar.set_xticks(list(x_ticks))
    ax_bar.set_ylim(0, 105)
    ax_bar.grid(True, axis="y", color="skyblue", alpha=0.4, linestyle="--")
    ax_bar.set_axisbelow(True)

    # ── Panel 4: PC1 & PC2 Projection Histograms + KDE ───────────────────────
    for comp, label, color, idx in [
        (pc1, "PC1", "teal", 0),
        (pc2, "PC2", "darkorange", 1),
    ]:
        pct = evr[idx] * 100
        ax_kde.hist(
            comp, bins=60, color=color, alpha=0.45,
            edgecolor="k", linewidth=0.3, density=True, label=f"{label} ({pct:.1f}%)",
        )
        xr = np.linspace(comp.min(), comp.max(), 300)
        ax_kde.plot(xr, gaussian_kde(comp)(xr), color=color, lw=2.0)
        ax_kde.axvline(comp.mean(), color=color, ls="--", lw=1.2)

    ax_kde.set_xlabel("Projection value", fontsize=11, fontweight="bold")
    ax_kde.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax_kde.set_title("PC1 & PC2 Projection Distributions", fontsize=12, fontweight="bold")
    ax_kde.legend(fontsize=9)
    ax_kde.grid(True, color="lightgray", alpha=0.5, linestyle="--")

    # ── Save ─────────────────────────────────────────────────────────────────
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output
