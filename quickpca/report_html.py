"""Self-contained interactive HTML report (Plotly) for a PCA/FEL run.

The report bundles the full Plotly runtime inline so the resulting ``.html`` file
is standalone and shareable — no network access or local server required.

Author: Gleb Novikov — The Visual Hub. MIT licensed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .types import FELResult, PCAResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from plotly.graph_objects import Figure


def _require_plotly():
    """Lazily import Plotly with a friendly install hint on failure."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - exercised only without plotly
        raise ImportError(
            "Plotly is required for interactive HTML reports: pip install quickpca[viz]"
        ) from exc
    return go, make_subplots


def build_figure(
    pca: PCAResult,
    fel: FELResult,
    title: str = "Essential Dynamics — Interactive PCA Report",
) -> Figure:
    """Build the multi-panel Plotly figure for a PCA/FEL run.

    Panel layout::

        Top-left     Free-Energy Landscape contour (PC1 vs PC2)
        Top-right    PC1/PC2 scatter coloured by frame index (time)
        Bottom-wide  Explained-variance bar chart (+ cumulative line)

    Parameters
    ----------
    pca:
        High-level PCA result from :func:`quickpca.compute_pca`.
    fel:
        Free-energy landscape from :func:`quickpca.compute_fel`.
    title:
        Figure-level title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go, make_subplots = _require_plotly()

    evr = pca.explained_variance_ratio
    F = fel.F
    xc, yc = fel.xcenters, fel.ycenters
    pc1, pc2 = fel.pc1, fel.pc2

    pc1_pct = evr[0] * 100 if evr.size > 0 else float("nan")
    pc2_pct = evr[1] * 100 if evr.size > 1 else float("nan")

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 2}, None],
        ],
        subplot_titles=(
            f"Free-Energy Landscape (T = {fel.temperature:.0f} K)",
            "PC1 vs PC2 — coloured by frame (time)",
            "Explained Variance per Principal Component",
        ),
        vertical_spacing=0.13,
        horizontal_spacing=0.12,
    )

    # -- Panel 1: Free-Energy Landscape contour -------------------------------
    # ``F`` is indexed ``F[xbin, ybin]``; transpose so rows align with ``yc``.
    F_plot = np.where(np.isnan(F), np.nanmax(F), F).T
    fig.add_trace(
        go.Contour(
            x=xc,
            y=yc,
            z=F_plot,
            colorscale="RdYlBu_r",
            colorbar={
                "title": {"text": "Free Energy<br>(kJ/mol)", "side": "right"},
                "len": 0.45,
                "x": 0.46,
                "y": 0.79,
            },
            contours={"coloring": "fill"},
            hovertemplate="PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>F=%{z:.2f} kJ/mol<extra></extra>",
            name="FEL",
        ),
        row=1,
        col=1,
    )
    # Start/end markers along the projected trajectory.
    fig.add_trace(
        go.Scatter(
            x=[pc1[0]],
            y=[pc2[0]],
            mode="markers",
            marker={
                "symbol": "star",
                "size": 16,
                "color": "lime",
                "line": {"color": "black", "width": 1},
            },
            name="Start",
            hovertemplate="Start<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[pc1[-1]],
            y=[pc2[-1]],
            mode="markers",
            marker={
                "symbol": "star",
                "size": 16,
                "color": "red",
                "line": {"color": "black", "width": 1},
            },
            name="End",
            hovertemplate="End<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # -- Panel 2: PC1/PC2 scatter coloured by frame index ---------------------
    frames = np.arange(pc1.size)
    fig.add_trace(
        go.Scatter(
            x=pc1,
            y=pc2,
            mode="markers",
            marker={
                "size": 5,
                "color": frames,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {
                    "title": {"text": "Frame", "side": "right"},
                    "len": 0.45,
                    "x": 1.0,
                    "y": 0.79,
                },
                "opacity": 0.8,
            },
            name="Projection",
            hovertemplate="Frame=%{marker.color}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # -- Panel 3: Explained-variance bar chart + cumulative line --------------
    n_show = int(min(pca.n_components, 10))
    pcs = np.arange(1, n_show + 1)
    bar_vals = evr[:n_show] * 100
    cumulative = np.cumsum(evr[:n_show]) * 100

    fig.add_trace(
        go.Bar(
            x=pcs,
            y=bar_vals,
            marker={"color": "steelblue", "line": {"color": "navy", "width": 1}},
            text=[f"{v:.1f}%" for v in bar_vals],
            textposition="outside",
            name="Per-PC variance",
            hovertemplate="PC%{x}<br>%{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=pcs,
            y=cumulative,
            mode="lines+markers",
            line={"color": "coral", "dash": "dash", "width": 2},
            marker={"size": 7, "color": "coral"},
            name="Cumulative",
            hovertemplate="PC%{x}<br>cumulative %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # -- Axis titles ----------------------------------------------------------
    fig.update_xaxes(title_text=f"PC1 ({pc1_pct:.1f}%)", row=1, col=1)
    fig.update_yaxes(title_text=f"PC2 ({pc2_pct:.1f}%)", row=1, col=1)
    fig.update_xaxes(title_text=f"PC1 ({pc1_pct:.1f}%)", row=1, col=2)
    fig.update_yaxes(title_text=f"PC2 ({pc2_pct:.1f}%)", row=1, col=2)
    fig.update_xaxes(title_text="Principal Component", dtick=1, row=2, col=1)
    fig.update_yaxes(title_text="Explained Variance (%)", row=2, col=1)

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        template="plotly_white",
        height=900,
        width=1200,
        bargap=0.25,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.08},
    )
    return fig


def build_html_report(
    pca: PCAResult,
    fel: FELResult,
    output: str = "PCA_Report.html",
    title: str = "Essential Dynamics — Interactive PCA Report",
) -> str:
    """Render a standalone interactive HTML report and save it to ``output``.

    The Plotly runtime is embedded inline (``include_plotlyjs="inline"``) so the
    written file is fully self-contained and can be shared or opened offline.

    Parameters
    ----------
    pca:
        High-level PCA result from :func:`quickpca.compute_pca`.
    fel:
        Free-energy landscape from :func:`quickpca.compute_fel`.
    output:
        Destination ``.html`` path.
    title:
        Figure-level title.

    Returns
    -------
    str
        The ``output`` path.
    """
    fig = build_figure(pca, fel, title=title)
    fig.write_html(
        output,
        include_plotlyjs="inline",
        full_html=True,
        auto_open=False,
    )
    return output
