"""Tests for the headless report figure."""

from __future__ import annotations

from quickpca.fel import compute_fel
from quickpca.pca import compute_pca
from quickpca.report import plot_report


def test_plot_report_writes_png(synthetic_coords, tmp_path):
    pca = compute_pca(synthetic_coords, n_components=10, backend="numpy")
    fel = compute_fel(pca.projections)

    out = tmp_path / "report.png"
    result = plot_report(pca, fel, output=str(out))

    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 5_000
