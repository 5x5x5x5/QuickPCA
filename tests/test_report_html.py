"""Tests for the interactive Plotly HTML report."""

from __future__ import annotations

import pytest

pytest.importorskip("plotly")

from quickpca.fel import compute_fel  # noqa: E402
from quickpca.pca import compute_pca  # noqa: E402
from quickpca.report_html import build_html_report  # noqa: E402


def test_build_html_report_writes_standalone_html(synthetic_coords, tmp_path):
    pca = compute_pca(synthetic_coords, n_components=10, backend="numpy")
    fel = compute_fel(pca.projections)

    out = tmp_path / "report.html"
    result = build_html_report(pca, fel, output=str(out))

    assert result == str(out)
    assert out.exists()
    # Inline Plotly runtime makes the standalone file large (> 50 KB).
    assert out.stat().st_size > 50_000

    text = out.read_text(encoding="utf-8")
    assert "<html" in text.lower()
    assert "plotly" in text.lower()


def test_build_html_report_custom_title(synthetic_coords, tmp_path):
    pca = compute_pca(synthetic_coords, n_components=5, backend="numpy")
    fel = compute_fel(pca.projections)

    out = tmp_path / "titled.html"
    title = "My Custom QuickPCA Report"
    build_html_report(pca, fel, output=str(out), title=title)

    assert title in out.read_text(encoding="utf-8")
