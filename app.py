"""Streamlit web demo for QuickPCA.

This file is meant to be executed via ``streamlit run app.py`` (or
``quickpca web``). The Streamlit UI is intentionally thin: every piece of heavy
logic lives in small, pure helper functions below so they can be unit-tested
without a running Streamlit server.

Author: Gleb Novikov — The Visual Hub. MIT licensed.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from quickpca import (  # noqa: E402
    available_backends,
    compute_fel,
    compute_pca,
    plot_report,
)
from quickpca.pca import compute_pca_from_files  # noqa: E402
from quickpca.types import PCAResult  # noqa: E402

# ── Bundled sample data ──────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent
SAMPLE_TOPOLOGY = REPO_ROOT / "data" / "reference.pdb"
SAMPLE_TRAJECTORY = REPO_ROOT / "data" / "trajectory.nc"


def sample_data_available() -> bool:
    """Return True if the bundled sample topology and trajectory both exist."""
    return SAMPLE_TOPOLOGY.is_file() and SAMPLE_TRAJECTORY.is_file()


# ── Pure pipeline helpers (no Streamlit) ─────────────────────────────────────


@dataclass
class ReportResult:
    """Outcome of a full PCA/FEL pipeline run for the web demo."""

    pca: PCAResult
    png_path: str


def run_pipeline(
    topology: str,
    trajectory: str | None,
    *,
    output_png: str,
    backend: str = "numpy",
    selection: str = "name CA",
    n_components: int = 10,
    temperature: float = 300.0,
    interval: int = 1,
) -> ReportResult:
    """Run ``load → PCA → FEL → report`` on files and write a PNG.

    This is the single pure entry point the Streamlit UI (and the tests) call.
    It performs no Streamlit interaction and simply returns the PCA result and
    the path of the rendered report PNG. The load + PCA half reuses the library
    helper :func:`quickpca.pca.compute_pca_from_files` so the web path stays in
    lock-step with the CLI/library behavior.
    """
    pca = compute_pca_from_files(
        topology,
        trajectory,
        selection=selection,
        interval=interval,
        n_components=n_components,
        backend=backend,
    )
    fel = compute_fel(pca.projections, temperature=temperature)
    png_path = plot_report(pca, fel, output=output_png, temperature=temperature)
    return ReportResult(pca=pca, png_path=png_path)


def run_pipeline_on_coords(
    coords,
    *,
    output_png: str,
    backend: str = "numpy",
    n_components: int = 10,
    temperature: float = 300.0,
) -> ReportResult:
    """Run ``PCA → FEL → report`` directly on a ``(F, N, 3)`` coordinate array.

    Used by tests so the pipeline can be exercised on synthetic coordinates
    without MDAnalysis or any trajectory files on disk.
    """
    pca = compute_pca(coords, n_components=n_components, backend=backend)
    fel = compute_fel(pca.projections, temperature=temperature)
    png_path = plot_report(pca, fel, output=output_png, temperature=temperature)
    return ReportResult(pca=pca, png_path=png_path)


def explained_variance_table(pca: PCAResult, n: int = 10) -> list[dict[str, object]]:
    """Build a small list-of-rows table of explained variance for display."""
    n = min(n, pca.n_components)
    rows: list[dict[str, object]] = []
    for i in range(n):
        rows.append(
            {
                "PC": i + 1,
                "Explained variance (%)": round(
                    float(pca.explained_variance_ratio[i]) * 100, 2
                ),
                "Cumulative (%)": round(
                    float(pca.cumulative_variance[i]) * 100, 2
                ),
            }
        )
    return rows


def _persist_upload(upload, directory: str) -> str:
    """Write a Streamlit ``UploadedFile`` to ``directory`` and return its path.

    The upload's ``name`` is reduced to its basename so a crafted filename
    cannot escape ``directory`` via path traversal.
    """
    name = os.path.basename(upload.name) or "upload"
    dest = os.path.join(directory, name)
    with open(dest, "wb") as fh:
        fh.write(upload.getbuffer())
    return dest


# ── Streamlit UI ─────────────────────────────────────────────────────────────


def main() -> None:  # pragma: no cover - requires a Streamlit runtime
    """Render the QuickPCA Streamlit demo."""
    import streamlit as st

    st.set_page_config(page_title="QuickPCA Demo", page_icon="🧬", layout="wide")
    st.title("QuickPCA — Essential-Dynamics Web Demo")
    st.caption(
        "Headless PCA, cross-correlation and Free-Energy Landscapes for MD "
        "trajectories. By Gleb Novikov — The Visual Hub."
    )

    with st.sidebar:
        st.header("Input data")
        has_sample = sample_data_available()
        source_options = ["Upload files"]
        if has_sample:
            source_options.insert(0, "Bundled sample data")
        source = st.radio("Data source", source_options)

        uploaded_top = uploaded_traj = None
        if source == "Upload files":
            uploaded_top = st.file_uploader(
                "Topology (PDB, GRO, PSF, ...)",
                type=None,
                key="topology",
            )
            uploaded_traj = st.file_uploader(
                "Trajectory (NC, XTC, DCD, ...) — optional",
                type=None,
                key="trajectory",
            )

        st.header("Parameters")
        backends = available_backends()
        backend = st.selectbox("Backend", backends, index=0)
        selection = st.text_input("Atom selection", value="name CA")
        n_components = st.slider("Number of components", 2, 20, 10)
        temperature = st.number_input(
            "Temperature (K)", min_value=1.0, max_value=1000.0, value=300.0, step=10.0
        )
        interval = st.number_input(
            "Frame interval (stride)", min_value=1, max_value=100, value=1, step=1
        )

        run_clicked = st.button("Run PCA", type="primary")

    if not run_clicked:
        st.info("Configure the inputs in the sidebar and click **Run PCA**.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        if source == "Bundled sample data":
            topology = str(SAMPLE_TOPOLOGY)
            trajectory: str | None = str(SAMPLE_TRAJECTORY)
        else:
            if uploaded_top is None:
                st.error("Please upload a topology file.")
                return
            topology = _persist_upload(uploaded_top, tmp)
            trajectory = (
                _persist_upload(uploaded_traj, tmp)
                if uploaded_traj is not None
                else None
            )

        output_png = os.path.join(tmp, "PCA_Report.png")
        try:
            with st.spinner("Loading trajectory and running PCA..."):
                result = run_pipeline(
                    topology,
                    trajectory,
                    output_png=output_png,
                    backend=backend,
                    selection=selection,
                    n_components=int(n_components),
                    temperature=float(temperature),
                    interval=int(interval),
                )
        except Exception as exc:  # noqa: BLE001 - surface a clean UI error
            st.error(f"Pipeline failed: {exc}")
            return

        st.success(
            f"Done — {result.pca.n_frames} frames, {result.pca.n_atoms} atoms, "
            f"backend '{result.pca.backend}'."
        )

        st.subheader("PCA report")
        st.image(result.png_path, use_container_width=True)

        st.subheader("Explained variance")
        st.dataframe(
            explained_variance_table(result.pca, n=int(n_components)),
            hide_index=True,
        )

        with open(result.png_path, "rb") as fh:
            st.download_button(
                "Download report PNG",
                data=fh.read(),
                file_name="QuickPCA_Report.png",
                mime="image/png",
            )


if __name__ == "__main__":  # pragma: no cover - only via `streamlit run`
    main()
