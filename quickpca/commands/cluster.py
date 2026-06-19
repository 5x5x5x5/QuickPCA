"""``cluster`` subcommand: conformational-state clustering in PC space.

Projects an MD trajectory onto its principal components and groups frames into
discrete conformational states with K-means, then renders a PC1/PC2 scatter
coloured by cluster with the cluster centroids marked.

Author: Gleb Novikov — The Visual Hub. MIT licensed.
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import for headless safety

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402


def cluster_pcs(
    projections: np.ndarray,
    n_clusters: int = 4,
    n_dims: int = 2,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster frames in principal-component space with K-means.

    Parameters
    ----------
    projections:
        PCA projection array of shape ``(n_frames, n_components)``.
    n_clusters:
        Number of conformational states (K-means clusters) to fit.
    n_dims:
        Number of leading principal components to cluster on.
    random_state:
        Seed forwarded to :class:`sklearn.cluster.KMeans` for reproducibility.

    Returns
    -------
    tuple of numpy.ndarray
        ``(labels, centers)`` where ``labels`` has shape ``(n_frames,)`` giving
        the cluster index per frame, and ``centers`` has shape
        ``(n_clusters, n_dims)`` giving the cluster centroids in PC space.
    """
    projections = np.asarray(projections, dtype=np.float64)
    if projections.ndim != 2:
        raise ValueError(
            f"projections must be 2-D (n_frames, n_components), got shape {projections.shape}."
        )

    n_frames, n_components = projections.shape
    n_dims = max(1, min(n_dims, n_components))
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}.")
    if n_clusters > n_frames:
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot exceed the number of frames ({n_frames})."
        )

    X = projections[:, :n_dims]
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    return labels.astype(np.int64), km.cluster_centers_


def plot_clusters(
    projections: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    output: str = "clusters.png",
    explained_variance_ratio: np.ndarray | None = None,
) -> str:
    """Render a PC1/PC2 scatter coloured by cluster with centroids marked.

    Returns
    -------
    str
        The ``output`` path.
    """
    projections = np.asarray(projections, dtype=np.float64)
    labels = np.asarray(labels)
    centers = np.asarray(centers, dtype=np.float64)

    pc1 = projections[:, 0]
    pc2 = projections[:, 1] if projections.shape[1] > 1 else np.zeros_like(pc1)

    if explained_variance_ratio is not None and len(explained_variance_ratio) >= 2:
        xlabel = f"PC1 ({explained_variance_ratio[0] * 100:.1f}%)"
        ylabel = f"PC2 ({explained_variance_ratio[1] * 100:.1f}%)"
    else:
        xlabel, ylabel = "PC1", "PC2"

    fig, ax = plt.subplots(figsize=(9, 7))

    sc = ax.scatter(
        pc1,
        pc2,
        c=labels,
        cmap="tab10",
        s=14,
        alpha=0.75,
        edgecolors="none",
        rasterized=True,
    )

    # Centroids: cluster_centers_ live in the clustered PC subspace, whose first
    # two columns correspond to PC1/PC2.
    if centers.shape[1] >= 2:
        cx, cy = centers[:, 0], centers[:, 1]
    else:
        cx, cy = centers[:, 0], np.zeros(centers.shape[0])

    ax.scatter(
        cx,
        cy,
        c="black",
        s=240,
        marker="X",
        edgecolors="white",
        linewidths=1.5,
        zorder=5,
        label="Centroids",
    )
    for i, (x, y) in enumerate(zip(cx, cy, strict=False)):
        ax.annotate(
            str(i),
            (x, y),
            color="white",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )

    n_clusters = centers.shape[0]
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cluster", fontsize=10)

    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(
        f"Conformational-State Clustering — {n_clusters} clusters",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best", frameon=True)
    ax.grid(True, color="lightgray", alpha=0.5, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output


def _cmd_cluster(args: argparse.Namespace) -> int:
    """Load → PCA → cluster_pcs → scatter plot of conformational states."""
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
        labels, centers = cluster_pcs(pca.projections, n_clusters=args.clusters)
        path = plot_clusters(
            pca.projections,
            labels,
            centers,
            output=args.output,
            explained_variance_ratio=pca.explained_variance_ratio,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(f"error: {exc}", file=sys.stderr)
        return 1

    counts = np.bincount(labels, minlength=args.clusters)
    total = int(counts.sum())
    print(f"Clustered {total} frames into {args.clusters} conformational states:")
    for i, n in enumerate(counts):
        pct = (100.0 * n / total) if total else 0.0
        print(f"  cluster {i}: {int(n):>6d} frames ({pct:5.1f}%)")
    print(f"Saved cluster scatter → {path}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``cluster`` subcommand on the shared subparsers object."""
    p = subparsers.add_parser(
        "cluster",
        help="Cluster trajectory frames into conformational states in PC space.",
    )
    p.add_argument("topology", help="Topology/structure file (e.g. PDB).")
    p.add_argument("trajectory", nargs="?", default=None, help="Trajectory file.")
    p.add_argument("--selection", "-s", default="name CA", help="Atom selection.")
    p.add_argument("--interval", "-i", type=int, default=1, help="Frame stride.")
    p.add_argument("--ncomp", type=int, default=10, help="Number of PCs.")
    p.add_argument("--clusters", "-k", type=int, default=4, help="Number of K-means clusters.")
    p.add_argument("--output", "-o", default="clusters.png", help="Output PNG path.")
    p.set_defaults(func=_cmd_cluster)
