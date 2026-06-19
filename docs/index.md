# QuickPCA

**Welcome to the universe of eigenvectors.**

QuickPCA is a lightweight Python toolkit for **Essential Dynamics Analysis** of
molecular-dynamics (MD) trajectories. It loads a trajectory, performs Principal
Component Analysis, recovers a residue cross-correlation map, builds a 2-D
Free-Energy Landscape, and renders a publication-ready report — all from a
single command or a few lines of Python.

Originally a PyMOL drag-and-drop script, QuickPCA is now a **headless,
pip-installable package** with a pluggable compute layer (NumPy today, JAX for
GPU/TPU acceleration) and a modern CLI.

---

## Why PCA for MD?

Molecular-dynamics trajectories carry an enormous amount of data. A protein with
1,000 atoms has 3,000 Cartesian coordinates per frame; 10,000 frames is already
30 million numbers. PCA compresses this high-dimensional dataset down to a
handful of **collective coordinates** (principal components), separating the
large-amplitude functional motions — the *essential subspace* — from
uncorrelated thermal noise. The first two components (PC1, PC2) typically capture
the dominant conformational change in a way you can actually plot and reason about.

## What makes it fast

Unlike the classical approach of building and diagonalising a
\(3N \times 3N\) covariance matrix, QuickPCA performs an **SVD** directly on the
\((F \times 3N)\) coordinate matrix. This skips the costly covariance
diagonalisation while producing identical principal components, and it is more
numerically stable. The residue cross-correlation matrix is then recovered
**analytically** from the PCA eigenvectors and eigenvalues, without revisiting
the raw trajectory.

## Features

- Structural alignment with the **Kabsch algorithm** (rotation + translation, reflection-corrected)
- **Principal Component Analysis** via full SVD (essential dynamics)
- **Free-Energy Landscape** over PC1 vs PC2 (Boltzmann inversion)
- **Residue cross-correlation** matrix recovered from PCA modes
- Explained-variance and cumulative-variance profiles
- PC projection distributions (histogram + KDE)
- Automated 2×2 **report figure** (PNG)
- Pluggable **backends**: pure NumPy, optional **JAX** (GPU/TPU)
- A modern **CLI** (`quickpca run`, `quickpca backends`, …) plus a clean Python API

## A 30-second taste

```bash
pip install quickpca
quickpca run data/reference.pdb data/trajectory.nc
# → Saved PCA report → PCA_Report.png
```

```python
from quickpca import compute_pca_from_files, compute_fel, plot_report

pca = compute_pca_from_files("data/reference.pdb", "data/trajectory.nc")
fel = compute_fel(pca.projections, temperature=300.0)
plot_report(pca, fel, output="PCA_Report.png")
```

Head to the [Installation](installation.md) and [Quickstart](quickstart.md)
guides to get going, dig into the [Theory](theory.md) behind the method, or
browse the full [API reference](api.md).

## Author & license

QuickPCA was developed by **Gleb Novikov** (The Visual Hub) and is released under
the **MIT License**. If QuickPCA contributes to results in a publication, thesis,
or report, appropriate citation is strongly encouraged.
