# Quickstart

This page takes you from an installed package to a finished PCA report — first
from the command line, then from Python.

## 1. From the command line

The `run` subcommand performs the full pipeline: **load → align → PCA → FEL →
report**.

```bash
quickpca run data/reference.pdb data/trajectory.nc
```

On success it prints the path of the figure it wrote:

```text
Saved PCA report → PCA_Report.png
```

The first argument is the **topology** (e.g. a PDB), the second is the optional
**trajectory** (NetCDF, XTC, TRR, DCD, …). By default QuickPCA selects the
`name CA` atoms, keeps up to 10 principal components, runs Kabsch alignment, and
builds the Free-Energy Landscape at 300 K.

### Common options

```bash
quickpca run data/reference.pdb data/trajectory.nc \
    --selection "name CA" \
    --ncomp 10 \
    --interval 1 \
    --temp 300 \
    --backend numpy \
    --output PCA_Report.png
```

See the [CLI reference](cli.md) for every flag, and [Backends](backends.md) for
running on JAX.

## 2. From Python

### One-liner from files

The highest-level entry point loads the trajectory and runs PCA in a single call:

```python
from quickpca import compute_pca_from_files, compute_fel, plot_report

pca = compute_pca_from_files(
    "data/reference.pdb",
    "data/trajectory.nc",
    selection="name CA",
    n_components=10,
    backend="numpy",
    align=True,
)

fel = compute_fel(pca.projections, temperature=300.0)
path = plot_report(pca, fel, output="PCA_Report.png")
print(f"Report written to {path}")
```

### Step by step

If you already have coordinates, or want to inspect intermediate results, drive
each stage yourself:

```python
import numpy as np
from quickpca import load_trajectory, compute_pca, compute_fel, plot_report

# 1. Load a trajectory into a Trajectory dataclass.
traj = load_trajectory("data/reference.pdb", "data/trajectory.nc", selection="name CA")
print(traj.n_frames, "frames ×", traj.n_atoms, "atoms")

# traj.coords has shape (n_frames, n_atoms, 3)
pca = compute_pca(traj.coords, n_components=10, backend="numpy", align=True)

# Explore the PCAResult.
print("Explained variance ratio:", pca.explained_variance_ratio[:5])
print("Cumulative variance:", pca.cumulative_variance[:5])
print("Cross-correlation shape:", pca.cross_correlation.shape)

# 2. Free-Energy Landscape over PC1/PC2.
fel = compute_fel(pca.projections, temperature=300.0, n_bins=50, sigma=1.0)
print("kBT =", fel.kBT, "kJ/mol")

# 3. Render the 2×2 report figure.
plot_report(pca, fel, output="PCA_Report.png", temperature=300.0)
```

## What you get

The report is a single PNG with four panels:

| Panel        | Content |
|--------------|---------|
| Top-left     | Free-Energy Landscape (PC1 vs PC2) with start/end markers |
| Top-right    | Residue cross-correlation matrix |
| Bottom-left  | Explained-variance bar chart (first 10 PCs) + cumulative curve |
| Bottom-right | PC1 & PC2 projection histograms with KDE |

The numerical results live on the returned
[`PCAResult`](api.md#quickpca.types.PCAResult) and
[`FELResult`](api.md#quickpca.types.FELResult) dataclasses, so you can also feed
them into your own plots or downstream analysis.

## Next steps

- Read the [Theory](theory.md) behind essential dynamics and Boltzmann inversion.
- Switch to the [JAX backend](backends.md) for GPU/TPU acceleration.
- Browse the full [API reference](api.md).
