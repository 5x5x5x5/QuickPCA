# QuickPCA — Brand Assets

Visual identity for **QuickPCA** (Essential Dynamics Analysis for MD trajectories).
All assets are generated locally and reproducibly — no external downloads.

## Files

| File | Size | Purpose |
| --- | --- | --- |
| `logo.svg` | 800×260 (vector) | Hand-authored, self-contained logo: a PCA scatter cloud inside its 1-σ covariance ellipse with PC1/PC2 eigenvector arrows, plus the QuickPCA wordmark. Use this as the canonical, scalable mark. |
| `logo.png` | 800 px wide | Raster render of `logo.svg` (via `cairosvg`). Suitable for the README header and places that need a bitmap. |
| `social-preview.png` | 1280×640 | GitHub social-preview banner: wordmark, tagline, and a two-basin free-energy-landscape motif. Set under *Settings → Social preview*. |
| `example_report.png` | 4-panel figure | A real QuickPCA report (Free-Energy Landscape, residue cross-correlation, explained variance, PC1/PC2 projections), produced by running the actual pipeline. |

## Regenerating

```bash
# from the repository root, with quickpca installed (pip install -e ".[dev]")
python assets/generate_assets.py            # logo.png + social-preview.png + example_report.png
python assets/generate_assets.py --no-report  # skip the report step
```

`generate_assets.py` renders the logo from `logo.svg` when `cairosvg` is
available and otherwise draws an equivalent logo with matplotlib, so it always
succeeds. The example report uses the bundled `data/` trajectory when present
(MDAnalysis required) and transparently falls back to synthetic collective-motion
coordinates otherwise.

The example report can also be reproduced directly with the CLI:

```bash
quickpca run data/reference.pdb data/trajectory.nc --interval 5 -o assets/example_report.png
```

## Brand palette

| Token | Hex | Use |
| --- | --- | --- |
| Indigo | `#3b2f8f` | Scatter points, primary accent |
| Blue | `#3457b0` | "PCA" wordmark, ellipse outline |
| Teal | `#16a3a3` | Gradient accent, underline rule |
| Coral | `#e8543c` | PC1 (major) eigenvector axis |
| PC2 Blue | `#2f6fd0` | PC2 (minor) eigenvector axis |

## Attribution

Designed by **Gleb Novikov — The Visual Hub**. Released under the MIT License,
consistent with the QuickPCA package.
