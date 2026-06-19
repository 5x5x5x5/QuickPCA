# API reference

The complete public API of QuickPCA, generated from the source docstrings by
[mkdocstrings](https://mkdocstrings.github.io/). Everything below is exported
from the top-level `quickpca` package:

```python
from quickpca import (
    compute_pca,
    compute_pca_from_files,
    compute_fel,
    plot_report,
    load_trajectory,
    get_backend,
    available_backends,
    PCAResult,
    FELResult,
    PCADecomposition,
    Trajectory,
    ReportConfig,
)
```

---

## PCA pipeline

::: quickpca.pca.compute_pca

::: quickpca.pca.compute_pca_from_files

## Free-Energy Landscape

::: quickpca.fel.compute_fel

## Cross-correlation

::: quickpca.crosscorr.cross_correlation

## Reporting

::: quickpca.report.plot_report

## Trajectory I/O

::: quickpca.io.loader.load_trajectory

## Backends

::: quickpca.backends.get_backend

::: quickpca.backends.available_backends

::: quickpca.backends.register_backend

::: quickpca.backends.base.Backend

::: quickpca.backends.numpy_backend.NumpyBackend

## Data types

::: quickpca.types.Trajectory

::: quickpca.types.PCAResult

::: quickpca.types.PCADecomposition

::: quickpca.types.FELResult

## Configuration

::: quickpca.config.ReportConfig
