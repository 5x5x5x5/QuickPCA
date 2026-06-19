# Installation

QuickPCA requires **Python 3.10 or newer**.

## Install from PyPI

```bash
pip install quickpca
```

This pulls in the core scientific stack QuickPCA depends on:

- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/) (headless `Agg` rendering)
- [MDAnalysis](https://www.mdanalysis.org/) for trajectory I/O

With just the core install you can already run the full
[CLI pipeline](quickstart.md) on the NumPy backend.

## Optional extras

QuickPCA ships several optional dependency groups. Install one or more with the
standard `pip` extras syntax:

```bash
pip install "quickpca[jax]"          # GPU/TPU-capable JAX backend
pip install "quickpca[viz]"          # interactive Plotly visualisations
pip install "quickpca[web]"          # Streamlit web app
pip install "quickpca[jax,viz,web]"  # combine extras
```

| Extra       | Adds                                   | Use it for |
|-------------|----------------------------------------|------------|
| `jax`       | `jax`, `jaxlib`                         | the accelerated [JAX backend](backends.md) on CPU/GPU/TPU |
| `viz`       | `plotly`, `kaleido`                     | interactive landscapes and figure export |
| `web`       | `streamlit`                            | the browser-based web app |
| `docs`      | `mkdocs-material`, `mkdocstrings[python]`, `mkdocs-gen-files`, `pymdown-extensions` | building this documentation site |
| `notebooks` | `jupyter`, `nbconvert`, `ipykernel`, `papermill` | running and executing example notebooks |
| `dev`       | `pytest`, `pytest-cov`, `ruff`, `mypy`, `pre-commit` | local development (see [Contributing](contributing.md)) |
| `all`       | `jax`, `viz` and `web` extras combined | a fully featured install |

!!! tip "GPU and TPU JAX"
    The `jax` extra installs the CPU build of JAX by default. For CUDA or TPU
    wheels, follow the official
    [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
    and install the platform-specific `jaxlib` for your hardware. See the
    [Backends](backends.md) guide for details.

## Install from source

To work against the latest development tree:

```bash
git clone https://github.com/5x5x5x5/quickpca.git
cd quickpca
pip install -e ".[dev]"
```

The editable install plus the `dev` extra gives you the test suite and linting
tools. See [Contributing](contributing.md) for the full developer workflow.

## Verify the install

```bash
quickpca --version
quickpca backends      # lists the registered compute backends
```

```python
import quickpca
print(quickpca.__version__)
```
