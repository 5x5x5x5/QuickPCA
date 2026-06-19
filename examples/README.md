# QuickPCA Examples

A set of didactic Jupyter notebooks that walk through QuickPCA's public API on the
bundled sample trajectory (`../data/reference.pdb` + `../data/trajectory.nc`).
Every notebook uses only the documented public functions:

```python
from quickpca import (
    compute_pca, compute_pca_from_files, compute_fel,
    plot_report, load_trajectory, get_backend,
)
```

## Notebooks

| Notebook | What it covers |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | The full pipeline end to end: `load_trajectory` -> `compute_pca` -> `compute_fel` -> `plot_report`, with the report shown inline and an explanation of every panel. |
| [`02_jax_gpu.ipynb`](02_jax_gpu.ipynb) | Running the same PCA on the `numpy` vs `jax` backends via `get_backend` / `compute_pca(..., backend=...)`, confirming the results agree, a small timing comparison, and notes on GPU/TPU dispatch. |
| [`03_free_energy_landscape.ipynb`](03_free_energy_landscape.ipynb) | A deep dive into `compute_fel`: Boltzmann inversion, the role of `temperature`, `n_bins` and `sigma`, custom matplotlib plots, and how to read minima and basins. |

Read them in order — `01` introduces the concepts the others build on.

## Setup

From the repository root, install QuickPCA with the optional extras these
notebooks use (Jupyter, JAX, and plotting/visualisation):

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -e ".[notebooks,jax,viz]"
```

The `jax` extra is optional. The JAX backend registers automatically once it is
installed; `02_jax_gpu.ipynb` is written to run cleanly either way and tells you
which backends it found. To accelerate the `jax` backend on hardware, install a
device-specific jaxlib instead, e.g. `pip install -U "jax[cuda12]"` (GPU) or
`pip install -U "jax[tpu]"` (TPU).

## Running

Launch Jupyter and open a notebook:

```bash
jupyter lab            # or: jupyter notebook
```

Run all notebooks **from inside this `examples/` directory** so the relative
`../data/...` paths resolve. To execute one headlessly from the command line:

```bash
cd examples
jupyter nbconvert --to notebook --execute 01_quickstart.ipynb --output /tmp/out.ipynb
```

`01_quickstart.ipynb` writes a `quickstart_report.png` next to itself when run;
it is a regenerable artifact and is not tracked in git.

---

QuickPCA was developed by **Gleb Novikov**. Released under the MIT License.
If QuickPCA contributes to published work, please cite it.
