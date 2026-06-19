# Command-line interface

QuickPCA installs a single console entry point, `quickpca`, backed by an
`argparse` parser with **auto-discovered subcommands**. You can also invoke it as
a module:

```bash
quickpca --help
python -m quickpca --help
quickpca --version
```

## Architecture: a discoverable command suite

The CLI ships two built-in subcommands — `run` and `backends` — and then
**discovers** any additional subcommands placed in the `quickpca.commands`
package. Each plugin module defines a `register(subparsers)` function that adds
its own parser and wires up a handler. A broken plugin is reported as a warning
and skipped, so it can never crash the CLI.

This makes QuickPCA's command line an extensible **analysis suite**. Alongside
the core `run` and `backends`, the wider toolset includes subcommands such as:

- **`rmsf`** — per-residue root-mean-square fluctuation profiles
- **`cluster`** — conformational clustering in the essential subspace
- **`convergence`** — sampling/convergence diagnostics for the PCA subspace
- **`interactive`** — interactive (Plotly) reports and landscapes
- **`web`** — launch the Streamlit web app

!!! note
    The exact set of available subcommands depends on which optional components
    are installed. Run `quickpca --help` to see every subcommand registered in
    your environment, and `quickpca <subcommand> --help` for its options.

---

## `quickpca run`

Run the full pipeline: **load → align → PCA → FEL → report**.

```bash
quickpca run TOPOLOGY [TRAJECTORY] [options]
```

### Positional arguments

| Argument     | Description |
|--------------|-------------|
| `topology`   | Topology / structure file (e.g. PDB). **Required.** |
| `trajectory` | Trajectory file (NetCDF, XTC, TRR, DCD, …). Optional; if omitted, only the topology coordinates are used. |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--selection`, `-s` | `name CA` | MDAnalysis atom-selection string. |
| `--ncomp` | `10` | Number of principal components to retain. |
| `--interval`, `-i` | `1` | Frame stride when reading the trajectory. |
| `--temp` | `300.0` | Temperature in kelvin for the Free-Energy Landscape. |
| `--nbins` | `50` | Number of histogram bins per axis for the FEL. |
| `--sigma` | `1.0` | Gaussian-smoothing sigma applied to the FEL density. |
| `--backend`, `-b` | `numpy` | Compute backend (e.g. `numpy`, `jax`). |
| `--output`, `-o` | `PCA_Report.png` | Output PNG path. |
| `--no-align` | _(off)_ | Skip Kabsch alignment of frames. |

### Examples

```bash
# Defaults: CA atoms, 10 PCs, 300 K, NumPy backend.
quickpca run data/reference.pdb data/trajectory.nc

# Backbone selection, every 5th frame, 350 K, JAX backend.
quickpca run data/reference.pdb data/trajectory.nc \
    --selection "backbone" --interval 5 --temp 350 --backend jax

# Skip alignment (e.g. trajectory is already aligned) and rename the output.
quickpca run data/reference.pdb data/trajectory.nc \
    --no-align --output run01_report.png
```

On success the command prints `Saved PCA report → <path>` and exits `0`. On
failure it prints `error: <message>` to standard error and exits `1`.

---

## `quickpca backends`

List the compute backends registered in the current environment, one per line.

```bash
quickpca backends
```

```text
numpy
```

After installing the `jax` extra (`pip install "quickpca[jax]"`) the JAX backend
self-registers and appears here too:

```text
jax
numpy
```

See the [Backends](backends.md) guide for how the registry works and how to
select a backend.

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0`  | Success. |
| `1`  | A runtime error occurred (printed to stderr), or no subcommand was given (help is shown). |
