# Sample benchmark results

Generated with `python benchmarks/bench_backends.py --repeats 5`.

These numbers are illustrative and machine-specific (see the caveats in
`README.md`). Re-run on your own hardware -- especially a GPU/TPU host -- to
reproduce the JAX speedups, which do not show up on CPU at these sizes.

## Environment

- OS: Linux 6.18.5 (x86_64, glibc 2.39)
- Python: 3.11.15
- NumPy: 2.4.6
- JAX: 0.10.2 (CPU device only on this host)

> Note: the JAX **backend** (`quickpca/backends/jax_backend.py`) was not yet
> registered in the package snapshot used for this run, so only the `numpy`
> backend is timed below. Once the JAX backend is present, the same command
> emits an extra `jax (ms)` column and a `speedup (numpy/jax)` column
> automatically. The benchmark requires no edits to pick it up -- it iterates
> over `available_backends()`.

## Timings (median of 5 repeats)

| operation    | atoms | frames | numpy (ms) |
| ------------ | ----- | ------ | ---------- |
| align_frames | 50    | 200    | 6.253      |
| align_frames | 200   | 200    | 8.277      |
| align_frames | 200   | 1000   | 41.864     |
| align_frames | 500   | 1000   | 53.375     |
| pca          | 50    | 200    | 5.473      |
| pca          | 200   | 200    | 20.974     |
| pca          | 200   | 1000   | 145.034    |
| pca          | 500   | 1000   | 507.292    |

The `align_frames` cost scales roughly with the number of frames (per-frame SVD
of a 3x3 matrix). The full-SVD `pca` cost is dominated by the SVD of the
`(F, 3N)` design matrix and is sensitive to which dimension is larger.
