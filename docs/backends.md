# Backends

QuickPCA separates the **analysis pipeline** from the **numerically heavy
kernels**. Those kernels — Kabsch frame alignment and full-SVD PCA — live behind
a small pluggable *backend* interface, so the same high-level code can run on
pure NumPy or on JAX-accelerated hardware without changes.

## The backend interface

Every backend is a subclass of `Backend` implementing two primitives:

- `align_frames(frames, ref)` — Kabsch-align each frame `(F, N, 3)` onto a
  reference `(N, 3)`.
- `pca(X, n_components)` — full-SVD PCA on a raw (not pre-centered)
  `(F, D)` matrix, returning a
  [`PCADecomposition`](api.md#quickpca.types.PCADecomposition).

See the [API reference](api.md#quickpca.backends.base.Backend) for the full protocol.

## The registry

Backends self-register by name through a decorator and are looked up at runtime:

```python
from quickpca import available_backends, get_backend

print(available_backends())   # e.g. ['numpy']  (or ['jax', 'numpy'])

backend = get_backend("numpy")
print(backend.name)           # 'numpy'
```

`get_backend` raises a `ValueError` listing the available backends if you request
one that is not registered. The CLI exposes the same listing through
[`quickpca backends`](cli.md#quickpca-backends).

## NumPy backend (default)

The `numpy` backend is the always-available reference implementation, built on
`numpy.linalg.svd`. It:

- Centres each frame on its own centroid, computes the optimal rotation via SVD
  of the cross-covariance, and **corrects reflections** using the sign of the
  determinant before translating back onto the reference centroid.
- Performs full SVD on the centred coordinate matrix and applies the same
  **sign convention as scikit-learn** (`svd_flip`), so results match
  `PCA(svd_solver="full")` exactly.

It needs no extra dependencies and is the right choice for small and
medium-sized trajectories.

## JAX backend (optional, accelerated)

Installing the `jax` extra makes a JAX-powered backend available:

```bash
pip install "quickpca[jax]"
```

The backend is imported opportunistically on package import, so once it is
installed it self-registers and appears in `available_backends()`. JAX
implements the same Kabsch + full-SVD primitives but runs them through XLA,
enabling vectorised execution and just-in-time compilation.

### Selecting JAX

From the CLI:

```bash
quickpca run data/reference.pdb data/trajectory.nc --backend jax
```

From Python:

```python
from quickpca import compute_pca

pca = compute_pca(coords, n_components=10, backend="jax")
```

### GPU and TPU

JAX is the route to **GPU and TPU** acceleration, which pays off most on large
trajectories (many frames and/or many atoms), where the per-frame alignment and
the SVD dominate the runtime.

The `jax` extra installs the default (CPU) build. To run on accelerators,
install the platform-specific `jaxlib` for your hardware following the official
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html):

```bash
# Example: CUDA 12 GPUs (consult the JAX docs for the current command).
pip install -U "jax[cuda12]"
```

Once a GPU/TPU build of JAX is installed, the JAX backend will use the
accelerator automatically — no QuickPCA code changes are required.

!!! tip "Numerical parity"
    Both backends follow the scikit-learn sign convention and the same Kabsch
    formulation, so switching backends should not change your scientific
    results — only the runtime.

## Benchmarking

Because the NumPy and JAX backends are drop-in interchangeable, the fairest way
to compare them is to time the same `compute_pca` call on representative data:

```python
import time
import numpy as np
from quickpca import compute_pca, available_backends

coords = np.random.default_rng(0).standard_normal((2000, 500, 3))

for name in available_backends():
    t0 = time.perf_counter()
    compute_pca(coords, n_components=10, backend=name)
    print(f"{name:8s} {time.perf_counter() - t0:7.3f} s")
```

!!! note
    JAX compiles kernels on first use, so discard the first timed run (warm-up)
    and benchmark steady-state calls. Expect the largest speedups on big inputs
    and on GPU/TPU hardware; for small trajectories the NumPy backend is often
    just as fast.
