# QuickPCA backend benchmarks

`bench_backends.py` times the two numerically heavy primitives in QuickPCA --
Kabsch frame alignment (`align_frames`) and full-SVD PCA (`pca`) -- for every
compute backend registered in your environment, and reports a markdown timing
table plus a NumPy/JAX speedup column.

## What it measures

For each problem size `(N atoms, F frames)` the script:

1. Generates deterministic random coordinates of shape `(F, N, 3)`. Each frame
   carries a small random rotation, jitter and translation, so `align_frames`
   has genuine work to do (it isn't aligning identical frames).
2. Times `backend.align_frames(frames, ref)`.
3. Flattens the aligned frames to the `(F, 3N)` design matrix (exactly as the
   high-level `quickpca.pca.compute_pca` pipeline does) and times
   `backend.pca(X, n_components)`.

Each measurement is a **warm-up call followed by several timed repeats**, and we
report the **median** of the repeats. Timing uses `time.perf_counter`.

When both the `numpy` and `jax` backends are registered, the table gains a
`speedup (numpy/jax)` column (NumPy median time divided by JAX median time;
values above `1.0` mean JAX is faster).

## Running

From the repository root, with the package installed (`pip install -e ".[dev,jax]"`):

```bash
# Fast smoke run (tiny sizes, fewer repeats)
python benchmarks/bench_backends.py --quick

# Full default sweep
python benchmarks/bench_backends.py

# Custom sizes (ATOMS:FRAMES pairs) and repeat count
python benchmarks/bench_backends.py --sizes "50:200,200:1000,500:1000" --repeats 7

# Only benchmark a specific backend
python benchmarks/bench_backends.py --backends numpy

# Also save a speedup bar chart (needs numpy + jax + matplotlib)
python benchmarks/bench_backends.py --chart
# -> writes benchmarks/speedup.png
```

### Options

| flag | description |
| ---- | ----------- |
| `--quick` | Use tiny sizes and only 3 repeats for a fast smoke run. |
| `--sizes "N:F,N:F"` | Comma-separated `ATOMS:FRAMES` pairs (e.g. `200:1000`). |
| `--repeats K` | Number of timed repeats per measurement (median is reported). |
| `--backends a,b` | Restrict to specific registered backends (default: all available). |
| `--chart [PATH]` | Save a NumPy/JAX speedup bar chart (default `benchmarks/speedup.png`). |

`available_backends()` decides what gets benchmarked. If the optional JAX
backend isn't installed/registered, the script transparently benchmarks NumPy
only and prints a note; it never errors out for a missing JAX backend.

## Caveats

- **JIT warm-up.** JAX compiles its kernels on first call. The script discards a
  warm-up call before timing so the reported median reflects steady-state cost,
  not one-off compilation. If you care about end-to-end latency on a single tiny
  job, remember that the first call is slower.
- **Async dispatch.** JAX executes asynchronously. The harness calls
  `block_until_ready()` on each result before stopping the clock, so the numbers
  are honest wall-clock time rather than dispatch time.
- **Small-size overhead.** At tiny sizes, Python/dispatch overhead dominates and
  JAX can look *slower* than NumPy. The speedups show up as the arrays grow. Use
  realistic sizes (hundreds of atoms, hundreds-to-thousands of frames) to see
  the win.
- **CPU vs GPU/TPU.** The dramatic JAX speedups appear on accelerators. On CPU
  the gap is narrower and, for some operations, NumPy's LAPACK can be
  competitive. Run on a GPU/TPU host (`jax.devices()` to check) to reproduce the
  "go big" numbers. The backend code is device-agnostic; only the timings
  change.
- **Machine variance.** Absolute milliseconds depend on your CPU/accelerator,
  BLAS/LAPACK build and thread count. Compare *relative* speedups, and re-run on
  your own hardware rather than trusting the committed `results.md`.

See `results.md` for a sample run captured on the benchmarking host.

---

QuickPCA is authored by Gleb Novikov (The Visual Hub) and is MIT licensed; these
benchmarks follow the same license.
