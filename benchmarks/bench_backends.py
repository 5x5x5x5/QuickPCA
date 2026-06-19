#!/usr/bin/env python3
"""Benchmark QuickPCA compute backends (NumPy vs JAX).

Generates random trajectory-shaped data ``(F, N, 3)`` across several sizes and
times the two numerically heavy primitives -- Kabsch frame alignment
(``align_frames``) and full-SVD PCA (``pca``) -- for every backend registered
in the running environment.

JAX is JIT-compiled, so the first call to a kernel pays a one-off compilation
cost. We therefore run a warm-up pass (discarded) before timing the
steady-state cost, and we block on device results so the wall-clock numbers are
honest. Timings use :func:`time.perf_counter`, repeated several times, and we
report the *median* to damp outliers.

Run ``python benchmarks/bench_backends.py --help`` for options.

QuickPCA is authored by Gleb Novikov (The Visual Hub), MIT licensed.
This benchmark harness follows the same license.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, is_dataclass

import numpy as np

from quickpca import available_backends, get_backend

# Default problem sizes: (n_atoms, n_frames). Atoms map to "name CA" selections
# of small-to-medium proteins; frames map to short-to-moderate trajectories.
DEFAULT_SIZES: tuple[tuple[int, int], ...] = (
    (50, 200),
    (200, 200),
    (200, 1000),
    (500, 1000),
)

# Tiny sizes used by --quick for a fast smoke run.
QUICK_SIZES: tuple[tuple[int, int], ...] = (
    (20, 50),
    (50, 100),
)

DEFAULT_REPEATS = 5
N_COMPONENTS = 10


@dataclass(frozen=True)
class Timing:
    """Median wall-clock time (seconds) for one backend/operation/size."""

    backend: str
    operation: str
    n_atoms: int
    n_frames: int
    median_s: float
    repeats: int


def make_data(n_atoms: int, n_frames: int, seed: int = 0) -> np.ndarray:
    """Return deterministic random coordinates of shape ``(F, N, 3)``.

    A small per-frame random rotation/translation is baked in so that
    ``align_frames`` has genuine work to do rather than aligning identical
    frames.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_atoms, 3))
    coords = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
    for f in range(n_frames):
        # Random small rotation via QR of a random matrix, plus jitter + shift.
        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        jitter = 0.1 * rng.standard_normal((n_atoms, 3))
        shift = rng.standard_normal(3)
        coords[f] = (base + jitter) @ q + shift
    return coords


def _block(value: object) -> None:
    """Force an async (JAX) result to materialise so timing is wall-honest.

    NumPy results are already concrete; JAX dispatches asynchronously, so we
    call ``block_until_ready`` (present on ``jax.Array``) before stopping the
    clock. ``align_frames`` returns a bare array, but ``pca`` returns a
    :class:`~quickpca.types.PCADecomposition` dataclass whose *fields* are the
    arrays we must block on -- so we recurse into dataclass fields rather than
    only checking the top-level object.
    """
    block = getattr(value, "block_until_ready", None)
    if callable(block):
        block()
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _block(getattr(value, field.name))


def _time_call(fn: Callable[[], object], repeats: int) -> float:
    """Warm up once, then time ``fn`` ``repeats`` times and return the median.

    The warm-up call absorbs JAX JIT compilation (and any first-touch
    allocation) so the reported median reflects steady-state cost.
    """
    _block(fn())  # warm-up: discarded (covers JIT compile)

    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        _block(result)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def bench_backend(
    name: str,
    datasets: Sequence[tuple[int, int, np.ndarray]],
    repeats: int,
) -> list[Timing]:
    """Time ``align_frames`` and ``pca`` for one backend across ``datasets``.

    ``datasets`` is a sequence of ``(n_atoms, n_frames, coords)`` tuples. The
    coordinate arrays are generated once by the caller and shared across
    backends, so every backend is timed on byte-identical input.
    """
    backend = get_backend(name)
    timings: list[Timing] = []

    # Closure factories bind the per-iteration data explicitly, so each timed
    # callable captures the right values (no late-binding loop-variable trap).
    def align_call(coords: np.ndarray, ref: np.ndarray) -> Callable[[], object]:
        return lambda: backend.align_frames(coords, ref)

    def pca_call(matrix: np.ndarray, k: int) -> Callable[[], object]:
        return lambda: backend.pca(matrix, k)

    for n_atoms, n_frames, coords in datasets:
        ref = coords[0]

        align_dt = _time_call(align_call(coords, ref), repeats)
        timings.append(
            Timing(name, "align_frames", n_atoms, n_frames, align_dt, repeats)
        )

        # PCA operates on the flattened (F, 3N) design matrix, mirroring the
        # high-level pipeline in quickpca.pca.compute_pca.
        aligned = backend.align_frames(coords, ref)
        x = np.asarray(aligned, dtype=np.float64).reshape(n_frames, 3 * n_atoms)
        n_comp = min(N_COMPONENTS, min(x.shape) - 1)

        pca_dt = _time_call(pca_call(x, n_comp), repeats)
        timings.append(Timing(name, "pca", n_atoms, n_frames, pca_dt, repeats))

    return timings


def _index(timings: Sequence[Timing]) -> dict[tuple[str, str, int, int], float]:
    """Map ``(backend, op, n_atoms, n_frames)`` to median seconds."""
    return {
        (t.backend, t.operation, t.n_atoms, t.n_frames): t.median_s
        for t in timings
    }


def format_table(
    timings: Sequence[Timing],
    backends: Sequence[str],
    sizes: Sequence[tuple[int, int]],
) -> str:
    """Render a markdown timing table with a NumPy/JAX speedup column.

    The speedup column is only added when both ``numpy`` and ``jax`` backends
    were benchmarked.
    """
    lookup = _index(timings)
    show_speedup = "numpy" in backends and "jax" in backends

    header = ["operation", "atoms", "frames"]
    header += [f"{b} (ms)" for b in backends]
    if show_speedup:
        header.append("speedup (numpy/jax)")

    rows: list[list[str]] = []
    for op in ("align_frames", "pca"):
        for n_atoms, n_frames in sizes:
            row = [op, str(n_atoms), str(n_frames)]
            for b in backends:
                secs = lookup.get((b, op, n_atoms, n_frames))
                row.append("-" if secs is None else f"{secs * 1e3:.3f}")
            if show_speedup:
                np_s = lookup.get(("numpy", op, n_atoms, n_frames))
                jax_s = lookup.get(("jax", op, n_atoms, n_frames))
                if np_s and jax_s and jax_s > 0:
                    row.append(f"{np_s / jax_s:.2f}x")
                else:
                    row.append("-")
            rows.append(row)

    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: Sequence[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(header))) + " |"
    lines = [fmt(header), sep]
    lines += [fmt(row) for row in rows]
    return "\n".join(lines)


def maybe_save_chart(
    timings: Sequence[Timing],
    backends: Sequence[str],
    sizes: Sequence[tuple[int, int]],
    path: str,
) -> bool:
    """Save a NumPy/JAX speedup bar chart to ``path``.

    Returns ``True`` if a chart was written. Requires both backends and
    matplotlib; returns ``False`` (without raising) otherwise.
    """
    if not ("numpy" in backends and "jax" in backends):
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    lookup = _index(timings)
    operations = ("align_frames", "pca")
    labels = [f"{n}x{f}" for n, f in sizes]
    x = np.arange(len(sizes))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, op in zip((-width / 2, width / 2), operations, strict=True):
        speedups = []
        for n_atoms, n_frames in sizes:
            np_s = lookup.get(("numpy", op, n_atoms, n_frames))
            jax_s = lookup.get(("jax", op, n_atoms, n_frames))
            speedups.append(np_s / jax_s if np_s and jax_s and jax_s > 0 else 0.0)
        ax.bar(x + offset, speedups, width, label=op)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="parity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("problem size (atoms x frames)")
    ax.set_ylabel("speedup (numpy time / jax time)")
    ax.set_title("QuickPCA: JAX speedup over NumPy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def parse_sizes(spec: str) -> list[tuple[int, int]]:
    """Parse ``--sizes`` of the form ``"N:F,N:F"`` into ``(atoms, frames)``."""
    sizes: list[tuple[int, int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            atoms_str, frames_str = token.split(":")
            atoms, frames = int(atoms_str), int(frames_str)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid size {token!r}; expected 'ATOMS:FRAMES' (e.g. 200:1000)."
            ) from exc
        if atoms < 2 or frames < 2:
            raise argparse.ArgumentTypeError(
                f"Size {token!r} too small; atoms and frames must each be >= 2."
            )
        sizes.append((atoms, frames))
    if not sizes:
        raise argparse.ArgumentTypeError("No valid sizes parsed from --sizes.")
    return sizes


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark QuickPCA backends (NumPy vs JAX).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use tiny sizes and fewer repeats for a fast smoke run.",
    )
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=None,
        help="Comma-separated ATOMS:FRAMES pairs, e.g. '50:200,500:1000'.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help=f"Timed repeats per measurement (default: {DEFAULT_REPEATS}).",
    )
    parser.add_argument(
        "--backends",
        default=None,
        help="Comma-separated backend names to benchmark (default: all available).",
    )
    parser.add_argument(
        "--chart",
        nargs="?",
        const="benchmarks/speedup.png",
        default=None,
        metavar="PATH",
        help="Save a speedup bar chart (needs numpy+jax and matplotlib).",
    )
    return parser


def resolve_backends(requested: str | None) -> list[str]:
    """Resolve the backend list, validating any explicit request."""
    available = available_backends()
    if requested is None:
        return available
    chosen = [b.strip() for b in requested.split(",") if b.strip()]
    missing = [b for b in chosen if b not in available]
    if missing:
        raise SystemExit(
            f"Requested backend(s) not available: {missing}. "
            f"Available: {available}"
        )
    return chosen


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and print a markdown report. Returns an exit code."""
    args = build_parser().parse_args(argv)

    if args.quick:
        sizes = args.sizes if args.sizes is not None else list(QUICK_SIZES)
        repeats = args.repeats if args.repeats is not None else 3
    else:
        sizes = args.sizes if args.sizes is not None else list(DEFAULT_SIZES)
        repeats = args.repeats if args.repeats is not None else DEFAULT_REPEATS

    if repeats < 1:
        raise SystemExit("--repeats must be >= 1.")

    backends = resolve_backends(args.backends)
    if not backends:
        raise SystemExit("No backends available to benchmark.")

    print("# QuickPCA backend benchmark\n")
    print(f"- backends: {', '.join(backends)}")
    print(f"- sizes (atoms x frames): {', '.join(f'{a}x{f}' for a, f in sizes)}")
    print(f"- repeats (median of): {repeats}")
    if "jax" not in backends:
        print("- note: JAX backend not registered; benchmarking NumPy only.")
    print()

    # Generate each size's data once and reuse it across all backends so every
    # backend is timed on byte-identical input (and we avoid regenerating the
    # same deterministic arrays per backend).
    datasets = [
        (n_atoms, n_frames, make_data(n_atoms, n_frames))
        for n_atoms, n_frames in sizes
    ]

    all_timings: list[Timing] = []
    for name in backends:
        all_timings.extend(bench_backend(name, datasets, repeats))

    # Sanity-check: every timing must be finite and strictly positive.
    for t in all_timings:
        if not (t.median_s > 0 and np.isfinite(t.median_s)):
            raise SystemExit(
                f"Non-finite/zero timing for {t.backend}/{t.operation} "
                f"at {t.n_atoms}x{t.n_frames}: {t.median_s}"
            )

    print(format_table(all_timings, backends, sizes))
    print()

    if args.chart is not None:
        if maybe_save_chart(all_timings, backends, sizes, args.chart):
            print(f"Saved speedup chart to {args.chart}")
        else:
            print(
                "Skipped chart: requires both 'numpy' and 'jax' backends "
                "plus matplotlib."
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
