"""Thin PyMOL integration layer for QuickPCA.

This module adapts PyMOL's ``cmd`` API to the headless ``quickpca`` package.
It deliberately contains *no* PCA/FEL math of its own: coordinates are extracted
from PyMOL states and handed straight to the package's public API
(:func:`quickpca.compute_pca`, :func:`quickpca.compute_fel`,
:func:`quickpca.plot_report`).

``pymol`` is imported lazily so that this module can be imported in a headless
environment (and unit-tested with a fake ``cmd`` object).

Author: Gleb Novikov — The Visual Hub. MIT licensed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quickpca import compute_fel, compute_pca, plot_report

# A 2-D Free-Energy Landscape needs at least PC1 and PC2, which in turn needs
# at least three frames; mirrors the original quickPCA.py guard.
MIN_FRAMES = 3


def coords_from_pymol(
    cmd: Any,
    obj_name: str,
    selection: str = "polymer and name CA",
) -> np.ndarray:
    """Extract per-state coordinates from a PyMOL object.

    Iterates over every state of ``obj_name`` and stacks the atom coordinates
    matching ``selection`` into a single array. States whose selection matches
    no atoms are skipped with a warning (as the original script did), rather
    than aborting the whole run.

    Parameters
    ----------
    cmd:
        A PyMOL ``cmd``-like object exposing ``count_states(obj_name)`` and
        ``get_model(selection, state=...)``. Passing a fake object makes this
        function unit-testable without PyMOL.
    obj_name:
        Name of the loaded PyMOL object.
    selection:
        Atom selection expression (default: alpha carbons of polymers).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_states, n_atoms, 3)``.

    Raises
    ------
    ValueError
        If no states are present, if no state yields any atoms, or if atom
        counts are inconsistent across the states that do match.
    """
    n_states = int(cmd.count_states(obj_name))
    if n_states < 1:
        raise ValueError(f"Object '{obj_name}' has no states to extract.")

    sele = f"({obj_name}) and ({selection})"
    frames: list[np.ndarray] = []

    for state in range(1, n_states + 1):
        model = cmd.get_model(sele, state=state)
        coords = np.array([a.coord for a in model.atom], dtype=np.float64)
        if coords.size == 0:
            print(f"   ⚠️  State {state}: selection matched no atoms — skipping.")
            continue
        frames.append(coords)

    if not frames:
        raise ValueError(f"Selection '{selection}' matched no atoms in any state of '{obj_name}'.")

    n_atoms = frames[0].shape[0]
    if any(f.shape[0] != n_atoms for f in frames):
        raise ValueError(
            "Inconsistent atom count across states; cannot stack into a coordinate array."
        )

    return np.stack(frames, axis=0)


def run_from_pymol(
    obj_name: str | None = None,
    *,
    cmd: Any | None = None,
    selection: str = "polymer and name CA",
    n_components: int = 10,
    temperature: float = 300.0,
    n_bins: int = 50,
    sigma: float = 1.0,
    backend: str = "numpy",
    output: str = "PCA_Report.png",
) -> str:
    """Run the full QuickPCA pipeline against a PyMOL object.

    Extracts coordinates from PyMOL, then delegates all analysis to the
    ``quickpca`` package: :func:`compute_pca` -> :func:`compute_fel` ->
    :func:`plot_report`.

    Parameters
    ----------
    obj_name:
        Name of the PyMOL object to analyse. If ``None``, the first loaded
        object is used.
    cmd:
        A PyMOL ``cmd``-like object. If ``None``, ``pymol.cmd`` is imported
        lazily (so this module stays importable without PyMOL).
    selection:
        Atom selection expression for coordinate extraction.
    n_components:
        Number of principal components to retain.
    temperature:
        Temperature (Kelvin) for the Boltzmann-inverted Free-Energy Landscape.
    n_bins:
        Histogram bins per axis for the Free-Energy Landscape.
    sigma:
        Gaussian smoothing sigma (in bin units) for the Free-Energy Landscape.
    backend:
        Name of a registered ``quickpca`` compute backend.
    output:
        Path of the PNG report to write.

    Returns
    -------
    str
        The ``output`` path of the saved report.

    Raises
    ------
    ValueError
        If no PyMOL objects are loaded and ``obj_name`` is ``None``, or if
        fewer than three usable frames are available for a 2-D landscape.
    """
    if cmd is None:
        from pymol import cmd as _cmd

        cmd = _cmd

    if obj_name is None:
        names = cmd.get_names("objects")
        if not names:
            raise ValueError("No objects loaded in PyMOL. Load a topology + trajectory first.")
        obj_name = names[0]

    coords = coords_from_pymol(cmd, obj_name, selection=selection)
    if coords.shape[0] < MIN_FRAMES:
        raise ValueError(
            f"Need at least {MIN_FRAMES} frames for a PC1/PC2 landscape, found {coords.shape[0]}."
        )

    pca = compute_pca(coords, n_components=n_components, backend=backend)
    fel = compute_fel(pca.projections, temperature=temperature, n_bins=n_bins, sigma=sigma)
    return plot_report(pca, fel, output=output, temperature=temperature)
