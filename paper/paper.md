---
title: "QuickPCA: GPU-accelerated essential dynamics analysis of molecular dynamics trajectories"
tags:
  - Python
  - molecular dynamics
  - PCA
  - essential dynamics
  - JAX
authors:
  - name: Gleb Novikov
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Affiliation placeholder
    index: 1
date: 19 June 2026
bibliography: paper.bib
---

# Summary

Molecular dynamics (MD) simulations produce high-dimensional trajectories in
which a small number of collective, large-amplitude motions typically encode the
functionally relevant conformational changes of a biomolecule. Essential dynamics
analysis [@amadei1993] isolates these motions by applying principal component
analysis (PCA) to the time series of atomic coordinates, separating the dominant
"essential" subspace from high-frequency thermal noise. `QuickPCA` is a
lightweight, headless Python package that performs essential dynamics analysis on
MD trajectories and produces a publication-ready report comprising a
Boltzmann-inverted free-energy landscape, a residue cross-correlation map, an
explained-variance profile, and the distributions of the leading principal
component projections.

`QuickPCA` is built around singular-value decomposition (SVD) PCA applied
directly to the `(n_frames × 3N)` coordinate matrix, Kabsch rigid-body alignment
[@kabsch1976], an analytic reconstruction of the residue cross-correlation matrix
from the PCA modes, and Boltzmann inversion of the principal-component density to
obtain a two-dimensional free-energy landscape. Numerically heavy primitives are
isolated behind a pluggable backend interface, so the same pipeline can run on a
pure-NumPy [@harris2020] reference backend or, where a GPU is available, on a JAX
backend [@jax2018] without any change to user code. Trajectory input is handled
through MDAnalysis [@michaud-agrawal2011; @gowers2016], and the package provides a
command-line interface alongside the original PyMOL drag-and-drop workflow.

# Statement of need

Essential dynamics is a standard step in the interpretation of MD simulations,
yet much of the tooling that practitioners reach for is tied to interactive
graphical environments. The original `QuickPCA` was distributed as a single
script that runs inside the PyMOL GUI: the user drops the script into a session
and a report is generated for the loaded object. This is convenient at the bench
but ill-suited to modern computational workflows. In-GUI scripts cannot be
installed with `pip`, are awkward to version and test, do not run on headless
compute nodes or in continuous-integration pipelines, and are difficult to embed
inside larger analysis scripts that batch-process many trajectories.

`QuickPCA` addresses these gaps by repackaging the analysis as a proper,
pip-installable Python library with a clean public API, a command-line entry
point, and a typed, tested codebase. Because the heavy linear algebra is hidden
behind a backend abstraction, the compute engine is selectable at run time: the
default NumPy backend [@harris2020] runs anywhere, and the same interface admits
a JAX backend [@jax2018] that can offload alignment and decomposition to a GPU
with no change to the analysis code. The decomposition is performed by SVD directly on the
coordinate matrix rather than by diagonalising the covariance matrix, mirroring
the full-SVD solver of scikit-learn [@pedregosa2011]; this avoids forming and
diagonalising the large `3N × 3N` covariance matrix, which is both faster and
more numerically stable while producing identical principal components. The
combination of a headless, scriptable interface, a fast and pluggable compute
backend, and a single command that emits a complete report makes `QuickPCA`
useful for high-throughput screening of trajectory ensembles, for reproducible
analysis in automated pipelines, and as a teaching tool for essential dynamics.

# Implementation

`QuickPCA` is organised as a small set of composable modules orchestrated by a
high-level pipeline.

**SVD-based PCA.** Aligned frames are flattened into an `(n_frames × 3N)` matrix
that is mean-centred and decomposed by full SVD. Principal components,
explained-variance ratios, and projections are recovered directly from the
singular vectors and values, reproducing the results of the scikit-learn
`PCA(svd_solver="full")` solver [@pedregosa2011], including its sign convention,
without explicitly building the covariance matrix.

**Kabsch alignment.** Prior to decomposition, every frame is superposed onto a
reference frame using the Kabsch algorithm [@kabsch1976]. Each frame is centred
on its own centroid and optimally rotated onto the reference, with reflections
corrected through the sign of the determinant of the rotation matrix, so that
only internal motion contributes to the principal components.

**Analytic cross-correlation.** The residue cross-correlation matrix is
reconstructed analytically from the PCA eigenvectors weighted by their absolute
eigenvalues, rather than by a second pass over the raw trajectory. The
displacement covariance is assembled from the retained modes and normalised to
correlation form, yielding a symmetric per-residue map of correlated and
anti-correlated motions.

**Free-energy landscape.** A two-dimensional free-energy landscape is obtained by
Boltzmann inversion of the smoothed density of the first two principal-component
projections, `F = -k_B T ln(rho)`, shifted so that the global minimum is zero.
The density is estimated on a regular grid and Gaussian-smoothed, giving an
interpretable map of metastable basins along the essential coordinates.

**Pluggable backends.** The two numerically heavy primitives, Kabsch alignment
and full-SVD PCA, are defined by an abstract backend protocol. A pure-NumPy
backend [@harris2020] serves as the reference implementation. The same protocol
accommodates an optional JAX backend [@jax2018], declared as an extras
dependency, that targets GPU and TPU hardware and just-in-time compilation for
the same two primitives. Backends self-register on import, so the compute engine
can be selected by name from the CLI or the API.

**Trajectory I/O and interfaces.** Trajectories are read through MDAnalysis
[@michaud-agrawal2011; @gowers2016], which supports common topology and
trajectory formats and atom-selection strings, with configurable frame striding.
The package exposes a command-line interface with an extensible subcommand
mechanism and a headless matplotlib [@hunter2007] report renderer that produces
the four-panel figure. The original PyMOL drag-and-drop script is retained
alongside the library, preserving the in-GUI workflow for interactive users.

# Acknowledgements

`QuickPCA` was developed by Gleb Novikov at The Visual Hub. We acknowledge the
open-source scientific Python ecosystem, in particular NumPy [@harris2020],
scikit-learn [@pedregosa2011], matplotlib [@hunter2007], MDAnalysis
[@michaud-agrawal2011; @gowers2016], and JAX [@jax2018], on which this work
builds.

# References
