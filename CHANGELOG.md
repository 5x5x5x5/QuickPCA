# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-06-19

### Added

- Headless `quickpca` package refactored from the original PyMOL script, usable
  without a GUI.
- Pluggable compute backends with a NumPy default and an optional JAX backend,
  selectable via `register_backend` / `get_backend`.
- MDAnalysis-based trajectory loader supporting topology/trajectory inputs and
  atom selections.
- Command-line interface with auto-discovered subcommands (`run`, `backends`)
  and a drop-in `quickpca/commands/` package.
- Essential-dynamics PCA, cross-correlation, and Free-Energy Landscape
  computations with plotted reports.
- Documentation site and continuous-integration pipeline.

[Unreleased]: https://github.com/5x5x5x5/quickpca/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/5x5x5x5/quickpca/releases/tag/v1.0.0
