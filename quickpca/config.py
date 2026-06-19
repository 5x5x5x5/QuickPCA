"""Default configuration for QuickPCA report generation."""

from __future__ import annotations

from dataclasses import dataclass

# Module-level constants (also the defaults of :class:`ReportConfig`).
SELECTION: str = "name CA"
N_COMPONENTS: int = 10
N_BINS: int = 50
SIGMA: float = 1.0
TEMPERATURE: float = 300.0
INTERVAL: int = 1
BACKEND: str = "numpy"
OUTPUT: str = "PCA_Report.png"


@dataclass
class ReportConfig:
    """Tunable parameters for a full PCA/FEL report run."""

    selection: str = SELECTION
    n_components: int = N_COMPONENTS
    n_bins: int = N_BINS
    sigma: float = SIGMA
    temperature: float = TEMPERATURE
    interval: int = INTERVAL
    backend: str = BACKEND
    output: str = OUTPUT
