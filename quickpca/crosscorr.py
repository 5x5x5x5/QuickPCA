"""Residue cross-correlation matrix computed analytically from PCA modes."""

from __future__ import annotations

import numpy as np


def cross_correlation(
    components: np.ndarray,
    explained_variance: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """Compute the (n_atoms, n_atoms) cross-correlation matrix.

    Reconstructs the displacement covariance from the eigenvectors weighted by
    the absolute eigenvalues, then normalises to correlation form.

    Parameters
    ----------
    components:
        PCA eigenvectors of shape ``(n_components, 3 * n_atoms)``.
    explained_variance:
        Eigenvalues of shape ``(n_components,)``.
    n_atoms:
        Number of atoms.

    Returns
    -------
    np.ndarray
        Symmetric ``float32`` matrix with diagonal ~1.
    """
    components = np.asarray(components, dtype=np.float64)
    eigs = np.asarray(explained_variance, dtype=np.float64)
    n_comp = components.shape[0]

    evecs_3d = components.reshape(n_comp, n_atoms, 3)
    cov = np.einsum("kia,kja,k->ij", evecs_3d, evecs_3d, np.abs(eigs))
    var = np.diag(cov)
    denom = np.sqrt(np.outer(var, var))
    cc = np.where(denom > 0, cov / denom, 0.0).astype(np.float32)
    return cc
