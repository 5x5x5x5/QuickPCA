"""JAX compute backend.

A GPU/TPU-ready drop-in for the NumPy reference backend. It performs the same
jit-compiled Kabsch alignment (vectorised over frames with :func:`jax.vmap`)
and full-SVD PCA, applying the identical sklearn ``svd_flip`` sign convention so
its outputs match :class:`~quickpca.backends.numpy_backend.NumpyBackend` within
floating-point tolerance.

JAX is imported lazily inside this module; the package backend registry guards
its absence with a ``try``/``except`` import, so simply dropping this file in
makes the ``"jax"`` backend available when JAX is installed.

Author: Gleb Novikov — The Visual Hub. MIT licensed.
"""

from __future__ import annotations

from functools import partial

import jax

# Double precision is required so results match the float64 NumPy backend
# within tolerance; JAX defaults to float32 otherwise. This must be set before
# any array is created. It is a process-wide flag, but enabling it is a no-op
# when already on and never reduces precision elsewhere.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ..types import PCADecomposition  # noqa: E402
from . import register_backend  # noqa: E402
from .base import Backend  # noqa: E402


@jax.jit
def _align_single(coords: jax.Array, ref_centered: jax.Array, ref_com: jax.Array) -> jax.Array:
    """Kabsch-align one frame onto a pre-centred reference.

    Parameters
    ----------
    coords:
        Coordinates of a single frame, shape ``(N, 3)``.
    ref_centered:
        Reference coordinates centred on their own centroid, shape ``(N, 3)``.
    ref_com:
        Centroid of the reference, shape ``(3,)``.

    Returns
    -------
    jax.Array
        The aligned frame of shape ``(N, 3)``.
    """
    coords_centered = coords - coords.mean(axis=0)
    h = coords_centered.T @ ref_centered
    u, _, vt = jnp.linalg.svd(h)
    # Reflection correction: flip the sign of the last column when needed.
    d = jnp.sign(jnp.linalg.det(vt.T @ u.T))
    diag = jnp.diag(jnp.array([1.0, 1.0, d], dtype=coords.dtype))
    rot = vt.T @ diag @ u.T
    return coords_centered @ rot.T + ref_com


@jax.jit
def _align_frames(frames: jax.Array, ref: jax.Array) -> jax.Array:
    """Kabsch-align every frame onto ``ref`` (vmapped over frames)."""
    ref_com = ref.mean(axis=0)
    ref_centered = ref - ref_com
    return jax.vmap(_align_single, in_axes=(0, None, None))(frames, ref_centered, ref_com)


@partial(jax.jit, static_argnums=(1,))
def _pca(
    x: jax.Array, n_components: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Full-SVD PCA with sklearn ``svd_flip`` sign convention.

    Parameters
    ----------
    x:
        Raw (not pre-centred) data of shape ``(F, D)``.
    n_components:
        Number of leading components to keep.

    Returns
    -------
    tuple of jax.Array
        ``(projections, components, explained_variance,
        explained_variance_ratio, mean)``.
    """
    n_samples = x.shape[0]
    mean = x.mean(axis=0)
    x_centered = x - mean

    # Full (economy-sized) SVD: x_centered = u @ diag(s) @ vt.
    u, s, vt = jnp.linalg.svd(x_centered, full_matrices=False)

    # Sign convention matching sklearn (svd_flip on u/v columns): pick the sign
    # of the entry with the largest absolute value in each column of u.
    max_abs_cols = jnp.argmax(jnp.abs(u), axis=0)
    signs = jnp.sign(u[max_abs_cols, jnp.arange(u.shape[1])])
    vt = vt * signs[:, jnp.newaxis]

    components = vt[:n_components]
    explained_variance = (s**2) / (n_samples - 1)
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var

    projections = (x_centered @ components.T)[:, :n_components]

    return (
        projections,
        components,
        explained_variance[:n_components],
        explained_variance_ratio[:n_components],
        mean,
    )


@register_backend
class JaxBackend(Backend):
    """JAX compute backend (GPU/TPU-ready, runs on CPU by default).

    Numerically equivalent to
    :class:`~quickpca.backends.numpy_backend.NumpyBackend`: the same Kabsch
    alignment and ``PCA(svd_solver="full")``-style decomposition, with the
    identical ``svd_flip`` sign convention. All heavy kernels are jit-compiled
    and run on JAX's default device; results are returned as NumPy ``float64``
    arrays at the public boundary.
    """

    name = "jax"

    def align_frames(self, frames: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Kabsch-align each frame onto ``ref``.

        Each frame is centred on its own centroid, rotated onto ``ref`` (centred
        on the reference centroid), then translated back to the reference
        centroid. Reflections are corrected via the sign of the determinant.

        Parameters
        ----------
        frames:
            Array of shape ``(F, N, 3)``.
        ref:
            Reference coordinates of shape ``(N, 3)``.

        Returns
        -------
        np.ndarray
            Aligned frames of shape ``(F, N, 3)`` as ``float64``.
        """
        frames_j = jnp.asarray(frames, dtype=jnp.float64)
        ref_j = jnp.asarray(ref, dtype=jnp.float64)
        aligned = _align_frames(frames_j, ref_j)
        return np.asarray(aligned, dtype=np.float64)

    def pca(self, X: np.ndarray, n_components: int) -> PCADecomposition:
        """Full-SVD PCA equivalent to sklearn ``PCA(svd_solver="full")``.

        Parameters
        ----------
        X:
            Raw (not pre-centred) data of shape ``(F, D)``.
        n_components:
            Number of leading components to retain.

        Returns
        -------
        PCADecomposition
            Decomposition with NumPy ``float64`` arrays.
        """
        x_j = jnp.asarray(X, dtype=jnp.float64)
        (
            projections,
            components,
            explained_variance,
            explained_variance_ratio,
            mean,
        ) = _pca(x_j, int(n_components))

        return PCADecomposition(
            projections=np.asarray(projections, dtype=np.float64),
            components=np.asarray(components, dtype=np.float64),
            explained_variance=np.asarray(explained_variance, dtype=np.float64),
            explained_variance_ratio=np.asarray(explained_variance_ratio, dtype=np.float64),
            mean=np.asarray(mean, dtype=np.float64),
        )
