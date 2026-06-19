"""Backend registry for QuickPCA.

Backends self-register via the :func:`register_backend` decorator. Discovery
does not rely on entry points; instead built-in backends are imported here so
they register on package import. The (optional) JAX backend is imported
opportunistically so that simply dropping ``jax_backend.py`` into this package
makes it available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .base import Backend

_REGISTRY: dict[str, type] = {}


def register_backend(cls):
    """Class decorator registering a :class:`Backend` subclass by ``cls.name``."""
    _REGISTRY[cls.name] = cls
    return cls


def get_backend(name: str = "numpy") -> Backend:
    """Return an instance of the registered backend ``name``.

    Raises
    ------
    ValueError
        If ``name`` is not registered, listing the available backends.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown backend {name!r}. Available backends: {available_backends()}"
        ) from None
    return cls()


def available_backends() -> list[str]:
    """Return the sorted list of registered backend names."""
    return sorted(_REGISTRY)


# Import built-ins so they self-register via @register_backend.
from .numpy_backend import NumpyBackend  # noqa: E402,F401

# Attempt to import the optional JAX backend (added later by the JAX worker).
try:  # pragma: no cover - optional dependency
    from . import jax_backend  # type: ignore[attr-defined]  # noqa: F401
except Exception:  # pragma: no cover
    pass
