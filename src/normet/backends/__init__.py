"""AutoML modeling backend interfaces.

Defines the ``Backend`` protocol and a global ``backend_registry`` that
consumers (``model/train.py``, ``model/io.py``, ``model/predict.py``) use
to dispatch training, persistence, and prediction without hard-coded
backend names.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from ..utils._config import DEFAULT_SEED

# ---------------------------------------------------------------------------
# Protocol  (what a backend must provide)
# ---------------------------------------------------------------------------


@runtime_checkable
class Backend(Protocol):
    """Interface every normet backend must satisfy.

    A backend wraps three operations — *train*, *save*, *load* — for one
    AutoML framework (e.g. FLAML).  The ``.name`` attribute is used as the
    lookup key in :class:`BackendRegistry`.
    """

    name: str

    def train(
        self,
        df: pd.DataFrame,
        value: str = "value",
        feature_names: list[str] | None = None,
        variables: list[str] | None = None,
        model_config: dict[str, Any] | None = None,
        seed: int = DEFAULT_SEED,
        verbose: bool = False,
        n_cores: int | None = None,
    ) -> object:
        """Train a model and return it with a ``.backend`` attribute set.

        Parameters
        ----------
        feature_names : list of str, optional
            Names of predictor columns in *df*.
        variables : list of str, optional
            .. deprecated::
                Use *feature_names* instead.

        .. versionchanged:: 0.3.0
            ``variables`` is deprecated in favour of ``feature_names``.
        """

    def save(
        self,
        model: object,
        path: str = ".",
        filename: str = "automl.joblib",
    ) -> str:
        """Persist a trained model to disk."""

    def load(
        self,
        path: str = ".",
        filename: str | None = None,
    ) -> object:
        """Load a previously saved model from disk."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class BackendRegistry:
    """Global registry of available :class:`Backend` implementations."""

    def __init__(self) -> None:
        self._backends: dict[str, Backend] = {}

    def register(self, backend: Backend) -> None:
        """Register a Backend-compatible object.

        Raises
        ------
        TypeError
            If *backend* does not satisfy the :class:`Backend` protocol.
        """
        if not isinstance(backend, Backend):
            raise TypeError(
                f"Expected a Backend, got {type(backend).__name__}. "
                f"Make sure the object has 'name', 'train', 'save', and 'load'."
            )
        self._backends[backend.name.lower()] = backend

    def get(self, name: str) -> Backend:
        """Look up a backend by name (case-insensitive).

        Raises
        ------
        ValueError
            If no backend with that name is registered.
        """
        key = name.lower()
        if key not in self._backends:
            available = ", ".join(sorted(self._backends))
            raise ValueError(f"Unknown backend '{name}'. Available: {available}")
        return self._backends[key]

    def has(self, name: str) -> bool:
        """Check whether a backend is registered."""
        return name.lower() in self._backends

    @property
    def available(self) -> list[str]:
        """Sorted list of registered backend names."""
        return sorted(self._backends)


# ---------------------------------------------------------------------------
# Global singleton — populated at import time
# ---------------------------------------------------------------------------

backend_registry = BackendRegistry()

from .flaml_backend import backend as _flaml_backend  # noqa: E402

backend_registry.register(_flaml_backend)

from .lgb_backend import backend as _lgb_backend  # noqa: E402

backend_registry.register(_lgb_backend)

__all__ = [
    "Backend",
    "BackendRegistry",
    "backend_registry",
    "train_flaml",
    "save_flaml",
    "load_flaml",
    "train_lgb",
    "save_lgb",
    "load_lgb",
    "LgbModel",
]

# Re-export the low-level functions for anyone that still uses them directly
from .flaml_backend import load_flaml, save_flaml, train_flaml  # noqa: E402, F811
from .lgb_backend import LgbModel, load_lgb, save_lgb, train_lgb  # noqa: E402, F811
