# src/normet/model/io.py
"""Save and load trained models via the registered AutoML backend."""

from __future__ import annotations

from pathlib import Path

from ..backends import backend_registry
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["load_model", "save_model"]


# -------------------------
# Public API
# -------------------------
def load_model(
    path: str | Path = ".",
    backend: str = "flaml",
    filename: str = "automl.joblib",
) -> object:
    """
    Load a previously saved model.

    Parameters
    ----------
    path : str | pathlib.Path, default="."
        Path to the saved model file or directory.
        Interpretation is backend-specific.
    backend : str, default="flaml"
        Backend name registered in :data:`backend_registry`.
    filename : str, default "automl.joblib"
        Expected filename if ``path`` is a directory; ignored when ``path``
        is a file. Interpretation is backend-specific.

    Returns
    -------
    object
        The loaded model.

    Raises
    ------
    FileNotFoundError
        If no suitable model file is found.
    ValueError
        If backend is not registered.
    """
    be = backend_registry.get(backend)
    # A file path is the model; ``filename`` only names the file inside a
    # directory. Passing the default "automl.joblib" unconditionally made
    # ``load_model(p)`` look for ``p/automl.joblib`` after ``p`` came back
    # from ``save_model``, so a saved model could not be loaded by its own
    # path (and a directory holding both names loaded the wrong one).
    filename_arg: str | None = None if Path(path).is_file() else filename
    return be.load(path=str(path), filename=filename_arg)


def save_model(
    model: object,
    path: str | Path = ".",
    filename: str = "automl.joblib",
) -> str:
    """
    Save a trained model by delegating to the appropriate backend saver.

    Parameters
    ----------
    model : object
        Trained model with a ``backend`` attribute set to a registered
        backend name.
    path : str | Path, default="."
        Destination directory. Created if it does not exist.
    filename : str, default="automl.joblib"
        Output filename. Interpretation is backend-specific.

    Returns
    -------
    str
        Path to the saved artifact.

    Raises
    ------
    AttributeError
        If model does not define a ``backend`` attribute.
    ValueError
        If backend is not registered.
    """
    model_backend = getattr(model, "backend", None)
    if model_backend is None:
        raise AttributeError("Model must have a 'backend' attribute matching a registered backend.")

    name = str(model_backend).lower()
    be = backend_registry.get(name)
    return be.save(model, path=str(path), filename=filename)
