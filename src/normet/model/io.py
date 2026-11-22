# src/normet/model/io.py
"""Save and load trained models via the registered AutoML backend."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..backends import backend_registry
from ..utils.features import model_feature_names
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["load_model", "load_model_metadata", "save_model"]


def _metadata_path(model_file: Path) -> Path:
    return model_file.with_suffix(model_file.suffix + ".meta.json")


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "tolist"):
        return _json_safe(obj.tolist())
    if hasattr(obj, "item"):
        return _json_safe(obj.item())
    return repr(obj)


def _resolve_model_file(path: str | Path, filename: str | None) -> Path | None:
    p = Path(path)
    if p.is_file():
        return p
    folder = p if p.is_dir() else p.parent
    if not folder.exists():
        return None
    if filename:
        cand = folder / filename
        return cand if cand.exists() else None
    candidates = [
        f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in {".joblib", ".pkl"}
    ]
    return max(candidates, key=lambda f: f.stat().st_mtime) if candidates else None


def load_model_metadata(
    path: str | Path = ".",
    filename: str = "automl.joblib",
) -> dict[str, Any] | None:
    """Load the JSON metadata sidecar written beside a saved model, if present."""
    p = Path(path)
    if p.is_file() and str(p).endswith(".meta.json"):
        meta_file = p
    else:
        resolved = _resolve_model_file(p, None if p.is_file() else filename)
        if resolved is None:
            return None
        meta_file = _metadata_path(resolved)
    if not meta_file.is_file():
        return None
    try:
        raw = json.loads(meta_file.read_text(encoding="utf-8"))
        return dict(raw) if isinstance(raw, dict) else {"metadata": raw}
    except Exception as exc:
        log.warning("Could not parse model metadata file %s: %s", meta_file, exc)
        return None


# -------------------------
# Public API
# -------------------------
def load_model(
    path: str | Path = ".",
    backend: str = "flaml",
    filename: str = "automl.joblib",
    *,
    return_metadata: bool = False,
) -> Any:
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
    return_metadata : bool, default False
        If True, return ``(model, metadata_dict_or_none)`` by also reading the
        adjacent ``.meta.json`` sidecar when available.

    Returns
    -------
    object or tuple
        The loaded model, or ``(model, metadata)`` if ``return_metadata=True``.

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
    model = be.load(path=str(path), filename=filename_arg)
    if return_metadata:
        return model, load_model_metadata(path=path, filename=filename)
    return model


def save_model(
    model: object,
    path: str | Path = ".",
    filename: str = "automl.joblib",
    *,
    metadata: Mapping[str, Any] | None = None,
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
    metadata : mapping, optional
        If provided, write a structured JSON sidecar alongside the saved model
        recording ``normet_version``, ``backend``, ``saved_at``, ``features``,
        and caller metadata (e.g. ``log_transform``, ``smearing_factor``,
        metrics).

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
    saved_path = be.save(model, path=str(path), filename=filename)

    if metadata is not None:
        from ..utils.cache import package_version

        try:
            feats: list[str] | None = model_feature_names(model)
        except Exception:
            feats = None
        payload: dict[str, Any] = {
            "normet_version": package_version(),
            "backend": name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "features": feats,
            "metadata": _json_safe(dict(metadata)),
        }
        meta_file = _metadata_path(Path(saved_path))
        meta_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Saved model metadata sidecar to %s", meta_file)

    return saved_path
