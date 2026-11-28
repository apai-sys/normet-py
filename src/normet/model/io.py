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


def _discover_model_file(folder: Path, filename: str | None) -> Path | None:
    """Resolve the one model file a directory holds, if it is unambiguous.

    ``filename`` wins when it names an existing file. Otherwise a directory
    holding exactly one ``*.joblib``/``*.pkl`` model resolves to that file, so
    a model saved under a custom name can be re-loaded from its directory.
    Discovery is *additive*: a directory with no candidate returns ``None`` and
    the caller hands ``filename`` to the backend as before (the backend decides
    or raises its own error). Several candidates raise instead of silently
    picking one.
    """
    if filename:
        named = folder / filename
        if named.is_file():
            return named
    candidates = sorted(
        f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in {".joblib", ".pkl"}
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    names = ", ".join(f.name for f in candidates)
    raise ValueError(
        f"Several model files found in '{folder}': {names}. "
        "Pass `filename=` to choose which one to load."
    )


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
        Expected filename if ``path`` is a directory. When that file is absent
        and the directory holds exactly one ``*.joblib``/``*.pkl`` model, that
        single file is loaded; a directory with several candidates raises
        ``ValueError`` unless ``filename`` names one, and an empty directory is
        left to the backend's own resolution/error. Ignored when ``path`` is a
        file. Interpretation is backend-specific.
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
        If backend is not registered, or a directory holds several model files
        and ``filename`` names none of them.
    """
    be = backend_registry.get(backend)
    # A file path is the model; ``filename`` only names the file inside a
    # directory. Passing the default "automl.joblib" unconditionally made
    # ``load_model(p)`` look for ``p/automl.joblib`` after ``p`` came back
    # from ``save_model``, so a saved model could not be loaded by its own
    # path (and a directory holding both names loaded the wrong one).
    p = Path(path)
    resolved: Path | None = None
    if p.is_file():
        filename_arg: str | None = None
    elif p.is_dir():
        # Honour ``filename`` when present, otherwise fall back to the single
        # model file the directory holds (so ``save_model(model, d, "m.joblib")``
        # round-trips through ``load_model(d)``). An empty directory keeps the
        # historical behaviour: ``filename`` goes to the backend, which decides
        # (or raises its own error).
        resolved = _discover_model_file(p, filename)
        filename_arg = resolved.name if resolved is not None else filename
    else:
        filename_arg = filename
    model = be.load(path=str(path), filename=filename_arg)
    if return_metadata:
        return model, load_model_metadata(
            path=str(resolved) if resolved is not None else path,
            filename=filename,
        )
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
