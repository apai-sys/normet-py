# src/normet/utils/provenance.py
"""
Lightweight provenance wrappers.

``NormetRun`` bundles a result, the trained model (optional), the prepared
data (optional), and a metadata dict describing *how* the result was produced.
Use :func:`make_run` to construct one, and :func:`save_run` / :func:`load_run`
to persist a runnable archive (joblib pickle + JSON sidecar).

This module never touches existing returns — existing pipelines (`do_all`,
`normalise`, etc.) keep their tuple/DataFrame shapes. Users opt in.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ._lazy import require
from .cache import config_hash, dataframe_hash
from .logging import get_logger

log = get_logger(__name__)

__all__ = ["NormetRun", "make_run", "save_run", "load_run"]


def _normet_version() -> str:
    try:
        from importlib.metadata import version  # py3.8+

        return version("normet")
    except Exception:
        return "0.0.0+unknown"


@dataclass
class NormetRun:
    """
    Container bundling a result, optional artefacts, and provenance metadata.

    Attributes
    ----------
    result : Any
        Primary output — typically a ``pandas.DataFrame``.
    model : object, optional
        Trained model.
    df_prep : pandas.DataFrame, optional
        Prepared/featured data the model was trained on.
    metadata : dict
        Provenance dictionary; see :func:`make_run`.
    """

    result: Any
    model: Any | None = None
    df_prep: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a short human-readable summary."""
        kind = self.metadata.get("kind", "run")
        ts = self.metadata.get("timestamp", "?")
        ver = self.metadata.get("normet_version", "?")
        return f"<NormetRun kind={kind} normet={ver} at={ts}>"


def make_run(
    result: Any,
    *,
    kind: str,
    config: dict[str, Any] | None = None,
    df: pd.DataFrame | None = None,
    df_prep: pd.DataFrame | None = None,
    model: Any | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> NormetRun:
    """
    Build a :class:`NormetRun` with auto-filled provenance metadata.

    Parameters
    ----------
    result : Any
        The result you want to archive (e.g., the normalised DataFrame).
    kind : str
        Short label describing the pipeline (e.g., "do_all", "normalise", "scm").
    config : dict, optional
        Configuration that produced the result (model_config, n_samples, etc.).
        Hashed into ``metadata.config_hash`` for cache keys.
    df : pandas.DataFrame, optional
        Input data; its content hash is stored as ``metadata.data_hash``.
    df_prep : pandas.DataFrame, optional
        Stored on the run for downstream reuse (e.g., modStats).
    model : object, optional
        Trained model object; stored on the run.
    seed : int, optional
        Random seed used. Stored verbatim in metadata.
    extra : dict, optional
        Free-form additions merged into ``metadata.extra``.

    Returns
    -------
    NormetRun
    """
    meta: dict[str, Any] = {
        "kind": kind,
        "normet_version": _normet_version(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
    }
    if df is not None:
        try:
            meta["data_hash"] = dataframe_hash(df)
            meta["data_shape"] = list(df.shape)
        except Exception as e:
            log.debug("data_hash failed: %s", e)
    if config is not None:
        meta["config"] = config
        try:
            meta["config_hash"] = config_hash(config)
        except Exception as e:
            log.debug("config_hash failed: %s", e)
    if extra:
        meta["extra"] = dict(extra)
    return NormetRun(result=result, model=model, df_prep=df_prep, metadata=meta)


def _coerce_json_safe(obj: Any) -> Any:
    """Best-effort conversion of metadata into JSON-serialisable primitives."""
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    if isinstance(obj, list | tuple):
        return [_coerce_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _coerce_json_safe(v) for k, v in obj.items()}
    return repr(obj)


def save_run(
    run: NormetRun,
    path: str | Path,
    *,
    compress: int = 3,
) -> dict[str, str]:
    """
    Persist a :class:`NormetRun` to disk.

    Layout: ``{path}.joblib`` + ``{path}.meta.json`` sidecar.

    Parameters
    ----------
    run : NormetRun
        Run to save.
    path : str | Path
        Base path; suffixes are appended automatically.
    compress : int, default 3
        Forwarded to :func:`joblib.dump` (0 disables compression).

    Returns
    -------
    dict
        Mapping ``{"artifact": ..., "metadata": ...}`` with the saved paths.
    """
    joblib = require("joblib", hint="pip install joblib")
    p = Path(path)
    if p.suffix == ".joblib":
        p = p.with_suffix("")
    p.parent.mkdir(parents=True, exist_ok=True)
    art = p.with_suffix(".joblib")
    meta = p.with_suffix(".meta.json")

    joblib.dump(run, str(art), compress=compress)
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(_coerce_json_safe(run.metadata), f, indent=2, sort_keys=True)
    log.info("Saved NormetRun → %s (+ %s)", art, meta.name)
    return {"artifact": str(art), "metadata": str(meta)}


def load_run(path: str | Path) -> NormetRun:
    """
    Load a :class:`NormetRun` previously saved via :func:`save_run`.

    Parameters
    ----------
    path : str | Path
        Either the ``.joblib`` artefact path or the base path used at save.

    Returns
    -------
    NormetRun
    """
    joblib = require("joblib", hint="pip install joblib")
    p = Path(path)
    art = p if p.suffix == ".joblib" else p.with_suffix(".joblib")
    if not art.exists():
        raise FileNotFoundError(f"NormetRun artifact not found: {art}")
    run = joblib.load(str(art))
    if not isinstance(run, NormetRun):
        raise TypeError(f"Loaded object is not a NormetRun: {type(run).__name__}")
    return run
