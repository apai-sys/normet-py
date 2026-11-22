# src/normet/backends/flaml_backend.py
"""FLAML AutoML backend: training, persistence, and the ``flaml`` registry entry."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils._config import DEFAULT_SEED
from ..utils._lazy import require
from ..utils.logging import get_logger
from ._shared import resolve_sample_weight

log = get_logger(__name__)

__all__ = [
    "train_flaml",
    "save_flaml",
    "load_flaml",
]


def _import_flaml_automl() -> Any:
    """Dynamically import FLAML's AutoML class."""
    AutoML = require("flaml.automl:AutoML", hint="pip install flaml")
    return AutoML


def save_flaml(
    model: Any,
    path: str | Path = ".",
    filename: str = "automl.joblib",
) -> str:
    """
    Save a FLAML AutoML model to the specified directory with the given filename.

    Parameters
    ----------
    model : AutoML
        The FLAML AutoML model to save.
    path : str | Path, default="."
        Directory path to save the model.
    filename : str, default="automl.joblib"
        Desired filename. If no extension is given, ``.joblib`` will be added.

    Returns
    -------
    str
        The path of the saved model.
    """
    joblib = require("joblib", hint="pip install joblib")  # <-- added
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)

    # Ensure extension
    if not Path(filename).suffix:
        filename = f"{filename}.joblib"

    model_path = folder / filename

    joblib.dump(model, str(model_path))
    log.info("Saved FLAML model to %s", model_path)
    return str(model_path)


def load_flaml(
    path: str | Path = ".",
    filename: str | None = None,
) -> object:
    """
    Load a FLAML model saved with ``save_flaml``.

    Resolution rules
    ----------------
    - If ``filename`` is provided, load exactly ``path/filename``.
    - Otherwise, scan ``path`` and pick the most recently modified
      ``.joblib`` or ``.pkl`` file.

    Parameters
    ----------
    path : str | Path, default "."
        Directory containing the saved model, or a file path if you pass the
        file directly via ``filename=None`` and ``path`` is a file.
    filename : str | None, optional
        Specific filename to load (e.g., "automl.joblib"). If None, auto-pick.

    Returns
    -------
    object
        The loaded FLAML AutoML object. Ensures ``backend == "flaml"``.

    Raises
    ------
    FileNotFoundError
        If no suitable model file is found.
    ImportError
        If ``joblib`` is not installed.
    """
    joblib = require("joblib", hint="pip install joblib")
    p = Path(path)

    # If user passed a direct file path via path and no filename
    if filename is None and p.is_file():
        target = p
        if target.suffix.lower() not in {".joblib", ".pkl"}:
            raise FileNotFoundError(
                f"Unsupported model file (expect .joblib/.pkl): '{target.name}'"
            )
    else:
        folder = p if p.is_dir() else p.parent
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: '{folder}'")

        if filename:
            target = folder / filename
            if not target.exists():
                raise FileNotFoundError(f"Specified model file not found: {target}")
        else:
            candidates = [
                f
                for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in {".joblib", ".pkl"}
            ]
            if not candidates:
                raise FileNotFoundError(
                    f"No FLAML model files (.joblib/.pkl) found under '{folder}'."
                )
            target = max(candidates, key=lambda f: f.stat().st_mtime)

    model = joblib.load(str(target))
    if getattr(model, "backend", None) != "flaml":
        try:
            model.backend = "flaml"
        except Exception as e:
            log.debug("Could not set .backend on loaded model: %s", e)
    log.info("Loaded FLAML model from %s", target)
    return model


def train_flaml(
    df: pd.DataFrame,
    target: str = "value",
    covariates: list[str] | None = None,
    variables: list[str] | None = None,
    model_config: dict[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
    verbose: bool = False,
    n_cores: int | None = None,
    sample_weight: str | np.ndarray | pd.Series | None = None,
) -> object:
    """
    Train a model with FLAML AutoML and tag it with ``backend='flaml'``.

    Parameters
    ----------
    df : pandas.DataFrame
        Training dataset containing predictors and target column.
        If a ``'set'`` column is present, rows with ``set == 'training'`` are used.
    target : str, default="value"
        Name of the target column.
    covariates : list of str, optional
        Predictor column names. Must be non-empty and unique.
    variables : list of str, optional
        .. deprecated::
            Use *covariates* instead.
    model_config : dict, optional
        FLAML configuration overrides. Recognized keys and **defaults**:
          - ``max_iter`` : int (default 30)
          - ``time_budget`` : int (default -1, i.e. no wall-clock bound)
          - ``metric`` : str (default "r2")
          - ``estimator_list`` : list[str] (default ["lgbm"])
          - ``task`` : {"regression","classification"} (default "regression")
          - ``eval_method`` : {"auto","cv","holdout"} (default "auto")
          - ``save_model`` : bool (default False)
          - ``path`` : str (default ".")
          - ``filename`` : str (default "automl.joblib")
          - ``verbose`` : bool (defaults to this function's ``verbose``)

        With no ``model_config`` the search is bounded by the number of
        trials (``max_iter``), so model selection is reproducible for a
        given ``seed`` and data. Passing ``time_budget`` explicitly asks
        for wall-clock search instead: the default ``max_iter`` is dropped
        and only the caller's clock applies. Passing both keeps both, and
        FLAML stops at whichever is reached first.

    seed : int, default=7654321
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to log progress.

    Returns
    -------
    object
        A trained FLAML AutoML model, tagged with ``backend="flaml"``.

    Raises
    ------
    ValueError
        If covariates are missing/empty, duplicated, or columns not found.

    .. versionchanged:: 0.3.0
        ``variables`` is deprecated in favour of ``covariates``.
    """
    if variables is not None:
        warnings.warn(
            "`variables` is deprecated, use `covariates` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if covariates is None:
            covariates = variables
    if not covariates:
        raise ValueError("`covariates` must be a non-empty list.")
    if len(covariates) != len(set(covariates)):
        raise ValueError("`covariates` contains duplicates.")
    missing = set(covariates + [target]) - set(df.columns)
    if missing:
        raise ValueError(f"Columns not found in df: {sorted(missing)}")

    # Pick training rows if a split is present
    if "set" in df.columns:
        train_mask = (df["set"] == "training").to_numpy()
        df_sub = df.loc[train_mask]
        if df_sub.empty:
            train_mask = np.ones(len(df), dtype=bool)
            df_sub = df
    else:
        train_mask = np.ones(len(df), dtype=bool)
        df_sub = df
    df_train = df_sub[[target] + covariates]

    if df_train[covariates].shape[1] == 0:
        raise ValueError("No predictor columns available after preprocessing.")

    weights_vec = resolve_sample_weight(sample_weight, df, df_sub, train_mask)

    # Defaults
    default_cfg: dict[str, Any] = {
        # Trial-bounded, not wall-clock: see the comment on ``max_iter``
        # below. 30 trials of the default LightGBM search took 14 s on a
        # 1,440-row synthetic frame and 64 s on 50,000 rows -- at or under
        # the 90 s wall-clock budget this replaces, while no longer
        # depending on how loaded the node was.
        "max_iter": 30,
        "time_budget": -1,
        "metric": "r2",
        "estimator_list": ["lgbm"],
        "task": "regression",
        "eval_method": "auto",
        "save_model": False,
        "path": ".",
        "filename": "automl.joblib",
        "verbose": verbose,
    }
    # ``split_type`` / ``n_splits`` let callers request time-ordered internal
    # validation (split_type="time"), which prevents the model-selection step
    # from being fooled by temporally-leaked random folds on autocorrelated
    # air-quality series. Only forwarded to FLAML when explicitly set.
    if n_cores is not None:
        default_cfg.setdefault("n_jobs", n_cores)
    if model_config:
        # A caller who names a wall-clock budget is asking for wall-clock
        # search; the default trial cap is not an extra bound to carry along.
        if "time_budget" in model_config and "max_iter" not in model_config:
            default_cfg.pop("max_iter", None)
        default_cfg.update(model_config)

    # Build kwargs for AutoML.fit
    #
    # ``max_iter`` caps the search by NUMBER OF TRIALS rather than by wall
    # clock. It is the default because model selection must be reproducible:
    # ``time_budget`` is wall-clock, so the same script on the same data can
    # return a different model depending on how loaded the node was, and the
    # result carries no sign that the search was cut short. The default
    # ``time_budget=-1`` disables the clock; a caller that passes
    # ``time_budget`` explicitly gets wall-clock-only search (the default
    # ``max_iter`` is dropped above), and a caller that wants both bounds
    # passes both -- FLAML stops at whichever is reached first.

    passthrough = {
        "time_budget",
        "max_iter",
        "metric",
        "estimator_list",
        "task",
        "eval_method",
        "verbose",
        "n_jobs",
        "split_type",
        "n_splits",
        "groups",
        # Bounds the per-estimator search space (e.g. capping LGBM's
        # num_leaves, which defaults to a [4, 32768] range) rather than the
        # search *effort* (time_budget/max_iter). Needed because CFO can
        # commit to a huge-leaf config early in the trial sequence -- at the
        # same best_iteration as sites that landed on a tiny model -- so
        # shrinking the budget does not reliably shrink the result; only
        # narrowing the space itself does. See flaml.AutoML.fit's own
        # docstring for the {"estimator": {"param": {"domain": ...}}} shape.
        "custom_hp",
    }
    _backend_only_keys = {"save_model", "path", "filename", "cv_strategy", "cv_folds"}
    _ignored = set(model_config or {}) - passthrough - _backend_only_keys
    if _ignored:
        log.warning(
            "FLAML backend ignoring unsupported model_config key(s): %s",
            ", ".join(sorted(_ignored)),
        )
    automl_kwargs = {k: default_cfg[k] for k in passthrough if k in default_cfg}

    X_tr = df_train[covariates]
    y_tr = df_train[target]
    cv_strategy = str((model_config or {}).get("cv_strategy", "auto")).lower()
    if cv_strategy not in {"auto", "random", "blocked", "time"}:
        raise ValueError(
            f"Unknown FLAML `cv_strategy` {cv_strategy!r}; expected 'auto', 'random', 'blocked', or 'time'."
        )
    if cv_strategy in {"blocked", "time"} and "split_type" not in (model_config or {}):
        if "date" in df_sub.columns and pd.api.types.is_datetime64_any_dtype(df_sub["date"]):
            sort_pos = np.argsort(df_sub["date"].to_numpy(), kind="stable")
            X_tr = X_tr.iloc[sort_pos].reset_index(drop=True)
            y_tr = y_tr.iloc[sort_pos].reset_index(drop=True)
            if weights_vec is not None:
                weights_vec = weights_vec[sort_pos]
        if cv_strategy == "blocked" and len(X_tr) >= 2:
            n_sp = int(default_cfg.get("n_splits") or default_cfg.get("cv_folds") or 5)
            n_sp = max(2, min(n_sp, len(X_tr)))
            groups = np.empty(len(X_tr), dtype=np.int64)
            for gid, blk in enumerate(np.array_split(np.arange(len(X_tr)), n_sp)):
                groups[blk] = gid
            automl_kwargs["split_type"] = "group"
            automl_kwargs["eval_method"] = "cv"
            automl_kwargs["n_splits"] = n_sp
            automl_kwargs["groups"] = groups
        elif cv_strategy == "time":
            automl_kwargs["split_type"] = "time"
    if weights_vec is not None:
        automl_kwargs["sample_weight"] = weights_vec

    AutoML = _import_flaml_automl()
    automl = AutoML()

    (log.info if verbose else log.debug)(
        "Training FLAML AutoML: X shape=%s, target='%s'", X_tr.shape, target
    )

    automl.fit(
        X_train=X_tr,
        y_train=y_tr,
        seed=seed,
        **automl_kwargs,
    )

    (log.info if verbose else log.debug)(
        "FLAML best_estimator=%s | best_config=%s", automl.best_estimator, automl.best_config
    )

    # Optional persistence
    if default_cfg.get("save_model", False):
        save_flaml(
            automl,
            path=default_cfg.get("path", "."),
            filename=default_cfg.get("filename", "automl.joblib"),
        )

    automl.backend = "flaml"
    return automl


class _FlamlBackend:
    """Backend-compatible wrapper around the FLAML-specific functions."""

    name = "flaml"

    def train(
        self,
        df: pd.DataFrame,
        target: str = "value",
        covariates: list[str] | None = None,
        variables: list[str] | None = None,
        model_config: dict[str, Any] | None = None,
        seed: int = DEFAULT_SEED,
        verbose: bool = False,
        n_cores: int | None = None,
        sample_weight: str | np.ndarray | pd.Series | None = None,
    ) -> object:
        return train_flaml(
            df,
            target=target,
            covariates=covariates,
            variables=variables,
            model_config=model_config,
            seed=seed,
            verbose=verbose,
            n_cores=n_cores,
            sample_weight=sample_weight,
        )

    def save(
        self,
        model: object,
        path: str = ".",
        filename: str = "automl.joblib",
    ) -> str:
        return save_flaml(model, path=path, filename=filename)

    def load(
        self,
        path: str = ".",
        filename: str | None = None,
    ) -> object:
        return load_flaml(path=path, filename=filename)


backend = _FlamlBackend()
