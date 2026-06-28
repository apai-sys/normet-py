# src/normet/analysis/pdp.py
"""Partial dependence: :func:`pdp` (single feature) and :func:`pdp_grid`."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..backends import backend_registry
from ..model.predict import ml_predict
from ..utils.features import extract_features
from ..utils.logging import get_logger

log = get_logger(__name__)


def pdp(
    df: pd.DataFrame,
    model: object,
    *,
    var_list: list[str] | None = None,
    training_only: bool = True,
    n_cores: int | None = None,
    grid_points: int = 50,
    quantile_range: tuple[float, float] = (0.01, 0.99),
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute Partial Dependence values for one or more features.

    Works with any registered backend (e.g. ``'flaml'``, ``'lightgbm'``)
    via the shared :func:`ml_predict` interface.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing features (and optionally a 'set' column == 'training').
    model : object
        Trained model with a `backend` attribute matching a registered
        backend and a predict interface supported by `ml_predict`.
    var_list : List[str] | None
        Variables to compute PDP for. If None, use model feature names.
    training_only : bool, default True
        If True and df has a 'set' column, use only rows with `set == "training"`.
    n_cores : int | None
        Parallel workers. Default: all cores - 1.
    grid_points : int, default 50
        Number of evaluation points on the value grid.
    quantile_range : (float, float), default (0.01, 0.99)
        Range of the feature values to cover.
    verbose : bool, default False
        If True, emit info logs.

    Returns
    -------
    pandas.DataFrame
        Columns: ['variable', 'value', 'pdp_mean', 'pdp_std'].
    """
    # --- model type guard ---
    model_type = getattr(model, "backend", None)
    if not model_type or not backend_registry.has(model_type):
        available = backend_registry.available
        raise TypeError(
            f"Unsupported model backend '{model_type}'. Expected one of: {', '.join(available)}"
        )

    # --- resolve features ---
    try:
        feature_names = [str(c) for c in extract_features(model)]
    except Exception as exc:
        if var_list:
            feature_names = [str(c) for c in var_list if str(c) in df.columns]
        else:
            raise ValueError("Cannot infer model features; please provide `var_list`.") from exc

    feature_names = [c for c in feature_names if c in df.columns]
    if not feature_names:
        raise ValueError("No valid model features present in `df`.")

    # PDP target variables
    if var_list is None:
        vars_for_pdp = feature_names
    else:
        vars_for_pdp = [v for v in var_list if v in feature_names]
        missing = [v for v in var_list if v not in feature_names]
        if missing:
            (log.info if verbose else log.debug)(
                "Skipping vars not present in features: %s", missing
            )

    # --- choose data subset ---
    if "set" in df.columns and training_only:
        X_df = df.loc[df["set"] == "training", feature_names].copy()
        if X_df.empty:
            X_df = df[feature_names].copy()
    else:
        X_df = df[feature_names].copy()

    def _grid(series: pd.Series) -> np.ndarray | None:
        s = pd.to_numeric(series, errors="coerce")
        s = s[np.isfinite(s)]
        if s.empty:
            return None
        lo, hi = np.quantile(s, quantile_range)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            v = lo if np.isfinite(lo) else hi
            return None if not np.isfinite(v) else np.array([float(v)])
        return np.linspace(float(lo), float(hi), int(max(2, grid_points)))

    X_df = X_df.copy()
    n_cores_eff = max(1, n_cores if n_cores is not None else (os.cpu_count() or 2) - 1)

    def _one(var: str) -> pd.DataFrame:
        grid = _grid(X_df[var])
        if grid is None or len(grid) == 0:
            (log.info if verbose else log.debug)(
                "Variable '%s' has insufficient numeric spread; skipping.", var
            )
            return pd.DataFrame(columns=["variable", "value", "pdp_mean", "pdp_std"])

        X_work = X_df.copy()
        means: list[float] = []
        stds: list[float] = []
        for g in grid:
            X_work[var] = g
            yhat = ml_predict(model, X_work)
            yhat = np.asarray(yhat, dtype=float)
            means.append(float(np.nanmean(yhat)) if yhat.size else np.nan)
            stds.append(float(np.nanstd(yhat)) if yhat.size else np.nan)

        return pd.DataFrame({"variable": var, "value": grid, "pdp_mean": means, "pdp_std": stds})

    pieces = Parallel(n_jobs=n_cores_eff)(delayed(_one)(v) for v in vars_for_pdp)
    return (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame(columns=["variable", "value", "pdp_mean", "pdp_std"])
    )
