# src/normet/analysis/pdp.py
"""Partial dependence: :func:`pdp` (single feature) and :func:`pdp_grid`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backends import backend_registry
from ..model.predict import ml_predict
from ..utils._config import resolve_workers
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
        Worker threads. Default: all cores - 1. PDP is dispatched over
        threads, so no worker copies the frame; the predictor releases the
        GIL while scoring.
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
    n_cores_eff = resolve_workers(n_cores)

    def _one(var: str) -> pd.DataFrame:
        grid = _grid(X_df[var])
        if grid is None or len(grid) == 0:
            (log.info if verbose else log.debug)(
                "Variable '%s' has insufficient numeric spread; skipping.", var
            )
            return pd.DataFrame(columns=["variable", "value", "pdp_mean", "pdp_std"])

        # One prediction per stacked block, not one per grid point: every
        # point changes only ``var``, so stack one copy of the frame per
        # point (``np.repeat`` puts point g into block g) and read each
        # point's mean/SD off the result. Fifty small predict calls each pay
        # the frame -> matrix conversion and the backend dispatch; one
        # stacked call pays them once. The stack is capped at ~1e6 rows
        # (~100 MB of float64 features) so a large training frame is not
        # inflated by the full grid.
        n = len(X_df)
        per_block = max(1, 1_000_000 // max(n, 1))
        means: list[float] = []
        stds: list[float] = []
        for start in range(0, len(grid), per_block):
            g = grid[start : start + per_block]
            X_stack = pd.concat([X_df] * len(g), ignore_index=True)
            X_stack[var] = np.repeat(g, n)
            yhat = np.asarray(ml_predict(model, X_stack), dtype=float)
            for j in range(len(g)):
                block_y = yhat[j * n : (j + 1) * n]
                means.append(float(np.nanmean(block_y)) if block_y.size else np.nan)
                stds.append(float(np.nanstd(block_y)) if block_y.size else np.nan)

        return pd.DataFrame({"variable": var, "value": grid, "pdp_mean": means, "pdp_std": stds})

    # Imported here, not at module level: joblib (and loky under it) is ~100 ms
    # of `import normet`, and only the parallel branch needs it.
    from joblib import Parallel, delayed

    # Threads, not processes: the tasks predict through LightGBM (which
    # releases the GIL) on a frame shared with the parent, while a loky worker
    # re-imports normet and pickles the frame per task. Measured at 4,344
    # training rows: three variables took 5.0 s through the process pool the
    # default used to pick, 1.7 s on four threads and 2.6 s serially; the
    # full 13-variable panel is 5.1 s on the default seven threads vs 6.4 s
    # serially.
    pieces = Parallel(n_jobs=n_cores_eff, prefer="threads")(delayed(_one)(v) for v in vars_for_pdp)
    return (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame(columns=["variable", "value", "pdp_mean", "pdp_std"])
    )
