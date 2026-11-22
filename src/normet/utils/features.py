# src/normet/utils/features.py
"""Feature introspection and permutation importance helpers."""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..utils._config import DEFAULT_SEED, resolve_workers
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["extract_features", "model_feature_names", "permutation_importance"]


def model_feature_names(model: Any) -> list[str]:
    """Extract predictor column names in the model's exact training order."""
    model_type = getattr(model, "backend", None)

    if model_type == "flaml":
        est = getattr(model, "model", None)
        booster = None
        try:
            lgb_est = getattr(est, "estimator", est)
            booster = getattr(lgb_est, "booster_", None) or getattr(lgb_est, "booster", None)
        except Exception:
            booster = None
        if booster is not None:
            try:
                names = list(map(str, booster.feature_name()))
                if names:
                    return names
            except Exception:
                pass
        for obj in (model, est, getattr(est, "estimator", None)):
            if obj is None:
                continue
            for attr in ("feature_names_in_", "feature_name_"):
                vals = getattr(obj, attr, None)
                if vals is not None:
                    names = list(map(str, list(vals)))
                    if names:
                        return names
        raise AttributeError("FLAML estimator does not expose feature names.")

    if model_type == "lightgbm":
        try:
            names = list(map(str, model.feature_name()))
            if names:
                return names
        except Exception as e:
            raise AttributeError("LightGBM model does not expose feature names.") from e
        raise AttributeError("LightGBM model does not expose feature names.")

    if (
        model_type in {"sklearn", "rf"}
        or hasattr(model, "feature_names_in_")
        or hasattr(model, "feature_name_")
    ):
        for attr in ("feature_names_in_", "feature_name_"):
            vals = getattr(model, attr, None)
            if vals is not None:
                names = list(map(str, list(vals)))
                if names:
                    return names
        raise AttributeError("Model does not expose feature names.")

    raise TypeError(
        f"Unsupported model type '{model_type}'. Expected one of: 'flaml', 'lightgbm', 'sklearn'."
    )


def extract_features(model: Any, importance_ascending: bool = False) -> list[str]:
    """
    Extract and sort feature names from an AutoML model.

    Supports models trained via FLAML. LightGBM booster
    importances are preferred when available; otherwise falls back to
    scikit-learn–style attributes.

    Parameters
    ----------
    model : object
        Trained AutoML model with attribute ``backend`` in {"flaml"}.
    importance_ascending : bool, default=False
        If True, sort features by importance ascending (least → most).
        If False (default), sort descending (most → least).

    Returns
    -------
    List[str]
        Feature names ordered by importance. If all importances are equal
        or unavailable, returns the raw feature order.

    Raises
    ------
    AttributeError
        If the model does not expose feature names/importances.
    TypeError
        If ``backend`` is unsupported.
    """
    model_type = getattr(model, "backend", None)

    # -------------------------
    # FLAML / sklearn-like path
    # -------------------------
    if model_type == "flaml":
        est = getattr(model, "model", None)  # FLAML AutoML.model
        booster = None  # LightGBM booster if present
        feature_names: list[str] | None = None
        importances: list[float] | None = None

        # Try to grab LightGBM booster for consistent importances
        try:
            lgb_est = getattr(est, "estimator", est)  # handle nested estimators
            booster = getattr(lgb_est, "booster_", None) or getattr(lgb_est, "booster", None)
        except Exception as e:
            log.debug("Could not access LightGBM booster on FLAML estimator: %s", e)
            booster = None

        if booster is not None:
            try:
                feature_names = list(map(str, booster.feature_name()))
                importances = list(booster.feature_importance(importance_type="gain"))
            except Exception as e:
                log.debug("LightGBM booster feature importance failed: %s", e)
                booster = None  # fall through to sklearn-style

        # Fallbacks: sklearn-style attributes
        if booster is None:
            # Probe a few plausible objects for attrs
            candidates = [model, est, getattr(est, "estimator", None)]

            def _first_attr(obj: Any, names: tuple[str, ...]) -> Any:
                for n in names:
                    if obj is not None and hasattr(obj, n):
                        return getattr(obj, n)
                return None

            for obj in candidates:
                feature_names = _first_attr(obj, ("feature_name_", "feature_names_in_"))
                importances = _first_attr(obj, ("feature_importances_",))
                if feature_names is not None and importances is not None:
                    feature_names = list(map(str, list(feature_names)))
                    importances = list(importances)
                    log.debug("Using sklearn-style feature importances.")
                    break

        if feature_names is None or importances is None:
            raise AttributeError("FLAML estimator does not expose feature names/importances.")

        _fn: list[str] = list(feature_names)
        _imps: list[float] = list(importances)

        # Sanitize importances to floats; treat non-finite as 0
        clean_imps: list[float] = []
        for i in _imps:
            try:
                v = float(i)
                if np.isnan(v) or not np.isfinite(v):
                    v = 0.0
            except Exception:
                v = 0.0
            clean_imps.append(v)

        # If all identical, just return the raw order
        if len(set(clean_imps)) <= 1:
            return [str(n) for n in _fn]

        # Order by (importance, name) for deterministic tie-breaking
        order = sorted(
            range(len(_fn)),
            key=lambda k: (clean_imps[k], str(_fn[k])),
            reverse=not importance_ascending,
        )
        return [str(_fn[i]) for i in order]

    # -------------------------
    # LightGBM path
    # -------------------------
    elif model_type == "lightgbm":
        try:
            feature_names = list(model.feature_name())
            importances = list(model.feature_importance(importance_type="gain"))
        except Exception as e:
            raise AttributeError("LightGBM model does not expose feature names/importances.") from e

        # Sanitize importances
        clean_imps = []
        for i in importances:
            try:
                v = float(i)
                if np.isnan(v) or not np.isfinite(v):
                    v = 0.0
            except Exception:
                v = 0.0
            clean_imps.append(v)

        if len(set(clean_imps)) <= 1:
            return [str(n) for n in feature_names]
        order = sorted(
            range(len(feature_names)),
            key=lambda k: (clean_imps[k], str(feature_names[k])),
            reverse=not importance_ascending,
        )
        return [str(feature_names[i]) for i in order]

    # -------------------------
    # Generic sklearn-compatible path
    # -------------------------
    elif model_type in {"sklearn", "rf"} or hasattr(model, "feature_names_in_"):
        try:
            fn = list(model.feature_names_in_)
        except Exception as e:
            log.debug("feature_names_in_ unavailable; falling back to feature_name_: %s", e)
            fn = list(getattr(model, "feature_name_", [])) or list(
                getattr(model, "feature_names_in_", [])
            )
        try:
            importances = list(model.feature_importances_)
        except Exception as e:
            log.debug("feature_importances_ unavailable; using insertion order: %s", e)
            importances = []
        if not fn:
            raise AttributeError("Model does not expose feature names.")
        clean_imps = []
        for i in importances or []:
            try:
                v = float(i)
                if np.isnan(v) or not np.isfinite(v):
                    v = 0.0
            except Exception:
                v = 0.0
            clean_imps.append(v)
        if len(set(clean_imps)) <= 1 or not clean_imps:
            return [str(n) for n in fn]
        order = sorted(
            range(len(fn)),
            key=lambda k: (clean_imps[k], str(fn[k])),
            reverse=not importance_ascending,
        )
        return [str(fn[i]) for i in order]

    # -------------------------
    # Unsupported backend
    # -------------------------
    else:
        raise TypeError(
            f"Unsupported model type '{model_type}'. Expected one of: 'flaml', 'lightgbm', 'sklearn'."
        )


def _eval_metric_score(y_pred: np.ndarray, y_true: np.ndarray, metric_key: str) -> float:
    """Compute scalar score for :func:`permutation_importance`."""
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if not np.any(mask):
        return float("nan")
    yp = y_pred[mask]
    yt = y_true[mask]
    if metric_key == "RMSE":
        return float(np.sqrt(np.mean((yp - yt) ** 2)))
    if metric_key in ("MGE", "MAE"):
        return float(np.mean(np.abs(yp - yt)))
    if metric_key == "NSE":
        sst = float(np.sum((yt - np.mean(yt)) ** 2))
        return float(1.0 - np.sum((yp - yt) ** 2) / sst) if sst > 1e-12 else float("nan")
    # R2 (Pearson r^2)
    sp = float(np.std(yp))
    st = float(np.std(yt))
    if sp <= 1e-12 or st <= 1e-12:
        return float("nan")
    r = float(np.corrcoef(yp, yt)[0, 1])
    return float(r * r)


def permutation_importance(
    df: pd.DataFrame,
    model: object,
    *,
    target: str = "value",
    covariates: Sequence[str] | None = None,
    subset: str | None = "testing",
    metric: str = "RMSE",
    n_repeats: int = 10,
    block_size: int = 1,
    log_transform: bool = False,
    back_transform: str = "arithmetic",
    smearing_factor: float = 1.0,
    seed: int = DEFAULT_SEED,
    n_cores: int | None = None,
) -> pd.DataFrame:
    """Out-of-sample (optionally block-permuted) permutation feature importance.

    Unlike tree split gain on training data, block permutation measures each
    predictor's true contribution to held-out generalization while preserving
    short-range autocorrelation inside blocks of length ``block_size``.
    """
    from ..model.predict import ml_predict

    metric_map = {"RMSE": "RMSE", "MGE": "MGE", "MAE": "MAE", "NSE": "NSE", "R2": "R2"}
    metric_key = metric_map.get(metric.upper() if metric.upper() != "R2" else "R2")
    if metric_key is None:
        raise ValueError(
            f"Unsupported `metric` {metric!r}; choose from ('RMSE', 'MGE', 'MAE', 'NSE', 'R2')."
        )
    nr_obj: object = n_repeats
    if isinstance(nr_obj, bool) or not isinstance(nr_obj, numbers.Integral) or int(nr_obj) < 1:
        raise ValueError(f"`n_repeats` must be >= 1, got {n_repeats!r}.")
    bs_obj: object = block_size
    if isinstance(bs_obj, bool) or not isinstance(bs_obj, numbers.Integral) or int(bs_obj) < 1:
        raise ValueError(f"`block_size` must be >= 1, got {block_size!r}.")
    if back_transform not in ("arithmetic", "smearing", "geometric"):
        raise ValueError(
            f"`back_transform` must be 'arithmetic', 'smearing', or 'geometric', got {back_transform!r}."
        )
    sf = float(smearing_factor)
    if not np.isfinite(sf) or sf <= 0.0:
        raise ValueError(
            f"`smearing_factor` must be a positive finite float, got {smearing_factor!r}."
        )
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    if covariates is not None:
        eval_vars = [str(c) for c in covariates]
    else:
        eval_vars = model_feature_names(model)
    eval_vars = [c for c in eval_vars if c in df.columns]
    if not eval_vars:
        raise ValueError("No evaluation covariates found in `df`.")

    work = df
    if "set" in work.columns and subset not in (None, "all"):
        sub_df = work.loc[work["set"] == subset]
        if not sub_df.empty:
            work = sub_df
    work = work.copy()
    if "date" in work.columns and pd.api.types.is_datetime64_any_dtype(work["date"]):
        work = work.sort_values("date", kind="stable")
    work = work.reset_index(drop=True)

    y_raw = pd.to_numeric(work[target], errors="coerce").to_numpy(dtype=float)
    y_true = np.expm1(y_raw) if log_transform else y_raw
    n_cores_eff = resolve_workers(n_cores)

    def _predict_orig(frame: pd.DataFrame) -> np.ndarray:
        yhat = np.asarray(ml_predict(model, frame, num_threads=n_cores_eff), dtype=float)
        if not log_transform:
            return yhat
        if back_transform == "smearing":
            return np.clip(sf * np.exp(yhat) - 1.0, 0.0, None)
        return np.clip(np.expm1(yhat), 0.0, None)

    base_score = _eval_metric_score(_predict_orig(work), y_true, metric_key)
    higher_is_better = metric_key in ("NSE", "R2")

    n = len(work)
    b_len = int(block_size)
    repeats = int(n_repeats)
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    blocks = (
        [np.arange(start, min(n, start + b_len)) for start in range(0, n, b_len)]
        if b_len > 1
        else []
    )
    for col in eval_vars:
        diffs: list[float] = []
        col_vals = work[col].to_numpy()
        X_perm = work.copy()
        for _ in range(repeats):
            if b_len == 1 or len(blocks) <= 1:
                perm_idx = rng.permutation(n)
            else:
                b_order = rng.permutation(len(blocks))
                perm_idx = np.concatenate([blocks[j] for j in b_order])
            X_perm[col] = col_vals[perm_idx]
            perm_score = _eval_metric_score(_predict_orig(X_perm), y_true, metric_key)
            imp = (base_score - perm_score) if higher_is_better else (perm_score - base_score)
            diffs.append(float(imp))
        arr_d = np.asarray(diffs, dtype=float)
        rows.append(
            {
                "variable": col,
                "importance_mean": float(np.nanmean(arr_d)),
                "importance_std": float(np.nanstd(arr_d, ddof=1)) if repeats > 1 else 0.0,
                "baseline_score": float(base_score),
                "metric": metric,
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values("importance_mean", ascending=False, kind="mergesort").reset_index(
        drop=True
    )
