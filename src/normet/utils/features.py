# src/normet/utils/features.py
"""Feature-importance helpers, including :func:`extract_features`."""

from __future__ import annotations

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)


def extract_features(model, importance_ascending: bool = False) -> list[str]:
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

            def _first_attr(obj, names):
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
