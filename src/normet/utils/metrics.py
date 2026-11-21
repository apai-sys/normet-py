# src/normet/utils/metrics.py
"""Model performance metrics, including the time-stratified :func:`modStats`."""

from __future__ import annotations

import numbers
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd

from ._config import DEFAULT_SEED
from .logging import get_logger

log = get_logger(__name__)

__all__ = ["Stats", "modStats", "exceedance_stats", "compare_models"]

# ---------------------------------------------------------------------
# Low-level metrics on arrays
# ---------------------------------------------------------------------


def _numeric_column(values: Any) -> np.ndarray:
    """A column of numbers as float64, missing values as NaN.

    A nullable ``Int64``/``Float64`` column comes out of ``Series.to_numpy``
    as an object array, and ``np.isfinite`` raises ``TypeError`` on it; a
    numeric-text column does not convert either. ``pd.to_numeric`` reads
    both, and its unparseable entries stay NaN for the finite mask to drop.
    """
    s = values if isinstance(values, pd.Series) else pd.Series(np.asarray(values).reshape(-1))
    return np.asarray(pd.to_numeric(s, errors="coerce"), dtype=float)


def _fac2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Fraction of predictions within a factor of 2 of observations (FAC2).

    FAC2 = mean( 0.5 <= y_pred / y_true <= 2 )

    Notes
    -----
    - We guard against division by zero by adding a tiny epsilon to y_true.
    - Only finite ratios are used; if none are finite, returns NaN.
    """
    epsilon = 1e-9
    ratio = np.divide(
        y_pred,
        y_true + epsilon,
        out=np.full_like(y_pred, np.nan, dtype=float),
        where=np.isfinite(y_true) & (y_true != 0),
    )
    mask = np.isfinite(ratio)
    if not np.any(mask):
        return np.nan
    r = ratio[mask]
    return float(np.mean((r >= 0.5) & (r <= 2.0)))


def _lag1_autocorr(arr: np.ndarray) -> float:
    """Finite lag-1 Pearson autocorrelation, defaulting to 0 when degenerate."""
    if len(arr) < 3:
        return 0.0
    a, b = arr[:-1], arr[1:]
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    val = float(np.corrcoef(a, b)[0, 1])
    return val if np.isfinite(val) else 0.0


def _stats_from_arrays(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    statistic: list[str],
    *,
    effective_n: bool = False,
) -> pd.DataFrame:
    """
    Compute a set of scalar statistics from two 1D arrays.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values (same length as y_true).
    y_true : np.ndarray
        Observed (ground-truth) values.
    statistic : List[str]
        A list of metric names to compute. Supported keys include:
        ``{"n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA",
        "R2", "NSE", "R2_oos", "KGE", "NRMSE", "FB", "FE", "MG", "VG",
        "Spearman", "slope", "intercept", "logMAE", "logRMSE"}``.
    effective_n : bool, default False
        If True, adjust the two-sided p-value of ``r`` for lag-1 serial
        autocorrelation using Bartlett's effective sample size approximation.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame containing the requested metrics.
        If "r" is requested, also includes "p_level" with significance stars.
    """
    # Keep only rows where both arrays are finite
    yhat = _numeric_column(y_pred)
    yobs = _numeric_column(y_true)
    mask = np.isfinite(yhat) & np.isfinite(yobs)
    yhat = yhat[mask]
    yobs = yobs[mask]

    n = yhat.size
    if n == 0:
        # If no valid pairs, return NaNs for requested metrics.
        keys = set(statistic)
        if "r" in statistic:
            keys |= {"p_level"}
        out = {k: np.nan for k in keys}
        if "n" in statistic:
            out["n"] = 0
        return pd.DataFrame([out])

    diff = yhat - yobs
    adiff = np.abs(diff)
    mu_pred = float(np.mean(yhat))
    mu_obs = float(np.mean(yobs))

    res: dict[str, float | int | str] = {}

    # Count
    if "n" in statistic:
        res["n"] = int(n)

    # Factor-of-two accuracy
    if "FAC2" in statistic:
        res["FAC2"] = _fac2(yhat, yobs)

    # Mean bias / Mean absolute error / RMSE / NRMSE
    if "MB" in statistic:
        res["MB"] = float(np.mean(diff))
    if "MGE" in statistic:
        res["MGE"] = float(np.mean(adiff))
    if "RMSE" in statistic or "NRMSE" in statistic:
        rmse_val = float(np.sqrt(np.mean(diff * diff)))
        if "RMSE" in statistic:
            res["RMSE"] = rmse_val
        if "NRMSE" in statistic:
            res["NRMSE"] = float(rmse_val / mu_obs) if abs(mu_obs) > 1e-12 else np.nan

    # True out-of-sample R^2 / Nash-Sutcliffe Efficiency
    if "NSE" in statistic or "R2_oos" in statistic:
        sst = float(np.sum((yobs - mu_obs) ** 2))
        nse_val = float(1.0 - float(np.sum(diff * diff)) / sst) if sst > 1e-12 else np.nan
        if "NSE" in statistic:
            res["NSE"] = nse_val
        if "R2_oos" in statistic:
            res["R2_oos"] = nse_val

    # Fractional Bias / Fractional Error (Boylan & Russell 2006)
    if "FB" in statistic or "FE" in statistic:
        mean_sum = mu_pred + mu_obs
        if "FB" in statistic:
            res["FB"] = float(2.0 * np.mean(diff) / mean_sum) if abs(mean_sum) > 1e-12 else np.nan
        if "FE" in statistic:
            res["FE"] = float(2.0 * np.mean(adiff) / mean_sum) if abs(mean_sum) > 1e-12 else np.nan

    # Chang & Hanna (2004) geometric mean bias (MG) and geometric variance (VG)
    if "MG" in statistic or "VG" in statistic:
        pos = (yhat > 0.0) & (yobs > 0.0)
        if np.any(pos):
            lr = np.log(yobs[pos]) - np.log(yhat[pos])
            if "MG" in statistic:
                res["MG"] = float(np.exp(np.clip(np.mean(lr), -50.0, 50.0)))
            if "VG" in statistic:
                res["VG"] = float(np.exp(np.clip(np.mean(lr * lr), -50.0, 50.0)))
        else:
            if "MG" in statistic:
                res["MG"] = np.nan
            if "VG" in statistic:
                res["VG"] = np.nan

    # Log-scale errors: accuracy in log1p space, where a fixed relative miss
    # maps to a bounded error at every level while its absolute error grows
    # without bound. This is the scale a log-transformed fit optimises, so
    # these are the right yardsticks when comparing against a linear fit on
    # skewed, multi-decade data. Pairs
    # with either side below -1 -- outside log1p's domain, e.g. de-weathered
    # residuals -- are dropped from these two metrics only; every other
    # statistic keeps them.
    if "logMAE" in statistic or "logRMSE" in statistic:
        lmask = (yhat > -1.0) & (yobs > -1.0)
        if bool(lmask.any()):
            ldiff = np.log1p(yhat[lmask]) - np.log1p(yobs[lmask])
            if "logMAE" in statistic:
                res["logMAE"] = float(np.mean(np.abs(ldiff)))
            if "logRMSE" in statistic:
                res["logRMSE"] = float(np.sqrt(np.mean(ldiff * ldiff)))

    # Normalized bias / error (sum-based)
    sum_obs = float(np.sum(yobs))
    if "NMB" in statistic:
        res["NMB"] = float(np.sum(diff) / sum_obs) if sum_obs != 0.0 else np.nan
    if "NMGE" in statistic:
        res["NMGE"] = float(np.sum(adiff) / sum_obs) if sum_obs != 0.0 else np.nan

    # Coefficient of efficiency (COE) & Index of agreement (IOA)
    denom_abs_obs = float(np.sum(np.abs(yobs - mu_obs)))
    if "COE" in statistic:
        res["COE"] = (
            float(1.0 - (np.sum(adiff) / denom_abs_obs)) if denom_abs_obs != 0.0 else np.nan
        )
    if "IOA" in statistic:
        lhs = float(np.sum(adiff))
        rhs = float(2.0 * denom_abs_obs)
        if rhs == 0.0 and lhs == 0.0:
            res["IOA"] = 1.0
        elif rhs == 0.0:
            res["IOA"] = np.nan
        else:
            # Common bounded formulation
            res["IOA"] = float(1.0 - lhs / rhs) if lhs <= rhs else float(rhs / lhs - 1.0)

    # Correlation and p-value (also used for R^2 and KGE)
    r_val = np.nan
    p_val = np.nan
    if any(k in statistic for k in ("r", "R2", "KGE")):
        try:
            from scipy import stats as _stats  # deferred: ~0.8 s to import

            r_val, p_val = _stats.pearsonr(yhat, yobs)
            if effective_n and n >= 4 and np.isfinite(r_val):
                rho1 = _lag1_autocorr(yhat)
                rho2 = _lag1_autocorr(yobs)
                denom_eff = max(1e-6, 1.0 + rho1 * rho2)
                n_eff = max(3.0, min(float(n), float(n) * max(1e-6, 1.0 - rho1 * rho2) / denom_eff))
                r2_clip = min(float(r_val * r_val), 1.0 - 1e-15)
                t_stat = abs(float(r_val)) * float(
                    np.sqrt((n_eff - 2.0) / max(1e-15, 1.0 - r2_clip))
                )
                p_val = float(2.0 * _stats.t.sf(t_stat, df=n_eff - 2.0))
        except Exception:
            r_val, p_val = (np.nan, np.nan)

    # Kling-Gupta Efficiency (KGE)
    if "KGE" in statistic:
        sd_pred = float(np.std(yhat))
        sd_obs = float(np.std(yobs))
        if np.isfinite(r_val) and sd_obs > 1e-12 and abs(mu_obs) > 1e-12:
            alpha_k = sd_pred / sd_obs
            beta_k = mu_pred / mu_obs
            res["KGE"] = float(
                1.0
                - np.sqrt((float(r_val) - 1.0) ** 2 + (alpha_k - 1.0) ** 2 + (beta_k - 1.0) ** 2)
            )
        else:
            res["KGE"] = np.nan

    # Spearman rank correlation
    if "Spearman" in statistic:
        if n >= 2 and float(np.std(yhat)) > 1e-12 and float(np.std(yobs)) > 1e-12:
            from scipy import stats as _stats

            res["Spearman"] = float(_stats.spearmanr(yhat, yobs).statistic)
        else:
            res["Spearman"] = np.nan

    # Calibration slope & intercept of y_pred on y_true
    if "slope" in statistic or "intercept" in statistic:
        yc_o = yobs - mu_obs
        denom_s = float(np.sum(yc_o * yc_o))
        if denom_s > 1e-12:
            sl_val = float(np.sum(yc_o * (yhat - mu_pred)) / denom_s)
            ic_val = float(mu_pred - sl_val * mu_obs)
        else:
            sl_val, ic_val = np.nan, np.nan
        if "slope" in statistic:
            res["slope"] = sl_val
        if "intercept" in statistic:
            res["intercept"] = ic_val

    # Pearson correlation with significance stars
    if "r" in statistic:
        res["r"] = float(r_val)
        # Significance stars based on two-sided p-value:
        #   "" (ns), "+" (p<0.1), "*" (p<0.05), "**" (p<0.01), "***" (p<0.001)
        if not np.isfinite(p_val) or p_val >= 0.1:
            res["p_level"] = ""
        elif p_val >= 0.05:
            res["p_level"] = "+"
        elif p_val >= 0.01:
            res["p_level"] = "*"
        elif p_val >= 0.001:
            res["p_level"] = "**"
        else:
            res["p_level"] = "***"

    # Coefficient of determination
    if "R2" in statistic:
        res["R2"] = float(r_val * r_val) if np.isfinite(r_val) else np.nan

    # Ensure all requested keys exist (and p_level if r is requested)
    keys_needed = set(statistic)
    if "r" in statistic:
        keys_needed |= {"p_level"}
    for k in keys_needed:
        res.setdefault(k, np.nan)

    return pd.DataFrame([res])


# ---------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------

_DEFAULT_STATS: list[str] = [
    "n",
    "FAC2",
    "MB",
    "MGE",
    "NMB",
    "NMGE",
    "RMSE",
    "r",
    "COE",
    "IOA",
    "R2",
]


def Stats(
    df: pd.DataFrame,
    mod: str = "value_predict",
    obs: str = "value",
    statistic: list[str] | None = None,
    *,
    effective_n: bool = False,
) -> pd.DataFrame:
    """
    Compute requested statistics from two DataFrame columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing columns with predictions and observations.
    mod : str, default "value_predict"
        Column name with model predictions.
    obs : str, default "value"
        Column name with ground-truth observations.
    statistic : List[str] | None
        Which metrics to compute. If None, a comprehensive default set is used.
    effective_n : bool, default False
        Adjust correlation ``r`` significance for lag-1 serial autocorrelation.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame with the requested statistics.
    """
    if statistic is None:
        statistic = _DEFAULT_STATS

    # Friendly validation for missing columns
    missing = [c for c in (mod, obs) if c not in df.columns]
    if missing:
        raise ValueError(f"Stats: columns not found in DataFrame: {missing}")

    y_pred = _numeric_column(df[mod])
    y_true = _numeric_column(df[obs])
    return _stats_from_arrays(y_pred, y_true, statistic, effective_n=effective_n)


_TIME_GROUP_TOKENS = {"season", "month", "hour", "weekday", "year", "day_of_year"}


def _season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"


def _expand_time_token(df: pd.DataFrame, token: str) -> pd.Series:
    """Derive a grouping series from `date` for time-based tokens."""
    if "date" not in df.columns:
        raise ValueError(f"`by='{token}'` requires a 'date' column.")
    dt = pd.to_datetime(df["date"])
    if token == "season":
        return dt.dt.month.map(_season_from_month).rename("season")
    if token == "month":
        return dt.dt.month.rename("month")
    if token == "hour":
        return dt.dt.hour.rename("hour")
    if token == "weekday":
        return (dt.dt.weekday + 1).rename("weekday")
    if token == "year":
        return dt.dt.year.rename("year")
    if token == "day_of_year":
        return dt.dt.dayofyear.rename("day_of_year")
    raise ValueError(f"Unknown time token '{token}'.")


def _wrap_log_predictor(
    df: pd.DataFrame,
    model: object,
    raw_predictor: Callable[[object, pd.DataFrame], np.ndarray],
    *,
    back_transform: str,
    smearing_factor: float | None,
) -> tuple[pd.DataFrame, Callable[[object, pd.DataFrame], np.ndarray]]:
    """Convert log1p-scale frame and predictor to original concentration scale."""
    from .prepare import compute_duan_smearing

    if back_transform not in ("arithmetic", "smearing", "geometric"):
        raise ValueError(
            f"`back_transform` must be 'arithmetic', 'smearing', or 'geometric', got {back_transform!r}."
        )
    phi = 1.0
    if back_transform == "smearing":
        if smearing_factor is not None:
            phi = float(smearing_factor)
            if not np.isfinite(phi) or phi <= 0.0:
                raise ValueError(
                    f"`smearing_factor` must be positive finite, got {smearing_factor!r}."
                )
        else:
            ref = (
                df[df["set"] == "training"]
                if ("set" in df.columns and bool((df["set"] == "training").any()))
                else df
            )
            tr_preds = np.asarray(raw_predictor(model, ref), dtype=float)
            phi = compute_duan_smearing(ref["value"], tr_preds)

    df_orig = df.copy()
    df_orig["value"] = np.expm1(
        pd.to_numeric(df_orig["value"], errors="coerce").to_numpy(dtype=float)
    )

    def _orig_pred(m: object, x: pd.DataFrame) -> np.ndarray:
        z = np.asarray(raw_predictor(m, x), dtype=float)
        if back_transform == "smearing":
            return np.clip(phi * np.exp(z) - 1.0, 0.0, None)
        return np.clip(np.expm1(z), 0.0, None)

    return df_orig, _orig_pred


def modStats(
    df: pd.DataFrame,
    model: object,
    subset: str | None = None,
    statistic: list[str] | None = None,
    predictor: Callable[[object, pd.DataFrame], np.ndarray] | None = None,
    *,
    by: str | list[str] | None = None,
    include_all: bool = True,
    log_transform: bool = False,
    back_transform: str = "arithmetic",
    smearing_factor: float | None = None,
    effective_n: bool = False,
) -> pd.DataFrame:
    """
    Predict with a model on a DataFrame and compute statistics.

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared dataset. Must contain target column "value".
        If a "set" column exists and `subset` is specified, only that subset is evaluated.
    model : object
        Trained model compatible with the package's `ml_predict` interface.
    subset : {"training","testing","all"} | None
        Which split to evaluate. If None and "set" exists, returns one row
        per split plus "all".
    statistic : List[str] | None
        Metrics to compute; defaults to a comprehensive set.
    predictor : callable(model, df) -> np.ndarray | None
        Optional override for the prediction function. If None, uses
        `..model.predict.ml_predict`.
    by : str | list of str | None, keyword-only
        Column name(s) to group results by. Special string tokens derived
        from ``date`` are also accepted: ``"season"``, ``"month"``, ``"hour"``,
        ``"weekday"``, ``"year"``, ``"day_of_year"``. When ``by`` is given,
        the per-``set`` aggregation is *replaced* by a per-group aggregation.
    include_all : bool, default True
        When ``by`` is given, also append a single row with metrics computed
        across all rows (group label ``"all"``).
    log_transform : bool, default False
        If True and ``predictor`` is None, assume ``model`` was trained on
        ``log1p(value)`` and evaluate predictions against original units.
    back_transform : {"arithmetic", "smearing", "geometric"}, default "arithmetic"
        Inversion rule when ``log_transform=True`` (``"smearing"`` multiplies
        ``exp(z)`` by Duan's smearing factor before subtracting 1).
    smearing_factor : float or None, default None
        Optional precomputed Duan smearing factor; derived from ``"training"``
        rows when ``None`` and ``back_transform="smearing"``.
    effective_n : bool, default False
        Adjust correlation ``r`` significance for lag-1 serial autocorrelation.

    Returns
    -------
    pandas.DataFrame
        Tidy DataFrame of metrics. Has either a ``set`` column (default) or
        one column per ``by`` key (when ``by`` is given).
    """
    if statistic is None:
        statistic = _DEFAULT_STATS

    # Lazy import to avoid potential circular imports
    if predictor is None:
        from ..model.predict import ml_predict as _ml_predict

        base_pred: Callable[..., Any] = _ml_predict
    else:
        base_pred = predictor

    if log_transform:
        df_eval, predict_fn = _wrap_log_predictor(
            df,
            model,
            base_pred,
            back_transform=back_transform,
            smearing_factor=smearing_factor,
        )
    else:
        df_eval = df
        predict_fn = base_pred

    def _one(df_in: pd.DataFrame, tag: str) -> pd.DataFrame:
        y_pred = predict_fn(model, df_in)
        y_true = df_in["value"].to_numpy()
        st = _stats_from_arrays(y_pred, y_true, statistic, effective_n=effective_n)
        st["set"] = tag
        return st

    # ----- grouped path -----
    if by is not None:
        keys: list[str] = [by] if isinstance(by, str) else list(by)
        # Build a working frame with grouping keys materialised.
        work = df_eval.copy()
        group_cols: list[str] = []
        for k in keys:
            if k in _TIME_GROUP_TOKENS:
                work[k] = _expand_time_token(work, k).to_numpy()
                group_cols.append(k)
            elif k in work.columns:
                group_cols.append(k)
            else:
                raise ValueError(f"`by` key '{k}' is neither a column nor a known time token.")

        pieces: list[pd.DataFrame] = []
        for keys_val, sub in work.groupby(group_cols, dropna=False, sort=True):
            keys_val = keys_val if isinstance(keys_val, tuple) else (keys_val,)
            y_pred = predict_fn(model, sub)
            y_true = sub["value"].to_numpy()
            row = _stats_from_arrays(y_pred, y_true, statistic, effective_n=effective_n)
            for col_name, val in zip(group_cols, keys_val, strict=False):
                row[col_name] = cast(Any, val)
            pieces.append(row)

        if include_all:
            y_pred = predict_fn(model, work)
            y_true = work["value"].to_numpy()
            row = _stats_from_arrays(y_pred, y_true, statistic, effective_n=effective_n)
            for col_name in group_cols:
                row[col_name] = "all"
            pieces.append(row)

        if not pieces:
            return pd.DataFrame()

        out = pd.concat(pieces, ignore_index=True)
        # Move group columns to the front for readability.
        ordered = group_cols + [c for c in out.columns if c not in group_cols]
        return out[ordered]

    # ----- legacy path: by `set` column / `subset` -----
    if subset is not None:
        if subset != "all":
            if "set" not in df_eval.columns:
                raise ValueError("DataFrame has no 'set' column but a `subset` was requested.")
            df_use = df_eval[df_eval["set"] == subset]
        else:
            df_use = df_eval
        return _one(df_use, subset)

    if "set" not in df_eval.columns:
        return _one(df_eval, "all")

    pieces = []
    for s in pd.unique(df_eval["set"]):
        pieces.append(_one(df_eval[df_eval["set"] == s], s))
    pieces.append(_one(df_eval, "all"))
    return pd.concat(pieces, ignore_index=True)


def exceedance_stats(
    y_pred: Any,
    y_true: Any,
    threshold: float,
) -> pd.DataFrame:
    """Compute contingency & upper-tail skill metrics above an air-quality *threshold*.

    Returns a single-row DataFrame with ``threshold``, ``n``, ``n_obs_exc``,
    ``n_pred_exc``, ``hits``, ``misses``, ``false_alarms``, ``correct_negatives``,
    ``POD`` (hit rate), ``FAR`` (false-alarm ratio), ``CSI`` (critical success
    index), ``FBI`` (frequency bias index), ``HSS`` (Heidke skill score), and
    ``tail_RMSE``.
    """
    thresh = float(threshold)
    if not np.isfinite(thresh):
        raise ValueError(f"`threshold` must be finite, got {threshold!r}.")

    yp = _numeric_column(y_pred)
    yt = _numeric_column(y_true)
    mask = np.isfinite(yp) & np.isfinite(yt)
    yp = yp[mask]
    yt = yt[mask]
    n = int(yp.size)

    if n == 0:
        return pd.DataFrame(
            [
                {
                    "threshold": thresh,
                    "n": 0,
                    "n_obs_exc": 0,
                    "n_pred_exc": 0,
                    "hits": 0,
                    "misses": 0,
                    "false_alarms": 0,
                    "correct_negatives": 0,
                    "POD": np.nan,
                    "FAR": np.nan,
                    "CSI": np.nan,
                    "FBI": np.nan,
                    "HSS": np.nan,
                    "tail_RMSE": np.nan,
                }
            ]
        )

    obs_exc = yt >= thresh
    pred_exc = yp >= thresh
    a = int(np.sum(pred_exc & obs_exc))
    b = int(np.sum(pred_exc & (~obs_exc)))
    c = int(np.sum((~pred_exc) & obs_exc))
    d = int(np.sum((~pred_exc) & (~obs_exc)))

    pod = float(a / (a + c)) if (a + c) > 0 else np.nan
    far = float(b / (a + b)) if (a + b) > 0 else np.nan
    csi = float(a / (a + b + c)) if (a + b + c) > 0 else np.nan
    fbi = float((a + b) / (a + c)) if (a + c) > 0 else np.nan
    hss_denom = float((a + c) * (c + d) + (a + b) * (b + d))
    hss = float(2.0 * (a * d - b * c) / hss_denom) if hss_denom > 0.0 else np.nan
    tail_rmse = (
        float(np.sqrt(np.mean((yp[obs_exc] - yt[obs_exc]) ** 2))) if np.any(obs_exc) else np.nan
    )

    return pd.DataFrame(
        [
            {
                "threshold": thresh,
                "n": n,
                "n_obs_exc": int(obs_exc.sum()),
                "n_pred_exc": int(pred_exc.sum()),
                "hits": a,
                "misses": c,
                "false_alarms": b,
                "correct_negatives": d,
                "POD": pod,
                "FAR": far,
                "CSI": csi,
                "FBI": fbi,
                "HSS": hss,
                "tail_RMSE": tail_rmse,
            }
        ]
    )


def compare_models(
    df: pd.DataFrame,
    model_a: object,
    model_b: object,
    *,
    target: str = "value",
    subset: str | None = "testing",
    metric: str = "RMSE",
    block_size: int = 24,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    log_transform_a: bool = False,
    log_transform_b: bool = False,
    back_transform: str = "arithmetic",
    smearing_a: float = 1.0,
    smearing_b: float = 1.0,
    predictor_a: Callable[[object, pd.DataFrame], np.ndarray] | None = None,
    predictor_b: Callable[[object, pd.DataFrame], np.ndarray] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Compare two models on the same aligned observations using block bootstrap & DM test.

    Computes scalar metrics for ``model_a`` and ``model_b`` on aligned
    observations, a circular moving-block bootstrap confidence interval for
    ``score_b - score_a``, and a two-sided Diebold-Mariano test with Newey-West
    HAC variance estimation.
    """
    from scipy import stats as _stats

    from ..model.predict import ml_predict

    valid_metrics = {"RMSE", "MGE", "MAE", "NSE", "R2", "KGE"}
    metric_norm = (
        "MGE"
        if metric.upper() == "MAE"
        else (metric if metric in valid_metrics else metric.upper())
    )
    if metric_norm not in valid_metrics:
        raise ValueError(f"Unsupported `metric` {metric!r}; choose from {sorted(valid_metrics)}.")
    bs_obj: object = block_size
    if isinstance(bs_obj, bool) or not isinstance(bs_obj, numbers.Integral) or int(bs_obj) < 1:
        raise ValueError(f"`block_size` must be >= 1, got {block_size!r}.")
    nb_obj: object = n_bootstrap
    if isinstance(nb_obj, bool) or not isinstance(nb_obj, numbers.Integral) or int(nb_obj) < 1:
        raise ValueError(f"`n_bootstrap` must be >= 1, got {n_bootstrap!r}.")
    if not (0.0 < float(confidence_level) < 1.0):
        raise ValueError("`confidence_level` must be strictly between 0 and 1.")
    if back_transform not in ("arithmetic", "smearing", "geometric"):
        raise ValueError(
            f"`back_transform` must be 'arithmetic', 'smearing', or 'geometric', got {back_transform!r}."
        )
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    work = df
    if "set" in work.columns and subset not in (None, "all"):
        sub_df = work.loc[work["set"] == subset]
        if not sub_df.empty:
            work = sub_df
    work = work.copy()
    if "date" in work.columns and pd.api.types.is_datetime64_any_dtype(work["date"]):
        work = work.sort_values("date", kind="stable")

    y_raw = _numeric_column(work[target])

    fn_a = predictor_a or ml_predict
    fn_b = predictor_b or ml_predict
    pa = np.asarray(fn_a(model_a, work), dtype=float).reshape(-1)
    pb = np.asarray(fn_b(model_b, work), dtype=float).reshape(-1)

    if log_transform_a:
        pa = (
            np.clip(float(smearing_a) * np.exp(pa) - 1.0, 0.0, None)
            if back_transform == "smearing"
            else np.clip(np.expm1(pa), 0.0, None)
        )
    if log_transform_b:
        pb = (
            np.clip(float(smearing_b) * np.exp(pb) - 1.0, 0.0, None)
            if back_transform == "smearing"
            else np.clip(np.expm1(pb), 0.0, None)
        )
    if (
        log_transform_a
        and log_transform_b
        and "log1p" not in target
        or (log_transform_a or log_transform_b)
        and float(np.nanmedian(pa)) > 5.0 * max(1.0, float(np.nanmedian(y_raw)))
    ):
        y_eval = np.expm1(y_raw)
    else:
        y_eval = y_raw

    mask = np.isfinite(y_eval) & np.isfinite(pa) & np.isfinite(pb)
    yt = y_eval[mask]
    pa_m = pa[mask]
    pb_m = pb[mask]
    n = int(yt.size)
    if n < 3:
        raise ValueError(f"Need at least 3 aligned finite observations to compare models; got {n}.")

    score_a = float(cast(Any, _stats_from_arrays(pa_m, yt, [metric_norm])[metric_norm].iloc[0]))
    score_b = float(cast(Any, _stats_from_arrays(pb_m, yt, [metric_norm])[metric_norm].iloc[0]))
    diff_val = float(score_b - score_a)

    # Circular moving-block bootstrap for difference CI
    rng = np.random.default_rng(seed)
    b_len = max(1, min(int(block_size), n))
    n_blocks = int(np.ceil(n / b_len))
    offsets = np.arange(b_len, dtype=np.int64)
    boot_diffs = np.empty(int(n_bootstrap), dtype=float)
    for b_idx in range(int(n_bootstrap)):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + offsets[None, :]).ravel()[:n] % n
        sa_b = float(
            cast(Any, _stats_from_arrays(pa_m[idx], yt[idx], [metric_norm])[metric_norm].iloc[0])
        )
        sb_b = float(
            cast(Any, _stats_from_arrays(pb_m[idx], yt[idx], [metric_norm])[metric_norm].iloc[0])
        )
        boot_diffs[b_idx] = sb_b - sa_b

    alpha = (1.0 - float(confidence_level)) / 2.0
    fin_bd = boot_diffs[np.isfinite(boot_diffs)]
    if fin_bd.size > 0:
        ci_low = float(np.quantile(fin_bd, alpha))
        ci_high = float(np.quantile(fin_bd, 1.0 - alpha))
    else:
        ci_low, ci_high = np.nan, np.nan

    # Newey-West HAC Diebold-Mariano test on loss differential d = L(a) - L(b)
    if metric_norm in ("MGE",):
        loss_a = np.abs(pa_m - yt)
        loss_b = np.abs(pb_m - yt)
    else:
        loss_a = (pa_m - yt) ** 2
        loss_b = (pb_m - yt) ** 2
    d = loss_a - loss_b
    d_mean = float(np.mean(d))
    dc = d - d_mean
    gamma0 = float(np.dot(dc, dc) / n)
    max_lag = min(max(0, b_len - 1), n - 1)
    lrv = gamma0
    for lag in range(1, max_lag + 1):
        w_bartlett = 1.0 - lag / (max_lag + 1.0)
        gamma_k = float(np.dot(dc[lag:], dc[:-lag]) / n)
        lrv += 2.0 * w_bartlett * gamma_k

    if not np.isfinite(lrv) or lrv <= 1e-18:
        dm_stat = 0.0
        dm_pval = 1.0
    else:
        dm_stat = float(d_mean / np.sqrt(lrv / n))
        dm_pval = float(2.0 * _stats.t.sf(abs(dm_stat), df=n - 1))

    summary_df = pd.DataFrame(
        [
            {
                "metric": metric,
                "n": n,
                "score_a": score_a,
                "score_b": score_b,
                "diff_b_minus_a": diff_val,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "dm_stat": dm_stat,
                "dm_pvalue": dm_pval,
            }
        ]
    )
    return {
        "metric": metric,
        "n": n,
        "score_a": score_a,
        "score_b": score_b,
        "diff": diff_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "dm_stat": dm_stat,
        "dm_pvalue": dm_pval,
        "summary": summary_df,
    }
