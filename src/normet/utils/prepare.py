# src/normet/utils/prepare.py
"""Data preparation: imputation, date features, train/test/season splits, and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from .logging import get_logger

log = get_logger(__name__)

__all__ = [
    "prepare_data",
    "process_date",
    "check_data",
    "impute_values",
    "add_date_variables",
    "split_into_sets",
]


def prepare_data(
    df: pd.DataFrame,
    value: str,
    feature_names: list[str],
    na_rm: bool = True,
    split_method: str = "random",
    fraction: float = 0.75,
    seed: int = 7_654_321,
) -> pd.DataFrame:
    """
    Clean, validate, and split the input DataFrame in a single pipeline.

    Steps:
      1) Ensure a datetime column named ``date`` is present.
      2) Validate target and features.
      3) Impute/drop missing values.
      4) Add derived date variables (unix, julian day, weekday, hour).
      5) Split into training/testing sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input dataset containing at least the target column and date/time info.
    value : str
        Target column name in ``df``.
    feature_names : list of str
        Predictor variable names to keep (must exist in ``df``).
    na_rm : bool, default True
        If True, drop rows where the target is NA. Also imputes other NAs.
    split_method : {"random","ts","season","month"}, default "random"
        Train/test split method.
    fraction : float, default 0.75
        Training fraction for data splitting.
    seed : int, default 7654321
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Processed dataset with:
          - ``date`` column
          - ``value`` column (target)
          - predictors
          - derived date features
          - ``set`` column indicating "training"/"testing".
    """
    log.debug("Preparing data with split_method=%s, fraction=%.3f", split_method, fraction)
    df_out = (
        df.pipe(process_date)
        .pipe(check_data, feature_names=feature_names, value=value)
        .pipe(impute_values, na_rm=na_rm)
        .pipe(add_date_variables)
        .pipe(split_into_sets, split_method=split_method, fraction=fraction, seed=seed)
        .reset_index(drop=True)
    )
    log.info("Prepared data: %d rows, %d columns", len(df_out), df_out.shape[1])
    return df_out


def process_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has a datetime column named ``date``.

    - If index is DatetimeIndex, reset and rename to ``date``.
    - If no datetime column found, attempts to coerce common names.
    - If multiple datetime columns, raises error unless unambiguous.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a datetime index or column.

    Returns
    -------
    pandas.DataFrame
        Copy of input with a single datetime column ``date``.

    Raises
    ------
    ValueError
        If no datetime information found or multiple ambiguous columns.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        idx_name = df.index.name or "index"
        df = df.reset_index().rename(columns={idx_name: "date"})

    time_cols = list(df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)

    if len(time_cols) == 0:
        candidates = [
            c for c in df.columns if str(c).lower() in {"date", "datetime", "time", "timestamp"}
        ]
        for c in candidates:
            try:
                coerced = pd.to_datetime(df[c], errors="raise", utc=False)
                df = df.copy()
                df[c] = coerced
                time_cols = [c]
                log.debug("Coerced column '%s' to datetime64[ns].", c)
                break
            except Exception:
                continue

    if len(time_cols) == 0:
        raise ValueError("No datetime information found in index or columns.")
    if len(time_cols) > 1:
        preferred = [c for c in time_cols if str(c).lower() in {"date", "timestamp"}]
        if len(preferred) == 1:
            date_col = preferred[0]
        else:
            raise ValueError(f"More than one datetime column found: {time_cols}")
    else:
        date_col = time_cols[0]

    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    return df


def check_data(df: pd.DataFrame, feature_names: list[str], value: str) -> pd.DataFrame:
    """
    Validate target column and restrict DataFrame to relevant variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset (must contain ``value`` and ``date``).
    feature_names : list of str
        Requested predictor columns.
    value : str
        Target variable name.

    Returns
    -------
    pandas.DataFrame
        Subset with predictors, ``date``, and target renamed to ``value``.

    Raises
    ------
    ValueError
        If target missing, or ``date`` not datetime, or has NA.
    """
    if value not in df.columns:
        raise ValueError(f"The target variable `{value}` is not in the DataFrame columns.")

    selected = [c for c in feature_names if c in df.columns]
    if not selected:
        log.warning("No requested features found; proceeding with 'date' + target only.")
    selected.extend(["date", value])
    df_sel = df[selected].copy()

    if not pdt.is_datetime64_any_dtype(df_sel["date"]):
        raise ValueError("`date` must be datetime64[ns] or datetimetz.")

    if value != "value":
        df_sel = df_sel.rename(columns={value: "value"})

    if df_sel["date"].isna().any():
        raise ValueError("`date` must not contain missing (NA) values.")

    return df_sel


def impute_values(df: pd.DataFrame, na_rm: bool) -> pd.DataFrame:
    """
    Impute or drop missing values in predictors and target.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset with target and features.
    na_rm : bool
        If True, drop rows with NA in target ``value``.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with missing values handled.
    """
    out = df.copy()

    if na_rm:
        before = len(out)
        out = out.dropna(subset=["value"]).reset_index(drop=True)
        dropped = before - len(out)
        if dropped:
            log.info("Dropped %d rows with NA in target.", dropped)

    for col in out.select_dtypes(include=[np.number]).columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())

    for col in out.select_dtypes(include=["object", "category"]).columns:
        if out[col].isna().any():
            mode_series = out[col].mode(dropna=True)
            if not mode_series.empty:
                out[col] = out[col].fillna(mode_series.iloc[0])
            else:
                log.warning("Column '%s' has only NA values; left unchanged.", col)

    return out


def add_date_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic date/time-derived features from ``date``.

    Adds:
      - ``date_unix`` : seconds since epoch
      - ``day_julian``: day of year
      - ``weekday``   : day of week (1=Mon..7=Sun, categorical)
      - ``hour``      : hour of day

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset with column ``date``.

    Returns
    -------
    pandas.DataFrame
        Copy of dataset with added time-derived variables.
    """
    out = df.copy()
    dt = pd.DatetimeIndex(out["date"])
    if dt.tz is not None:
        dt = dt.tz_convert("UTC").tz_localize(None)

    out.loc[:, "date_unix"] = dt.view(np.int64) // 10**9  # type: ignore[attr-defined]
    out.loc[:, "day_julian"] = dt.dayofyear
    out["weekday"] = pd.Categorical(dt.weekday + 1)
    out.loc[:, "hour"] = dt.hour
    return out


def split_into_sets(
    df: pd.DataFrame, split_method: str, fraction: float, seed: int
) -> pd.DataFrame:
    """
    Split dataset into training/testing subsets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset with ``date`` column.
    split_method : {"random","ts","season","month"}
        Splitting strategy:
          - "random": random sample by fraction.
          - "ts": sequential split by time order.
          - "season": stratified by climatological season (DJF, MAM, JJA, SON).
          - "month": stratified by calendar month.
    fraction : float
        Proportion of rows per group to assign to training.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Dataset with added column ``set`` ("training" or "testing").

    Raises
    ------
    ValueError
        If ``split_method`` is invalid.
    """
    out = df.sort_values("date").reset_index(drop=True)

    if split_method == "random":
        out["set"] = "testing"
        out.loc[out.sample(frac=fraction, random_state=seed).index, "set"] = "training"

    elif split_method == "ts":
        n = len(out)
        cut = int(fraction * n)
        out["set"] = np.where(np.arange(n) < cut, "training", "testing")

    elif split_method == "season":
        season_map = {
            12: "DJF",
            1: "DJF",
            2: "DJF",
            3: "MAM",
            4: "MAM",
            5: "MAM",
            6: "JJA",
            7: "JJA",
            8: "JJA",
            9: "SON",
            10: "SON",
            11: "SON",
        }
        out["season"] = out["date"].dt.month.map(season_map)
        out["set"] = "testing"
        rng = np.random.default_rng(seed)
        for s in out["season"].dropna().unique():
            idx = out.index[out["season"] == s]
            k = int(fraction * len(idx))
            if k > 0:
                train_idx = rng.choice(idx.to_numpy(), size=k, replace=False)
                out.loc[train_idx, "set"] = "training"
        out = out.drop(columns=["season"])

    elif split_method == "month":
        out["month"] = out["date"].dt.month
        out["set"] = "testing"
        rng = np.random.default_rng(seed)
        for m in range(1, 13):
            idx = out.index[out["month"] == m]
            k = int(fraction * len(idx))
            if k > 0:
                train_idx = rng.choice(idx.to_numpy(), size=k, replace=False)
                out.loc[train_idx, "set"] = "training"
        out = out.drop(columns=["month"])

    else:
        raise ValueError(f"Unknown split_method '{split_method}'.")

    return out
