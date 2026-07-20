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
    target: str,
    covariates: list[str],
    dropna: bool = True,
    split_method: str = "random",
    train_fraction: float = 0.75,
    seed: int = 7_654_321,
) -> pd.DataFrame:
    """
    Clean, validate, and split the input DataFrame in a single pipeline.

    Steps:
      1) Ensure a datetime column named ``date`` is present.
      2) Validate target and covariates.
      3) Impute/drop missing values.
      4) Add derived date variables (unix, julian day, weekday, hour).
      5) Split into training/testing sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input dataset containing at least the target column and date/time info.
    target : str
        Target column name in ``df``.
    covariates : list of str
        External predictor variable names to keep (must exist in ``df``).
        Time variables (``date_unix``/``day_julian``/``weekday``/``hour``)
        are added automatically by this pipeline -- don't list them here;
        see :func:`add_date_variables`.
    dropna : bool, default True
        If True, drop rows where the target is NA. Also imputes other NAs.
    split_method : {"random","ts","month_ts","season_ts"}, default "random"
        Train/test split method. See :func:`split_into_sets` for the exact
        mechanics -- in particular, "month_ts"/"season_ts" hold out a
        contiguous block at a randomised (seeded) position within every
        period, rather than always the trailing slice; read that
        docstring's note for why that matters and its remaining caveats
        before choosing them over "random".
    train_fraction : float, default 0.75
        Training fraction for data splitting.
    seed : int, default 7654321
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Processed dataset with:
          - ``date`` column
          - ``value`` column (target, renamed internally)
          - covariates
          - derived date features
          - ``set`` column indicating "training"/"testing".
    """
    log.debug(
        "Preparing data with split_method=%s, train_fraction=%.3f", split_method, train_fraction
    )
    df_out = (
        df.pipe(process_date)
        .pipe(check_data, covariates=covariates, target=target)
        .pipe(impute_values, dropna=dropna)
        .pipe(add_date_variables)
        .pipe(split_into_sets, split_method=split_method, train_fraction=train_fraction, seed=seed)
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


def check_data(df: pd.DataFrame, covariates: list[str], target: str) -> pd.DataFrame:
    """
    Validate target column and restrict DataFrame to relevant variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset (must contain ``target`` and ``date``).
    covariates : list of str
        Requested predictor columns.
    target : str
        Target variable name.

    Returns
    -------
    pandas.DataFrame
        Subset with covariates, ``date``, and target renamed to ``value``.

    Raises
    ------
    ValueError
        If target missing, or ``date`` not datetime, or has NA.
    """
    if target not in df.columns:
        raise ValueError(f"The target variable `{target}` is not in the DataFrame columns.")

    # Excludes "date"/target here (added back once below) so a caller who
    # accidentally includes either in `covariates` doesn't end up with
    # duplicate-named columns in `df_sel` -- `df[selected]` on a list with
    # a repeated name silently returns a 2-column-wide slice under that
    # name, which corrupts everything downstream with a confusing failure
    # far from the actual mistake.
    selected = [c for c in covariates if c in df.columns and c not in ("date", target)]
    if not selected:
        log.warning("No requested covariates found; proceeding with 'date' + target only.")
    selected.extend(["date", target])
    df_sel = df[selected].copy()

    if not pdt.is_datetime64_any_dtype(df_sel["date"]):
        raise ValueError("`date` must be datetime64[ns] or datetimetz.")

    if target != "value":
        df_sel = df_sel.rename(columns={target: "value"})

    if df_sel["date"].isna().any():
        raise ValueError("`date` must not contain missing (NA) values.")

    return df_sel


def impute_values(df: pd.DataFrame, dropna: bool) -> pd.DataFrame:
    """
    Impute or drop missing values in predictors and target.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset with target and features.
    dropna : bool
        If True, drop rows with NA in target ``value``.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with missing values handled.
    """
    out = df.copy()

    if dropna:
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

    .. note::
        This always computes and adds all four columns, regardless of
        whether any of them will actually be used to train a model --
        they're opt-in, not mandatory. :func:`normet.build_model` /
        :func:`normet.train_model` only use whichever of these four end up
        in the caller's ``predictors``; a subset (e.g. only ``weekday``
        and ``hour``, omitting ``date_unix``/``day_julian``) or none at all
        (training purely on meteorology, traffic counts, or other
        non-temporal predictors) both work -- just don't list the ones you
        don't want. :func:`normet.decompose`/:func:`normet.decom_emi`
        adapt automatically, producing a component only for whichever time
        variables actually ended up in the model.

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


def _mark_random_window_training(
    out: pd.DataFrame, group_index: pd.Index, train_fraction: float, rng: np.random.Generator
) -> None:
    """Mark one group's rows "training" outside a random contiguous test window.

    The window (size ``n - int(train_fraction * n)``) is placed at a random
    start position within the group, drawn from ``rng`` -- unlike a fixed
    trailing slice, this doesn't anchor the held-out block to the same
    relative calendar position in every period instance. See
    :func:`split_into_sets` for why this matters. Mutates ``out["set"]`` in
    place for this group's rows; ``out["set"]`` must already default to
    "testing".
    """
    n = len(group_index)
    cut = int(train_fraction * n)
    test_len = n - cut
    max_start = max(n - test_len, 0)
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    train_idx = group_index[:start].append(group_index[start + test_len :])
    out.loc[train_idx, "set"] = "training"


def split_into_sets(
    df: pd.DataFrame, split_method: str, train_fraction: float, seed: int
) -> pd.DataFrame:
    """
    Split dataset into training/testing subsets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset with ``date`` column.
    split_method : {"random","ts","month_ts","season_ts"}
        Splitting strategy:
          - "random": random sample by fraction.
          - "ts": sequential split by time order (single global cutoff).
          - "month_ts": chronological split within each individual
            (year, month); a contiguous (1 - train_fraction) block is held
            out as testing, at a random position within the month (seeded
            by ``seed``, reproducible).
          - "season_ts": chronological split within each individual
            (meteorological year, season) --- DJF/MAM/JJA/SON per year,
            with December assigned to the following year's DJF block; a
            contiguous (1 - train_fraction) block is held out as testing,
            at a random position within the season (seeded by ``seed``,
            reproducible).

        .. note::
            Before this random-position placement, "month_ts"/"season_ts"
            held out a block at a **fixed relative position** within every
            period -- always the trailing ``1 - train_fraction`` -- which
            meant any calendar window consistently falling in that
            trailing slice (e.g. late December, with
            ``train_fraction=0.75``'s trailing ~8 days of a 31-day month)
            was excluded from training in *every* year of the record, not
            just some of them. That created a permanent blind spot: a
            decomposition/feature-importance step built on such a model
            could systematically fail to represent whatever real signal
            lived in that window (observed directly in practice: a genuine
            network-corroborated holiday-period activity drop was absent
            from the recovered components under the old fixed-trailing
            "month_ts" but present under "random"). Randomising the
            window's start position per period (via ``seed``) removes that
            permanent alignment -- a given calendar window is no longer
            excluded in every year. Note this doesn't make a *single* run
            immune to coincidentally missing a specific window in some
            subset of years (the window position is still fixed once
            ``seed`` is fixed, just no longer calendar-anchored); "random"
            remains the safest default for data with a recurring feature
            you must not systematically lose from training.
    train_fraction : float
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
        out.loc[out.sample(frac=train_fraction, random_state=seed).index, "set"] = "training"

    elif split_method == "ts":
        n = len(out)
        cut = int(train_fraction * n)
        out["set"] = np.where(np.arange(n) < cut, "training", "testing")

    elif split_method == "month_ts":
        # Chronological split within each individual (year, month) instance:
        # a contiguous (1 - train_fraction) block, at a random position
        # (seeded), is held out as testing; the rest of the month is
        # training. Combines the "ts" split's within-month temporal
        # ordering (a contiguous block is held out, not randomly
        # interleaved with training hours from the same days) with
        # full-record coverage (every calendar month, across every year,
        # contributes training rows) -- avoiding the trend-feature
        # extrapolation failure a single global "ts" cutoff causes for
        # monotonic features like a Unix-time trend.
        out["ym"] = out["date"].dt.to_period("M")
        out["set"] = "testing"
        rng = np.random.default_rng(seed)
        for _, grp in out.groupby("ym"):
            _mark_random_window_training(out, grp.index, train_fraction, rng)
        out = out.drop(columns=["ym"])

    elif split_method == "season_ts":
        # Chronological split within each individual (climatological year,
        # season) instance: a contiguous (1 - train_fraction) block, at a
        # random position (seeded), is held out as testing. Same
        # construction as "month_ts" but grouped by season instead of
        # calendar month. December is assigned to the following year's DJF
        # block (standard meteorological-year convention), so each DJF
        # group is a single contiguous Dec-Jan-Feb span rather than being
        # split across two groups.
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
        out["season_year"] = out["date"].dt.year
        out.loc[out["date"].dt.month == 12, "season_year"] += 1
        out["set"] = "testing"
        rng = np.random.default_rng(seed)
        for _, grp in out.groupby(["season_year", "season"]):
            _mark_random_window_training(out, grp.index, train_fraction, rng)
        out = out.drop(columns=["season", "season_year"])

    else:
        raise ValueError(f"Unknown split_method '{split_method}'.")

    return out
