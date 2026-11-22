# src/normet/utils/prepare.py
"""Data preparation: imputation, date features, train/test/season splits, and validation."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

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
    "log1p_target",
    "expm1_columns",
    "compute_duan_smearing",
]


def prepare_data(
    df: pd.DataFrame,
    target: str,
    covariates: list[str],
    dropna: bool = True,
    split_method: str = "random",
    train_fraction: float = 0.75,
    seed: int = 7_654_321,
    *,
    impute: str = "median",
    max_gap: int | None = 6,
    embargo: int = 0,
    circular_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Clean, validate, and split the input DataFrame in a single pipeline.

    Steps:
      1) Ensure a datetime column named ``date`` is present.
      2) Validate target and covariates (and drop NA targets before splitting when ``dropna=True``).
      3) Add derived date variables (unix, julian day, weekday, hour).
      4) Split into training/testing sets (with optional ``embargo`` around test blocks).
      5) Impute missing predictor values (see ``impute``; ``"train_median"``
         restricts the statistics to the training rows).

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
        If True, drop rows where the target is NA before splitting.
    split_method : {"random","ts","month_ts","season_ts","blocked"}, default "random"
        Train/test split method. See :func:`split_into_sets`.
    train_fraction : float, default 0.75
        Training fraction for data splitting.
    seed : int, default 7654321
        Random seed for reproducibility.
    impute : {"median","train_median","time_interp","none"}, default "median"
        Missing-value handling applied *after* splitting.
        ``"median"`` (the default, and the behaviour before the strategy was
        configurable) fills numeric NAs from the full-frame median and
        categorical NAs from the full-frame mode; ``"train_median"`` opts into
        training-set-only statistics, so test rows never leak into the fill
        values.
    max_gap : int or None, default 6
        Maximum consecutive missing steps filled by linear temporal interpolation
        when ``impute="time_interp"``. Larger gaps fall back to training medians.
    embargo : int, default 0
        Number of adjacent training rows marked ``"embargo"`` around each contiguous
        testing block (useful for autocorrelated hourly series).
    circular_cols : sequence of str, optional
        Angular degree columns (e.g. ``("wd",)``) interpolated via ``sin``/``cos``
        when ``impute="time_interp"``.

    Returns
    -------
    pandas.DataFrame
        Processed dataset with:
          - ``date`` column
          - ``value`` column (target, renamed internally)
          - covariates
          - derived date features
          - ``set`` column indicating "training"/"testing" (and "embargo" if ``embargo > 0``).
    """
    log.debug(
        "Preparing data with split_method=%s, train_fraction=%.3f", split_method, train_fraction
    )
    df_out = df.pipe(process_date).pipe(check_data, covariates=covariates, target=target)
    if dropna and "value" in df_out.columns:
        before = len(df_out)
        df_out = df_out.dropna(subset=["value"]).reset_index(drop=True)
        dropped = before - len(df_out)
        if dropped:
            log.info("Dropped %d rows with NA in target.", dropped)
    df_out = (
        df_out.pipe(add_date_variables)
        .pipe(
            split_into_sets,
            split_method=split_method,
            train_fraction=train_fraction,
            seed=seed,
            embargo=embargo,
        )
        .pipe(
            impute_values,
            dropna=False,
            strategy=impute,
            max_gap=max_gap,
            circular_cols=circular_cols,
        )
        .reset_index(drop=True)
    )
    log.info("Prepared data: %d rows, %d columns", len(df_out), df_out.shape[1])
    return df_out


def _coerce_datetime(values: pd.Series) -> pd.Series | None:
    """One column read as datetimes, or None when it does not parse.

    ``pd.to_datetime(..., utc=False)`` keeps a single UTC offset, but a
    column spanning a daylight-saving change has two: pandas 2.x returns an
    object column of timestamps (with a FutureWarning) and pandas 3 raises
    "Mixed timezones detected". Both used to make the column unusable here.
    The offsets have no one zone to keep, so a mixed column is read on UTC
    instead -- the same convention as :func:`normet.utils._time.to_datetime_coerced`
    and ``normet.io.tables``.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", ".*parsing datetimes with mixed time zones", FutureWarning
            )
            out = pd.to_datetime(values, errors="raise", utc=False)
    except (ValueError, TypeError, OverflowError):
        out = None
    if out is not None and pdt.is_datetime64_any_dtype(out):
        return out
    try:
        return pd.to_datetime(values, errors="raise", utc=True)
    except (ValueError, TypeError, OverflowError):
        return None


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
        # A column named "date" outranks one named "time": a table with both
        # (time,date,NO2) is a clock beside a full date, the reverse of the
        # order the header happens to use.
        priority = {"date": 0, "datetime": 1, "timestamp": 2, "time": 3}
        candidates = sorted(
            (c for c in df.columns if str(c).lower() in priority),
            key=lambda c: priority[str(c).lower()],
        )
        for c in candidates:
            coerced = _coerce_datetime(df[c])
            if coerced is not None:
                df = df.copy()
                df[c] = coerced
                time_cols = [c]
                log.debug("Coerced column '%s' to datetime64[ns].", c)
                break

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


def _fill_numeric_median(out: pd.DataFrame, col: str, ref_mask: np.ndarray) -> None:
    """Fill remaining NA in numeric *col* using median fit on *ref_mask* rows."""
    ref_vals = pd.to_numeric(out.loc[ref_mask, col], errors="coerce").to_numpy(dtype=float)
    finite_ref = ref_vals[np.isfinite(ref_vals)]
    if finite_ref.size > 0:
        median = float(np.median(finite_ref))
    else:
        all_vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        finite_all = all_vals[np.isfinite(all_vals)]
        if finite_all.size == 0:
            return
        median = float(np.median(finite_all))
    try:
        out[col] = out[col].fillna(median)
    except (TypeError, ValueError):
        # A nullable Int64 column accepts only a whole-number fill;
        # a 1.5 median raises "Invalid value '1.5' for dtype
        # 'Int64'" (whether it did depended on the data). The column
        # has to become float to carry the median.
        out[col] = out[col].astype(float).fillna(median)


def impute_values(
    df: pd.DataFrame,
    dropna: bool = True,
    *,
    strategy: str = "median",
    max_gap: int | None = 6,
    circular_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Impute or drop missing values in predictors and target.

    ``"train_median"`` (and the fallback used by ``"time_interp"``) estimates
    summary statistics strictly from the ``"training"`` rows of a frame
    carrying a ``"set"`` column, so held-out test distributions never leak into
    the fill values. ``"median"`` is the legacy full-frame fill.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset with target and features.
    dropna : bool, default True
        If True, drop rows with NA in target ``value``.
    strategy : {"median", "train_median", "time_interp", "none"}, default "median"
        - ``"median"``: fill numeric NAs with the full-frame median and
          categorical NAs with the full-frame mode (legacy behaviour).
        - ``"train_median"``: the same fill, but the median/mode is fit on the
          ``"training"`` rows only (opt-in leak-free behaviour).
        - ``"time_interp"``: temporally interpolate numeric predictors over short
          gaps up to ``max_gap`` rows (using ``sin``/``cos`` interpolation for
          ``circular_cols``), then fill any remaining gaps with training medians.
        - ``"none"``: only drop NA targets (if ``dropna=True``); leave predictor
          NAs intact for tree backends with native missing-value routing.
    max_gap : int or None, default 6
        Maximum consecutive missing steps filled by linear interpolation when
        ``strategy="time_interp"``. ``None`` places no gap limit.
    circular_cols : sequence of str, optional
        Degree columns (e.g. wind direction ``"wd"``) to interpolate on the circle
        via ``sin``/``cos`` when ``strategy="time_interp"``.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with missing values handled.
    """
    valid_strategies = {"median", "train_median", "time_interp", "none"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown `impute` strategy {strategy!r}; expected one of {sorted(valid_strategies)}."
        )
    if max_gap is not None and int(max_gap) < 0:
        raise ValueError(f"`max_gap` must be non-negative or None, got {max_gap!r}.")

    out = df.copy()

    if dropna and "value" in out.columns:
        before = len(out)
        out = out.dropna(subset=["value"]).reset_index(drop=True)
        dropped = before - len(out)
        if dropped:
            log.info("Dropped %d rows with NA in target.", dropped)

    if strategy == "none" or out.empty:
        return out

    has_train = "set" in out.columns and bool((out["set"] == "training").any())
    # Only the opt-in strategies restrict their statistics to the training
    # rows; plain "median" must keep the legacy full-frame fill, otherwise
    # existing prepare_data()/do_all() results change without an opt-in.
    use_train = has_train and strategy in ("train_median", "time_interp")
    ref_mask = (out["set"] == "training").to_numpy() if use_train else np.ones(len(out), dtype=bool)
    exclude_interp = {"value", "date_unix", "day_julian", "hour"}
    circular_set = set(circular_cols or ())

    if strategy == "time_interp":
        order_idx = (
            out["date"].sort_values(kind="stable").index if "date" in out.columns else out.index
        )
        limit_val = int(max_gap) if max_gap is not None else None
        for col in out.select_dtypes(include=[np.number]).columns:
            if not out[col].isna().any() or col in exclude_interp:
                continue
            if limit_val is None or limit_val > 0:
                s_sorted = out.loc[order_idx, col].astype(float)
                na_sorted = s_sorted.isna().to_numpy()
                if col in circular_set:
                    rad = np.deg2rad(s_sorted.to_numpy(dtype=float))
                    sin_s = pd.Series(np.sin(rad), index=order_idx).interpolate(
                        method="linear", limit=limit_val, limit_direction="both"
                    )
                    cos_s = pd.Series(np.cos(rad), index=order_idx).interpolate(
                        method="linear", limit=limit_val, limit_direction="both"
                    )
                    sin_v = sin_s.to_numpy(dtype=float)
                    cos_v = cos_s.to_numpy(dtype=float)
                    ok = (
                        na_sorted
                        & np.isfinite(sin_v)
                        & np.isfinite(cos_v)
                        & (np.hypot(sin_v, cos_v) > 1e-12)
                    )
                    filled = s_sorted.to_numpy(dtype=float).copy()
                    filled[ok] = np.rad2deg(np.arctan2(sin_v[ok], cos_v[ok])) % 360.0
                    out.loc[order_idx, col] = filled
                else:
                    interp_s = s_sorted.interpolate(
                        method="linear", limit=limit_val, limit_direction="both"
                    )
                    try:
                        out.loc[order_idx, col] = interp_s.to_numpy()
                    except (TypeError, ValueError):
                        out[col] = out[col].astype(float)
                        out.loc[order_idx, col] = interp_s.to_numpy(dtype=float)

    for col in out.select_dtypes(include=[np.number]).columns:
        if out[col].isna().any():
            _fill_numeric_median(out, col, ref_mask)

    for col in out.select_dtypes(include=["object", "category"]).columns:
        if col in {"date", "set"}:
            continue
        if out[col].isna().any():
            mode_series = out.loc[ref_mask, col].mode(dropna=True)
            if mode_series.empty:
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

    # astype("int64") counts in the index's own resolution, not nanoseconds:
    # pandas >= 2.0 keeps a series' unit (datetime64[s]/[ms]/[us], e.g. from
    # Parquet) through ``DatetimeIndex``, so without the explicit ns conversion
    # ``date_unix`` came out in seconds/ms/us -- 1.7e9 -> 1.7 for a s-resolution
    # column -- and the trend feature silently lost its scale.
    out["date_unix"] = dt.as_unit("ns").astype("int64").to_numpy() // 10**9
    out["day_julian"] = dt.dayofyear.to_numpy()
    out["weekday"] = pd.Categorical(dt.weekday + 1)
    out["hour"] = dt.hour.to_numpy()
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
    df: pd.DataFrame,
    split_method: str,
    train_fraction: float,
    seed: int,
    *,
    embargo: int = 0,
) -> pd.DataFrame:
    """
    Split dataset into training/testing subsets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset with ``date`` column.
    split_method : {"random","ts","month_ts","season_ts","blocked"}
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
          - "blocked": partitions the sorted record into contiguous temporal
            blocks and holds out a seeded subset of blocks as ``"testing"``
            while assigning the rest to ``"training"``.

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
    embargo : int, default 0
        When ``> 0``, marks up to ``embargo`` training rows immediately before and
        after every contiguous ``"testing"`` block as ``"embargo"`` so adjacent
        autocorrelated timestamps never straddle the train/test boundary.

    Returns
    -------
    pandas.DataFrame
        Dataset with added column ``set`` ("training", "testing", or "embargo").

    Raises
    ------
    ValueError
        If ``split_method`` is invalid or ``embargo < 0``.
    """
    emb = int(embargo)
    if emb < 0:
        raise ValueError(f"`embargo` must be >= 0, got {embargo!r}.")

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

    elif split_method == "blocked":
        n = len(out)
        if n <= 1 or train_fraction >= 1.0:
            out["set"] = "training"
        elif train_fraction <= 0.0:
            out["set"] = "testing"
        else:
            n_blocks = min(n, max(4, min(10, n // 24)))
            blocks = np.array_split(np.arange(n, dtype=np.int64), n_blocks)
            n_test = max(
                1, min(len(blocks) - 1, int(round((1.0 - float(train_fraction)) * len(blocks))))
            )
            rng = np.random.default_rng(seed)
            test_ids = set(rng.choice(len(blocks), size=n_test, replace=False).tolist())
            labels = np.full(n, "training", dtype=object)
            for bid in test_ids:
                labels[blocks[bid]] = "testing"
            out["set"] = labels

    else:
        raise ValueError(f"Unknown split_method '{split_method}'.")

    if emb > 0 and len(out) > 0:
        labels = out["set"].to_numpy(dtype=object).copy()
        n = len(labels)
        is_test = labels == "testing"
        i = 0
        while i < n:
            if not is_test[i]:
                i += 1
                continue
            start = i
            while i < n and is_test[i]:
                i += 1
            stop = i
            for k in range(max(0, start - emb), start):
                if labels[k] == "training":
                    labels[k] = "embargo"
            for k in range(stop, min(n, stop + emb)):
                if labels[k] == "training":
                    labels[k] = "embargo"
        if not np.any(labels == "training"):
            raise ValueError(f"`embargo={emb}` excludes all training rows across {n} samples.")
        out["set"] = labels

    return out


def log1p_target(df: pd.DataFrame, target: str = "value") -> pd.DataFrame:
    """Replace ``target`` with ``log1p(target)``, for a log-scale fit.

    A pollutant that ranges from sub-unit to hundreds of µg/m³ spends most of
    its variance in its high hours, so a model fitted on the raw scale buys
    accuracy there at the expense of the low values -- and those are the ones a
    policy or a health assessment reads. Fitting on ``log1p`` makes errors count
    in relative rather than absolute terms: a fixed relative miss maps to a
    bounded error at every level (its weight grows from half a relative unit at
    level 1 toward the full unit), while its absolute error grows without bound.
    The transform also keeps the target non-negative by construction when the
    result is back-transformed.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame carrying ``target``; already prepared (dates parsed, target
        named ``value``) -- this is the step between :func:`prepare_data` and
        the split-aware fit.
    target : str, default "value"
        Column to transform, in place on a copy.

    Returns
    -------
    pandas.DataFrame
        A copy with ``target`` replaced by its ``log1p``.

    Raises
    ------
    ValueError
        If ``target`` is missing, or carries negative values: ``log1p`` is
        undefined below -1, and a concentration that is negative in the
        original units (a de-weathered residual, a difference) is not a
        quantity a log-scale fit can represent.
    """
    if target not in df.columns:
        raise ValueError(f"`log1p_target`: column {target!r} not found.")
    values = pd.to_numeric(df[target], errors="coerce")
    negative = values < 0
    if bool(negative.any()):
        n_bad = int(negative.sum())
        worst = float(values[negative].min())
        raise ValueError(
            f"`log_transform` needs non-negative {target!r}, but {n_bad} rows are "
            f"negative (min {worst:g}). Log-scale normalisation cannot represent them; "
            "drop or floor those rows first, or leave `log_transform=False`."
        )
    out = df.copy()
    out[target] = np.log1p(values.to_numpy(dtype=float))
    return out


def expm1_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Invert :func:`log1p_target`: every numeric column back through ``expm1``.

    Applied to the whole result, not just the point estimate, because every
    column is a value of the same target -- the observations, the normalised
    series and each quantile. ``expm1`` is monotone, so a quantile of the log
    space is the quantile of the original space, and the back-transform cannot
    produce a negative concentration.
    """
    out = df.copy()
    numeric = out.select_dtypes(include=[np.number]).columns
    if len(numeric):
        out[numeric] = np.expm1(out[numeric].to_numpy(dtype=float))
    return out


def compute_duan_smearing(
    y_true_log: np.ndarray | pd.Series,
    y_pred_log: np.ndarray | pd.Series,
) -> float:
    """Compute Duan's (1983) nonparametric smearing factor on the log scale.

    For a model fitted on ``z = log1p(y)`` returning ``z_hat = E[z | X]`` with
    log-scale residual ``r = z - z_hat``, back-transforming by ``expm1(z_hat)``
    alone estimates the conditional geometric mean shifted by ``-1`` rather
    than ``E[y | X]``. Multiplying ``exp(z_hat)`` by the empirical smearing
    factor ``phi = mean(exp(r))`` corrects the log-normal/skewed retransformation
    bias without assuming Gaussian residuals: ``E[y | X] ≈ phi * exp(z_hat) - 1``.

    Parameters
    ----------
    y_true_log : array-like or Series
        Observed target values on the ``log1p`` scale.
    y_pred_log : array-like or Series
        Model predictions on the ``log1p`` scale (same length as ``y_true_log``).

    Returns
    -------
    float
        Duan smearing multiplier ``>= 0`` (defaults to ``1.0`` when no finite pairs exist).
    """
    y_t = np.asarray(y_true_log, dtype=float).reshape(-1)
    y_p = np.asarray(y_pred_log, dtype=float).reshape(-1)
    if y_t.shape != y_p.shape:
        raise ValueError(
            f"`y_true_log` and `y_pred_log` must have matching length ({y_t.size} vs {y_p.size})."
        )
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    if not np.any(mask):
        return 1.0
    resid = np.clip(y_t[mask] - y_p[mask], -50.0, 50.0)
    return float(np.mean(np.exp(resid)))
