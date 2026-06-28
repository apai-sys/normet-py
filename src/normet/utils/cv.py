# src/normet/utils/cv.py
"""
Time-series cross-validation utilities.

Provides walk-forward (a.k.a. forward-chaining) splits that respect temporal
ordering — unlike random K-fold, no future data ever leaks into training.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from .logging import get_logger
from .metrics import _DEFAULT_STATS, _stats_from_arrays

log = get_logger(__name__)

__all__ = ["time_series_cv", "cv_score"]


def time_series_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    *,
    gap: int = 0,
    test_size: int | None = None,
    max_train_size: int | None = None,
    date_col: str = "date",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Yield walk-forward train/test index pairs in temporal order.

    Indices are positional with respect to the *date-sorted* view of ``df``
    (not the original index). For each fold, the test segment immediately
    follows the train segment (plus an optional ``gap``).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a datetime-like ``date_col`` or be indexed by datetime.
    n_splits : int, default 5
        Number of folds.
    gap : int, default 0
        Rows skipped between the end of train and the start of test.
    test_size : int, optional
        Test segment size. If ``None``, splits the tail evenly into ``n_splits``.
    max_train_size : int, optional
        If given, train is a sliding window of this size; otherwise, train
        is expanding (anchored at the start).
    date_col : str, default "date"
        Datetime column to sort by.

    Yields
    ------
    (train_idx, test_idx) : tuple of numpy.ndarray
        Positional indices into the date-sorted frame.

    Raises
    ------
    ValueError
        If parameters are inconsistent with the data length.
    """
    if n_splits < 1:
        raise ValueError("`n_splits` must be >= 1.")
    if gap < 0:
        raise ValueError("`gap` must be >= 0.")

    work = df
    if date_col not in work.columns:
        if isinstance(work.index, pd.DatetimeIndex):
            work = work.reset_index().rename(columns={work.index.name or "index": date_col})
        else:
            raise ValueError(f"`{date_col}` not found and index is not a DatetimeIndex.")
    if not pd.api.types.is_datetime64_any_dtype(work[date_col]):
        raise ValueError(f"`{date_col}` must be datetime-like.")

    n = len(work)
    if n < n_splits + 1:
        raise ValueError(f"Need at least n_splits+1 ({n_splits + 1}) rows; got {n}.")

    sort_pos = np.argsort(work[date_col].to_numpy(), kind="stable")
    # Tests carved out from the tail
    if test_size is None:
        test_size = max(1, n // (n_splits + 1))
    total_test = n_splits * test_size + gap * n_splits
    if total_test >= n:
        raise ValueError(
            f"With n={n}, n_splits={n_splits}, test_size={test_size}, gap={gap}, "
            "no room left for training. Reduce n_splits or test_size."
        )

    # Compute fold boundaries from the end backward, then yield forward.
    end_indices = [n - i * test_size for i in range(n_splits)][::-1]
    for test_end in end_indices:
        test_start = test_end - test_size
        train_end = test_start - gap
        if max_train_size is not None and train_end > max_train_size:
            train_start = train_end - max_train_size
        else:
            train_start = 0
        if train_end - train_start < 1:
            log.warning(
                "Empty train fold (train_end=%d, train_start=%d) — skipping.",
                train_end,
                train_start,
            )
            continue
        train_idx = sort_pos[train_start:train_end]
        test_idx = sort_pos[test_start:test_end]
        yield train_idx, test_idx


def cv_score(
    df: pd.DataFrame,
    value: str = "value",
    feature_names: list[str] | None = None,
    *,
    backend: str = "flaml",
    n_splits: int = 5,
    gap: int = 0,
    test_size: int | None = None,
    max_train_size: int | None = None,
    statistic: list[str] | None = None,
    model_config: dict[str, Any] | None = None,
    seed: int = 7_654_321,
    n_cores: int | None = None,
    verbose: bool = False,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Train + evaluate the configured AutoML backend across walk-forward folds.

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared data with a datetime column ``date_col``, target column,
        and predictor columns.
    value : str, default "value"
        Target column name.
    feature_names : list of str, optional
        Predictor columns. Must be non-empty.
    backend : {"flaml"}, default "flaml"
        AutoML backend.
    n_splits, gap, test_size, max_train_size, date_col :
        Forwarded to :func:`time_series_cv`.
    statistic : list of str, optional
        Metric keys (see :func:`normet.utils.metrics.modStats`). Defaults to
        the comprehensive set.
    model_config : dict, optional
        Backend-specific training config.
    seed, n_cores, verbose :
        Forwarded to the backend trainer.

    Returns
    -------
    pandas.DataFrame
        One row per fold (plus an optional pooled row), with metric columns
        and metadata: ``fold``, ``train_start``, ``train_end``, ``test_start``,
        ``test_end``, ``n_train``, ``n_test``.
    """
    if not feature_names:
        raise ValueError("`feature_names` must be a non-empty list.")
    if value not in df.columns:
        raise ValueError(f"Target column '{value}' not found.")
    stats_keys = statistic or _DEFAULT_STATS

    from ..model.predict import ml_predict
    from ..model.train import train_model

    work = df.copy()
    if date_col in work.columns and pd.api.types.is_datetime64_any_dtype(work[date_col]):
        work = work.sort_values(date_col).reset_index(drop=True)
    elif isinstance(work.index, pd.DatetimeIndex):
        work = work.reset_index().rename(columns={work.index.name or "index": date_col})
        work = work.sort_values(date_col).reset_index(drop=True)
    else:
        raise ValueError(
            f"`{date_col}` must be present and datetime-like (or use a DatetimeIndex)."
        )

    folds: list[pd.DataFrame] = []
    for i, (tr_idx, te_idx) in enumerate(
        time_series_cv(
            work,
            n_splits=n_splits,
            gap=gap,
            test_size=test_size,
            max_train_size=max_train_size,
            date_col=date_col,
        )
    ):
        df_tr = work.iloc[tr_idx]
        df_te = work.iloc[te_idx]
        model = train_model(
            df=df_tr,
            value=value,
            backend=backend,
            feature_names=list(feature_names),
            model_config=model_config,
            seed=seed,
            n_cores=n_cores,
            verbose=verbose,
        )
        y_pred = ml_predict(model, df_te)
        y_true = df_te[value].to_numpy()
        row = _stats_from_arrays(y_pred, y_true, stats_keys)
        row["fold"] = i
        row["train_start"] = df_tr[date_col].iloc[0]
        row["train_end"] = df_tr[date_col].iloc[-1]
        row["test_start"] = df_te[date_col].iloc[0]
        row["test_end"] = df_te[date_col].iloc[-1]
        row["n_train"] = len(df_tr)
        row["n_test"] = len(df_te)
        folds.append(row)

    if not folds:
        return pd.DataFrame()

    out = pd.concat(folds, ignore_index=True)
    meta_cols = ["fold", "train_start", "train_end", "test_start", "test_end", "n_train", "n_test"]
    ordered = meta_cols + [c for c in out.columns if c not in meta_cols]
    return out[ordered]
