# src/normet/utils/cv.py
"""
Time-series cross-validation utilities.

Provides walk-forward (a.k.a. forward-chaining) splits that respect temporal
ordering — unlike random K-fold, no future data ever leaks into training.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .logging import get_logger
from .metrics import _DEFAULT_STATS, _stats_from_arrays
from .prepare import compute_duan_smearing, impute_values, log1p_target

log = get_logger(__name__)

__all__ = ["time_series_cv", "blocked_time_cv", "cv_score"]


def blocked_time_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    *,
    embargo: int = 0,
    date_col: str = "date",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Yield contiguous blocked K-fold train/test index pairs with optional embargo.

    Unlike walk-forward CV (:func:`time_series_cv`), every observation falls
    into exactly one held-out validation block across the ``n_splits`` folds,
    so the union of validation blocks covers the full record out-of-fold.
    For each fold ``k``, rows within ``embargo`` positions immediately before or
    after the validation block are excluded from that fold's training set to
    suppress serial-correlation leakage across block boundaries.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a datetime-like ``date_col`` or be indexed by datetime.
    n_splits : int, default 5
        Number of contiguous temporal blocks (must be >= 2).
    embargo : int, default 0
        Number of rows skipped immediately before and after each test block.
    date_col : str, default "date"
        Datetime column used to establish temporal order.

    Yields
    ------
    (train_idx, test_idx) : tuple of numpy.ndarray
        Positional indices into ``df`` (for ``df.iloc``), ordered by time.
    """
    if int(n_splits) < 2:
        raise ValueError("`n_splits` must be >= 2 for blocked_time_cv.")
    if int(embargo) < 0:
        raise ValueError("`embargo` must be >= 0.")

    work = df
    if date_col not in work.columns:
        if isinstance(work.index, pd.DatetimeIndex):
            work = work.reset_index().rename(columns={work.index.name or "index": date_col})
        else:
            raise ValueError(f"`{date_col}` not found and index is not a DatetimeIndex.")
    if not pd.api.types.is_datetime64_any_dtype(work[date_col]):
        raise ValueError(f"`{date_col}` must be datetime-like.")

    n = len(work)
    k_splits = int(n_splits)
    emb = int(embargo)
    if n < k_splits:
        raise ValueError(f"Need at least n_splits ({k_splits}) rows; got {n}.")

    sort_pos = np.argsort(work[date_col].to_numpy(), kind="stable")
    blocks = np.array_split(np.arange(n, dtype=np.int64), k_splits)
    for blk in blocks:
        s = int(blk[0])
        e = int(blk[-1]) + 1
        left = np.arange(0, max(0, s - emb), dtype=np.int64)
        right = np.arange(min(n, e + emb), n, dtype=np.int64)
        tr_pos = np.concatenate([left, right])
        if tr_pos.size == 0:
            raise ValueError(
                f"With n={n}, n_splits={k_splits}, embargo={emb}, no room left for training."
            )
        yield sort_pos[tr_pos], sort_pos[blk]


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

    Indices are positional (``iloc``) into ``df`` itself, gathered in date
    order -- not positions into a separately sorted copy, so ``df.iloc[idx]``
    selects the fold's rows. For each fold, the test segment immediately
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
        Positional indices into ``df`` (for ``df.iloc``), in date order.

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
    # The earliest fold is the binding one: its test tail starts at
    # ``n - n_splits * test_size`` and its gap precedes that. Multiplying the
    # gap by the number of splits counted rows the later gaps reuse.
    train_rows = n - n_splits * test_size - gap
    if train_rows < 1:
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
    target: str = "value",
    covariates: list[str] | None = None,
    *,
    backend: str = "flaml",
    strategy: str = "walk_forward",
    n_splits: int = 5,
    gap: int = 0,
    embargo: int = 0,
    test_size: int | None = None,
    max_train_size: int | None = None,
    statistic: list[str] | None = None,
    model_config: dict[str, Any] | None = None,
    impute: str = "none",
    max_gap: int | None = 6,
    circular_cols: Sequence[str] | None = None,
    log_transform: bool = False,
    back_transform: str = "arithmetic",
    include_pooled: bool = False,
    return_oof: bool = False,
    seed: int = 7_654_321,
    n_cores: int | None = None,
    verbose: bool = False,
    date_col: str = "date",
) -> Any:
    """
    Train + evaluate the configured AutoML backend across temporal CV folds.

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared data with a datetime column ``date_col``, target column,
        and predictor columns.
    target : str, default "value"
        Target column name.
    covariates : list of str, optional
        Predictor columns. Must be non-empty.
    backend : {"flaml", "lightgbm"}, default "flaml"
        AutoML backend.
    strategy : {"walk_forward", "blocked"}, default "walk_forward"
        Cross-validation splitter: forward-chaining ``walk_forward`` via
        :func:`time_series_cv` or full-coverage ``blocked`` via :func:`blocked_time_cv`.
    n_splits, gap, embargo, test_size, max_train_size, date_col :
        Forwarded to the chosen CV splitter.
    statistic : list of str, optional
        Metric keys (see :func:`normet.utils.metrics.modStats`). Defaults to
        the comprehensive set.
    model_config : dict, optional
        Backend-specific training config.
    impute, max_gap, circular_cols :
        When ``impute != "none"``, fit predictor imputation strictly within
        each fold's training set to eliminate cross-fold leakage.
    log_transform : bool, default False
        If True, fit each fold on ``log1p(target)`` and evaluate OOF metrics
        on original units using ``back_transform``.
    back_transform : {"arithmetic", "smearing", "geometric"}, default "arithmetic"
        Log-scale inversion rule when ``log_transform=True``.
    include_pooled : bool, default False
        If True, append a ``fold="pooled"`` summary row computed over all
        concatenated out-of-fold predictions.
    return_oof : bool, default False
        If True, return ``(cv_table, oof_df)`` where ``oof_df`` carries
        ``["date", "observed", "predicted", "fold"]``.
    seed, n_cores, verbose :
        Forwarded to the backend trainer.

    Returns
    -------
    pandas.DataFrame or (pandas.DataFrame, pandas.DataFrame)
        Per-fold metric summary (plus optional pooled row and OOF predictions).
    """
    if not covariates:
        raise ValueError("`covariates` must be a non-empty list.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    strat_norm = str(strategy or "walk_forward").lower()
    if strat_norm not in ("walk_forward", "blocked"):
        raise ValueError(f"`strategy` must be 'walk_forward' or 'blocked', got {strategy!r}.")
    if back_transform not in ("arithmetic", "smearing", "geometric"):
        raise ValueError(
            f"`back_transform` must be 'arithmetic', 'smearing', or 'geometric', got {back_transform!r}."
        )
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

    splitter = (
        blocked_time_cv(
            work,
            n_splits=n_splits,
            embargo=embargo if embargo > 0 else gap,
            date_col=date_col,
        )
        if strat_norm == "blocked"
        else time_series_cv(
            work,
            n_splits=n_splits,
            gap=gap,
            test_size=test_size,
            max_train_size=max_train_size,
            date_col=date_col,
        )
    )

    folds: list[pd.DataFrame] = []
    oof_dates: list[np.ndarray] = []
    oof_obs: list[np.ndarray] = []
    oof_pred: list[np.ndarray] = []
    oof_fold: list[np.ndarray] = []

    for i, (tr_idx, te_idx) in enumerate(splitter):
        if impute != "none":
            fold_work = work.copy()
            labels = np.full(len(fold_work), "embargo", dtype=object)
            labels[tr_idx] = "training"
            labels[te_idx] = "testing"
            fold_work["set"] = labels
            fold_work = impute_values(
                fold_work,
                dropna=False,
                strategy=impute,
                max_gap=max_gap,
                circular_cols=circular_cols,
            )
            df_tr = fold_work.iloc[tr_idx].copy()
            df_te = fold_work.iloc[te_idx].copy()
        else:
            df_tr = work.iloc[tr_idx].copy()
            df_te = work.iloc[te_idx].copy()

        y_true = pd.to_numeric(df_te[target], errors="coerce").to_numpy(dtype=float)
        if log_transform:
            df_tr = log1p_target(df_tr, target)

        model = train_model(
            df=df_tr,
            target=target,
            backend=backend,
            covariates=list(covariates),
            model_config=model_config,
            seed=seed,
            n_cores=n_cores,
            verbose=verbose,
        )
        raw_pred = np.asarray(ml_predict(model, df_te), dtype=float)
        if log_transform:
            phi = 1.0
            if back_transform == "smearing":
                tr_raw = np.asarray(ml_predict(model, df_tr), dtype=float)
                phi = compute_duan_smearing(df_tr[target], tr_raw)
            y_pred = (
                np.clip(phi * np.exp(raw_pred) - 1.0, 0.0, None)
                if back_transform == "smearing"
                else np.clip(np.expm1(raw_pred), 0.0, None)
            )
        else:
            y_pred = raw_pred

        row = _stats_from_arrays(y_pred, y_true, stats_keys)
        row["fold"] = i
        row["train_start"] = df_tr[date_col].iloc[0]
        row["train_end"] = df_tr[date_col].iloc[-1]
        row["test_start"] = df_te[date_col].iloc[0]
        row["test_end"] = df_te[date_col].iloc[-1]
        row["n_train"] = len(df_tr)
        row["n_test"] = len(df_te)
        folds.append(row)

        oof_dates.append(df_te[date_col].to_numpy())
        oof_obs.append(y_true)
        oof_pred.append(y_pred)
        oof_fold.append(np.full(len(df_te), i, dtype=int))

    empty_oof = pd.DataFrame(columns=[date_col, "observed", "predicted", "fold"])
    if not folds:
        return (pd.DataFrame(), empty_oof) if return_oof else pd.DataFrame()

    if include_pooled and oof_obs:
        all_obs = np.concatenate(oof_obs)
        all_pred = np.concatenate(oof_pred)
        pooled_row = _stats_from_arrays(all_pred, all_obs, stats_keys)
        pooled_row["fold"] = "pooled"
        pooled_row["train_start"] = pd.concat([r["train_start"] for r in folds]).min()
        pooled_row["train_end"] = pd.concat([r["train_end"] for r in folds]).max()
        pooled_row["test_start"] = pd.concat([r["test_start"] for r in folds]).min()
        pooled_row["test_end"] = pd.concat([r["test_end"] for r in folds]).max()
        pooled_row["n_train"] = int(sum(int(r["n_train"].iloc[0]) for r in folds))
        pooled_row["n_test"] = int(len(all_obs))
        folds.append(pooled_row)

    out = pd.concat(folds, ignore_index=True)
    meta_cols = ["fold", "train_start", "train_end", "test_start", "test_end", "n_train", "n_test"]
    ordered = meta_cols + [c for c in out.columns if c not in meta_cols]
    out_df = out[ordered]

    if return_oof:
        oof_df = pd.DataFrame(
            {
                date_col: np.concatenate(oof_dates),
                "observed": np.concatenate(oof_obs),
                "predicted": np.concatenate(oof_pred),
                "fold": np.concatenate(oof_fold),
            }
        )
        return out_df, oof_df
    return out_df
