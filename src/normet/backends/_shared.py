"""Shared internal helpers for AutoML/tree training backends."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def resolve_sample_weight(
    sample_weight: str | np.ndarray | pd.Series | None,
    df_full: pd.DataFrame,
    df_sub: pd.DataFrame,
    train_mask: np.ndarray | None,
) -> np.ndarray | None:
    """Resolve and validate finite non-negative sample weights aligned to *df_sub*."""
    if sample_weight is None:
        return None
    if isinstance(sample_weight, str):
        if sample_weight not in df_sub.columns:
            raise ValueError(f"sample_weight column {sample_weight!r} not found in DataFrame.")
        arr = pd.to_numeric(df_sub[sample_weight], errors="coerce").to_numpy(dtype=float)
    elif isinstance(sample_weight, pd.Series) and len(sample_weight) == len(df_full):
        if sample_weight.index.equals(df_full.index):
            arr = pd.to_numeric(sample_weight.loc[df_sub.index], errors="coerce").to_numpy(
                dtype=float
            )
        else:
            raw_s = pd.to_numeric(sample_weight, errors="coerce").to_numpy(dtype=float)
            arr = raw_s[train_mask] if train_mask is not None else raw_s
    else:
        raw = np.asarray(sample_weight, dtype=float).reshape(-1)
        if len(raw) == len(df_sub):
            arr = raw
        elif len(raw) == len(df_full) and train_mask is not None:
            arr = raw[train_mask]
        else:
            raise ValueError(
                f"sample_weight length ({len(raw)}) does not match training rows ({len(df_sub)}) "
                f"or input frame ({len(df_full)})."
            )

    if not np.all(np.isfinite(arr)):
        # The documented contract for `build_model`/`train_model`: non-finite
        # observation weights warn and are dropped -- they enter the fit as
        # weight 0, so the affected rows carry no weight. Raising here instead
        # meant the two entry points disagreed (build_model median-imputed the
        # NaNs, a direct train_model call raised).
        bad = ~np.isfinite(arr)
        warnings.warn(
            f"{int(np.sum(bad))} non-finite `sample_weight` value(s) were dropped "
            "(given weight 0).",
            UserWarning,
            stacklevel=2,
        )
        arr = np.where(bad, 0.0, arr)
    if np.any(arr < 0.0):
        raise ValueError("`sample_weight` must contain non-negative values.")
    return arr
