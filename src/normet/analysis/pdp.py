# src/normet/analysis/pdp.py
"""Partial dependence and accumulated local effects (:func:`pdp`, :func:`ale`)."""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd

from ..backends import backend_registry
from ..model.predict import ml_predict
from ..utils._config import resolve_workers
from ..utils.features import extract_features
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["pdp", "ale"]


def _validate_back_transform(back_transform: str, smearing_factor: float) -> float:
    """Validate ``back_transform`` mode and positive finite ``smearing_factor``."""
    if back_transform not in ("arithmetic", "smearing", "geometric"):
        raise ValueError(
            "`back_transform` must be 'arithmetic', 'smearing', or 'geometric', "
            f"got {back_transform!r}."
        )
    sf_obj: object = smearing_factor
    if isinstance(sf_obj, bool) or not isinstance(sf_obj, numbers.Real):
        raise ValueError(f"`smearing_factor` must be a positive finite float, got {sf_obj!r}.")
    sf = float(sf_obj)
    if not np.isfinite(sf) or sf <= 0.0:
        raise ValueError(f"`smearing_factor` must be a positive finite float, got {sf_obj!r}.")
    return sf


def _back_transform_preds(
    yhat: np.ndarray,
    *,
    log_transform: bool,
    back_transform: str,
    smearing_factor: float,
) -> np.ndarray:
    """Invert log1p predictions to original non-negative concentration scale."""
    if not log_transform:
        return yhat
    if back_transform == "smearing":
        return np.clip(smearing_factor * np.exp(yhat) - 1.0, 0.0, None)
    return np.clip(np.expm1(yhat), 0.0, None)


def _resolve_model_and_data(
    df: pd.DataFrame,
    model: object,
    var_list: list[str] | None,
    training_only: bool,
    verbose: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Validate backend, resolve feature list and target variables, and slice data subset."""
    model_type = getattr(model, "backend", None)
    if not model_type or not backend_registry.has(model_type):
        available = backend_registry.available
        raise TypeError(
            f"Unsupported model backend '{model_type}'. Expected one of: {', '.join(available)}"
        )

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

    if var_list is None:
        vars_for_eval = feature_names
    else:
        vars_for_eval = [v for v in var_list if v in feature_names]
        missing = [v for v in var_list if v not in feature_names]
        if missing:
            (log.info if verbose else log.debug)(
                "Skipping vars not present in features: %s", missing
            )

    if "set" in df.columns and training_only:
        X_df = df.loc[df["set"] == "training", feature_names].copy()
        if X_df.empty:
            X_df = df[feature_names].copy()
    else:
        X_df = df[feature_names].copy()

    return X_df, vars_for_eval


def pdp(
    df: pd.DataFrame,
    model: object,
    *,
    var_list: list[str] | None = None,
    training_only: bool = True,
    n_cores: int | None = None,
    grid_points: int = 50,
    quantile_range: tuple[float, float] = (0.01, 0.99),
    log_transform: bool = False,
    back_transform: str = "geometric",
    smearing_factor: float = 1.0,
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
    log_transform : bool, default False
        If True, ``model`` predicts ``log1p(y)`` and results are inverted back
        to original concentration units.
    back_transform : {"arithmetic", "smearing", "geometric"}, default "geometric"
        How to invert log-space predictions when ``log_transform=True``:
        ``"geometric"`` (the default) takes ``expm1(mean_z)`` and the
        delta-method SD ``exp(mean_z) * std_z``, matching normalise's legacy
        inversion; ``"arithmetic"`` applies ``expm1`` before averaging across
        the sample; ``"smearing"`` uses ``smearing_factor * exp(z) - 1``
        before averaging.
    smearing_factor : float, default 1.0
        Duan (1983) residual smearing factor used when ``log_transform=True``
        and ``back_transform="smearing"``.
    verbose : bool, default False
        If True, emit info logs.

    Returns
    -------
    pandas.DataFrame
        Columns: ['variable', 'value', 'pdp_mean', 'pdp_std'].
    """
    sf = _validate_back_transform(back_transform, smearing_factor)
    X_df, vars_for_pdp = _resolve_model_and_data(df, model, var_list, training_only, verbose)

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

    n_cores_eff = resolve_workers(n_cores)
    empty_cols = ["variable", "value", "pdp_mean", "pdp_std"]

    def _one(var: str) -> pd.DataFrame:
        grid = _grid(X_df[var])
        if grid is None or len(grid) == 0:
            (log.info if verbose else log.debug)(
                "Variable '%s' has insufficient numeric spread; skipping.", var
            )
            return pd.DataFrame(columns=empty_cols)

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
            # One prediction thread per variable: the pool above is the
            # parallelism, and a per-call LightGBM pool sized from the OpenMP
            # defaults would open one full-width pool per worker (13 variables
            # × all cores on 13 workers). Prediction is bit-identical across
            # thread counts (ml_predict), so this only removes the contention.
            yhat = np.asarray(ml_predict(model, X_stack, num_threads=1), dtype=float)
            if log_transform and back_transform != "geometric":
                yhat = _back_transform_preds(
                    yhat,
                    log_transform=True,
                    back_transform=back_transform,
                    smearing_factor=sf,
                )
            for j in range(len(g)):
                block_y = yhat[j * n : (j + 1) * n]
                if block_y.size == 0 or not np.isfinite(block_y).any():
                    means.append(np.nan)
                    stds.append(np.nan)
                elif log_transform and back_transform == "geometric":
                    mean_z = float(np.nanmean(block_y))
                    std_z = float(np.nanstd(block_y))
                    means.append(
                        max(0.0, float(np.expm1(mean_z))) if np.isfinite(mean_z) else np.nan
                    )
                    stds.append(
                        float(np.exp(mean_z) * std_z)
                        if (np.isfinite(mean_z) and np.isfinite(std_z))
                        else np.nan
                    )
                else:
                    means.append(float(np.nanmean(block_y)))
                    stds.append(float(np.nanstd(block_y)))

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
    non_empty = [p for p in pieces if not p.empty]
    return (
        pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame(columns=empty_cols)
    )


def ale(
    df: pd.DataFrame,
    model: object,
    *,
    var_list: list[str] | None = None,
    training_only: bool = True,
    n_bins: int = 20,
    quantile_range: tuple[float, float] = (0.01, 0.99),
    log_transform: bool = False,
    back_transform: str = "geometric",
    smearing_factor: float = 1.0,
    n_cores: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute first-order Accumulated Local Effects (ALE; Apley & Zhu 2020).

    Unlike :func:`pdp`, ALE integrates local conditional finite differences
    within empirical quantile bins so correlated meteorological covariates
    (e.g. temperature vs boundary-layer height) are evaluated on-manifold rather
    than extrapolated into unrealistic combinations.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing features (and optionally ``'set' == 'training'``).
    model : object
        Trained model supported by :func:`ml_predict`.
    var_list : list of str or None, default None
        Features to evaluate; defaults to all model features present in ``df``.
    training_only : bool, default True
        Use only ``set == "training"`` rows when present.
    n_bins : int, default 20
        Target number of quantile bins across ``quantile_range``.
    quantile_range : tuple of float, default (0.01, 0.99)
        Lower and upper quantiles defining the evaluation domain ``[q_lo, q_hi]``.
    log_transform : bool, default False
        If True, invert predictions from ``log1p`` space before computing local
        effects.
    back_transform : {"arithmetic", "smearing", "geometric"}, default "geometric"
        Inversion rule when ``log_transform=True``; ``"geometric"`` is the
        legacy ``expm1``-of-the-mean behaviour.
    smearing_factor : float, default 1.0
        Duan (1983) smearing factor used when ``back_transform="smearing"``.
    n_cores : int or None, default None
        Worker thread count across variables.
    verbose : bool, default False
        If True, log info messages when skipping features.

    Returns
    -------
    pandas.DataFrame
        Columns: ``['variable', 'value', 'ale', 'bin_low', 'bin_high', 'n_bin']``,
        centered so the bin-count weighted mean of ``ale`` per variable is 0.
    """
    sf = _validate_back_transform(back_transform, smearing_factor)
    nb_obj: object = n_bins
    if isinstance(nb_obj, bool) or not isinstance(nb_obj, numbers.Integral) or int(nb_obj) < 1:
        raise ValueError(f"`n_bins` must be a positive integer, got {n_bins!r}.")
    n_bins_int = int(nb_obj)

    if len(quantile_range) != 2:
        raise ValueError(f"`quantile_range` must have length 2, got {quantile_range!r}.")
    q_lo, q_hi = float(quantile_range[0]), float(quantile_range[1])
    if (
        not np.isfinite(q_lo)
        or not np.isfinite(q_hi)
        or q_lo < 0.0
        or q_hi > 1.0
        or not (q_lo < q_hi)
    ):
        raise ValueError(
            f"`quantile_range` must satisfy 0 <= q_lo < q_hi <= 1, got {quantile_range!r}."
        )

    X_df, vars_for_ale = _resolve_model_and_data(df, model, var_list, training_only, verbose)
    n_cores_eff = resolve_workers(n_cores)
    empty_cols = ["variable", "value", "ale", "bin_low", "bin_high", "n_bin"]

    def _one(var: str) -> pd.DataFrame:
        s = pd.to_numeric(X_df[var], errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(s)
        if int(valid_mask.sum()) < 2:
            (log.info if verbose else log.debug)(
                "Variable '%s' has fewer than 2 finite values; skipping.", var
            )
            return pd.DataFrame(columns=empty_cols)

        s_valid = s[valid_mask]
        edges = np.unique(np.quantile(s_valid, np.linspace(q_lo, q_hi, n_bins_int + 1)))
        if len(edges) < 2:
            (log.info if verbose else log.debug)(
                "Variable '%s' has degenerate quantile edges; skipping.", var
            )
            return pd.DataFrame(columns=empty_cols)

        in_range = valid_mask & (s >= edges[0]) & (s <= edges[-1])
        X_in = X_df.loc[in_range].copy()
        x_in = s[in_range]
        if X_in.empty:
            return pd.DataFrame(columns=empty_cols)

        n_intervals = len(edges) - 1
        bin_idx = np.clip(np.searchsorted(edges, x_in, side="left") - 1, 0, n_intervals - 1)

        X_lo = X_in.copy()
        X_hi = X_in.copy()
        X_lo[var] = edges[bin_idx]
        X_hi[var] = edges[bin_idx + 1]

        n_in = len(X_in)
        per_chunk = max(1, 500_000)
        diffs_chunks: list[np.ndarray] = []
        for start in range(0, n_in, per_chunk):
            end = min(n_in, start + per_chunk)
            X_pair = pd.concat([X_lo.iloc[start:end], X_hi.iloc[start:end]], ignore_index=True)
            yhat_pair = np.asarray(ml_predict(model, X_pair, num_threads=1), dtype=float)
            yhat_pair = _back_transform_preds(
                yhat_pair,
                log_transform=log_transform,
                back_transform=back_transform,
                smearing_factor=sf,
            )
            m = end - start
            diffs_chunks.append(yhat_pair[m:] - yhat_pair[:m])

        diffs = np.concatenate(diffs_chunks)

        deltas: list[float] = []
        bin_lows: list[float] = []
        bin_highs: list[float] = []
        values: list[float] = []
        n_bin_list: list[int] = []
        for k in range(n_intervals):
            mask_k = bin_idx == k
            n_k = int(mask_k.sum())
            if n_k == 0:
                continue
            d_k = diffs[mask_k]
            if not np.isfinite(d_k).any():
                continue
            lo_k = float(edges[k])
            hi_k = float(edges[k + 1])
            deltas.append(float(np.nanmean(d_k)))
            bin_lows.append(lo_k)
            bin_highs.append(hi_k)
            values.append(0.5 * (lo_k + hi_k))
            n_bin_list.append(n_k)

        if not deltas:
            return pd.DataFrame(columns=empty_cols)

        f_edge = np.concatenate([[0.0], np.cumsum(np.asarray(deltas, dtype=float))])
        ale_raw = 0.5 * (f_edge[:-1] + f_edge[1:])
        n_arr = np.asarray(n_bin_list, dtype=float)
        ale_centered = ale_raw - float(np.sum(n_arr * ale_raw) / np.sum(n_arr))

        return pd.DataFrame(
            {
                "variable": var,
                "value": values,
                "ale": ale_centered,
                "bin_low": bin_lows,
                "bin_high": bin_highs,
                "n_bin": n_bin_list,
            }
        )

    from joblib import Parallel, delayed

    pieces = Parallel(n_jobs=n_cores_eff, prefer="threads")(delayed(_one)(v) for v in vars_for_ale)
    non_empty = [p for p in pieces if not p.empty]
    return (
        pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame(columns=empty_cols)
    )
