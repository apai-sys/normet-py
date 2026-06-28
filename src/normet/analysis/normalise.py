# src/normet/analysis/normalise.py
"""Monte Carlo weather normalisation ("deweathering").

Provides :func:`normalise` (resample-and-predict) and :func:`normalise_auto`
(adaptive resampling until the result converges), plus :class:`NormaliseConfig`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import ConfigError, DataError, ModelError
from ..model.predict import ml_predict
from ..utils._config import resolve_config
from ..utils.logging import get_logger
from ..utils.prepare import check_data, process_date

log = get_logger(__name__)


@dataclass
class NormaliseConfig:
    """Consolidated configuration for :func:`normalise`.

    Every field has a default so the dataclass can be constructed with only
    the overrides that differ from the standard values.
    """

    feature_names: list[str] | None = None
    variables_resample: list[str] | None = None
    n_samples: int = 300
    replace: bool = True
    aggregate: bool = True
    seed: int = 7_654_321
    n_cores: int | None = None
    resample_df: pd.DataFrame | None = None
    memory_save: bool = False
    use_gpu: bool = False
    verbose: bool = False
    return_quantiles: Sequence[float] | None = None
    conditional_on: Mapping[str, Any] | None = None


def _resolve_normalise_config(config: NormaliseConfig | None = None, **kwargs) -> NormaliseConfig:
    return resolve_config(NormaliseConfig, config, **kwargs)


def _format_quantile_name(q: float) -> str:
    """Map 0.025 → 'q025', 0.5 → 'q500', 0.975 → 'q975'."""
    if not (0.0 <= float(q) <= 1.0):
        raise ValueError(f"Quantile must be in [0,1]: got {q}")
    return f"q{int(round(float(q) * 1000)):03d}"


def _apply_conditional_filter(
    pool: pd.DataFrame,
    conditional_on: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Restrict ``pool`` to rows matching every key/value condition.

    - Scalar value → exact match.
    - Iterable (list/tuple/set/Series/ndarray) → ``isin`` semantics.
    - Tuple ``(lo, hi)`` is treated as iterable membership (use callable for ranges).
    - Callable → boolean mask applied to that column.
    """
    mask = pd.Series(True, index=pool.index)
    for col, cond in conditional_on.items():
        if col not in pool.columns:
            raise ValueError(f"`conditional_on` key '{col}' not found in resample pool columns.")
        s = pool[col]
        if callable(cond):
            mask &= s.map(cond).astype(bool)
        elif isinstance(cond, list | tuple | set | pd.Series | np.ndarray):
            mask &= s.isin(list(cond))
        else:
            mask &= s == cond
    return pool.loc[mask]


def generate_resampled(
    df: pd.DataFrame,
    variables_resample: list[str],
    replace: bool,
    seed: int,
    resample_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate a resampled copy of the dataset.

    Selected predictors are replaced with values drawn from a weather
    reference pool.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset to be resampled. Must include the target ``value`` and
        a ``date`` column.
    variables_resample : List[str]
        Predictor columns to resample from ``resample_df``.
    replace : bool
        If True, sample with replacement. If False, sample without replacement.
    seed : int
        Random seed for reproducibility of the resampling.
    resample_df : pandas.DataFrame
        Pool of data used to resample the specified predictors. Must contain
        all columns listed in ``variables_resample``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with:
          - specified ``variables_resample`` columns replaced by resampled values,
          - a new column ``seed`` indicating the resampling seed used.
    """
    missing = [c for c in variables_resample if c not in resample_df.columns]
    if missing:
        raise ValueError(f"`resample_df` is missing columns: {missing}")

    pool = (
        resample_df[variables_resample]
        .sample(n=len(df), replace=replace, random_state=seed)
        .reset_index(drop=True)
    )

    out = df.copy(deep=False).reset_index(drop=True)
    out.loc[:, variables_resample] = pool.to_numpy()
    out.loc[:, "seed"] = seed
    return out


def normalise(
    df: pd.DataFrame,
    model: object,
    *,
    config: NormaliseConfig | None = None,
    feature_names: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Normalise a time series using a trained model and Monte Carlo resampling.

    This function resamples meteorological variables (or user-specified
    predictors), predicts with the supplied model, and aggregates results
    to provide deweathered estimates of the target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing at least ``date`` (datetime64) and target
        column ``value``.
    model : object
        Trained model with a ``predict`` method.
    config : NormaliseConfig, optional
        Consolidated config object. Individual keyword arguments (``n_samples``,
        ``feature_names``, ``variables_resample``, …) override the corresponding
        field when provided.
    feature_names : list[str], optional
        Deprecated — prefer ``config.feature_names`` or passing via kwargs.
        Predictor columns used by the model.
    **kwargs
        Supported shorthand for overriding individual :class:`NormaliseConfig`
        fields (e.g. ``n_samples=300``, ``n_cores=4``) without constructing a
        config object. Any field passed both via ``config`` and as a keyword is
        resolved in favour of the keyword.

    Examples
    --------
    >>> import pandas as pd
    >>> from normet import normalise, build_model
    >>> df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=48, freq="h"),
    ...                    "value": range(48), "t2m": 10.0, "blh": 500.0})
    >>> df, model = build_model(df, value="value", feature_names=["t2m", "blh"])
    >>> result = normalise(df, model, n_samples=5, n_cores=1)  # doctest: +SKIP

    Returns
    -------
    pandas.DataFrame
        If ``aggregate=True``:
            Indexed by ``date`` with columns:
              - observed
              - normalised
              - one column per requested quantile (when ``return_quantiles``)
        If ``aggregate=False``:
            Indexed by ``date`` with columns:
              - observed
              - one column per seed (e.g., ``12345``).
    """
    if feature_names is not None:
        kwargs.setdefault("feature_names", feature_names)
    _cfg = _resolve_normalise_config(config=config, **kwargs)

    if _cfg.feature_names is None:
        raise ConfigError("`feature_names` must be provided (either directly or via config).")

    df = process_date(df).pipe(check_data, _cfg.feature_names, "value")
    if "date" not in df.columns:
        raise DataError("`df` must contain a 'date' column.")

    resample_df = df if _cfg.resample_df is None else _cfg.resample_df
    time_vars = {"date_unix", "day_julian", "weekday", "hour"}
    variables_resample = _cfg.variables_resample or [
        c for c in _cfg.feature_names if c not in time_vars
    ]

    if _cfg.conditional_on:
        before = len(resample_df)
        resample_df = _apply_conditional_filter(resample_df, _cfg.conditional_on)
        if resample_df.empty:
            raise DataError(
                f"`conditional_on` filter left no rows in the resample pool (was {before}). "
                "Loosen the condition or pass a wider resample_df."
            )
        (log.info if _cfg.verbose else log.debug)(
            "conditional_on filter: %d → %d rows in resample pool.", before, len(resample_df)
        )

    missing = [c for c in variables_resample if c not in resample_df.columns]
    if missing:
        raise DataError(f"`resample_df` is missing columns required for resampling: {missing}")

    n_cores_eff = max(1, _cfg.n_cores if _cfg.n_cores is not None else (os.cpu_count() or 2) - 1)

    rng = np.random.default_rng(_cfg.seed)
    random_seeds = rng.choice(1_000_000, size=_cfg.n_samples, replace=False)

    # GPU only accelerates the sampling/gather stage, not ml_predict (which runs on CPU).
    # memory_save=True path uses joblib and bypasses the GPU gather entirely.
    if _cfg.use_gpu and _cfg.memory_save:
        log.warning(
            "use_gpu=True has no effect when memory_save=True (joblib path bypasses GPU gather). "
            "Set memory_save=False to enable GPU-accelerated resampling."
        )

    is_gpu = False
    if _cfg.use_gpu and not _cfg.memory_save:
        try:
            import cupy as _cupy  # noqa: F401 — presence check only

            is_gpu = True
        except ImportError:
            log.warning("use_gpu=True but cupy is not installed. Falling back to numpy.")

    (log.info if _cfg.verbose else log.debug)(
        "Normalising with %d resamples (aggregate=%s, memory_save=%s, n_cores=%d, gpu=%s).",
        _cfg.n_samples,
        _cfg.aggregate,
        _cfg.memory_save,
        n_cores_eff,
        is_gpu,
    )

    def process_one(seed_i: int) -> pd.DataFrame | None:
        try:
            df_resampled = generate_resampled(
                df, variables_resample, _cfg.replace, int(seed_i), resample_df
            )
            preds = ml_predict(model, df_resampled)
            return pd.DataFrame(
                {
                    "date": df_resampled["date"].to_numpy(),
                    "observed": df_resampled["value"].to_numpy(),
                    "normalised": preds,
                    "seed": int(seed_i),
                }
            )
        except Exception:
            log.exception("Error in seed %d", seed_i)
            return None

    if _cfg.memory_save:
        from joblib import Parallel, delayed

        # prefer="threads" keeps the (large) model + frame shared in-process
        # rather than pickling them to worker processes. This relies on the
        # heavy work (model.predict in FLAML/LightGBM, NumPy/pandas ops)
        # releasing the GIL; a pure-Python predict path would not parallelise.
        results_raw = Parallel(n_jobs=n_cores_eff, prefer="threads")(
            delayed(process_one)(int(s)) for s in random_seeds
        )
        results: list[pd.DataFrame] = [r for r in results_raw if r is not None]
        if not results:
            raise ModelError("No successful resamples produced results.")
        df_result = pd.concat(results, ignore_index=True)
    else:
        n_rows = len(df)
        n_pool = len(resample_df)

        # Generate all indices on CPU with numpy (fast PRNG, per-seed reproducibility,
        # no CUDA kernel-launch overhead for 300 small RNG objects).
        indices_np = np.empty((_cfg.n_samples, n_rows), dtype=np.int64)
        for i, s_val in enumerate(random_seeds):
            rng_s = np.random.default_rng(int(s_val))
            indices_np[i] = rng_s.choice(n_pool, size=n_rows, replace=_cfg.replace)

        indices_flat = indices_np.flatten()
        pool_arr = resample_df[variables_resample].to_numpy()

        if is_gpu:
            import cupy as cp

            # Transfer pool and flat indices to GPU; perform the gather there.
            pool_gpu = cp.array(pool_arr)
            idx_gpu = cp.array(indices_flat)
            resampled_data: np.ndarray = cp.take(pool_gpu, idx_gpu, axis=0).get()
        else:
            resampled_data = pool_arr[indices_flat]

        # Pre-build column → position map to avoid O(n) list.index() per column.
        resample_pos = {c: i for i, c in enumerate(variables_resample)}

        df_all_dict = {}
        for c in df.columns:
            if c in resample_pos:
                df_all_dict[c] = resampled_data[:, resample_pos[c]]
            else:
                df_all_dict[c] = np.tile(df[c].to_numpy(), _cfg.n_samples)

        df_all = pd.DataFrame(df_all_dict)
        df_all["seed"] = np.repeat(random_seeds, n_rows)

        preds = ml_predict(model, df_all)

        df_result = pd.DataFrame(
            {
                "date": df_all["date"].to_numpy(),
                "observed": df_all["value"].to_numpy(),
                "normalised": preds,
                "seed": df_all["seed"].to_numpy(),
            }
        )

    if _cfg.aggregate:
        (log.info if _cfg.verbose else log.debug)("Aggregating %d predictions.", _cfg.n_samples)
        gb = df_result.groupby("date", as_index=True)
        df_out = gb[["observed", "normalised"]].mean()
        if _cfg.return_quantiles:
            q_arr = sorted({float(q) for q in _cfg.return_quantiles})
            q_df = gb["normalised"].quantile(q_arr).unstack(level=-1)  # type: ignore[arg-type]
            q_df.columns = pd.Index([_format_quantile_name(float(q)) for q in q_df.columns])
            df_out = df_out.join(q_df, how="left")
    else:
        observed = df_result.drop_duplicates(subset=["date"]).set_index("date")[["observed"]]
        wide = df_result.pivot(index="date", columns="seed", values="normalised")
        df_out = pd.concat([observed, wide], axis=1)
        if _cfg.return_quantiles:
            log.debug(
                "`return_quantiles` ignored when aggregate=False (wide table already exposes all seeds)."
            )

    (log.info if _cfg.verbose else log.debug)("Finished normalisation.")
    return df_out


def normalise_auto(
    df: pd.DataFrame,
    model: object,
    *,
    feature_names: list[str] | None = None,
    variables_resample: list[str] | None = None,
    resample_df: pd.DataFrame | None = None,
    convergence_tol: float | str = "0.5%",
    stability_streak: int = 5,
    batch_size: int = 100,
    max_samples: int = 5000,
    verbose: bool = True,
    **normalise_kwargs,
) -> dict:
    """Run weather normalisation in batches until the result converges.

    Instead of guessing ``n_samples``, this function runs resampling in batches
    of ``batch_size``, tracks the running mean, and stops when the relative
    change in the global mean stays below ``convergence_tol`` for
    ``stability_streak`` consecutive checks.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset with ``date`` and target ``value`` columns.
    model : object
        Trained model with a ``predict`` method.
    feature_names : list[str], optional
        Predictor columns used by the model (required).
    variables_resample : list[str], optional
        Predictor columns to resample. Defaults to all non-time features.
    resample_df : pandas.DataFrame, optional
        External resampling pool. Defaults to ``df``.
    convergence_tol : float or str, default "0.5%"
        Stopping threshold for the relative change in the global mean.
        Pass a percentage string (``"0.5%"``) or a plain fraction (``0.005``).
    stability_streak : int, default 5
        Number of consecutive checks below tolerance required to stop.
    batch_size : int, default 100
        Resampling iterations per batch.
    max_samples : int, default 5000
        Hard upper limit; a warning is raised if this is reached without
        strict convergence.
    verbose : bool, default True
        Print progress to the logger.
    **normalise_kwargs :
        Extra keyword arguments forwarded to :func:`normalise` (e.g.
        ``n_cores``, ``seed``, ``memory_save``).

    Returns
    -------
    dict with keys:
        ``best_n`` : int — total samples used.
        ``res``    : pandas.DataFrame with columns ``date``, ``observed``,
                     ``normalised``.
    """
    # --- 0. Parse convergence_tol ---
    if isinstance(convergence_tol, str):
        if "%" in convergence_tol:
            convergence_tol = float(convergence_tol.replace("%", "").strip()) / 100.0
        else:
            convergence_tol = float(convergence_tol)

    if resample_df is None:
        resample_df = df

    # --- 1. Initialise accumulators ---
    # Running sums per date so we can compute incremental means without
    # storing all raw predictions.
    # date → {sum_norm, n_total, observed}. Built by zipping the columns instead
    # of iterrows() (far cheaper on large frames); duplicate dates keep the last
    # observed value, matching the previous row-by-row assignment.
    obs_col = df["value"] if "value" in df.columns else pd.Series(np.nan, index=df.index)
    acc: dict[object, dict] = {
        d: {"sum_norm": 0.0, "n_total": 0, "observed": float(o)}
        for d, o in zip(df["date"], obs_col, strict=False)
    }

    total_n = 0
    stable_count = 0
    prev_global_mean = 0.0
    converged = False

    if verbose:
        log.info(
            "Starting auto-normalisation | tol=%.3f%% | batch=%d | max=%d",
            convergence_tol * 100,
            batch_size,
            max_samples,
        )

    # --- 2. Main loop ---
    while total_n < max_samples:
        batch_result = normalise(
            df,
            model,
            feature_names=feature_names,
            variables_resample=variables_resample,
            resample_df=resample_df,
            n_samples=batch_size,
            aggregate=True,
            verbose=False,
            **normalise_kwargs,
        )

        # batch_result is indexed by date; iterate only the column we need.
        for date_val, norm in batch_result["normalised"].items():
            if date_val in acc:
                acc[date_val]["sum_norm"] += float(norm) * batch_size
                acc[date_val]["n_total"] += batch_size

        total_n += batch_size

        # --- 3. Convergence check ---
        if total_n > batch_size:
            daily_means = [v["sum_norm"] / v["n_total"] for v in acc.values() if v["n_total"] > 0]
            current_global_mean = float(np.mean(daily_means)) if daily_means else 0.0

            if prev_global_mean != 0.0:
                rel_change = abs((current_global_mean - prev_global_mean) / prev_global_mean)
            else:
                rel_change = float("inf")

            if rel_change < convergence_tol:
                stable_count += 1
            else:
                stable_count = 0

            if verbose:
                log.info(
                    "n=%d | global_mean=%.4f | rel_change=%.5f%% | streak=%d/%d",
                    total_n,
                    current_global_mean,
                    rel_change * 100,
                    stable_count,
                    stability_streak,
                )

            if stable_count >= stability_streak:
                if verbose:
                    log.info(
                        "Convergence reached at n=%d (change=%.5f%%).",
                        total_n,
                        rel_change * 100,
                    )
                converged = True
                break

            prev_global_mean = current_global_mean

    if not converged and verbose:
        import warnings

        warnings.warn(
            f"normalise_auto reached max_samples={max_samples} without strict convergence.",
            stacklevel=2,
        )

    # --- 4. Assemble output ---
    records = [
        {
            "date": d,
            "observed": v["observed"],
            "normalised": v["sum_norm"] / v["n_total"] if v["n_total"] > 0 else float("nan"),
        }
        for d, v in acc.items()
    ]
    res = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    return {"best_n": total_n, "res": res}
