# src/normet/analysis/normalise.py
"""Monte Carlo meteorological normalisation ("deweathering").

Provides :func:`normalise` (resample-and-predict) and :func:`normalise_auto`
(adaptive resampling until the result converges), plus :class:`NormaliseConfig`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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

    covariates: list[str] | None = None
    """The model's full trained feature set, not a subset of interest. If
    None, auto-derived from ``model`` via :func:`extract_features`. If
    given, must be a superset of the model's actual trained features --
    :func:`normalise` validates this and raises :class:`ConfigError`
    immediately on a mismatch. Use ``variables_resample`` to select which
    of these get resampled; do not trim ``covariates`` itself for that."""
    variables_resample: list[str] | None = None
    n_samples: int = 300
    replace: bool = True
    aggregate: bool = True
    seed: int = 7_654_321
    n_cores: int | None = None
    resample_df: pd.DataFrame | None = None
    memory_save: bool = False
    verbose: bool = False
    return_quantiles: Sequence[float] | None = None
    conditional_on: Mapping[str, Any] | None = None
    batch_size: int | None = None
    """Batch-reduce strategy (mirrors R's transient-memory pipeline).

    When batching is active and ``aggregate=True``, predictions are produced
    and immediately reduced into running sums *batch by batch* so peak memory
    is O(batch_size × n_rows) rather than O(n_samples × n_rows).  Each batch
    is discarded via explicit ``del`` before the next is allocated.

    - ``None`` (default): auto-derive the batch size from the data footprint,
      matching normet-R's heuristic -- the largest number of resampled copies
      of ``df`` that fits in a ~400 MB payload budget, clamped to
      ``[1, n_samples]``.
    - ``0``: disable batching (materialise the full n_samples × n_rows frame).
    - ``> 0``: use exactly this many samples per batch.

    Has no effect when ``aggregate=False`` (full wide table requires all seeds).
    """
    cache: str | Path | None = None
    """If given, memoize the resample-and-predict result to this directory
    (a :class:`joblib.Memory` location). Repeat calls with the same data,
    resample pool, model, and configuration are served from disk instead of
    re-running the Monte Carlo resampling -- useful since :func:`decompose`
    calls :func:`normalise` once per fixed time-variable and each call is a
    full ``n_samples``-draw resample-and-predict. Off by default. The model
    is fingerprinted via :func:`normet.utils.cache.model_hash`, so a re-fit
    model (even with identical config) correctly invalidates the cache.
    """
    resample_pools: Mapping[str, pd.DataFrame] | None = None
    """Extra pools that some of the resampled variables are drawn from instead
    of ``resample_df``, as ``{name: DataFrame}``.

    A pool's columns name the variables it serves: every variable in
    ``variables_resample`` that is a column of the pool is drawn from it, whole
    rows at a time (so the pool's variables keep their joint structure), and
    independently of ``resample_df`` and of the other pools. Variables no pool
    claims are drawn from ``resample_df`` as usual; a variable may be a column
    of at most one pool. The names are labels only.

    Without pools every resampled variable comes from one pool, so the result
    is the expectation over the *average* conditions in it -- with trajectory
    features among the variables, over the average air mass. A separate pool
    sets a reference instead: ``{"transport": clean_hours[traj_cols]}`` draws
    the trajectory features from clean-air-mass hours only while the local
    weather is still drawn from the whole record.

    ``conditional_on`` filters ``resample_df`` only; filter a pool yourself.
    """


def _resolve_normalise_config(
    config: NormaliseConfig | None = None, **kwargs: Any
) -> NormaliseConfig:
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


def _split_across_pools(
    variables_resample: Sequence[str],
    resample_df: pd.DataFrame,
    resample_pools: Mapping[str, pd.DataFrame] | None,
) -> list[tuple[int, list[str], pd.DataFrame]]:
    """Assign each resampled variable to the pool it is drawn from.

    Returns ``(stream, columns, pool)`` triples. Stream 0 is ``resample_df`` and
    serves every variable no extra pool claims; the extra pools follow in name
    order as streams 1, 2, ... A pool's stream is fixed by its name, not by
    which of its variables a given call resamples, so a variable's draws do not
    move when other variables are fixed -- the differences :func:`decom_met`
    takes between calls stay common-random-number differences.
    """
    if resample_pools is None:
        resample_pools = {}
    if not isinstance(resample_pools, Mapping):
        raise ConfigError("`resample_pools` must be a mapping of name -> DataFrame.")
    claimed: dict[str, str] = {}
    extra: list[tuple[int, list[str], pd.DataFrame]] = []
    for stream, name in enumerate(sorted(resample_pools, key=str), start=1):
        pool = resample_pools[name]
        if not isinstance(pool, pd.DataFrame):
            raise ConfigError(
                f"resample pool {name!r} must be a DataFrame, got {type(pool).__name__}."
            )
        cols = [c for c in variables_resample if c in pool.columns]
        if not cols:
            continue
        if len(pool) == 0:
            raise DataError(f"resample pool {name!r} has no rows to draw from.")
        for c in cols:
            if c in claimed:
                raise ConfigError(
                    f"variable {c!r} is a column of two resample pools ({claimed[c]!r} and "
                    f"{name!r}); each variable is drawn from exactly one pool."
                )
            claimed[c] = name
        extra.append((stream, cols, pool))
    base = [c for c in variables_resample if c not in claimed]
    return ([(0, base, resample_df)] if base else []) + extra


def _draw_rows(stream: int, seed: int, n_pool: int, n_rows: int, replace: bool) -> np.ndarray:
    """Row indices into one pool for one Monte Carlo draw.

    Stream 0 keeps the generator :func:`normalise` has always used, so results
    without extra pools are unchanged; stream k > 0 gets an independent
    generator derived from the same seed.
    """
    rng = np.random.default_rng(seed if stream == 0 else [seed, stream])
    return rng.choice(n_pool, size=n_rows, replace=replace)


def _stacked_columns(
    df: pd.DataFrame,
    draws: Sequence[tuple[int, list[str], np.ndarray]],
    seeds: Sequence[int] | np.ndarray,
    replace: bool,
) -> dict[str, np.ndarray]:
    """Columns of ``len(seeds)`` resampled copies of ``df``, stacked copy after copy.

    ``draws`` holds ``(stream, columns, values)`` with ``values`` the pool's
    ``columns`` as one array, so each copy takes whole pool rows.
    """
    n_rows = len(df)
    drawn: dict[str, np.ndarray] = {}
    for stream, cols, values in draws:
        idx = np.empty((len(seeds), n_rows), dtype=np.int64)
        for i, s in enumerate(seeds):
            idx[i] = _draw_rows(stream, int(s), len(values), n_rows, replace)
        picked = values[idx.ravel()]
        for j, c in enumerate(cols):
            drawn[c] = picked[:, j]
    return {
        c: drawn[c] if c in drawn else np.tile(df[c].to_numpy(), len(seeds)) for c in df.columns
    }


def generate_resampled(
    df: pd.DataFrame,
    variables_resample: list[str],
    replace: bool,
    seed: int,
    resample_df: pd.DataFrame,
    resample_pools: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Generate a resampled copy of the dataset.

    Selected predictors are replaced with values drawn from a meteorological
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
        every column of ``variables_resample`` that no ``resample_pools`` pool
        serves.
    resample_pools : mapping of str to pandas.DataFrame, optional
        Extra pools; see :attr:`NormaliseConfig.resample_pools`.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with:
          - specified ``variables_resample`` columns replaced by resampled values,
          - a new column ``seed`` indicating the resampling seed used.
    """
    draws = _split_across_pools(variables_resample, resample_df, resample_pools)
    base = draws[0][1] if draws and draws[0][0] == 0 else []
    missing = [c for c in base if c not in resample_df.columns]
    if missing:
        raise ValueError(f"`resample_df` is missing columns: {missing}")

    out = df.copy(deep=False).reset_index(drop=True)
    for stream, cols, pool in draws:
        # Stream 0 keeps the historical random_state=seed; see _draw_rows.
        state = seed if stream == 0 else np.random.default_rng([seed, stream])
        picked = pool[cols].sample(n=len(df), replace=replace, random_state=state)
        out.loc[:, cols] = picked.to_numpy()
    out.loc[:, "seed"] = seed
    return out


def normalise(
    df: pd.DataFrame,
    model: object,
    *,
    config: NormaliseConfig | None = None,
    covariates: list[str] | None = None,
    **kwargs: Any,
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
        ``covariates``, ``variables_resample``, …) override the corresponding
        field when provided.
    covariates : list[str], optional
        Deprecated — prefer ``config.covariates`` or passing via kwargs.
        Predictor columns used by the model. If omitted (and ``model`` is
        given), auto-derived from the model's own trained features via
        :func:`normet.utils.features.extract_features` -- the same
        detection :func:`decompose`/:func:`decom_emi`/:func:`decom_met`
        already perform before calling this function. If given explicitly,
        it must be a **superset** of the model's trained features (extra
        columns are fine and simply pass through unused); a smaller/wrong
        set raises :class:`ConfigError` immediately, rather than silently
        dropping columns via :func:`normet.utils.prepare.check_data` and
        failing later inside ``model.predict()`` with a confusing
        backend-level ``KeyError``. Use ``variables_resample`` (not a
        trimmed ``covariates``) to control which of the model's features
        get resampled.
    **kwargs
        Supported shorthand for overriding individual :class:`NormaliseConfig`
        fields (e.g. ``n_samples=300``, ``n_cores=4``, ``cache=".normet_cache"``)
        without constructing a config object. Any field passed both via
        ``config`` and as a keyword is resolved in favour of the keyword.

    Examples
    --------
    >>> import pandas as pd
    >>> from normet import normalise, build_model
    >>> df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=48, freq="h"),
    ...                    "value": range(48), "t2m": 10.0, "blh": 500.0})
    >>> df, model = build_model(df, target="value", covariates=["t2m", "blh"])
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
    if covariates is not None:
        kwargs.setdefault("covariates", covariates)
    _cfg = _resolve_normalise_config(config=config, **kwargs)

    if _cfg.covariates is None:
        if model is None:
            raise ConfigError("`covariates` must be provided (either directly or via config).")
        try:
            from ..utils.features import extract_features

            _cfg.covariates = [str(c) for c in extract_features(model)]
        except Exception as exc:
            raise ConfigError(
                "`covariates` must be provided (either directly or via config) -- "
                "it could not be auto-detected from `model`."
            ) from exc
    elif model is not None:
        try:
            from ..utils.features import extract_features

            model_feats = {str(c) for c in extract_features(model)}
        except Exception:
            model_feats = None
        if model_feats is not None:
            missing = sorted(model_feats - set(_cfg.covariates))
            if missing:
                raise ConfigError(
                    f"`covariates` is missing {missing}, which `model` requires for "
                    "prediction. `covariates` must be a superset of the model's trained "
                    "features -- pass the full feature list and use `variables_resample` to "
                    "choose which of those get resampled."
                )

    if _cfg.cache is None:
        return _normalise_uncached(df, model, _cfg)

    from ..utils.cache import config_hash, dataframe_hash, make_memory, model_hash

    # Every ingredient below is order-canonicalised: the result depends on which
    # columns and which resampled variables are involved, never on the order they
    # are listed in. Two places had to be fixed together -- the lists themselves
    # (hashed via `sorted(...)` in config_hash) *and* the column order of the
    # frames handed to dataframe_hash, which is order-sensitive. Callers that
    # reach the same set by different routes -- notably decompose(), whose
    # resample list is a shrinking sublist of a feature-importance or permutation
    # order -- otherwise recompute every single time.
    key_cols = sorted(set(_cfg.covariates) | {"value", "date"})
    df_keyed = process_date(df.copy()).pipe(check_data, _cfg.covariates, "value")
    resample_pool = df_keyed if _cfg.resample_df is None else _cfg.resample_df
    resample_key_cols = sorted(
        c for c in (_cfg.variables_resample or key_cols) if c in resample_pool.columns
    )
    # Extra pools join the key only when given, so keys of runs without them --
    # and the caches already on disk -- are unchanged. A pool can only serve
    # covariates, so those are the columns worth hashing.
    covariate_set = set(_cfg.covariates)
    pool_keys = [
        (str(name), dataframe_hash(pool[sorted(c for c in pool.columns if c in covariate_set)]))
        for name, pool in sorted((_cfg.resample_pools or {}).items(), key=lambda kv: str(kv[0]))
        if isinstance(pool, pd.DataFrame)
    ]
    cache_key = config_hash(
        *([pool_keys] if pool_keys else []),
        sorted(_cfg.covariates),
        sorted(_cfg.variables_resample) if _cfg.variables_resample is not None else None,
        _cfg.n_samples,
        _cfg.replace,
        _cfg.aggregate,
        _cfg.seed,
        _cfg.memory_save,
        list(_cfg.return_quantiles) if _cfg.return_quantiles else None,
        dict(_cfg.conditional_on) if _cfg.conditional_on else None,
        _cfg.batch_size,
        dataframe_hash(df_keyed[[c for c in key_cols if c in df_keyed.columns]]),
        dataframe_hash(resample_pool[resample_key_cols]),
        model_hash(model),
    )
    memory = make_memory(_cfg.cache)
    cached = memory.cache(_normalise_cached_call, ignore=["df", "model", "cfg"])
    return cached(cache_key, df=df, model=model, cfg=_cfg)


def _normalise_cached_call(
    _cache_key: str, *, df: pd.DataFrame, model: object, cfg: NormaliseConfig
) -> pd.DataFrame:
    """Cache-keyed wrapper so joblib can memoize on ``_cache_key`` alone (see :func:`normalise`)."""
    return _normalise_uncached(df, model, cfg)


def _normalise_uncached(df: pd.DataFrame, model: object, _cfg: NormaliseConfig) -> pd.DataFrame:
    """Uncached resample-and-predict core of :func:`normalise`."""
    # narrowing: normalise() always resolves _cfg.covariates (auto-detecting
    # from `model` or raising ConfigError) before calling this helper.
    assert _cfg.covariates is not None
    df = process_date(df).pipe(check_data, _cfg.covariates, "value")
    if "date" not in df.columns:
        raise DataError("`df` must contain a 'date' column.")

    resample_df = df if _cfg.resample_df is None else _cfg.resample_df
    time_vars = {"date_unix", "day_julian", "weekday", "hour"}
    variables_resample = (
        _cfg.variables_resample
        if _cfg.variables_resample is not None
        else [c for c in _cfg.covariates if c not in time_vars]
    )

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

    for name, pool in (_cfg.resample_pools or {}).items():
        # A pool none of whose columns is a covariate can never serve anything:
        # almost certainly a misspelt or wrongly subset frame, which would
        # otherwise be ignored in silence.
        if isinstance(pool, pd.DataFrame) and not set(pool.columns) & set(_cfg.covariates):
            raise ConfigError(
                f"resample pool {name!r} has no covariate among its columns "
                f"({list(pool.columns)[:6]}); a pool's columns name the variables drawn from it."
            )
    draws = _split_across_pools(variables_resample, resample_df, _cfg.resample_pools)
    base_cols = draws[0][1] if draws and draws[0][0] == 0 else []
    missing = [c for c in base_cols if c not in resample_df.columns]
    if missing:
        raise DataError(f"`resample_df` is missing columns required for resampling: {missing}")
    # Each pool's served columns as one array, built once for every draw.
    draw_arrays = [(stream, cols, pool[cols].to_numpy()) for stream, cols, pool in draws]

    n_cores_eff = max(1, _cfg.n_cores if _cfg.n_cores is not None else (os.cpu_count() or 2) - 1)

    rng = np.random.default_rng(_cfg.seed)
    random_seeds = rng.choice(1_000_000, size=_cfg.n_samples, replace=False)

    # Resolve the effective batch size. None -> auto-derive from the data
    # footprint (normet-R's heuristic: as many resampled copies of `df` as fit
    # in a ~400 MB payload budget); 0 -> batching disabled; >0 -> as given.
    if _cfg.batch_size is None:
        if _cfg.memory_save:
            # Explicit memory_save=True selects the joblib-threaded path;
            # don't let the auto default shadow that choice. An explicit
            # batch_size > 0 still takes precedence over memory_save.
            eff_batch_size = 0
        else:
            one_copy_bytes = int(df.memory_usage(deep=True).sum())
            safe_payload_bytes = 400 * 1024**2
            eff_batch_size = int(
                max(1, min(safe_payload_bytes // max(one_copy_bytes, 1), _cfg.n_samples))
            )
            (log.info if _cfg.verbose else log.debug)(
                "Auto-batching: one resampled copy ≈ %.1f MB -> batch_size=%d.",
                one_copy_bytes / 1024**2,
                eff_batch_size,
            )
    else:
        eff_batch_size = int(_cfg.batch_size)

    use_batch_reduce = eff_batch_size > 0 and _cfg.aggregate and not _cfg.return_quantiles

    (log.info if _cfg.verbose else log.debug)(
        "Normalising with %d resamples (aggregate=%s, memory_save=%s, "
        "batch_reduce=%s, n_cores=%d).",
        _cfg.n_samples,
        _cfg.aggregate,
        _cfg.memory_save,
        use_batch_reduce,
        n_cores_eff,
    )

    def process_one(seed_i: int) -> pd.DataFrame | None:
        try:
            df_resampled = generate_resampled(
                df,
                variables_resample,
                _cfg.replace,
                int(seed_i),
                resample_df,
                resample_pools=_cfg.resample_pools,
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

    # ── Batch-reduce path: O(batch_size × n_rows) peak memory ──────────────
    # Mirrors R's transient-memory pipeline: generate → predict → accumulate
    # → delete, one batch at a time.  Only available when aggregate=True.
    if use_batch_reduce:
        import gc

        n_rows = len(df)
        obs_arr = df["value"].to_numpy()
        dates_arr = df["date"].to_numpy()

        sum_norm = np.zeros(n_rows, dtype=np.float64)
        sum_obs = np.zeros(n_rows, dtype=np.float64)
        n_completed = 0

        seeds_batched = [
            random_seeds[i : i + eff_batch_size] for i in range(0, _cfg.n_samples, eff_batch_size)
        ]
        (log.info if _cfg.verbose else log.debug)(
            "Batch-reduce: %d batches of ≤%d (total %d resamples).",
            len(seeds_batched),
            eff_batch_size,
            _cfg.n_samples,
        )

        for b_idx, batch_seeds in enumerate(seeds_batched):
            b_size = len(batch_seeds)

            # Build prediction DataFrame without materialising the full n_samples × n_rows frame
            df_batch_dict = _stacked_columns(df, draw_arrays, batch_seeds, _cfg.replace)
            df_batch = pd.DataFrame(df_batch_dict)

            batch_preds = ml_predict(model, df_batch)  # shape (b_size × n_rows,)

            # Accumulate into running sums and immediately free batch arrays
            batch_preds_2d = batch_preds.reshape(b_size, n_rows)
            sum_norm += batch_preds_2d.sum(axis=0)
            sum_obs += np.tile(obs_arr, b_size).reshape(b_size, n_rows).sum(axis=0)
            n_completed += b_size

            del df_batch_dict, df_batch
            del batch_preds, batch_preds_2d
            gc.collect()

            (log.debug)(
                "Batch %d/%d done (%d/%d total resamples).",
                b_idx + 1,
                len(seeds_batched),
                n_completed,
                _cfg.n_samples,
            )

        df_out = pd.DataFrame(
            {
                "observed": sum_obs / n_completed,
                "normalised": sum_norm / n_completed,
            },
            index=pd.Index(dates_arr, name="date"),
        )
        df_out.index.name = "date"

    # ── joblib threaded path: memory_save=True ─────────────────────────────
    elif _cfg.memory_save:
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
        n_failed = len(results_raw) - len(results)
        if n_failed:
            log.warning(
                "%d/%d resample seeds failed and were dropped (see prior error logs); "
                "aggregate is based on %d samples instead of the requested %d.",
                n_failed,
                _cfg.n_samples,
                len(results),
                _cfg.n_samples,
            )
        df_result = pd.concat(results, ignore_index=True)

        if _cfg.aggregate:
            (log.info if _cfg.verbose else log.debug)("Aggregating %d predictions.", _cfg.n_samples)
            gb = df_result.groupby("date", as_index=True)
            df_out = gb[["observed", "normalised"]].mean()
            if _cfg.return_quantiles:
                q_arr = sorted({float(q) for q in _cfg.return_quantiles})
                q_df = gb["normalised"].quantile(np.asarray(q_arr)).unstack(level=-1)
                q_df.columns = pd.Index([_format_quantile_name(float(q)) for q in q_df.columns])
                df_out = df_out.join(q_df, how="left")
        else:
            observed = df_result.drop_duplicates(subset=["date"]).set_index("date")[["observed"]]
            wide = df_result.pivot(index="date", columns="seed", values="normalised")
            df_out = pd.concat([observed, wide], axis=1)

    # ── Vectorised path: default, O(n_samples × n_rows) ───────────────────
    else:
        n_rows = len(df)

        # All draws at once with numpy (fast PRNG, per-seed reproducibility).
        df_all = pd.DataFrame(_stacked_columns(df, draw_arrays, random_seeds, _cfg.replace))
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
                q_df = gb["normalised"].quantile(np.asarray(q_arr)).unstack(level=-1)
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
    covariates: list[str] | None = None,
    variables_resample: list[str] | None = None,
    resample_df: pd.DataFrame | None = None,
    convergence_metric: str = "series",
    convergence_tol: float | str | None = None,
    rse_percentile: float = 95.0,
    stability_streak: int | None = None,
    batch_size: int = 100,
    max_samples: int = 5000,
    seed: int = 7_654_321,
    verbose: bool = True,
    return_history: bool = False,
    **normalise_kwargs: Any,
) -> dict:
    """Run meteorological normalisation in batches until the result converges.

    Instead of guessing ``n_samples``, this function runs resampling in
    batches of ``batch_size`` and stops when a convergence criterion is
    satisfied for ``stability_streak`` consecutive checks.

    Two criteria are available via ``convergence_metric``:

    ``"series"`` (default)
        Stop when the ``rse_percentile``-th percentile of the **per-date
        relative standard error** of the cumulative mean falls below
        ``convergence_tol`` (default ``"3%"``). This targets the actual
        deliverable of deweathering -- the per-timestamp series: with the
        defaults, at least 95 % of timestamps have a Monte-Carlo standard
        error below 3 % of their value when sampling stops. The RSE is a
        CLT-based distance-to-limit measure, so the test does not weaken
        as the sample count grows.
    ``"global"`` (legacy)
        Stop when the relative batch-to-batch change of the **global mean**
        stays below ``convergence_tol`` (default ``"0.5%"``). Note two
        caveats, which motivated the new default above: (i) the global
        mean averages over all timestamps, suppressing Monte-Carlo noise
        by roughly ``sqrt(n_dates)``, so it can satisfy a tight tolerance
        while individual timestamps are still far from converged; and
        (ii) the increment of a cumulative mean shrinks like ``1/n_batches``
        automatically, so the effective threshold on each new batch's
        deviation grows linearly with the batch count -- the test gets
        weaker as sampling proceeds and cannot catch late, slow drift.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset with ``date`` and target ``value`` columns.
    model : object
        Trained model with a ``predict`` method.
    covariates : list[str], optional
        Predictor columns used by the model (required).
    variables_resample : list[str], optional
        Predictor columns to resample. Defaults to all non-time features.
    resample_df : pandas.DataFrame, optional
        External resampling pool. Defaults to ``df``.
    convergence_metric : {"series", "global"}, default "series"
        Convergence criterion (see above).
    convergence_tol : float or str, optional
        Stopping threshold. Pass a percentage string (``"3%"``) or a plain
        fraction (``0.03``). Defaults to ``"3%"`` for ``"series"`` and
        ``"0.5%"`` for ``"global"``.
    rse_percentile : float, default 95.0
        Percentile of the per-date relative-standard-error distribution
        tested against ``convergence_tol``. Only used by ``"series"``.
    stability_streak : int, optional
        Number of consecutive checks below tolerance required to stop.
        Defaults to 3 for ``"series"`` (the RSE declines smoothly as
        ``1/sqrt(n)``, so a long streak adds little) and 5 for ``"global"``
        (matching the legacy behaviour).
    batch_size : int, default 100
        Resampling iterations per batch.
    max_samples : int, default 5000
        Hard upper limit; a warning is raised if this is reached without
        strict convergence.
    verbose : bool, default True
        Print progress to the logger.
    seed : int, default 7654321
        Base random seed. Varied per batch internally (``seed + total_n``)
        so each batch is an independent Monte-Carlo draw -- required for
        the convergence check to be meaningful (a fixed seed would make
        every batch bit-identical, trivially "converged" from the second
        batch on regardless of true stability).
    return_history : bool, default False
        Also return a per-check history DataFrame (columns ``n``,
        ``metric``, ``global_mean``, ``stable_count``).
    **normalise_kwargs :
        Extra keyword arguments forwarded to :func:`normalise` (e.g.
        ``n_cores``, ``memory_save``). A ``seed`` passed here is ignored
        in favour of the explicit ``seed`` parameter above.

    Returns
    -------
    dict with keys:
        ``best_n``  : int — total samples used.
        ``res``     : pandas.DataFrame with columns ``date``, ``observed``,
                      ``normalised``.
        ``history`` : pandas.DataFrame (only when ``return_history=True``).
    """
    normalise_kwargs.pop("seed", None)

    # --- 0. Resolve criterion configuration ---
    metric = (convergence_metric or "series").lower()
    if metric not in ("series", "global"):
        raise ConfigError(
            f"convergence_metric must be 'series' or 'global', got {convergence_metric!r}."
        )
    if convergence_tol is None:
        convergence_tol = "3%" if metric == "series" else "0.5%"
    if isinstance(convergence_tol, str):
        if "%" in convergence_tol:
            convergence_tol = float(convergence_tol.replace("%", "").strip()) / 100.0
        else:
            convergence_tol = float(convergence_tol)
    if stability_streak is None:
        stability_streak = 3 if metric == "series" else 5

    if resample_df is None:
        resample_df = df

    # --- 1. Initialise accumulators ---
    # Per-date running sums of batch means (and their squares, for the
    # series criterion's per-date variance) as flat numpy arrays aligned to
    # a frozen date index -- far cheaper than per-date dict updates.
    obs_col = df["value"] if "value" in df.columns else pd.Series(np.nan, index=df.index)
    observed_by_date: dict[object, float] = {
        d: float(o) for d, o in zip(df["date"], obs_col, strict=False)
    }

    dates_idx: pd.Index | None = None  # frozen on the first batch
    sum_bm: np.ndarray | None = None  # per-date sum of batch means
    sum_bm2: np.ndarray | None = None  # per-date sum of squared batch means

    total_n = 0
    n_batches = 0
    stable_count = 0
    prev_global_mean = 0.0
    converged = False
    history: list[dict] = []

    if verbose:
        log.info(
            "Starting auto-normalisation | metric=%s | tol=%.3f%% | batch=%d | max=%d",
            metric,
            convergence_tol * 100,
            batch_size,
            max_samples,
        )

    # --- 2. Main loop ---
    while total_n < max_samples:
        batch_result = normalise(
            df,
            model,
            covariates=covariates,
            variables_resample=variables_resample,
            resample_df=resample_df,
            n_samples=batch_size,
            aggregate=True,
            seed=seed + total_n,
            verbose=False,
            **normalise_kwargs,
        )

        bm_series = batch_result["normalised"]
        if dates_idx is None:
            dates_idx = bm_series.index
            sum_bm = np.zeros(len(dates_idx))
            sum_bm2 = np.zeros(len(dates_idx))
        assert sum_bm is not None and sum_bm2 is not None
        bm = bm_series.reindex(dates_idx).to_numpy(dtype=float)
        sum_bm += bm
        sum_bm2 += bm**2

        total_n += batch_size
        n_batches += 1

        # --- 3. Convergence check ---
        if n_batches >= 2:
            cum_mean = sum_bm / n_batches
            current_global_mean = float(np.nanmean(cum_mean))

            if metric == "series":
                # Per-date relative standard error of the cumulative mean:
                # SE_t = sd(batch means at t)/sqrt(n_batches); RSE_t = SE_t/|mean_t|.
                var_bm = (sum_bm2 - sum_bm**2 / n_batches) / (n_batches - 1)
                se = np.sqrt(np.clip(var_bm, 0.0, None) / n_batches)
                denom = np.abs(cum_mean)
                ok = denom > 1e-12
                rse = se[ok] / denom[ok]
                metric_value = (
                    float(np.percentile(rse, rse_percentile)) if rse.size else float("inf")
                )
                label = f"P{rse_percentile:g}(RSE)"
            else:
                if prev_global_mean != 0.0:
                    metric_value = abs((current_global_mean - prev_global_mean) / prev_global_mean)
                else:
                    metric_value = float("inf")
                label = "rel_change"

            if metric_value < convergence_tol:
                stable_count += 1
            else:
                stable_count = 0

            history.append(
                {
                    "n": total_n,
                    "metric": metric_value,
                    "global_mean": current_global_mean,
                    "stable_count": stable_count,
                }
            )

            if verbose:
                log.info(
                    "n=%d | global_mean=%.4f | %s=%.5f%% | streak=%d/%d",
                    total_n,
                    current_global_mean,
                    label,
                    metric_value * 100,
                    stable_count,
                    stability_streak,
                )

            if stable_count >= stability_streak:
                if verbose:
                    log.info(
                        "Convergence reached at n=%d (%s=%.5f%%).",
                        total_n,
                        label,
                        metric_value * 100,
                    )
                converged = True
                break

            prev_global_mean = current_global_mean
        else:
            # Seed the global-metric baseline from the first batch so the
            # first real check (batch 2) compares against batch 1 rather
            # than against 0 (which would waste a check and shift the stop
            # point by one batch relative to the legacy behaviour).
            prev_global_mean = float(np.nanmean(sum_bm / n_batches))

    if not converged:
        # Deliberately not gated on `verbose`: silently returning an
        # unconverged result would defeat the point of an adaptive stop.
        import warnings

        warnings.warn(
            f"normalise_auto reached max_samples={max_samples} without strict convergence.",
            stacklevel=2,
        )

    # --- 4. Assemble output ---
    cum_mean = sum_bm / n_batches if n_batches and sum_bm is not None else np.array([])
    res = (
        pd.DataFrame(
            {
                "date": list(dates_idx) if dates_idx is not None else [],
                "observed": [observed_by_date.get(d, float("nan")) for d in dates_idx]
                if dates_idx is not None
                else [],
                "normalised": cum_mean,
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    out = {"best_n": total_n, "res": res}
    if return_history:
        out["history"] = pd.DataFrame(history)
    return out
