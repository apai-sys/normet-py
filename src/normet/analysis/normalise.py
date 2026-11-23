# src/normet/analysis/normalise.py
"""Monte Carlo meteorological normalisation ("deweathering").

Provides :func:`normalise` (resample-and-predict) and :func:`normalise_auto`
(adaptive resampling until the result converges), plus :class:`NormaliseConfig`.
"""

from __future__ import annotations

import numbers
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import ConfigError, DataError, ModelError
from ..model.predict import ml_predict
from ..utils import progress
from ..utils._config import DEFAULT_SEED, resolve_config, resolve_workers
from ..utils.logging import get_logger
from ..utils.metrics import _DEFAULT_STATS, _stats_from_arrays
from ..utils.prepare import (
    add_date_variables,
    check_data,
    compute_duan_smearing,
    impute_values,
    log1p_target,
    prepare_data,
    process_date,
)

log = get_logger(__name__)

#: Default purge width (rows) around each out-of-fold validation block.
#: One day of hourly data, enough to separate adjacent hours whose target and
#: weather are autocorrelated. Exposed so ``do_all(cross_fit=True)`` can use
#: the same default instead of duplicating the number.
_CROSSFIT_DEFAULT_EMBARGO = 24

#: Time covariates kept out of the DML tree's split space. Each is monotone (or
#: piecewise monotone) over a record, so a tree can use it to reproduce the
#: smooth emission trajectory that ``m(t)`` is meant to carry: in-sample it
#: then absorbs short events, and out-of-fold it extrapolates by clamping at the
#: training boundary. ``day_julian`` is a linear day counter, not a cyclic
#: calendar feature, so it belongs here beside ``date_unix`` and ``year``.
#: Truly cyclic features (``hour``, ``weekday``, sine/cosine terms) stay in.
_DML_SLOW_TIME_COVARIATES: frozenset[str] = frozenset({"date_unix", "year", "day_julian"})

_TIME_VARS_EXACT: frozenset[str] = frozenset(
    {
        "date_unix",
        "day_julian",
        "weekday",
        "hour",
        "month",
        "year",
        "dayofyear",
        "dayofweek",
        "day_of_year",
        "day_of_week",
    }
)

_TIME_VARS_CYCLICAL: frozenset[str] = frozenset(
    {
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "weekday_sin",
        "weekday_cos",
        "day_julian_sin",
        "day_julian_cos",
        "month_sin",
        "month_cos",
    }
)


def _is_time_feature(col: str) -> bool:
    """Return True if *col* is a deterministic calendar/time covariate."""
    return col in _TIME_VARS_EXACT or col in _TIME_VARS_CYCLICAL


def _back_transform_values(z: np.ndarray, *, smearing_factor: float = 1.0) -> np.ndarray:
    """Invert ``log1p(y)`` with optional Duan (1983) smearing multiplier ``phi``."""
    arr = np.asarray(z, dtype=float)
    if smearing_factor == 1.0:
        return np.expm1(arr)
    return float(smearing_factor) * np.exp(arr) - 1.0


def _resolve_paired_mode(paired: bool | str, *, log_transform: bool) -> str | None:
    """Normalise *paired* option to ``None`` or one of ``additive|ratio|log_ratio``."""
    if paired is False or paired is None:
        return None
    if paired is True:
        return "log_ratio" if log_transform else "additive"
    mode = str(paired).lower().strip()
    if mode not in {"additive", "ratio", "log_ratio"}:
        raise ConfigError(
            f"`paired` must be False, True, 'additive', 'ratio', or 'log_ratio', got {paired!r}."
        )
    return mode


def _compute_paired_series(
    obs_orig: np.ndarray,
    fact_orig: np.ndarray,
    norm_orig: np.ndarray,
    *,
    mode: str,
    floor: float,
) -> np.ndarray:
    """Compute paired de-weathered concentration preserving short-term anomalies."""
    if mode == "additive":
        return obs_orig - (fact_orig - norm_orig)
    if mode == "ratio":
        safe = np.where(np.abs(fact_orig) > floor, fact_orig, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            return obs_orig * (norm_orig / safe)
    # mode == "log_ratio": additive difference on log1p scale keeps positive ratio
    # well-behaved near zero and cancels multiplicative smearing terms identically.
    z_obs = np.log1p(np.maximum(obs_orig, 0.0))
    z_fact = np.log1p(np.maximum(fact_orig, 0.0))
    z_norm = np.log1p(np.maximum(norm_orig, 0.0))
    return np.maximum(np.expm1(z_obs - (z_fact - z_norm)), 0.0)


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
    """Workers for a parallel section. The per-seed path sizes its joblib pool
    with it; the batch-reduce path has no pool, so it sizes LightGBM's
    prediction threads instead (other backends ignore it). ``None`` uses the
    allocation minus one, as everywhere else."""
    resample_df: pd.DataFrame | None = None
    memory_save: bool = False
    verbose: bool = False
    return_quantiles: Sequence[float] | None = None
    conditional_on: Mapping[str, Any] | None = None
    """Filter the resampling pool to rows matching ``{column: condition}``.

    A condition is a scalar, a list/tuple/set/array of allowed values, or a
    callable. The condition joins the cache key by value: sequences are
    hashed by content (so a long array's truncated ``repr`` cannot make two
    different filters share a key) and a callable through its cloudpickle
    fingerprint -- its code and captured state, stable across processes --
    so changing a closure's captured value invalidates the entry. A callable
    cloudpickle cannot serialise falls back to its ``repr``, which ties the
    key to that process object.
    """
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
    model (even with identical config) correctly invalidates the cache. The
    installed normet version and the versions of the result-bearing
    dependencies (:func:`normet.utils.cache.dependency_versions`) join the
    key too, so an upgrade recomputes instead of serving a result the old
    code wrote. A source change that keeps the version (an editable
    checkout) is not covered -- clear the cache directory for one.
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
    log_transform: bool = False
    """Whether ``df`` and ``model`` live on the ``log1p`` scale.

    True says the caller fitted the model on ``log1p(value)`` and hands the
    frame in that space -- what :func:`normet.do_all` does with
    ``log_transform=True``. The resampling and the model evaluation happen
    on that scale (which is the point: an error then counts in relative
    rather than absolute terms, so a fixed relative miss maps to a bounded
    error at every level while its absolute error grows without bound). How
    the per-draw predictions are returned to the units of the input is
    chosen by :attr:`back_transform`:

    - ``"geometric"`` (the default) averages on the ``log1p`` scale and
      applies ``expm1`` once at the end -- the legacy behaviour, kept as the
      default so results produced before the option existed stay
      bit-identical;
    - ``"arithmetic"`` applies ``expm1`` to every draw before averaging, so
      the estimate targets the arithmetic mean rather than the geometric
      one (the two differ by Jensen's inequality whenever the draws vary);
    - ``"smearing"`` applies Duan's (1983) residual-correction factor to
      that arithmetic mean.

    In every mode ``observed``, ``normalised`` and the quantile columns
    come back in the units of the input. False, the default, leaves the
    result untouched.
    """
    stratify_by: str | Sequence[str] | None = None
    """Optional stratum column(s) (e.g. ``"month"``, ``"season"``, ``"hour"``,
    ``"weekday"``, or ``"traj_cluster"``) constraining each target row to resample
    only from pool rows sharing the same stratum. Eliminates physically impossible
    cross-season or cross-regime combinations when separating local weather
    from long-range transport. ``"month"``, ``"season"``, ``"hour"``, and
    ``"weekday"`` can be derived automatically from ``"date"`` if not already
    present as columns.
    """
    conditional_k: int | None = None
    """Optional positive integer ``k`` for kNN on-manifold conditional resampling.
    When given, resampled covariates for each row are drawn uniformly from the
    ``k`` nearest neighbours in its pool with respect to the currently fixed
    numerical covariates (standardised via pool mean/std). Combines naturally
    with ``stratify_by`` to search nearest neighbours within the same stratum.
    Not available on the chronos-2 backend.
    """
    back_transform: str = "geometric"
    """Retransformation mode when ``log_transform=True``:

    - ``"geometric"`` (default): averages in ``log1p`` space and applies ``expm1``
      once at the end. This is the legacy behaviour, kept as the default so
      ``log_transform=True`` results from earlier releases stay bit-identical.
    - ``"arithmetic"``: back-transform each resampled draw via ``expm1``
      before averaging, estimating the meteorological arithmetic mean rather than
      the biased geometric mean.
    - ``"smearing"``: additionally applies Duan's (1983) nonparametric smearing
      correction ``phi * exp(z_hat) - 1`` to remove training-residual Jensen bias.
    """
    smearing_factor: float | None = None
    """Optional explicit Duan smearing factor ``phi`` used when ``log_transform=True``
    and ``back_transform="smearing"``. When ``None``, estimated automatically from
    held-out (``set=="testing"``) or training model residuals.
    """
    return_mc_se: bool = False
    """If True and ``aggregate=True``, include a per-date ``mc_se`` column with the
    Monte Carlo standard error ``sd(draws) / sqrt(n_samples)`` of ``normalised``.
    """
    paired: bool | str = False
    """Optional paired / residual-preserving deweathering mode alongside level
    estimation (requires ``aggregate=True``). One of ``False`` (default), ``True``
    (``"log_ratio"`` if ``log_transform`` else ``"additive"``), ``"additive"``,
    ``"ratio"``, or ``"log_ratio"``. Appends ``factual``, ``weather_effect``, and
    ``paired`` columns to the output DataFrame.
    """
    paired_floor: float = 1e-6
    """Denominator magnitude guard for ``paired="ratio"``."""
    sampling: str = "iid"
    """Resampling scheme: ``"iid"`` (standard independent Monte Carlo) or
    ``"balanced"`` (variance-reduced cyclic balanced permutations over each pool /
    stratum / kNN neighbourhood).
    """
    block_hours: int | None = None
    """Optional contiguous weather-block length ``>= 2`` (rows). When given,
    contiguous weather trajectories of length ``block_hours`` are drawn instead of
    isolated timestamps, preserving temporal coherence for lag/rolling features.
    """
    block_mode: str = "independent"
    """How successive contiguous weather blocks are joined when ``block_hours > 1``:
    ``"independent"`` (default, uniform random block starts) or ``"markov"``
    (Markov-bridge transition selecting each next block start from the nearest
    pool states to the previous block's ending meteorology to avoid seam jumps).
    """
    control_variates: bool = False
    """If True and ``aggregate=True``, subtract a zero-mean linear control variate
    built from the resampled numeric covariates and their exact conditional pool
    expectation, reducing Monte Carlo variance without bias.
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


def _conditional_mask(
    pool: pd.DataFrame,
    conditional_on: Mapping[str, Any],
) -> pd.Series:
    """
    Boolean row mask for every key/value condition in ``conditional_on``.

    - Scalar value → exact match.
    - Iterable (list/tuple/set/Series/ndarray) → ``isin`` semantics.
    - Tuple ``(lo, hi)`` is treated as iterable membership (use callable for ranges).
    - Callable → boolean mask applied to that column.

    Kept separate from :func:`_apply_conditional_filter` so a caller can apply
    the same positional mask to a row-aligned frame (see ``normalise``'s
    stratified path) without a label-based ``.loc`` that expands a non-unique
    index.
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
    return mask


def _apply_conditional_filter(
    pool: pd.DataFrame,
    conditional_on: Mapping[str, Any],
) -> pd.DataFrame:
    """Restrict ``pool`` to the rows matching every condition (see :func:`_conditional_mask`)."""
    mask = np.asarray(_conditional_mask(pool, conditional_on), dtype=bool)
    return pool.loc[mask]


def _condition_key(cond: Any) -> Any:
    """A stable cache-key entry for one ``conditional_on`` condition.

    ``repr`` is enough for a scalar, but it truncates an ndarray or Series
    ("array([0, 1, ..., 999])"), so two long conditions that differ only in
    the elided middle shared a key and the cache served the other filter's
    draws; a set's repr is ordered by the per-process string hash, so the
    same condition missed after a restart. Values are hashed by content
    instead. A callable is hashed through :func:`callable_hash`, which
    fingerprints its code and captured state rather than its address.
    """
    from ..utils.cache import callable_hash, dataframe_hash

    if callable(cond):
        return callable_hash(cond)
    if isinstance(cond, set | frozenset):
        return sorted(repr(v) for v in cond)
    if isinstance(cond, list | tuple | pd.Series | np.ndarray):
        try:
            return dataframe_hash(pd.DataFrame({"v": list(cond)}), include_index=False)
        except (TypeError, ValueError):  # mixed or unhashable elements
            return [repr(v) for v in cond]
    return cond


def _split_across_pools(
    variables_resample: Sequence[str],
    resample_df: pd.DataFrame,
    resample_pools: Mapping[str, pd.DataFrame] | None,
) -> list[tuple[int, list[str], pd.DataFrame]]:
    """Assign each resampled variable to the pool it is drawn from.

    Returns ``(stream, columns, pool)`` triples. Stream 0 is ``resample_df`` and
    serves every variable no extra pool claims; an extra pool's stream is a
    hash of its name. So a variable's draws do not move when other variables
    are fixed -- the differences :func:`decom_met` takes between calls stay
    common-random-number differences -- nor when other pools are added,
    dropped or renamed.
    """
    if resample_pools is None:
        resample_pools = {}
    if not isinstance(resample_pools, Mapping):
        raise ConfigError("`resample_pools` must be a mapping of name -> DataFrame.")
    claimed: dict[str, str] = {}
    extra: list[tuple[int, list[str], pd.DataFrame]] = []
    for name in sorted(resample_pools, key=str):
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
        # crc32, not hash(): the stream must not change between processes.
        extra.append((zlib.crc32(str(name).encode("utf-8")) or 1, cols, pool))
    base = [c for c in variables_resample if c not in claimed]
    return ([(0, base, resample_df)] if base else []) + extra


def _parse_stratify_by(stratify_by: str | Sequence[str] | None) -> tuple[str, ...]:
    """Normalise ``stratify_by`` config into a tuple of column names."""
    if stratify_by is None:
        return ()
    if isinstance(stratify_by, str):
        s = stratify_by.strip()
        return (s,) if s else ()
    cols = tuple(str(c).strip() for c in stratify_by if str(c).strip())
    return cols


def _extract_strata(frame: pd.DataFrame, strat_cols: tuple[str, ...], *, label: str) -> np.ndarray:
    """Extract per-row stratum labels from *frame*, deriving ``month``/``season`` if needed."""
    parts: list[np.ndarray] = []
    dates_s: pd.Series | None = None
    for col in strat_cols:
        vals: np.ndarray
        if col in frame.columns:
            vals = frame[col].astype(object).to_numpy()
        elif col in ("month", "season", "hour", "weekday"):
            if dates_s is None:
                if "date" in frame.columns:
                    dates_s = pd.to_datetime(frame["date"], errors="coerce")
                elif isinstance(frame.index, pd.DatetimeIndex):
                    dates_s = pd.Series(frame.index, index=frame.index)
                else:
                    raise ConfigError(
                        f"stratify_by column {col!r} requires a 'date' column or DatetimeIndex in {label}."
                    )
            if dates_s.isna().any():
                raise ConfigError(f"{label} contains missing dates needed for stratify_by={col!r}.")
            if col == "month":
                vals = np.asarray(dates_s.dt.month.to_numpy(), dtype=object)
            elif col == "season":
                vals = np.asarray(
                    ((dates_s.dt.month.to_numpy() % 12) // 3).astype(int), dtype=object
                )
            elif col == "hour":
                vals = np.asarray(dates_s.dt.hour.to_numpy(), dtype=object)
            else:
                # Match add_date_variables() 1..7 weekday encoding
                vals = np.asarray(dates_s.dt.dayofweek.to_numpy() + 1, dtype=object)
        else:
            raise ConfigError(
                f"stratify_by column {col!r} not found in {label} "
                f"(available: {list(frame.columns)[:8]})."
            )
        parts.append(vals)

    if len(parts) == 1:
        return np.asarray([str(v) for v in parts[0]], dtype=object)
    return np.asarray(
        ["\x1f".join(str(p[i]) for p in parts) for i in range(len(frame))],
        dtype=object,
    )


def _draw_rows(stream: int, seed: int, n_pool: int, n_rows: int, replace: bool) -> np.ndarray:
    """Row indices into one pool for one Monte Carlo draw.

    This is the single source of draw indices: :func:`generate_resampled` and
    the batch/vectorised paths all go through it, so the same seed gives the
    same resample whichever path -- and whichever ``memory_save`` setting --
    a call happens to take. Stream 0 keeps the generator the default
    ``normalise`` path has always used, so its results are unchanged; stream
    k > 0 gets an independent generator derived from the same seed.
    """
    rng = np.random.default_rng(seed if stream == 0 else [seed, stream])
    return rng.choice(n_pool, size=n_rows, replace=replace)


def _draw_rows_balanced(
    stream: int,
    base_seed: int,
    sample_pos: int,
    n_pool: int,
    n_rows: int,
    *,
    cycle_cache: dict[tuple[Any, ...], Any] | None = None,
) -> np.ndarray:
    """Balanced cyclic permutation draw over ``0..n_pool-1`` for variance reduction.

    For each row ``r`` and each cycle of ``n_pool`` draws, every pool entry is
    visited identically once before any entry repeats, preserving exact uniform
    marginals while eliminating duplicate-draw Monte Carlo noise.
    """
    cycle = int(sample_pos) // n_pool
    pos_in_cycle = int(sample_pos) % n_pool
    key = ("plain", stream, cycle, n_pool, n_rows)
    cached = cycle_cache.get(key) if cycle_cache is not None else None
    if cached is None:
        rng = np.random.default_rng([base_seed, stream, cycle, 1340])
        perm = rng.permutation(n_pool)
        offsets = rng.integers(0, n_pool, size=n_rows, dtype=np.int64)
        cached = (perm, offsets)
        if cycle_cache is not None:
            cycle_cache[key] = cached
    perm, offsets = cached
    return perm[(offsets + pos_in_cycle) % n_pool]


def _draw_rows_stratified(
    stream: int,
    seed: int,
    df_strata: np.ndarray,
    pool_strata: np.ndarray,
    replace: bool,
) -> np.ndarray:
    """Draw pool row indices matching each target row's stratum deterministically."""
    rng = np.random.default_rng(seed if stream == 0 else [seed, stream])
    n_rows = len(df_strata)
    n_pool = len(pool_strata)
    out = np.empty(n_rows, dtype=np.int64)

    pool_idx: dict[str, np.ndarray] = {}
    for key in sorted(set(pool_strata)):
        pool_idx[key] = np.flatnonzero(pool_strata == key)

    full_pool = np.arange(n_pool, dtype=np.int64)
    missing_keys: list[str] = []
    for key in sorted(set(df_strata)):
        pos = np.flatnonzero(df_strata == key)
        candidates = pool_idx.get(key)
        if candidates is None or candidates.size == 0:
            missing_keys.append(str(key))
            candidates = full_pool
        use_replace = replace or len(pos) > len(candidates)
        out[pos] = rng.choice(candidates, size=len(pos), replace=use_replace)

    if missing_keys and stream in (0, 1):
        log.warning(
            "stratify_by: %d stratum value(s) absent from pool (%s); fell back to full pool.",
            len(missing_keys),
            ", ".join(missing_keys[:4]),
        )
    return out


def _draw_rows_stratified_balanced(
    stream: int,
    base_seed: int,
    sample_pos: int,
    df_strata: np.ndarray,
    pool_strata: np.ndarray,
    *,
    cycle_cache: dict[tuple[Any, ...], Any] | None = None,
) -> np.ndarray:
    """Balanced cyclic permutation draw within each stratum."""
    n_rows = len(df_strata)
    n_pool = len(pool_strata)
    out = np.empty(n_rows, dtype=np.int64)

    map_key = ("strata_map", stream)
    cached_map = cycle_cache.get(map_key) if cycle_cache is not None else None
    if cached_map is None:
        pool_idx = {key: np.flatnonzero(pool_strata == key) for key in sorted(set(pool_strata))}
        full_pool = np.arange(n_pool, dtype=np.int64)
        missing_keys: list[str] = []
        target_groups: list[tuple[int, np.ndarray, np.ndarray]] = []
        for s_idx, key in enumerate(sorted(set(df_strata))):
            pos = np.flatnonzero(df_strata == key)
            cands = pool_idx.get(key)
            if cands is None or cands.size == 0:
                missing_keys.append(str(key))
                cands = full_pool
            target_groups.append((s_idx, pos, cands))
        if missing_keys and stream in (0, 1) and int(sample_pos) == 0:
            log.warning(
                "stratify_by: %d stratum value(s) absent from pool (%s); fell back to full pool.",
                len(missing_keys),
                ", ".join(missing_keys[:4]),
            )
        cached_map = target_groups
        if cycle_cache is not None:
            cycle_cache[map_key] = cached_map

    for s_idx, pos, cands in cached_map:
        p_s = len(cands)
        cycle = int(sample_pos) // p_s
        pos_in_cycle = int(sample_pos) % p_s
        ckey = ("strata", stream, s_idx, cycle)
        cyc = cycle_cache.get(ckey) if cycle_cache is not None else None
        if cyc is None:
            rng = np.random.default_rng([base_seed, stream, s_idx, cycle, 1339])
            perm = rng.permutation(p_s)
            offsets = rng.integers(0, p_s, size=len(pos), dtype=np.int64)
            cyc = (perm, offsets)
            if cycle_cache is not None:
                cycle_cache[ckey] = cyc
        perm, offsets = cyc
        out[pos] = cands[perm[(offsets + pos_in_cycle) % p_s]]

    return out


def _standardize_pool_matrix(pool_values: np.ndarray) -> np.ndarray | None:
    """Return a finite, z-scored float64 view of numeric columns in *pool_values*."""
    if pool_values.ndim != 2 or pool_values.shape[0] == 0 or pool_values.shape[1] == 0:
        return None
    df_tmp = pd.DataFrame(pool_values)
    num = df_tmp.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    keep_cols = np.any(np.isfinite(num), axis=0)
    if not np.any(keep_cols):
        return None
    mat = num[:, keep_cols]
    mu = np.nanmean(mat, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.nanstd(mat, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    return np.where(np.isfinite(mat), (mat - mu) / sd, 0.0)


def _draw_rows_block(
    stream: int,
    seed: int,
    n_pool: int,
    n_rows: int,
    block_hours: int,
    *,
    df_strata: np.ndarray | None = None,
    pool_strata: np.ndarray | None = None,
    z_pool: np.ndarray | None = None,
    block_mode: str = "independent",
) -> np.ndarray:
    """Draw contiguous meteorological blocks of length ``block_hours`` from the pool."""
    rng = np.random.default_rng(seed if stream == 0 else [seed, stream])
    b_len = int(block_hours)
    out = np.empty(n_rows, dtype=np.int64)

    pool_idx_map: dict[str, np.ndarray] | None = None
    if df_strata is not None and pool_strata is not None:
        pool_idx_map = {key: np.flatnonzero(pool_strata == key) for key in sorted(set(pool_strata))}

    use_markov = block_mode == "markov" and z_pool is not None and len(z_pool) == n_pool
    all_starts_cache: np.ndarray | None = None

    for start_r in range(0, n_rows, b_len):
        end_r = min(n_rows, start_r + b_len)
        seg = end_r - start_r
        if pool_idx_map is not None and df_strata is not None:
            key = df_strata[start_r]
            cands = pool_idx_map.get(key)
            if cands is not None and cands.size > 0:
                if seg <= n_pool:
                    valid = cands[cands <= n_pool - seg]
                    pool_choices = valid if valid.size > 0 else cands
                else:
                    pool_choices = cands
                if use_markov and start_r > 0 and len(pool_choices) > 1:
                    assert z_pool is not None
                    ref_idx = (int(out[start_r - 1]) + 1) % n_pool
                    dists = np.sum((z_pool[pool_choices] - z_pool[ref_idx]) ** 2, axis=1)
                    k_bridge = min(len(pool_choices), max(4, min(16, len(pool_choices) // 20)))
                    if k_bridge < len(pool_choices):
                        top_idx = np.argpartition(dists, k_bridge - 1)[:k_bridge]
                        bridge_choices = pool_choices[top_idx]
                    else:
                        bridge_choices = pool_choices
                    start_p = int(bridge_choices[int(rng.integers(0, len(bridge_choices)))])
                else:
                    start_p = int(pool_choices[int(rng.integers(0, len(pool_choices)))])
            else:
                max_start = max(1, n_pool - seg + 1) if seg <= n_pool else n_pool
                if use_markov and start_r > 0 and max_start > 1:
                    assert z_pool is not None
                    ref_idx = (int(out[start_r - 1]) + 1) % n_pool
                    dists = np.sum((z_pool[:max_start] - z_pool[ref_idx]) ** 2, axis=1)
                    k_bridge = min(max_start, max(4, min(16, max_start // 20)))
                    top_idx = np.argpartition(dists, k_bridge - 1)[:k_bridge]
                    start_p = int(top_idx[int(rng.integers(0, len(top_idx)))])
                else:
                    start_p = int(rng.integers(0, max_start))
        else:
            max_start = max(1, n_pool - seg + 1) if seg <= n_pool else n_pool
            if use_markov and start_r > 0 and max_start > 1:
                assert z_pool is not None
                if all_starts_cache is None or len(all_starts_cache) != max_start:
                    all_starts_cache = np.arange(max_start, dtype=np.int64)
                ref_idx = (int(out[start_r - 1]) + 1) % n_pool
                dists = np.sum((z_pool[:max_start] - z_pool[ref_idx]) ** 2, axis=1)
                k_bridge = min(max_start, max(4, min(16, max_start // 20)))
                if k_bridge < max_start:
                    top_idx = np.argpartition(dists, k_bridge - 1)[:k_bridge]
                    start_p = int(top_idx[int(rng.integers(0, len(top_idx)))])
                else:
                    start_p = int(rng.integers(0, max_start))
            else:
                start_p = int(rng.integers(0, max_start))
        out[start_r:end_r] = (start_p + np.arange(seg, dtype=np.int64)) % n_pool
    return out


def _standardize_for_knn(
    df: pd.DataFrame, pool: pd.DataFrame, cond_cols: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score fixed numeric columns using the pool's mean and standard deviation."""
    cols = list(cond_cols)
    x_pool = pool[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    x_df = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    mu = np.nanmean(x_pool, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.nanstd(x_pool, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    z_pool = np.where(np.isfinite(x_pool), (x_pool - mu) / sd, 0.0)
    z_df = np.where(np.isfinite(x_df), (x_df - mu) / sd, 0.0)
    return z_df, z_pool


def _build_knn_indices(
    df: pd.DataFrame,
    pool: pd.DataFrame,
    cond_cols: Sequence[str],
    k: int,
    *,
    df_strata: np.ndarray | None = None,
    pool_strata: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Precompute ``(n_rows, k_eff)`` pool neighbour indices for on-manifold draws."""
    from scipy.spatial import cKDTree

    n_rows = len(df)
    n_pool = len(pool)
    if n_pool == 0:
        raise DataError("Resample pool has no rows for conditional_k matching.")
    k_target = max(1, int(k))
    z_df, z_pool = _standardize_for_knn(df, pool, cond_cols)

    if df_strata is not None and pool_strata is not None:
        width = min(k_target, n_pool)
        out = np.zeros((n_rows, width), dtype=np.int64)
        counts = np.full(n_rows, width, dtype=np.int64)
        need_counts = False
        pool_idx_map: dict[str, np.ndarray] = {
            str(key): np.flatnonzero(pool_strata == key) for key in sorted(set(pool_strata))
        }
        full_tree: cKDTree | None = None
        mapped: np.ndarray[Any, Any]
        for key in sorted(set(df_strata)):
            pos = np.flatnonzero(df_strata == key)
            cand = pool_idx_map.get(str(key))
            if cand is None or cand.size == 0:
                if full_tree is None:
                    full_tree = cKDTree(z_pool)
                k_eff = min(width, n_pool)
                _, loc = full_tree.query(z_df[pos], k=k_eff)
                mapped = np.asarray(loc, dtype=np.int64).reshape(len(pos), k_eff)
            else:
                tree = cKDTree(z_pool[cand])
                k_eff = min(width, cand.size)
                _, loc = tree.query(z_df[pos], k=k_eff)
                loc_2d = np.asarray(loc, dtype=np.int64).reshape(len(pos), k_eff)
                mapped = cand[loc_2d]
            out[pos, :k_eff] = mapped
            if k_eff < width:
                counts[pos] = k_eff
                need_counts = True
        return out, (counts if need_counts else None)

    k_eff = min(k_target, n_pool)
    tree = cKDTree(z_pool)
    _, loc = tree.query(z_df, k=k_eff)
    return np.asarray(loc, dtype=np.int64).reshape(n_rows, k_eff), None


def _draw_rows_knn(
    stream: int,
    seed: int,
    nbr_idx: np.ndarray,
    valid_counts: np.ndarray | None = None,
) -> np.ndarray:
    """Uniformly pick one of each target row's precomputed k nearest pool neighbours."""
    rng = np.random.default_rng(seed if stream == 0 else [seed, stream])
    n_rows = nbr_idx.shape[0]
    high: Any = valid_counts if valid_counts is not None else nbr_idx.shape[1]
    pick = rng.integers(0, high, size=n_rows)
    return nbr_idx[np.arange(n_rows), pick]


def _draw_rows_knn_balanced(
    stream: int,
    base_seed: int,
    sample_pos: int,
    nbr_idx: np.ndarray,
    valid_counts: np.ndarray | None = None,
    *,
    cycle_cache: dict[tuple[Any, ...], Any] | None = None,
) -> np.ndarray:
    """Balanced cyclic draw over each row's ``k_eff`` nearest pool neighbours."""
    # ``shape`` entries are ``Any`` under the bare ndarray annotation; typing
    # them as int picks the array-returning ``integers`` overload instead of
    # the scalar one (numpy >= 2.4 stubs), which mypy otherwise chooses.
    n_rows = int(nbr_idx.shape[0])
    width = int(nbr_idx.shape[1])
    offsets: np.ndarray[Any, Any]
    if valid_counts is None:
        cycle = int(sample_pos) // width
        pos_in_cycle = int(sample_pos) % width
        ckey = ("knn", stream, width, cycle, n_rows)
        cyc = cycle_cache.get(ckey) if cycle_cache is not None else None
        if cyc is None:
            rng = np.random.default_rng([base_seed, stream, cycle, 1337])
            perm = rng.permutation(width)
            offsets = rng.integers(0, width, size=n_rows, dtype=np.int64)
            cyc = (perm, offsets)
            if cycle_cache is not None:
                cycle_cache[ckey] = cyc
        perm, offsets = cyc
        pick = perm[(offsets + pos_in_cycle) % width]
        return nbr_idx[np.arange(n_rows), pick]

    pick_all = np.empty(n_rows, dtype=np.int64)
    for k_val in np.unique(valid_counts):
        k_int = int(k_val)
        pos = np.flatnonzero(valid_counts == k_int)
        cycle = int(sample_pos) // k_int
        pos_in_cycle = int(sample_pos) % k_int
        ckey = ("knn_var", stream, k_int, cycle, len(pos))
        cyc = cycle_cache.get(ckey) if cycle_cache is not None else None
        if cyc is None:
            rng = np.random.default_rng([base_seed, stream, k_int, cycle, 1338])
            perm = rng.permutation(k_int)
            offsets = rng.integers(0, k_int, size=len(pos), dtype=np.int64)
            cyc = (perm, offsets)
            if cycle_cache is not None:
                cycle_cache[ckey] = cyc
        perm, offsets = cyc
        pick_all[pos] = perm[(offsets + pos_in_cycle) % k_int]
    return nbr_idx[np.arange(n_rows), pick_all]


def _draw_sample_indices(
    stream: int,
    draw_seed: int,
    *,
    base_seed: int,
    sample_pos: int,
    n_pool: int,
    n_rows: int,
    replace: bool,
    sampling: str = "iid",
    block_hours: int | None = None,
    block_mode: str = "independent",
    z_pool: np.ndarray | None = None,
    df_strata: np.ndarray | None = None,
    p_strata: np.ndarray | None = None,
    knn_spec: tuple[np.ndarray, np.ndarray | None] | None = None,
    cycle_cache: dict[tuple[Any, ...], Any] | None = None,
) -> np.ndarray:
    """Dispatch single-sample pool index generation uniformly across all execution paths."""
    if block_hours is not None and block_hours > 1:
        return _draw_rows_block(
            stream,
            draw_seed,
            n_pool,
            n_rows,
            block_hours,
            df_strata=df_strata,
            pool_strata=p_strata,
            z_pool=z_pool,
            block_mode=block_mode,
        )
    if knn_spec is not None:
        if sampling == "balanced":
            return _draw_rows_knn_balanced(
                stream,
                base_seed,
                sample_pos,
                knn_spec[0],
                knn_spec[1],
                cycle_cache=cycle_cache,
            )
        return _draw_rows_knn(stream, draw_seed, knn_spec[0], knn_spec[1])
    if df_strata is not None and p_strata is not None:
        if sampling == "balanced":
            return _draw_rows_stratified_balanced(
                stream,
                base_seed,
                sample_pos,
                df_strata,
                p_strata,
                cycle_cache=cycle_cache,
            )
        return _draw_rows_stratified(stream, draw_seed, df_strata, p_strata, replace)
    if sampling == "balanced":
        return _draw_rows_balanced(
            stream,
            base_seed,
            sample_pos,
            n_pool,
            n_rows,
            cycle_cache=cycle_cache,
        )
    return _draw_rows(stream, draw_seed, n_pool, n_rows, replace)


def _stacked_columns(
    df: pd.DataFrame,
    draws: Sequence[tuple[int, list[str], np.ndarray, np.ndarray | None]],
    seeds: Sequence[int] | np.ndarray,
    replace: bool,
    df_strata: np.ndarray | None = None,
    knn_map: Mapping[int, tuple[np.ndarray, np.ndarray | None]] | None = None,
    *,
    base_seed: int | None = None,
    sample_positions: Sequence[int] | None = None,
    sampling: str = "iid",
    block_hours: int | None = None,
    block_mode: str = "independent",
    z_pool_map: Mapping[int, np.ndarray | None] | None = None,
) -> dict[str, np.ndarray]:
    """Columns of ``len(seeds)`` resampled copies of ``df``, stacked copy after copy.

    ``draws`` holds ``(stream, columns, values, pool_strata)`` with ``values``
    the pool's ``columns`` as one array, so each copy takes whole pool rows.
    """
    n_rows = len(df)
    b_seed = int(seeds[0]) if base_seed is None and len(seeds) else int(base_seed or 0)
    pos_list = list(sample_positions) if sample_positions is not None else list(range(len(seeds)))
    cycle_cache: dict[tuple[Any, ...], Any] = {}
    drawn: dict[str, np.ndarray] = {}
    for stream, cols, values, p_strata in draws:
        knn_spec = knn_map.get(stream) if knn_map is not None else None
        z_pool = z_pool_map.get(stream) if z_pool_map is not None else None
        idx = np.empty((len(seeds), n_rows), dtype=np.int64)
        for i, (s, pos) in enumerate(zip(seeds, pos_list, strict=True)):
            idx[i] = _draw_sample_indices(
                stream,
                int(s),
                base_seed=b_seed,
                sample_pos=int(pos),
                n_pool=len(values),
                n_rows=n_rows,
                replace=replace,
                sampling=sampling,
                block_hours=block_hours,
                block_mode=block_mode,
                z_pool=z_pool,
                df_strata=df_strata,
                p_strata=p_strata,
                knn_spec=knn_spec,
                cycle_cache=cycle_cache,
            )
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
    *,
    df_strata: np.ndarray | None = None,
    pool_strata_map: Mapping[int, np.ndarray] | None = None,
    knn_map: Mapping[int, tuple[np.ndarray, np.ndarray | None]] | None = None,
    sampling: str = "iid",
    block_hours: int | None = None,
    block_mode: str = "independent",
    z_pool_map: Mapping[int, np.ndarray | None] | None = None,
    base_seed: int | None = None,
    sample_pos: int = 0,
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

    b_seed = int(seed) if base_seed is None else int(base_seed)
    out = df.copy(deep=False).reset_index(drop=True)
    for stream, cols, pool in draws:
        # The same indices the batch/vectorised paths draw (``_draw_sample_indices``).
        # pandas' ``sample(random_state=seed)`` used the legacy RandomState
        # here for stream 0, so a ``normalise(..., memory_save=True)`` and a
        # default call with the same seed resampled differently.
        p_strata = pool_strata_map.get(stream) if pool_strata_map is not None else None
        knn_spec = knn_map.get(stream) if knn_map is not None else None
        z_pool = z_pool_map.get(stream) if z_pool_map is not None else None
        if (
            z_pool is None
            and block_mode == "markov"
            and block_hours is not None
            and block_hours > 1
        ):
            z_pool = _standardize_pool_matrix(pool[cols].to_numpy())
        idx = _draw_sample_indices(
            stream,
            int(seed),
            base_seed=b_seed,
            sample_pos=int(sample_pos),
            n_pool=len(pool),
            n_rows=len(df),
            replace=replace,
            sampling=sampling,
            block_hours=block_hours,
            block_mode=block_mode,
            z_pool=z_pool,
            df_strata=df_strata,
            p_strata=p_strata,
            knn_spec=knn_spec,
        )
        out.loc[:, cols] = pool[cols].to_numpy()[idx]
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
        ``log_transform=True`` reads ``df`` and ``model`` on the ``log1p``
        scale and returns the result in the original units -- see
        :attr:`NormaliseConfig.log_transform`.

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
    if _cfg.resample_pools is not None and not isinstance(_cfg.resample_pools, Mapping):
        raise ConfigError("`resample_pools` must be a mapping of name -> DataFrame.")
    if _cfg.conditional_k is not None:
        cond_k: object = _cfg.conditional_k
        if isinstance(cond_k, bool) or not isinstance(cond_k, numbers.Integral):
            raise ConfigError(
                f"`conditional_k` must be a positive integer or None, got {cond_k!r}."
            )
        if int(cond_k) < 1:
            raise ConfigError(f"`conditional_k` must be at least 1, got {cond_k!r}.")
    if _cfg.back_transform not in ("arithmetic", "smearing", "geometric"):
        raise ConfigError(
            "`back_transform` must be 'arithmetic', 'smearing', or 'geometric', "
            f"got {_cfg.back_transform!r}."
        )
    if _cfg.smearing_factor is not None:
        sf_obj: object = _cfg.smearing_factor
        if isinstance(sf_obj, bool) or not isinstance(sf_obj, numbers.Real):
            raise ConfigError(
                f"`smearing_factor` must be a positive finite float or None, got {sf_obj!r}."
            )
        sf_val = float(sf_obj)
        if not np.isfinite(sf_val) or sf_val <= 0.0:
            raise ConfigError(f"`smearing_factor` must be a positive finite float, got {sf_obj!r}.")
    _resolve_paired_mode(_cfg.paired, log_transform=bool(_cfg.log_transform))
    pf_obj: object = _cfg.paired_floor
    if (
        isinstance(pf_obj, bool)
        or not isinstance(pf_obj, numbers.Real)
        or not np.isfinite(float(pf_obj))
        or float(pf_obj) <= 0.0
    ):
        raise ConfigError(f"`paired_floor` must be a positive finite float, got {pf_obj!r}.")
    if _cfg.sampling not in ("iid", "balanced"):
        raise ConfigError(f"`sampling` must be 'iid' or 'balanced', got {_cfg.sampling!r}.")
    if _cfg.block_mode not in ("independent", "markov"):
        raise ConfigError(
            f"`block_mode` must be 'independent' or 'markov', got {_cfg.block_mode!r}."
        )
    if _cfg.block_hours is not None:
        bh_obj: object = _cfg.block_hours
        if isinstance(bh_obj, bool) or not isinstance(bh_obj, numbers.Integral):
            raise ConfigError(f"`block_hours` must be a positive integer or None, got {bh_obj!r}.")
        if int(bh_obj) < 1:
            raise ConfigError(f"`block_hours` must be at least 1, got {bh_obj!r}.")

    required_features: set[str] | None = None
    if _cfg.covariates is None:
        if model is None:
            raise ConfigError("`covariates` must be provided (either directly or via config).")
        try:
            from ..utils.features import extract_features

            features = [str(c) for c in extract_features(model)]
            required_features = set(features)
            _cfg.covariates = features
        except Exception as exc:
            raise ConfigError(
                "`covariates` must be provided (either directly or via config) -- "
                "it could not be auto-detected from `model`."
            ) from exc
    elif model is not None:
        try:
            from ..utils.features import extract_features

            required_features = {str(c) for c in extract_features(model)}
        except Exception:
            required_features = None
        if required_features is not None:
            missing = sorted(required_features - set(_cfg.covariates))
            if missing:
                raise ConfigError(
                    f"`covariates` is missing {missing}, which `model` requires for "
                    "prediction. `covariates` must be a superset of the model's trained "
                    "features -- pass the full feature list and use `variables_resample` to "
                    "choose which of those get resampled."
                )

    # A feature the model was trained on but the frame does not carry is
    # dropped in silence by ``check_data``, and an AutoML backend then
    # reindexes to its feature names, fills the holes with zeros and predicts
    # as if that meteorology were zero. The covariates check above only proves
    # the *list* is a superset; this proves the frame is too.
    if required_features:
        present = set(df.columns)
        if isinstance(df.index, pd.DatetimeIndex):
            present.add("date")
        absent = sorted(required_features - present)
        if absent:
            raise ConfigError(
                f"`df` is missing {absent}, which `model` requires for prediction. "
                "Pass the frame the model was trained on (or add the missing "
                "columns, e.g. `add_date_variables` for the calendar features) -- "
                "the backend would otherwise predict them as zero."
            )

    if _cfg.cache is None:
        return _normalise_uncached(df, model, _cfg)

    from ..utils.cache import (
        config_hash,
        dataframe_hash,
        dependency_versions,
        make_memory,
        model_hash,
        package_version,
    )

    # Every ingredient below is order-canonicalised: the result depends on which
    # columns and which resampled variables are involved, never on the order they
    # are listed in. Two places had to be fixed together -- the lists themselves
    # (hashed via `sorted(...)` in config_hash) *and* the column order of the
    # frames handed to dataframe_hash, which is order-sensitive. Callers that
    # reach the same set by different routes -- notably decompose(), whose
    # resample list is a shrinking sublist of a feature-importance or permutation
    # order -- otherwise recompute every single time.
    strat_cols = _parse_stratify_by(_cfg.stratify_by)
    key_cols = sorted(set(_cfg.covariates) | {"value", "date"})
    df_dated_key = process_date(df.copy())
    df_keyed = check_data(df_dated_key, _cfg.covariates, "value")
    # The automatic Duan smearing factor is estimated from the rows labelled
    # ``set == "testing"`` (``_resolve_smearing_factor``), so two frames with
    # identical date/value/covariates but a different split produce different
    # results and must not share a cache entry.
    set_key = dataframe_hash(df_dated_key[["set"]]) if "set" in df_dated_key.columns else None
    resample_pool = df_keyed if _cfg.resample_df is None else _cfg.resample_df
    # `month`/`season` strata are derived from a pool's own `date` column, so
    # changing the pool's date composition changes which rows serve each
    # stratum while leaving the served covariate values (and therefore the
    # rest of the key) identical. Hash the dates too, or a shared cache
    # returns draws from the old month mix.
    derived_strata_cols = {"date"} if {"month", "season"} & set(strat_cols) else set()
    resample_key_cols = sorted(
        c
        for c in set(_cfg.variables_resample or key_cols)
        | set(_cfg.conditional_on or {})
        | set(strat_cols)
        | derived_strata_cols
        if c in resample_pool.columns
    )
    pool_key_cols = (
        set(_cfg.covariates)
        | set(_cfg.variables_resample or ())
        | set(strat_cols)
        | derived_strata_cols
    )
    pool_keys = [
        (str(name), dataframe_hash(pool[sorted(c for c in pool.columns if c in pool_key_cols)]))
        for name, pool in sorted((_cfg.resample_pools or {}).items(), key=lambda kv: str(kv[0]))
        if isinstance(pool, pd.DataFrame)
    ]
    strata_key: Any = None
    if strat_cols:
        strata_arr = _extract_strata(df_dated_key, strat_cols, label="df")
        strata_key = dataframe_hash(pd.DataFrame({"_s": strata_arr}), include_index=False)
    conditional_key = (
        {
            str(col): _condition_key(cond)
            for col, cond in sorted(_cfg.conditional_on.items(), key=lambda kv: str(kv[0]))
        }
        if _cfg.conditional_on
        else None
    )
    cache_key = config_hash(
        package_version(),
        dependency_versions(),
        *([pool_keys] if pool_keys else []),
        *([list(strat_cols), strata_key] if strat_cols else []),
        *([("conditional_k", int(_cfg.conditional_k))] if _cfg.conditional_k is not None else []),
        sorted(_cfg.covariates),
        sorted(_cfg.variables_resample) if _cfg.variables_resample is not None else None,
        _cfg.n_samples,
        _cfg.replace,
        _cfg.aggregate,
        _cfg.seed,
        _cfg.memory_save,
        list(_cfg.return_quantiles) if _cfg.return_quantiles else None,
        conditional_key,
        _cfg.batch_size,
        # Joins the key only when on, so caches written by the linear path keep
        # their key (and their hits) -- the flag cannot change a linear result.
        *(
            [("log_transform", _cfg.back_transform, _cfg.smearing_factor)]
            if _cfg.log_transform
            else []
        ),
        *([("set", set_key)] if set_key is not None else []),
        *([("mc_se", True)] if _cfg.return_mc_se else []),
        *([("paired", _cfg.paired, float(_cfg.paired_floor))] if _cfg.paired else []),
        *([("sampling", _cfg.sampling)] if _cfg.sampling != "iid" else []),
        *([("block_hours", int(_cfg.block_hours))] if _cfg.block_hours is not None else []),
        *([("block_mode", _cfg.block_mode)] if _cfg.block_mode != "independent" else []),
        *([("control_variates", True)] if _cfg.control_variates else []),
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


def _resolve_smearing_factor(
    df_dated: pd.DataFrame,
    df_checked: pd.DataFrame,
    model: object,
    _cfg: NormaliseConfig,
    n_cores_eff: int,
) -> float:
    """Resolve Duan smearing factor ``phi`` when ``log_transform`` and ``back_transform='smearing'``."""
    if not _cfg.log_transform or _cfg.back_transform != "smearing":
        return 1.0
    if _cfg.smearing_factor is not None:
        return float(_cfg.smearing_factor)
    sub = df_checked
    if "set" in df_dated.columns and len(df_dated) == len(df_checked):
        val_num = pd.to_numeric(df_checked["value"], errors="coerce")
        test_mask = (df_dated["set"].to_numpy() == "testing") & val_num.notna().to_numpy()
        train_mask = (df_dated["set"].to_numpy() == "training") & val_num.notna().to_numpy()
        if int(test_mask.sum()) >= 10:
            sub = df_checked.loc[test_mask]
        elif bool(train_mask.any()):
            sub = df_checked.loc[train_mask]
    preds_sub = ml_predict(model, sub, num_threads=n_cores_eff)
    return compute_duan_smearing(sub["value"], preds_sub)


def _prepare_control_variate(
    df: pd.DataFrame,
    draw_arrays: Sequence[tuple[int, list[str], np.ndarray, np.ndarray | None]],
    model: object,
    *,
    n_cores_eff: int,
    is_log_arith: bool,
    phi: float,
    df_strata: np.ndarray | None,
    knn_map: Mapping[int, tuple[np.ndarray, np.ndarray | None]],
) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    """Fit a linear control-variate surrogate and compute exact pool expectations."""
    n_rows = len(df)
    if n_rows < 3 or not draw_arrays:
        return None

    cv_cols: list[str] = []
    mu_parts: list[np.ndarray] = []
    df_cols: list[np.ndarray] = []

    for stream, cols, pool_vals, p_strata in draw_arrays:
        if len(pool_vals) == 0:
            continue
        p_df = pd.DataFrame(pool_vals, columns=cols)
        knn_spec = knn_map.get(stream)
        for col in cols:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            p_num = pd.to_numeric(p_df[col], errors="coerce").to_numpy(dtype=np.float64)
            d_num = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            p_fin = p_num[np.isfinite(p_num)]
            if p_fin.size == 0 or float(np.std(p_fin)) <= 1e-12:
                continue
            g_mean = float(np.mean(p_fin))
            p_filled = np.where(np.isfinite(p_num), p_num, g_mean)
            d_filled = np.where(np.isfinite(d_num), d_num, g_mean)

            if knn_spec is not None:
                nbr_idx, valid_counts = knn_spec
                if valid_counts is None:
                    mu_col = p_filled[nbr_idx].mean(axis=1)
                else:
                    mu_col = np.empty(n_rows, dtype=np.float64)
                    for k_val in np.unique(valid_counts):
                        k_int = int(k_val)
                        pos = np.flatnonzero(valid_counts == k_int)
                        mu_col[pos] = p_filled[nbr_idx[pos, :k_int]].mean(axis=1)
            elif df_strata is not None and p_strata is not None:
                mu_col = np.full(n_rows, g_mean, dtype=np.float64)
                stratum_means = {
                    key: float(np.mean(p_filled[p_strata == key]))
                    for key in set(p_strata)
                    if np.any(p_strata == key)
                }
                for key in set(df_strata):
                    if key in stratum_means:
                        mu_col[df_strata == key] = stratum_means[key]
            else:
                mu_col = np.full(n_rows, g_mean, dtype=np.float64)

            cv_cols.append(col)
            mu_parts.append(mu_col)
            df_cols.append(d_filled)

    if not cv_cols:
        return None

    X_df = np.column_stack(df_cols)
    mu_x = np.column_stack(mu_parts)
    fact_raw = ml_predict(model, df, num_threads=n_cores_eff)
    y_eval = (
        _back_transform_values(fact_raw, smearing_factor=phi)
        if is_log_arith
        else np.asarray(fact_raw, dtype=np.float64)
    )
    finite_y = np.isfinite(y_eval)
    if int(finite_y.sum()) < 3:
        return None

    X_fit = X_df[finite_y]
    y_fit = y_eval[finite_y]
    x_mean = X_fit.mean(axis=0)
    x_sd = np.maximum(X_fit.std(axis=0), 1e-12)
    Z = (X_fit - x_mean) / x_sd
    yc = y_fit - float(np.mean(y_fit))
    p = Z.shape[1]
    gram = Z.T @ Z + (1e-4 * float(len(y_fit))) * np.eye(p)
    w = np.linalg.solve(gram, Z.T @ yc)
    beta = w / x_sd
    return cv_cols, beta.astype(np.float64), mu_x.astype(np.float64)


def _aggregate_draws_frame(
    df_result: pd.DataFrame,
    _cfg: NormaliseConfig,
    *,
    phi: float,
    is_log_arith: bool,
    is_log_geom: bool,
) -> pd.DataFrame:
    """Aggregate per-seed predictions into normalised series, quantiles, MC SE, or wide table."""
    if _cfg.aggregate:
        (log.info if _cfg.verbose else log.debug)("Aggregating %d predictions.", _cfg.n_samples)
        if is_log_arith:
            norm_vals = _back_transform_values(
                df_result["normalised"].to_numpy(dtype=float), smearing_factor=phi
            )
            if "_cv_adj" in df_result.columns:
                norm_vals = norm_vals - df_result["_cv_adj"].to_numpy(dtype=float)
            df_work = pd.DataFrame(
                {
                    "date": df_result["date"].to_numpy(),
                    "observed": np.expm1(df_result["observed"].to_numpy(dtype=float)),
                    "normalised": norm_vals,
                    "seed": df_result["seed"].to_numpy(),
                }
            )
        elif "_cv_adj" in df_result.columns:
            df_work = pd.DataFrame(
                {
                    "date": df_result["date"].to_numpy(),
                    "observed": df_result["observed"].to_numpy(dtype=float),
                    "normalised": df_result["normalised"].to_numpy(dtype=float)
                    - df_result["_cv_adj"].to_numpy(dtype=float),
                    "seed": df_result["seed"].to_numpy(),
                }
            )
        else:
            df_work = df_result

        gb = df_work.groupby("date", as_index=True)
        df_out = gb[["observed", "normalised"]].mean()
        mean_z = df_out["normalised"].to_numpy(dtype=float).copy() if is_log_geom else None
        if is_log_geom:
            df_out["observed"] = np.expm1(df_out["observed"].to_numpy(dtype=float))
            assert mean_z is not None
            df_out["normalised"] = np.expm1(mean_z)

        if _cfg.return_mc_se:
            per_seed = df_work.groupby(["date", "seed"], as_index=False)["normalised"].mean()
            gb_s = per_seed.groupby("date", as_index=True)["normalised"]
            cnt = gb_s.count()
            se_s = gb_s.std(ddof=1) / np.sqrt(cnt)
            se_vals = se_s.reindex(df_out.index).to_numpy(dtype=float)
            if is_log_geom:
                assert mean_z is not None
                se_vals = np.exp(mean_z) * se_vals
            df_out["mc_se"] = se_vals

        if _cfg.return_quantiles:
            q_arr = sorted({float(q) for q in _cfg.return_quantiles})
            if "_cv_adj" in df_result.columns:
                # Take the quantiles from the same adjusted draw values as
                # the mean. `_cv_adj` moves every draw, so building the
                # bands from the raw draws left them centred on a different
                # level than `normalised` (and, for arithmetic
                # back-transform, in a different space).
                gb_raw = df_work.groupby("date", as_index=True)
                q_df = gb_raw["normalised"].quantile(np.asarray(q_arr)).unstack(level=-1)
                if is_log_geom:
                    q_df = pd.DataFrame(
                        np.expm1(q_df.to_numpy(dtype=float)),
                        index=q_df.index,
                        columns=q_df.columns,
                    )
            else:
                # Without control variates keep the historical convention:
                # quantiles of the raw fits, one inverse transform at the
                # end (bit-identical to earlier releases).
                gb_raw = df_result.groupby("date", as_index=True)
                q_df = gb_raw["normalised"].quantile(np.asarray(q_arr)).unstack(level=-1)
                if _cfg.log_transform:
                    q_df = pd.DataFrame(
                        np.expm1(q_df.to_numpy(dtype=float)),
                        index=q_df.index,
                        columns=q_df.columns,
                    )
            q_df.columns = pd.Index([_format_quantile_name(float(q)) for q in q_df.columns])
            df_out = df_out.join(q_df, how="left")
        return df_out

    observed = df_result.groupby("date", as_index=True)[["observed"]].mean()
    wide = (
        df_result.groupby(["date", "seed"], as_index=False)["normalised"]
        .mean()
        .pivot(index="date", columns="seed", values="normalised")
    )
    if is_log_arith:
        observed["observed"] = np.expm1(observed["observed"].to_numpy(dtype=float))
        wide = pd.DataFrame(
            _back_transform_values(wide.to_numpy(dtype=float), smearing_factor=phi),
            index=wide.index,
            columns=wide.columns,
        )
    elif is_log_geom:
        observed["observed"] = np.expm1(observed["observed"].to_numpy(dtype=float))
        wide = pd.DataFrame(
            np.expm1(wide.to_numpy(dtype=float)),
            index=wide.index,
            columns=wide.columns,
        )
    if _cfg.return_quantiles:
        log.debug(
            "`return_quantiles` ignored when aggregate=False (wide table already exposes all seeds)."
        )
    return pd.concat([observed, wide], axis=1)


def _attach_paired_columns(
    df_out: pd.DataFrame,
    df: pd.DataFrame,
    model: object,
    _cfg: NormaliseConfig,
    *,
    phi: float,
    paired_mode: str | None,
    is_log_arith: bool,
    is_log_geom: bool,
    n_cores_eff: int,
    fact_raw_precomputed: np.ndarray | None = None,
) -> pd.DataFrame:
    """Append ``factual``, ``weather_effect``, and ``paired`` columns when ``paired`` is enabled."""
    if paired_mode is None or not _cfg.aggregate:
        return df_out
    fact_raw = (
        fact_raw_precomputed
        if fact_raw_precomputed is not None
        else ml_predict(model, df, num_threads=n_cores_eff)
    )
    if is_log_arith:
        fact_vals = _back_transform_values(fact_raw, smearing_factor=phi)
    elif is_log_geom:
        fact_vals = np.expm1(fact_raw)
    else:
        fact_vals = fact_raw
    fact_aligned = (
        pd.Series(fact_vals, index=pd.Index(df["date"].to_numpy(), name="date"))
        .groupby(level=0)
        .mean()
        .reindex(df_out.index)
        .to_numpy(dtype=float)
    )
    obs_arr = df_out["observed"].to_numpy(dtype=float)
    norm_arr = df_out["normalised"].to_numpy(dtype=float)
    df_out["factual"] = fact_aligned
    df_out["weather_effect"] = fact_aligned - norm_arr
    df_out["paired"] = _compute_paired_series(
        obs_arr,
        fact_aligned,
        norm_arr,
        mode=paired_mode,
        floor=float(_cfg.paired_floor),
    )
    return df_out


def _normalise_uncached(df: pd.DataFrame, model: object, _cfg: NormaliseConfig) -> pd.DataFrame:
    """Uncached resample-and-predict core of :func:`normalise`."""
    # narrowing: normalise() always resolves _cfg.covariates (auto-detecting
    # from `model` or raising ConfigError) before calling this helper.
    assert _cfg.covariates is not None
    df_dated = process_date(df)
    strat_cols = _parse_stratify_by(_cfg.stratify_by)
    df_strata = _extract_strata(df_dated, strat_cols, label="df") if strat_cols else None
    df = check_data(df_dated, _cfg.covariates, "value")
    if "date" not in df.columns:
        raise DataError("`df` must contain a 'date' column.")

    resample_df = df if _cfg.resample_df is None else _cfg.resample_df
    base_strata_source = df_dated if _cfg.resample_df is None else resample_df
    variables_resample = (
        _cfg.variables_resample
        if _cfg.variables_resample is not None
        else [c for c in _cfg.covariates if not _is_time_feature(c)]
    )

    if _cfg.conditional_on:
        before = len(resample_df)
        cond_mask = np.asarray(_conditional_mask(resample_df, _cfg.conditional_on), dtype=bool)
        resample_df = resample_df.loc[cond_mask]
        if _cfg.resample_df is None:
            # `check_data` keeps df's rows and order, so the same positional
            # mask selects the aligned rows of the strata source. A label-based
            # `.loc[resample_df.index]` expanded every repeated label when the
            # index was not unique (e.g. a concat of sites), making the strata
            # array longer than the pool values and raising IndexError.
            base_strata_source = base_strata_source.loc[cond_mask]
        else:
            base_strata_source = resample_df
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

    if _cfg.block_hours is not None and int(_cfg.block_hours) > 1:
        for _stream, _cols, pool in draws:
            if len(pool) < int(_cfg.block_hours):
                log.warning(
                    "block_hours=%d exceeds resample pool length %d; blocks wrap cyclically.",
                    int(_cfg.block_hours),
                    len(pool),
                )
                break

    pool_strata_map: dict[int, np.ndarray] = {}
    if strat_cols:
        base_s = _extract_strata(base_strata_source, strat_cols, label="resample_df")
        pool_strata_map[0] = base_s
        for name, pool in (_cfg.resample_pools or {}).items():
            if isinstance(pool, pd.DataFrame):
                stream_id = zlib.crc32(str(name).encode("utf-8")) or 1
                pool_strata_map[stream_id] = _extract_strata(
                    pool, strat_cols, label=f"resample_pools[{name!r}]"
                )

    # Each pool's served columns as one array, built once for every draw.
    draw_arrays = [
        (stream, cols, pool[cols].to_numpy(), pool_strata_map.get(stream))
        for stream, cols, pool in draws
    ]

    knn_map: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
    if _cfg.conditional_k is not None and variables_resample:
        cond_cols = [
            c
            for c in _cfg.covariates
            if c not in variables_resample
            and not _is_time_feature(c)
            and c in df.columns
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if cond_cols:
            for stream, _cols, pool in draws:
                p_cond = [
                    c
                    for c in cond_cols
                    if c in pool.columns and pd.api.types.is_numeric_dtype(pool[c])
                ]
                if p_cond:
                    knn_map[stream] = _build_knn_indices(
                        df,
                        pool,
                        p_cond,
                        int(_cfg.conditional_k),
                        df_strata=df_strata,
                        pool_strata=pool_strata_map.get(stream),
                    )

    n_cores_eff = resolve_workers(_cfg.n_cores)
    is_log_arith = bool(_cfg.log_transform and _cfg.back_transform in ("arithmetic", "smearing"))
    is_log_geom = bool(_cfg.log_transform and _cfg.back_transform == "geometric")
    phi = _resolve_smearing_factor(df_dated, df, model, _cfg, n_cores_eff)
    paired_mode = _resolve_paired_mode(_cfg.paired, log_transform=bool(_cfg.log_transform))

    z_pool_map: dict[int, np.ndarray | None] | None = None
    if _cfg.block_mode == "markov" and _cfg.block_hours is not None and int(_cfg.block_hours) > 1:
        z_pool_map = {
            stream: _standardize_pool_matrix(pool_vals)
            for stream, _cols, pool_vals, _p_strata in draw_arrays
        }

    cv_spec: tuple[list[str], np.ndarray, np.ndarray] | None = None
    if _cfg.control_variates and _cfg.aggregate and variables_resample:
        cv_spec = _prepare_control_variate(
            df,
            draw_arrays,
            model,
            n_cores_eff=n_cores_eff,
            is_log_arith=is_log_arith,
            phi=phi,
            df_strata=df_strata,
            knn_map=knn_map,
        )

    rng = np.random.default_rng(_cfg.seed)
    random_seeds = rng.choice(1_000_000, size=_cfg.n_samples, replace=False)

    if not variables_resample:
        preds = ml_predict(model, df, num_threads=n_cores_eff)
        obs_raw = df["value"].to_numpy(dtype=float)
        obs_vals = np.expm1(obs_raw) if _cfg.log_transform else obs_raw
        pred_vals = _back_transform_values(preds, smearing_factor=phi) if is_log_arith else preds
        df_one = pd.DataFrame(
            {
                "date": df["date"].to_numpy(),
                "observed": obs_vals,
                "normalised": pred_vals,
            }
        )
        gb_one = df_one.groupby("date", as_index=True)
        if _cfg.aggregate:
            df_out = gb_one[["observed", "normalised"]].mean()
            if is_log_geom:
                df_out["normalised"] = np.expm1(df_out["normalised"].to_numpy(dtype=float))
            if _cfg.return_mc_se:
                df_out["mc_se"] = 0.0 if _cfg.n_samples >= 2 else np.nan
            if _cfg.return_quantiles:
                q_arr = sorted({float(q) for q in _cfg.return_quantiles})
                # The resampling paths compute quantiles over their long
                # per-seed table, where every input row appears n_samples
                # times. With nothing resampled every seed predicts the same
                # value, so replicating each row n_samples times reproduces
                # that multiset exactly. Quantiling the raw rows instead
                # changed the result whenever timestamps repeat: two rows at
                # one timestamp predicting 6 and 7 gave 6.1 / 6.9 for the
                # 10 % / 90 % quantiles instead of the slow path's 6 / 7.
                # The quantile is taken on the model's scale, then inverted
                # once (a monotone transform commutes with the quantile).
                df_one_raw = pd.DataFrame({"date": df["date"].to_numpy(), "normalised": preds})
                q_source = df_one_raw.iloc[np.repeat(np.arange(len(df_one_raw)), _cfg.n_samples)]
                q_df = (
                    q_source.groupby("date", as_index=True)["normalised"]
                    .quantile(np.asarray(q_arr))
                    .unstack(level=-1)
                )
                if _cfg.log_transform:
                    q_df = pd.DataFrame(
                        np.expm1(q_df.to_numpy(dtype=float)), index=q_df.index, columns=q_df.columns
                    )
                q_df.columns = pd.Index([_format_quantile_name(float(q)) for q in q_df.columns])
                df_out = df_out.join(q_df, how="left")
            df_out = _attach_paired_columns(
                df_out,
                df,
                model,
                _cfg,
                phi=phi,
                paired_mode=paired_mode,
                is_log_arith=is_log_arith,
                is_log_geom=is_log_geom,
                n_cores_eff=n_cores_eff,
                fact_raw_precomputed=preds,
            )
        else:
            observed = gb_one[["observed"]].mean()
            vals = gb_one["normalised"].mean().to_numpy(dtype=float)
            if is_log_geom:
                vals = np.expm1(vals)
            sorted_seeds = sorted(int(s) for s in random_seeds)
            wide = pd.DataFrame(
                {s: vals for s in sorted_seeds},
                index=observed.index,
            )
            wide.columns = pd.Index(sorted_seeds, name="seed")
            df_out = pd.concat([observed, wide], axis=1)
        (log.info if _cfg.verbose else log.debug)(
            "Finished normalisation (no resampled variables; single pass)."
        )
        return df_out

    # Resolve the effective batch size. None -> auto-derive from the data
    # footprint (normet-R's heuristic: as many resampled copies of `df` as fit
    # in a ~400 MB payload budget); 0 -> batching disabled; >0 -> as given.
    if _cfg.batch_size is None:
        if _cfg.memory_save:
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

    use_batch_reduce = eff_batch_size > 0 and _cfg.aggregate

    (log.info if _cfg.verbose else log.debug)(
        "Normalising with %d resamples (aggregate=%s, memory_save=%s, "
        "batch_reduce=%s, n_cores=%d).",
        _cfg.n_samples,
        _cfg.aggregate,
        _cfg.memory_save,
        use_batch_reduce,
        n_cores_eff,
    )

    def _cv_adjustment(
        source: Mapping[str, np.ndarray] | pd.DataFrame,
        n_draws: int,
        n_rows_blk: int,
    ) -> np.ndarray | None:
        if cv_spec is None:
            return None
        cv_cols, beta, mu_x = cv_spec
        x_cols: list[np.ndarray] = []
        for j, col in enumerate(cv_cols):
            raw_c = source[col]
            arr_c = pd.to_numeric(raw_c, errors="coerce")
            vec = (
                arr_c.to_numpy(dtype=np.float64)
                if hasattr(arr_c, "to_numpy")
                else np.asarray(arr_c, dtype=np.float64)
            )
            mu_rep = np.tile(mu_x[:, j], n_draws)
            x_cols.append(np.where(np.isfinite(vec), vec, mu_rep))
        x_mat = np.column_stack(x_cols)
        mu_tiled = np.tile(mu_x, (n_draws, 1))
        adj = (x_mat - mu_tiled) @ beta
        return adj.reshape(n_draws, n_rows_blk)

    def process_one(sample_pos: int, seed_i: int) -> pd.DataFrame | None:
        try:
            df_resampled = generate_resampled(
                df,
                variables_resample,
                _cfg.replace,
                int(seed_i),
                resample_df,
                resample_pools=_cfg.resample_pools,
                df_strata=df_strata,
                pool_strata_map=pool_strata_map,
                knn_map=knn_map or None,
                sampling=_cfg.sampling,
                block_hours=_cfg.block_hours,
                block_mode=_cfg.block_mode,
                z_pool_map=z_pool_map,
                base_seed=_cfg.seed,
                sample_pos=int(sample_pos),
            )
            preds = ml_predict(model, df_resampled, num_threads=1)
            out_dict: dict[str, Any] = {
                "date": df_resampled["date"].to_numpy(),
                "observed": df_resampled["value"].to_numpy(),
                "normalised": preds,
                "seed": int(seed_i),
            }
            adj = _cv_adjustment(df_resampled, 1, len(df_resampled))
            if adj is not None:
                out_dict["_cv_adj"] = adj[0]
            return pd.DataFrame(out_dict)
        except Exception:
            log.exception("Error in seed %d", seed_i)
            return None

    # ── Batch-reduce path: O(batch_size × n_rows) peak memory ──────────────
    # Supports streaming Welford MC SE and compact scalar quantile buffer.
    if use_batch_reduce:
        import gc

        n_rows = len(df)
        obs_arr = df["value"].to_numpy(dtype=float)
        dates_arr = df["date"].to_numpy()
        dates_unique = bool(pd.Index(dates_arr).is_unique)

        sum_norm = np.zeros(n_rows, dtype=np.float64)
        sum_obs = np.zeros(n_rows, dtype=np.float64)
        n_completed = 0

        # Chan parallel Welford accumulators for MC SE when requested
        w_count = 0
        inv_date_idx: np.ndarray | None = None
        date_counts: np.ndarray | None = None
        if _cfg.return_mc_se:
            if dates_unique:
                w_mean = np.zeros(n_rows, dtype=np.float64)
                w_m2 = np.zeros(n_rows, dtype=np.float64)
            else:
                unique_dates, inv_date_idx = np.unique(dates_arr, return_inverse=True)
                date_counts = np.bincount(inv_date_idx).astype(np.float64)
                w_mean = np.zeros(len(unique_dates), dtype=np.float64)
                w_m2 = np.zeros(len(unique_dates), dtype=np.float64)

        # Compact scalar prediction buffer for quantiles (avoids full feature copy in RAM)
        q_buf: np.ndarray | None = (
            np.empty((_cfg.n_samples, n_rows), dtype=np.float64) if _cfg.return_quantiles else None
        )
        # Control-variate adjustments are buffered beside the raw draws so the
        # quantiles can be built from adjusted values; without control
        # variates the raw-draw path is kept bit-identical to earlier releases.
        q_adj_buf: np.ndarray | None = (
            np.empty((_cfg.n_samples, n_rows), dtype=np.float64)
            if (_cfg.return_quantiles and _cfg.control_variates and cv_spec is not None)
            else None
        )

        seeds_batched = [
            random_seeds[i : i + eff_batch_size] for i in range(0, _cfg.n_samples, eff_batch_size)
        ]
        (log.info if _cfg.verbose else log.debug)(
            "Batch-reduce: %d batches of ≤%d (total %d resamples).",
            len(seeds_batched),
            eff_batch_size,
            _cfg.n_samples,
        )

        sample_offset = 0
        for b_idx, batch_seeds in enumerate(seeds_batched):
            progress.report(b_idx, len(seeds_batched))
            b_size = len(batch_seeds)
            batch_positions = list(range(sample_offset, sample_offset + b_size))

            df_batch_dict = _stacked_columns(
                df,
                draw_arrays,
                batch_seeds,
                _cfg.replace,
                df_strata=df_strata,
                knn_map=knn_map or None,
                base_seed=_cfg.seed,
                sample_positions=batch_positions,
                sampling=_cfg.sampling,
                block_hours=_cfg.block_hours,
                block_mode=_cfg.block_mode,
                z_pool_map=z_pool_map,
            )
            df_batch = pd.DataFrame(df_batch_dict)
            batch_preds = ml_predict(model, df_batch, num_threads=n_cores_eff)
            batch_z_2d = batch_preds.reshape(b_size, n_rows)

            if q_buf is not None:
                q_buf[sample_offset : sample_offset + b_size] = batch_z_2d

            batch_y_2d = (
                _back_transform_values(batch_z_2d, smearing_factor=phi)
                if is_log_arith
                else batch_z_2d
            )
            cv_adj = _cv_adjustment(df_batch_dict, b_size, n_rows)
            if cv_adj is not None:
                batch_y_2d = batch_y_2d - cv_adj
                if q_adj_buf is not None:
                    q_adj_buf[sample_offset : sample_offset + b_size] = cv_adj
            sum_norm += batch_y_2d.sum(axis=0)
            sum_obs += np.broadcast_to(obs_arr, (b_size, n_rows)).sum(axis=0)

            if _cfg.return_mc_se:
                if dates_unique:
                    w_data = batch_y_2d
                else:
                    assert inv_date_idx is not None and date_counts is not None
                    w_data = np.zeros((b_size, len(date_counts)), dtype=np.float64)
                    np.add.at(w_data.T, inv_date_idx, batch_y_2d.T)
                    w_data /= date_counts[None, :]
                b_mean = w_data.mean(axis=0)
                b_m2 = ((w_data - b_mean) ** 2).sum(axis=0)
                if w_count == 0:
                    w_mean = b_mean
                    w_m2 = b_m2
                    w_count = b_size
                else:
                    new_count = w_count + b_size
                    delta = b_mean - w_mean
                    w_mean = w_mean + delta * (b_size / new_count)
                    w_m2 = w_m2 + b_m2 + (delta**2) * (w_count * b_size / new_count)
                    w_count = new_count

            n_completed += b_size
            sample_offset += b_size

            del df_batch_dict, df_batch, batch_preds, batch_z_2d, batch_y_2d
            gc.collect()

            (log.debug)(
                "Batch %d/%d done (%d/%d total resamples).",
                b_idx + 1,
                len(seeds_batched),
                n_completed,
                _cfg.n_samples,
            )

        obs_mean = sum_obs / n_completed
        if _cfg.log_transform:
            obs_mean = np.expm1(obs_mean)
        norm_raw_mean = sum_norm / n_completed

        if dates_unique:
            order = np.argsort(dates_arr, kind="stable")
            sorted_idx = pd.Index(dates_arr[order], name="date")
            mean_z = norm_raw_mean[order] if is_log_geom else None
            norm_out = np.expm1(mean_z) if mean_z is not None else norm_raw_mean[order]
            df_out = pd.DataFrame(
                {"observed": obs_mean[order], "normalised": norm_out},
                index=sorted_idx,
            )
            if _cfg.return_mc_se:
                if n_completed >= 2:
                    se_vals = np.sqrt(np.clip(w_m2 / (n_completed - 1), 0.0, None) / n_completed)
                    se_vals = se_vals[order]
                    if mean_z is not None:
                        se_vals = np.exp(mean_z) * se_vals
                else:
                    se_vals = np.full(n_rows, np.nan, dtype=float)
                df_out["mc_se"] = se_vals
            if q_buf is not None:
                q_arr = sorted({float(q) for q in (_cfg.return_quantiles or ())})
                q_vals = q_buf[:, order]
                if q_adj_buf is not None:
                    # Adjusted draws, in the same space as `normalised`:
                    # back-transform first for arithmetic/smearing, then
                    # subtract the control-variate shift.
                    if is_log_arith:
                        q_vals = (
                            _back_transform_values(q_vals, smearing_factor=phi)
                            - q_adj_buf[:, order]
                        )
                        q_mat = np.quantile(q_vals, q_arr, axis=0)
                    else:
                        q_mat = np.quantile(q_vals - q_adj_buf[:, order], q_arr, axis=0)
                        if _cfg.log_transform:
                            q_mat = np.expm1(q_mat)
                else:
                    q_mat = np.quantile(q_vals, q_arr, axis=0)
                    if _cfg.log_transform:
                        q_mat = np.expm1(q_mat)
                for qi, q_val in enumerate(q_arr):
                    df_out[_format_quantile_name(q_val)] = q_mat[qi]
        else:
            df_out = pd.DataFrame(
                {"observed": obs_mean, "normalised": norm_raw_mean},
                index=pd.Index(dates_arr, name="date"),
            )
            df_out = df_out.groupby(level="date", as_index=True).mean()
            df_out.index.name = "date"
            mean_z = df_out["normalised"].to_numpy(dtype=float).copy() if is_log_geom else None
            if mean_z is not None:
                df_out["normalised"] = np.expm1(mean_z)
            if _cfg.return_mc_se:
                if n_completed >= 2:
                    se_vals = np.sqrt(np.clip(w_m2 / (n_completed - 1), 0.0, None) / n_completed)
                    if mean_z is not None:
                        se_vals = np.exp(mean_z) * se_vals
                else:
                    se_vals = np.full(len(df_out), np.nan, dtype=float)
                df_out["mc_se"] = se_vals
            if q_buf is not None:
                q_arr = sorted({float(q) for q in (_cfg.return_quantiles or ())})
                q_vals = q_buf
                q_log_after = _cfg.log_transform
                if q_adj_buf is not None:
                    if is_log_arith:
                        q_vals = _back_transform_values(q_buf, smearing_factor=phi) - q_adj_buf
                        q_log_after = False
                    else:
                        q_vals = q_buf - q_adj_buf
                q_src = pd.DataFrame(
                    {"date": np.tile(dates_arr, n_completed), "_pred": q_vals.ravel()}
                )
                q_df = (
                    q_src.groupby("date", as_index=True)["_pred"]
                    .quantile(np.asarray(q_arr))
                    .unstack(level=-1)
                )
                if q_log_after:
                    q_df = pd.DataFrame(
                        np.expm1(q_df.to_numpy(dtype=float)),
                        index=q_df.index,
                        columns=q_df.columns,
                    )
                q_df.columns = pd.Index([_format_quantile_name(float(q)) for q in q_df.columns])
                df_out = df_out.join(q_df, how="left")

        if q_buf is not None:
            del q_buf
            if q_adj_buf is not None:
                del q_adj_buf
            gc.collect()

    # ── joblib threaded path: memory_save=True ─────────────────────────────
    elif _cfg.memory_save:
        from joblib import Parallel, delayed

        results_raw = Parallel(n_jobs=n_cores_eff, prefer="threads")(
            delayed(process_one)(pos, int(s)) for pos, s in enumerate(random_seeds)
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
        df_out = _aggregate_draws_frame(
            df_result, _cfg, phi=phi, is_log_arith=is_log_arith, is_log_geom=is_log_geom
        )

    # ── Vectorised path: O(n_samples × n_rows) ────────────────────────────
    else:
        n_rows = len(df)
        df_all = pd.DataFrame(
            _stacked_columns(
                df,
                draw_arrays,
                random_seeds,
                _cfg.replace,
                df_strata=df_strata,
                knn_map=knn_map or None,
                base_seed=_cfg.seed,
                sample_positions=list(range(_cfg.n_samples)),
                sampling=_cfg.sampling,
                block_hours=_cfg.block_hours,
                block_mode=_cfg.block_mode,
                z_pool_map=z_pool_map,
            )
        )
        df_all["seed"] = np.repeat(random_seeds, n_rows)
        preds = ml_predict(model, df_all, num_threads=n_cores_eff)

        res_dict: dict[str, Any] = {
            "date": df_all["date"].to_numpy(),
            "observed": df_all["value"].to_numpy(),
            "normalised": preds,
            "seed": df_all["seed"].to_numpy(),
        }
        cv_adj_vec = _cv_adjustment(df_all, _cfg.n_samples, n_rows)
        if cv_adj_vec is not None:
            res_dict["_cv_adj"] = cv_adj_vec.ravel()
        df_result = pd.DataFrame(res_dict)
        df_out = _aggregate_draws_frame(
            df_result, _cfg, phi=phi, is_log_arith=is_log_arith, is_log_geom=is_log_geom
        )

    df_out = _attach_paired_columns(
        df_out,
        df,
        model,
        _cfg,
        phi=phi,
        paired_mode=paired_mode,
        is_log_arith=is_log_arith,
        is_log_geom=is_log_geom,
        n_cores_eff=n_cores_eff,
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
    return_mc_se: bool = False,
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
        Input dataset with ``date`` column (or DatetimeIndex) and target ``value``.
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
    return_mc_se : bool, default False
        If True, append a per-date ``mc_se`` column to ``res``.
    **normalise_kwargs : Any
        Extra keyword arguments forwarded to :func:`normalise` (e.g.
        ``n_cores``, ``memory_save``, ``log_transform``, ``back_transform``).
        A ``seed`` passed here is ignored in favour of ``seed`` above.

    Returns
    -------
    dict
        A dict with keys ``best_n`` (int, total samples used), ``converged``
        (bool), ``res`` (DataFrame with ``date``, ``observed``, ``normalised``,
        and optional ``mc_se``) and ``history`` (DataFrame, only when
        ``return_history=True``).
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

    df_dated = process_date(df)
    if resample_df is None:
        resample_df = df

    call_kwargs = dict(normalise_kwargs)
    is_log = bool(call_kwargs.get("log_transform", False))
    bt_mode = str(call_kwargs.get("back_transform", "geometric"))
    if bt_mode not in ("arithmetic", "smearing", "geometric"):
        raise ConfigError(
            f"`back_transform` must be 'arithmetic', 'smearing', or 'geometric', got {bt_mode!r}."
        )
    is_log_geom_auto = is_log and bt_mode == "geometric"
    if is_log and bt_mode == "smearing" and call_kwargs.get("smearing_factor") is None:
        covs_for_phi = covariates or _resolve_normalise_config(**call_kwargs).covariates
        if covs_for_phi is None and model is not None:
            try:
                from ..utils.features import extract_features

                covs_for_phi = [str(c) for c in extract_features(model)]
            except Exception:
                covs_for_phi = None
        if covs_for_phi is not None and "value" in df_dated.columns:
            df_phi = check_data(df_dated, list(covs_for_phi), "value")
            tmp_cfg = _resolve_normalise_config(
                covariates=list(covs_for_phi),
                log_transform=True,
                back_transform="smearing",
                n_cores=call_kwargs.get("n_cores"),
            )
            call_kwargs["smearing_factor"] = _resolve_smearing_factor(
                df_dated, df_phi, model, tmp_cfg, resolve_workers(tmp_cfg.n_cores)
            )
    if is_log_geom_auto:
        # Accumulate batch means on log1p scale so final expm1(mean) is invariant
        # to batch_size and agrees with single-call back_transform="geometric".
        call_kwargs["log_transform"] = False

    # --- 1. Initialise Welford accumulators ---
    if "value" in df_dated.columns:
        obs_series = df_dated.groupby("date")["value"].mean()
        if is_log:
            obs_series = pd.Series(
                np.expm1(obs_series.to_numpy(dtype=float)), index=obs_series.index
            )
        observed_by_date: dict[object, float] = {d: float(o) for d, o in obs_series.items()}
    else:
        observed_by_date = {}

    dates_idx: pd.Index | None = None
    mean_bm: np.ndarray | None = None
    m2_bm: np.ndarray | None = None

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
            df_dated,
            model,
            covariates=covariates,
            variables_resample=variables_resample,
            resample_df=resample_df,
            n_samples=batch_size,
            aggregate=True,
            seed=seed + total_n,
            verbose=False,
            **call_kwargs,
        )

        bm_series = batch_result["normalised"]
        if dates_idx is None:
            dates_idx = bm_series.index
            bm = bm_series.reindex(dates_idx).to_numpy(dtype=float)
            mean_bm = bm.copy()
            m2_bm = np.zeros_like(bm)
        else:
            assert mean_bm is not None and m2_bm is not None
            bm = bm_series.reindex(dates_idx).to_numpy(dtype=float)

        total_n += batch_size
        n_batches += 1
        if n_batches >= 2:
            assert mean_bm is not None and m2_bm is not None
            delta = bm - mean_bm
            mean_bm = mean_bm + delta / n_batches
            m2_bm = m2_bm + delta * (bm - mean_bm)

        # --- 3. Convergence check ---
        if n_batches >= 2:
            assert mean_bm is not None and m2_bm is not None
            cum_mean = np.expm1(mean_bm) if is_log_geom_auto else mean_bm
            current_global_mean = float(np.nanmean(cum_mean))

            if metric == "series":
                var_bm = np.clip(m2_bm, 0.0, None) / (n_batches - 1)
                se_raw = np.sqrt(var_bm / n_batches)
                se = np.exp(mean_bm) * se_raw if is_log_geom_auto else se_raw
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
            assert mean_bm is not None
            first_est = np.expm1(mean_bm) if is_log_geom_auto else mean_bm
            prev_global_mean = float(np.nanmean(first_est))

    if not converged:
        # Deliberately not gated on `verbose`: silently returning an
        # unconverged result would defeat the point of an adaptive stop.
        import warnings

        warnings.warn(
            f"normalise_auto reached max_samples={max_samples} without strict convergence.",
            stacklevel=2,
        )

    # --- 4. Assemble output ---
    if n_batches and mean_bm is not None:
        cum_mean_out = np.expm1(mean_bm) if is_log_geom_auto else mean_bm
        if n_batches >= 2 and m2_bm is not None:
            se_raw_out = np.sqrt(np.clip(m2_bm, 0.0, None) / (n_batches - 1) / n_batches)
            mc_se_out = np.exp(mean_bm) * se_raw_out if is_log_geom_auto else se_raw_out
        else:
            mc_se_out = np.full(len(cum_mean_out), np.nan, dtype=float)
    else:
        cum_mean_out = np.array([])
        mc_se_out = np.array([])

    res_dict: dict[str, Any] = {
        "date": list(dates_idx) if dates_idx is not None else [],
        "observed": [observed_by_date.get(d, float("nan")) for d in dates_idx]
        if dates_idx is not None
        else [],
        "normalised": cum_mean_out,
    }
    if return_mc_se:
        res_dict["mc_se"] = mc_se_out
    res = pd.DataFrame(res_dict).sort_values("date").reset_index(drop=True)

    out: dict[str, Any] = {"best_n": total_n, "converged": converged, "res": res}
    if return_history:
        out["history"] = pd.DataFrame(history)
    return out


@dataclass
class CrossFitNormaliseResult:
    """Result container returned by :func:`crossfit_normalise`."""

    normalised: pd.DataFrame
    oof_metrics: pd.DataFrame
    fold_models: list[object]
    n_splits: int
    embargo: int
    paired_mode: str
    log_transform: bool
    back_transform: str
    smearing_factors: list[float] = field(default_factory=list)


def crossfit_normalise(
    df: pd.DataFrame,
    target: str = "value",
    *,
    covariates: list[str],
    variables_resample: list[str] | None = None,
    backend: str = "lightgbm",
    model_config: dict[str, Any] | None = None,
    n_splits: int = 5,
    embargo: int = _CROSSFIT_DEFAULT_EMBARGO,
    paired: bool | str = "log_ratio",
    paired_floor: float = 1e-6,
    n_samples: int = 300,
    log_transform: bool = False,
    back_transform: str = "geometric",
    stratify_by: str | Sequence[str] | None = None,
    conditional_k: int | None = None,
    conditional_on: Mapping[str, Any] | None = None,
    resample_pools: Mapping[str, pd.DataFrame] | None = None,
    resample_df: pd.DataFrame | None = None,
    return_quantiles: Sequence[float] | None = None,
    return_mc_se: bool = False,
    sampling: str = "iid",
    block_hours: int | None = None,
    seed: int = DEFAULT_SEED,
    n_cores: int | None = None,
    memory_save: bool = False,
    cache: str | Path | None = None,
    impute: str = "train_median",
    max_gap: int | None = 6,
    circular_cols: Sequence[str] | None = None,
    verbose: bool = False,
) -> CrossFitNormaliseResult:
    """Temporally blocked, embargoed out-of-fold meteorological normalisation.

    Mitigates in-sample absorption of short-to-medium emission events by
    predicting each temporal block solely from a model fitted outside that
    block (excluding an ``embargo`` buffer on both sides) and pairing observed
    concentrations against the out-of-fold factual/counterfactual ratio.

    Parameters
    ----------
    df : pandas.DataFrame
        Input time series carrying ``date`` (or DatetimeIndex), ``target``, and
        predictors.
    target : str, default "value"
        Target concentration column in ``df``.
    covariates : list of str
        Model predictors (may include helper calendar features
        ``date_unix``/``day_julian``/``weekday``/``hour``).
    variables_resample : list of str, optional
        Variables to marginalise over; defaults to non-time predictors in
        ``covariates``.
    backend : str, default "lightgbm"
        Registered training backend.
    model_config : dict, optional
        Forwarded to :func:`normet.model.train.train_model`.
    n_splits : int, default 5
        Number of contiguous temporal blocks.
    embargo : int, default 24
        Rows excluded on both sides of each validation block when fitting its
        out-of-fold model.
    paired : bool or {"additive", "ratio", "log_ratio"}, default "log_ratio"
        Paired anomaly-preserving estimator alongside ``normalised`` level.
    paired_floor : float, default 1e-6
        Near-zero denominator floor when ``paired="ratio"``.
    n_samples, log_transform, back_transform, stratify_by, conditional_k, conditional_on, resample_pools, resample_df, return_quantiles, return_mc_se, sampling, block_hours, seed, n_cores, memory_save, cache :
        Forwarded to :func:`normalise` on each fold's validation block.
    impute, max_gap, circular_cols :
        Per-fold leak-free missing-value handling fit only on that fold's training rows.
    verbose : bool, default False
        Emit progress logs.

    Returns
    -------
    CrossFitNormaliseResult
    """
    from ..model.train import train_model
    from ..utils.cv import blocked_time_cv

    if not covariates:
        raise ValueError("`covariates` must be a non-empty list.")
    cov_list = list(covariates)
    paired_mode_resolved = _resolve_paired_mode(paired, log_transform=log_transform) or "none"

    # Build clean base frame with target NA rows removed and time variables added,
    # keeping missing predictors intact so each fold imputes from its own training set.
    df_base = prepare_data(
        df,
        target=target,
        covariates=cov_list,
        dropna=True,
        split_method="ts",
        train_fraction=1.0,
        seed=seed,
        impute="none",
    ).drop(columns=["set"])
    missing_cov = [c for c in cov_list if c not in df_base.columns]
    if missing_cov:
        raise ConfigError(
            f"Covariate column(s) {missing_cov} not found after date-feature generation."
        )

    ext_pool: pd.DataFrame | None = None
    if resample_df is not None:
        ext_pool = process_date(resample_df.copy())
        need_time = [v for v in _TIME_VARS_EXACT if v in cov_list and v not in ext_pool.columns]
        if need_time:
            ext_pool = add_date_variables(ext_pool)

    n_cores_eff = resolve_workers(n_cores)
    fold_frames: list[pd.DataFrame] = []
    fold_stats: list[pd.DataFrame] = []
    fold_models: list[object] = []
    smearing_factors: list[float] = []
    oof_obs_all: list[np.ndarray] = []
    oof_fact_all: list[np.ndarray] = []

    folds = list(blocked_time_cv(df_base, n_splits=n_splits, embargo=embargo, date_col="date"))
    for k, (tr_idx, te_idx) in enumerate(folds):
        df_fold = df_base.copy()
        set_labels = np.full(len(df_fold), "embargo", dtype=object)
        set_labels[tr_idx] = "training"
        set_labels[te_idx] = "testing"
        df_fold["set"] = set_labels
        # Every fold must impute from its own training rows alone. The plain
        # "median" strategy is a full-frame fill, which would mix the held-out
        # block (and the embargo) into the fill values, so it means train-only
        # statistics here.
        fold_impute = "train_median" if impute == "median" else impute
        df_fold = impute_values(
            df_fold,
            dropna=False,
            strategy=fold_impute,
            max_gap=max_gap,
            circular_cols=circular_cols,
        )

        df_tr = df_fold.iloc[tr_idx].copy()
        df_te = df_fold.iloc[te_idx].copy()
        pool_k = df_fold.drop(columns=["set"]) if ext_pool is None else ext_pool

        if log_transform:
            df_tr = log1p_target(df_tr, "value")
            df_te = log1p_target(df_te, "value")
            if ext_pool is None and "value" in pool_k.columns:
                pool_k = log1p_target(pool_k, "value")

        model_k = train_model(
            df=df_tr,
            target="value",
            backend=backend,
            covariates=cov_list,
            model_config=model_config,
            seed=int(seed) + k,
            verbose=bool(verbose and k == 0),
            n_cores=n_cores,
            cache=cache,
        )
        fold_models.append(model_k)

        phi_k = 1.0
        if log_transform and back_transform == "smearing":
            tr_preds = ml_predict(model_k, df_tr, num_threads=n_cores_eff)
            phi_k = compute_duan_smearing(df_tr["value"], tr_preds)
        smearing_factors.append(float(phi_k))

        norm_k = normalise(
            df_te,
            model_k,
            covariates=cov_list,
            variables_resample=variables_resample,
            n_samples=n_samples,
            aggregate=True,
            seed=int(seed),
            n_cores=n_cores,
            resample_df=pool_k,
            resample_pools=resample_pools,
            conditional_on=conditional_on,
            stratify_by=stratify_by,
            conditional_k=conditional_k,
            return_quantiles=return_quantiles,
            return_mc_se=return_mc_se,
            log_transform=log_transform,
            back_transform=back_transform,
            smearing_factor=phi_k if (log_transform and back_transform == "smearing") else None,
            paired=paired,
            paired_floor=paired_floor,
            sampling=sampling,
            block_hours=block_hours,
            memory_save=memory_save,
            cache=cache,
            verbose=False,
        )
        norm_k["fold"] = int(k)
        fold_frames.append(norm_k)

        if "factual" in norm_k.columns:
            y_fact = norm_k["factual"].to_numpy(dtype=float)
            y_obs = norm_k["observed"].to_numpy(dtype=float)
        else:
            raw_pred = ml_predict(model_k, df_te, num_threads=n_cores_eff)
            if log_transform:
                y_fact = _back_transform_values(
                    raw_pred, smearing_factor=phi_k if back_transform == "smearing" else 1.0
                )
                y_obs = np.expm1(df_te["value"].to_numpy(dtype=float))
            else:
                y_fact = raw_pred
                y_obs = df_te["value"].to_numpy(dtype=float)
        oof_obs_all.append(y_obs)
        oof_fact_all.append(y_fact)

        row = _stats_from_arrays(y_fact, y_obs, _DEFAULT_STATS)
        row["fold"] = str(k)
        row["n_train"] = int(len(df_tr))
        row["n_test"] = int(len(df_te))
        row["test_start"] = df_te["date"].iloc[0]
        row["test_end"] = df_te["date"].iloc[-1]
        fold_stats.append(row)

    out_df = pd.concat(fold_frames, axis=0).sort_index()
    out_df.index.name = "date"

    obs_cat = np.concatenate(oof_obs_all) if oof_obs_all else np.array([])
    fact_cat = np.concatenate(oof_fact_all) if oof_fact_all else np.array([])
    all_row = _stats_from_arrays(fact_cat, obs_cat, _DEFAULT_STATS)
    all_row["fold"] = "all"
    all_row["n_train"] = int(sum(int(r["n_train"].iloc[0]) for r in fold_stats))
    all_row["n_test"] = int(len(obs_cat))
    all_row["test_start"] = out_df.index.min()
    all_row["test_end"] = out_df.index.max()
    oof_metrics = pd.concat([*fold_stats, all_row], ignore_index=True)
    meta = ["fold", "n_train", "n_test", "test_start", "test_end"]
    oof_metrics = oof_metrics[meta + [c for c in oof_metrics.columns if c not in meta]]

    return CrossFitNormaliseResult(
        normalised=out_df,
        oof_metrics=oof_metrics,
        fold_models=fold_models,
        n_splits=len(folds),
        embargo=int(embargo),
        paired_mode=paired_mode_resolved,
        log_transform=bool(log_transform),
        back_transform=str(back_transform),
        smearing_factors=smearing_factors,
    )


@dataclass
class DMLNormaliseResult:
    """Result container returned by :func:`dml_normalise`."""

    normalised: pd.DataFrame
    oof_metrics: pd.DataFrame
    fold_models: list[object]
    cutoff_days: float
    n_harmonics: int
    n_splits: int
    embargo: int
    paired_mode: str
    log_transform: bool
    back_transform: str
    smearing_factors: list[float] = field(default_factory=list)


def _build_dml_basis(
    dates: pd.DatetimeIndex,
    *,
    t0: pd.Timestamp,
    span_seconds: float,
    cutoff_days: float,
    n_harmonics: int | None,
) -> tuple[np.ndarray, int]:
    """Construct smooth DCT-II low-pass cosine nuisance basis for :func:`dml_normalise`."""
    span_s = max(float(span_seconds), 3600.0)
    span_days = max(span_s / 86400.0, 1.0)
    k_harm = (
        max(1, int(n_harmonics))
        if n_harmonics is not None
        else max(1, min(64, int(round(2.0 * span_days / max(float(cutoff_days), 1.0)))))
    )
    u_t = (dates - t0).total_seconds().to_numpy(dtype=np.float64) / span_s
    cols: list[np.ndarray] = [np.cos(np.pi * float(k) * u_t) for k in range(1, k_harm + 1)]
    return np.column_stack(cols), k_harm


def dml_normalise(
    df: pd.DataFrame,
    target: str = "value",
    *,
    covariates: list[str],
    variables_resample: list[str] | None = None,
    backend: str = "lightgbm",
    model_config: dict[str, Any] | None = None,
    cutoff_days: float = 30.0,
    n_harmonics: int | None = None,
    ridge_alpha: float = 10.0,
    n_splits: int = 5,
    embargo: int = _CROSSFIT_DEFAULT_EMBARGO,
    paired: bool | str = "log_ratio",
    paired_floor: float = 1e-6,
    n_samples: int = 300,
    log_transform: bool = False,
    back_transform: str = "geometric",
    stratify_by: str | Sequence[str] | None = None,
    conditional_k: int | None = None,
    conditional_on: Mapping[str, Any] | None = None,
    resample_pools: Mapping[str, pd.DataFrame] | None = None,
    resample_df: pd.DataFrame | None = None,
    return_quantiles: Sequence[float] | None = None,
    return_mc_se: bool = False,
    sampling: str = "iid",
    control_variates: bool = False,
    block_hours: int | None = None,
    block_mode: str = "independent",
    seed: int = DEFAULT_SEED,
    n_cores: int | None = None,
    memory_save: bool = False,
    cache: str | Path | None = None,
    impute: str = "train_median",
    max_gap: int | None = 6,
    circular_cols: Sequence[str] | None = None,
    verbose: bool = False,
) -> DMLNormaliseResult:
    """Orthogonal partially linear Double-ML meteorological normalisation.

    Separates smooth low-frequency emission trajectories from high-dimensional
    nonlinear meteorological responses via Neyman-orthogonal residualisation
    (Robinson 1988; Chernozhukov et al. 2018), explicitly removing the slow
    time covariates (``date_unix``, ``year``, ``day_julian``) from the tree's
    split space so trees neither absorb short emission events in-sample nor
    produce boundary-clamping step artefacts out-of-fold.

    Parameters
    ----------
    df : pandas.DataFrame
        Input time series carrying ``date`` (or DatetimeIndex), ``target``, and
        predictors.
    target : str, default "value"
        Target concentration column in ``df``.
    covariates : list of str
        Predictor columns. ``date_unix``, ``year`` and ``day_julian`` are used
        solely to build the smooth temporal baseline and are excluded from the
        tree's split features.
    variables_resample : list of str, optional
        Variables to marginalise over; defaults to non-time predictors in
        ``covariates``.
    backend : str, default "lightgbm"
        Registered training backend.
    model_config : dict, optional
        Forwarded to :func:`normet.model.train.train_model`.
    cutoff_days : float, default 30.0
        Smoothness bandwidth (days) setting the highest slow Fourier frequency
        in the low-pass nuisance baseline when ``n_harmonics`` is ``None``.
    n_harmonics : int, optional
        Explicit number of slow Fourier harmonics in the nuisance baseline.
    ridge_alpha : float, default 10.0
        L2 ridge penalty applied to non-intercept nuisance basis coefficients.
    n_splits : int, default 5
        Number of contiguous temporal blocks.
    embargo : int, default 24
        Rows excluded on both sides of each validation block when fitting its
        out-of-fold model.
    paired : bool or {"additive", "ratio", "log_ratio"}, default "log_ratio"
        Paired anomaly-preserving estimator alongside ``normalised`` level.
    paired_floor : float, default 1e-6
        Near-zero denominator floor when ``paired="ratio"``.
    n_samples, log_transform, back_transform, stratify_by, conditional_k, conditional_on, resample_pools, resample_df, return_quantiles, return_mc_se, sampling, control_variates, block_hours, block_mode, seed, n_cores, memory_save, cache :
        Forwarded to :func:`normalise` on each fold's validation block.
    impute, max_gap, circular_cols :
        Per-fold leak-free missing-value handling fit only on that fold's training rows.
    verbose : bool, default False
        Emit progress logs.

    Returns
    -------
    DMLNormaliseResult
    """
    from ..model.train import train_model
    from ..utils.cv import blocked_time_cv

    if not covariates:
        raise ConfigError("`covariates` must be a non-empty list.")
    if float(cutoff_days) <= 0.0:
        raise ConfigError(f"`cutoff_days` must be positive, got {cutoff_days!r}.")
    if float(ridge_alpha) < 0.0:
        raise ConfigError(f"`ridge_alpha` must be non-negative, got {ridge_alpha!r}.")

    cov_list = list(covariates)
    tree_covs = [c for c in cov_list if c not in _DML_SLOW_TIME_COVARIATES]
    if not tree_covs:
        raise ConfigError(
            "dml_normalise requires at least one meteorological or cyclical covariate "
            "besides the slow time covariates ('date_unix', 'year', 'day_julian')."
        )
    paired_mode_resolved = _resolve_paired_mode(paired, log_transform=log_transform) or "none"

    df_base = prepare_data(
        df,
        target=target,
        covariates=cov_list,
        dropna=True,
        split_method="ts",
        train_fraction=1.0,
        seed=seed,
        impute="none",
    ).drop(columns=["set"])

    missing_cov = [c for c in tree_covs if c not in df_base.columns]
    if missing_cov:
        raise ConfigError(
            f"Covariate column(s) {missing_cov} not found after date-feature generation."
        )

    res_vars = (
        [c for c in variables_resample if c in tree_covs]
        if variables_resample is not None
        else [c for c in tree_covs if not _is_time_feature(c)]
    )

    ext_pool: pd.DataFrame | None = None
    if resample_df is not None:
        ext_pool = process_date(resample_df.copy())
        need_time = [v for v in _TIME_VARS_EXACT if v in tree_covs and v not in ext_pool.columns]
        if need_time:
            ext_pool = add_date_variables(ext_pool)

    all_dates = pd.DatetimeIndex(df_base["date"])
    t0 = all_dates.min()
    span_seconds = float((all_dates.max() - t0).total_seconds())
    phi_all, k_harm_used = _build_dml_basis(
        all_dates,
        t0=t0,
        span_seconds=span_seconds,
        cutoff_days=float(cutoff_days),
        n_harmonics=n_harmonics,
    )

    pen_diag = np.array(
        [(float(k) / float(k_harm_used)) ** 2 for k in range(1, k_harm_used + 1)],
        dtype=np.float64,
    )
    n_cores_eff = resolve_workers(n_cores)
    fold_models: list[object] = []
    smearing_factors: list[float] = []
    fold_payloads: list[dict[str, Any]] = []

    folds = list(blocked_time_cv(df_base, n_splits=n_splits, embargo=embargo, date_col="date"))
    for k, (tr_idx, te_idx) in enumerate(folds):
        df_fold = df_base.copy()
        set_labels = np.full(len(df_fold), "embargo", dtype=object)
        set_labels[tr_idx] = "training"
        set_labels[te_idx] = "testing"
        df_fold["set"] = set_labels
        fold_impute = "train_median" if impute == "median" else impute
        df_fold = impute_values(
            df_fold,
            dropna=False,
            strategy=fold_impute,
            max_gap=max_gap,
            circular_cols=circular_cols,
        )

        df_tr = df_fold.iloc[tr_idx].copy()
        df_te = df_fold.iloc[te_idx].copy()
        pool_k = df_fold.drop(columns=["set"]) if ext_pool is None else ext_pool

        if log_transform:
            df_tr = log1p_target(df_tr, "value")
            df_te = log1p_target(df_te, "value")
            if ext_pool is None and "value" in pool_k.columns:
                pool_k = log1p_target(pool_k, "value")

        y_tr = df_tr["value"].to_numpy(dtype=np.float64)
        y_te = df_te["value"].to_numpy(dtype=np.float64)

        phi_tr_raw = phi_all[tr_idx]
        phi_tr = phi_tr_raw - phi_tr_raw.mean(axis=0, keepdims=True)
        reg_tr = np.diag(max(float(ridge_alpha), 1e-6) * float(len(tr_idx)) * pen_diag)
        theta_tr = np.linalg.solve(
            phi_tr.T @ phi_tr + reg_tr,
            phi_tr.T @ (y_tr - float(np.mean(y_tr))),
        )
        m_tr_zero = phi_tr @ theta_tr

        df_tr_res = df_tr.copy()
        df_tr_res["value"] = y_tr - m_tr_zero
        model_k = train_model(
            df=df_tr_res,
            target="value",
            backend=backend,
            covariates=tree_covs,
            model_config=model_config,
            seed=int(seed) + k,
            verbose=bool(verbose and k == 0),
            n_cores=n_cores,
            cache=cache,
        )
        fold_models.append(model_k)

        phi_smear = 1.0
        if log_transform and back_transform == "smearing":
            tr_res_pred = ml_predict(model_k, df_tr_res, num_threads=n_cores_eff)
            phi_smear = compute_duan_smearing(y_tr, m_tr_zero + tr_res_pred)
        smearing_factors.append(float(phi_smear))

        g_fact = ml_predict(model_k, df_te, num_threads=n_cores_eff)
        df_te_res = df_te.copy()
        df_te_res["value"] = 0.0
        g_norm_df = normalise(
            df_te_res,
            model_k,
            covariates=tree_covs,
            variables_resample=res_vars,
            n_samples=n_samples,
            aggregate=True,
            seed=int(seed),
            n_cores=n_cores,
            resample_df=pool_k,
            resample_pools=resample_pools,
            conditional_on=conditional_on,
            stratify_by=stratify_by,
            conditional_k=conditional_k,
            return_quantiles=return_quantiles,
            return_mc_se=return_mc_se,
            # The draw-level target is 0, so log1p leaves it unchanged; setting
            # log_transform here only asks `normalise` to aggregate the draws on
            # the original (expm1) scale before averaging. Without it the fold
            # returned the mean in log space and `back_transform` was a no-op,
            # i.e. "arithmetic" silently produced the geometric level.
            log_transform=bool(log_transform),
            # Aggregate the raw draws on the expm1 scale. Smearing's Duan
            # factor is applied once, when the fold is combined with the slow
            # trend below; letting the fold apply it as well squared phi.
            back_transform="arithmetic" if back_transform == "smearing" else str(back_transform),
            paired=False,
            sampling=sampling,
            control_variates=control_variates,
            block_hours=block_hours,
            block_mode=block_mode,
            memory_save=memory_save,
            cache=cache,
            verbose=False,
        )

        te_dates = pd.Index(df_te["date"].to_numpy(), name="date")
        g_fact_aligned = (
            pd.Series(g_fact, index=te_dates)
            .groupby(level=0)
            .mean()
            .reindex(g_norm_df.index)
            .to_numpy(dtype=np.float64)
        )
        obs_aligned = (
            pd.Series(y_te, index=te_dates)
            .groupby(level=0)
            .mean()
            .reindex(g_norm_df.index)
            .to_numpy(dtype=np.float64)
        )
        phi_te_df = (
            pd.DataFrame(phi_all[te_idx], index=te_dates)
            .groupby(level=0)
            .mean()
            .reindex(g_norm_df.index)
        )
        fold_payloads.append(
            {
                "fold": int(k),
                "index": g_norm_df.index,
                "g_norm_df": g_norm_df,
                "g_norm": g_norm_df["normalised"].to_numpy(dtype=np.float64),
                "g_fact": g_fact_aligned,
                "obs": obs_aligned,
                "phi_te": phi_te_df.to_numpy(dtype=np.float64),
                "phi_smear": float(phi_smear),
                "n_train": int(len(df_tr)),
                "n_test": int(len(df_te)),
                "test_start": df_te["date"].iloc[0],
                "test_end": df_te["date"].iloc[-1],
            }
        )

    # Stage 2: Fit globally smooth slow low-pass trend m(t) across concatenated
    # cross-fitted weather-stripped residuals so boundary steps vanish.
    obs_cat_raw = np.concatenate([p["obs"] for p in fold_payloads])
    g_fact_cat = np.concatenate([p["g_fact"] for p in fold_payloads])
    phi_work_all = np.vstack([p["phi_te"] for p in fold_payloads])
    res_all = obs_cat_raw - g_fact_cat
    phi_full = np.column_stack([np.ones(len(res_all), dtype=np.float64), phi_work_all])
    reg_full = np.diag([0.0, *(float(ridge_alpha) * (max(len(res_all), 1) / 100.0) * pen_diag)])
    theta_full = np.linalg.solve(phi_full.T @ phi_full + reg_full, phi_full.T @ res_all)

    fold_frames: list[pd.DataFrame] = []
    fold_stats: list[pd.DataFrame] = []
    oof_obs_all: list[np.ndarray] = []
    oof_fact_all: list[np.ndarray] = []

    for p in fold_payloads:
        m_k_len = len(p["index"])
        phi_te_full = np.column_stack([np.ones(m_k_len, dtype=np.float64), p["phi_te"]])
        m_te_aligned = phi_te_full @ theta_full

        obs_aligned = p["obs"]
        g_fact_aligned = p["g_fact"]
        g_norm = p["g_norm"]
        g_norm_df = p["g_norm_df"]
        smear_use = float(p["phi_smear"]) if back_transform == "smearing" else 1.0

        if log_transform:
            obs_orig = np.expm1(obs_aligned)
            fact_orig = _back_transform_values(
                m_te_aligned + g_fact_aligned, smearing_factor=smear_use
            )
            # `g_norm` is the fold's draw aggregation on the expm1 scale (its
            # arithmetic or geometric mean, depending on `back_transform`).
            # log1p maps that residual level back to the additive log scale
            # where the slow trend m(t) lives; combining before the inverse
            # transform is what makes the two stages commute.
            norm_orig = _back_transform_values(
                m_te_aligned + np.log1p(g_norm), smearing_factor=smear_use
            )
        else:
            obs_orig = obs_aligned
            fact_orig = m_te_aligned + g_fact_aligned
            norm_orig = m_te_aligned + g_norm

        fold_out = pd.DataFrame(
            {"observed": obs_orig, "normalised": norm_orig},
            index=p["index"],
        )
        if return_mc_se and "mc_se" in g_norm_df.columns:
            mc_se_raw = g_norm_df["mc_se"].to_numpy(dtype=np.float64)
            if log_transform:
                # The fold already reports the residual-level MC SE on the
                # expm1 scale (log-space SE scaled by the level, or the SE of
                # the per-draw expm1 values). d(level)/d(residual) is exp(m),
                # so the total's SE is phi * exp(m) times it -- not
                # exp(m + g_norm), which applied that Jacobian twice.
                mc_se_raw = smear_use * np.exp(m_te_aligned) * mc_se_raw
            fold_out["mc_se"] = mc_se_raw
        if return_quantiles:
            for q_val in sorted({float(q) for q in return_quantiles}):
                q_col = _format_quantile_name(q_val)
                if q_col in g_norm_df.columns:
                    q_res = g_norm_df[q_col].to_numpy(dtype=np.float64)
                    if log_transform:
                        # Quantiles are equivariant under the monotone expm1:
                        # the total's q-th quantile is expm1(m + q_z), and
                        # q_z = log1p(q_res) because the fold's quantiles are
                        # already back-transformed per draw. The same Duan
                        # factor as the mean scales the smeared quantile.
                        q_tot = _back_transform_values(
                            m_te_aligned + np.log1p(q_res), smearing_factor=smear_use
                        )
                    else:
                        q_tot = m_te_aligned + q_res
                    fold_out[q_col] = q_tot

        if paired is not False and paired is not None:
            fold_out["factual"] = fact_orig
            fold_out["weather_effect"] = fact_orig - norm_orig
            fold_out["paired"] = _compute_paired_series(
                obs_orig,
                fact_orig,
                norm_orig,
                mode=paired_mode_resolved,
                floor=float(paired_floor),
            )
        fold_out["fold"] = int(p["fold"])
        fold_frames.append(fold_out)

        oof_obs_all.append(obs_orig)
        oof_fact_all.append(fact_orig)
        row = _stats_from_arrays(fact_orig, obs_orig, _DEFAULT_STATS)
        row["fold"] = str(p["fold"])
        row["n_train"] = p["n_train"]
        row["n_test"] = p["n_test"]
        row["test_start"] = p["test_start"]
        row["test_end"] = p["test_end"]
        fold_stats.append(row)

    out_df = pd.concat(fold_frames, axis=0).sort_index()
    out_df.index.name = "date"

    obs_cat = np.concatenate(oof_obs_all) if oof_obs_all else np.array([])
    fact_cat = np.concatenate(oof_fact_all) if oof_fact_all else np.array([])
    all_row = _stats_from_arrays(fact_cat, obs_cat, _DEFAULT_STATS)
    all_row["fold"] = "all"
    all_row["n_train"] = int(sum(int(r["n_train"].iloc[0]) for r in fold_stats))
    all_row["n_test"] = int(len(obs_cat))
    all_row["test_start"] = out_df.index.min()
    all_row["test_end"] = out_df.index.max()
    oof_metrics = pd.concat([*fold_stats, all_row], ignore_index=True)
    meta = ["fold", "n_train", "n_test", "test_start", "test_end"]
    oof_metrics = oof_metrics[meta + [c for c in oof_metrics.columns if c not in meta]]

    return DMLNormaliseResult(
        normalised=out_df,
        oof_metrics=oof_metrics,
        fold_models=fold_models,
        cutoff_days=float(cutoff_days),
        n_harmonics=k_harm_used,
        n_splits=len(folds),
        embargo=int(embargo),
        paired_mode=paired_mode_resolved,
        log_transform=bool(log_transform),
        back_transform=str(back_transform),
        smearing_factors=smearing_factors,
    )
