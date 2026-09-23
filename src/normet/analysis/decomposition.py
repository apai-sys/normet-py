# src/normet/analysis/decomposition.py
"""Split a normalised series into emission- and meteorology-driven components.

Provides :func:`decompose` (and the convenience wrappers :func:`decom_emi`,
:func:`decom_met`) plus :class:`DecomposeConfig`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import factorial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import ConfigError, DataError, ModelError
from ..foundation.estimator import CHRONOS_BACKEND, resolve_n_samples
from ..model.train import build_model
from ..utils._config import DEFAULT_SEED, resolve_config
from ..utils.features import extract_features
from ..utils.logging import get_logger
from ..utils.prepare import add_date_variables, process_date
from .normalise import normalise

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DecomposeConfig:
    """Consolidated configuration for :func:`decompose` / :func:`decom_emi` / :func:`decom_met`.

    Every field has a default so the dataclass can be constructed with only
    the overrides that differ from the standard values.
    """

    target: str = "value"
    covariates: list[str] | None = None
    backend: str | None = None
    split_method: str = "random"
    train_fraction: float = 0.75
    model_config: dict[str, Any] | None = None
    #: ``None`` resolves per backend: 300 for the AutoML backends, 8 for
    #: Chronos-2, where each sample is a full transformer forward pass.
    n_samples: int | None = None
    seed: int = DEFAULT_SEED
    n_cores: int | None = None
    memory_save: bool = False
    verbose: bool = False
    importance_ascending: bool = False
    method: str = "emission"
    variable_order: list[str] | None = None
    """Explicit meteorological-feature decomposition order for
    :func:`decom_met` (ignored by :func:`decom_emi`, which always uses its
    own hardcoded calendar order -- see its docstring). If None (default),
    order is derived from fitted feature importance via
    ``importance_ascending``, which can silently reorder "which component
    comes first" across refits of the same features/data with a different
    seed -- results aren't directly comparable run to run. Pass an
    explicit list (must be exactly the model's non-time-variable features,
    in any permutation) to get a decomposition order that stays fixed and
    comparable across runs regardless of the underlying model's importance
    ranking."""
    cache: str | Path | None = None
    """If given, memoize expensive sub-calls to this directory (a
    :class:`joblib.Memory` location): the internal :func:`build_model` fit
    (when ``model=None``) and every per-time-variable :func:`normalise`
    call in the decomposition loop -- ``decom_emi``/``decom_met`` call
    ``normalise`` once per fixed variable, each a full Monte Carlo
    resample-and-predict over ``n_samples`` draws. Off by default."""
    groups: Mapping[str, Sequence[str]] | None = None
    """:func:`decom_met` only. Attribute the meteorological features in named
    groups rather than one by one, e.g. ``{"local": met_cols, "transport":
    traj_cols}``; the result then has one contribution column per group. Every
    non-time model feature must be in exactly one group. A group is fixed and
    resampled as a unit, so its column is the effect of the group as a whole,
    interactions among its members included."""
    attribution: str | None = None
    """:func:`decom_met` only. How ``prediction - emi_total`` is split among the
    features or groups:

    - ``"sequential"``: fix them one at a time -- in ``variable_order``, else
      fitted-importance order, or in the order ``groups`` lists them -- and
      report each step's change. ``k + 1`` normalisations for ``k`` features or
      groups, but the split depends on the order.
    - ``"shapley"``: average each one's marginal effect over every order (its
      Shapley value), so no order is privileged; see ``n_permutations``.

    ``None`` (default) means ``"shapley"`` when ``groups`` is given, otherwise
    ``"sequential"`` -- the historical behaviour."""
    n_permutations: int | None = None
    """``attribution="shapley"`` only. ``None`` (default): exact Shapley values
    from all ``2**k`` coalitions of the ``k`` features or groups, allowed up to
    ``k = 10``. An integer: an estimate from that many random orders, drawn in
    antithetic pairs (an order and its reverse, so odd values round up); when
    that would evaluate as many coalitions as the exact values need, the exact
    values are computed instead. Either way the contributions add up exactly
    to ``prediction - emi_total``."""
    resample_df: pd.DataFrame | None = None
    """Pool the resampled variables are drawn from (default: ``df`` itself),
    forwarded to every :func:`normalise` call. Not on the chronos-2 backend."""
    resample_pools: Mapping[str, pd.DataFrame] | None = None
    """Extra pools some variables are drawn from instead, forwarded to every
    :func:`normalise` call -- see :attr:`NormaliseConfig.resample_pools`. Not
    on the chronos-2 backend."""
    conditional_on: Mapping[str, Any] | None = None
    """Filter on the ``resample_df`` pool, forwarded to every :func:`normalise`
    call. Not on the chronos-2 backend."""


_TIME_VARS = ("date_unix", "day_julian", "weekday", "hour")
# decom_met's own result columns, which a group may not be named after.
_MET_RESULT_COLUMNS = frozenset(
    {"date", "observed", "emi_total", "met_total", "met_base", "met_noise"}
)
# Exact Shapley values need 2**k normalisations for k features or groups.
_SHAPLEY_EXACT_MAX = 10


def _resolve_config(config: DecomposeConfig | None = None, **kwargs: Any) -> DecomposeConfig:
    return resolve_config(DecomposeConfig, config, **kwargs)


def _effective_cores(n_cores: int | None) -> int:
    """Resolve parallel worker count (>=1)."""
    return max(1, n_cores if n_cores is not None else (os.cpu_count() or 2) - 1)


def _log_decomposition_progress(
    verbose: bool, start: float, i: int, total: int, var_to_fix: str
) -> None:
    """Log a "Decomposing <var>" progress line with an ETA, shared by decom_emi/decom_met."""
    elapsed = time.time() - start
    eta = (elapsed / max(i - 1, 1)) * (total - (i - 1)) if i > 1 else None
    eta_str = (
        ""
        if eta is None
        else (
            f" | ETA: {eta:.1f}s"
            if eta < 60
            else f" | ETA: {eta / 60:.1f}m"
            if eta < 3600
            else f" | ETA: {eta / 3600:.1f}h"
        )
    )
    (log.info if verbose else log.debug)(
        "%s: Decomposing %s%s",
        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        var_to_fix,
        eta_str,
    )


# A "player" is one unit decom_met attributes to: (result column name, features),
# fixed at observed values or resampled as a whole.
_Player = tuple[str, list[str]]


def _players_from_groups(groups: Mapping[str, Sequence[str]], features: list[str]) -> list[_Player]:
    """Validate ``groups`` against the model's meteorological features."""
    if not isinstance(groups, Mapping) or not groups:
        raise ConfigError("`groups` must be a non-empty mapping of group name -> features.")
    feature_set = set(features)
    owner: dict[str, str] = {}
    players: list[_Player] = []
    for name, members in groups.items():
        if not isinstance(name, str) or not name:
            raise ConfigError(f"group names must be non-empty strings, got {name!r}.")
        if name in _MET_RESULT_COLUMNS:
            raise ConfigError(
                f"group name {name!r} clashes with a result column; rename the group."
            )
        feats = [members] if isinstance(members, str) else [str(f) for f in members]
        if not feats:
            raise ConfigError(f"group {name!r} is empty.")
        for f in feats:
            if f in _TIME_VARS:
                raise ConfigError(
                    f"group {name!r} lists the time variable {f!r}; decom_met holds the time "
                    "variables at their observed values and does not attribute them."
                )
            if f not in feature_set:
                raise ConfigError(
                    f"group {name!r} lists {f!r}, which is not a meteorological (non-time) "
                    f"feature of the model. Features: {features}."
                )
            if f in owner:
                where = (
                    f"twice in group {name!r}"
                    if owner[f] == name
                    else (f"in two groups ({owner[f]!r} and {name!r})")
                )
                raise ConfigError(f"feature {f!r} is listed {where}.")
            owner[f] = name
        players.append((name, feats))
    unassigned = [f for f in features if f not in owner]
    if unassigned:
        raise ConfigError(
            "every meteorological (non-time) model feature must be in exactly one group; "
            f"not in any: {unassigned}."
        )
    return players


def _attribution_method(cfg: DecomposeConfig) -> str:
    """Resolve and validate the attribution options that do not depend on the
    model's features, so a bad call fails before a model is trained or loaded."""
    method = cfg.attribution or ("shapley" if cfg.groups is not None else "sequential")
    if method not in ("sequential", "shapley"):
        raise ConfigError(
            f"`attribution` must be 'sequential' or 'shapley', got {cfg.attribution!r}."
        )
    if cfg.n_permutations is not None:
        if method != "shapley":
            raise ConfigError("`n_permutations` only applies to attribution='shapley'.")
        if int(cfg.n_permutations) < 1:
            raise ConfigError(f"`n_permutations` must be at least 1, got {cfg.n_permutations}.")
    if cfg.groups is not None and cfg.variable_order is not None:
        raise ConfigError(
            "`variable_order` orders single features; with `groups` the groups are the "
            "units, taken in the order they are listed."
        )
    if cfg.groups is None and cfg.variable_order is not None and method == "shapley":
        raise ConfigError(
            "`variable_order` has no effect with attribution='shapley', which averages "
            "over every order."
        )
    return method


def _attribution_plan(
    features: list[str],
    cfg: DecomposeConfig,
    default_order: Callable[[list[str]], list[str]],
) -> tuple[list[_Player], str]:
    """Decide what :func:`decom_met` attributes to (features or groups) and how.

    Returns ``(players, method)``, players in result-column order.
    ``default_order`` is only consulted for a sequential run over single
    features without ``variable_order``: the one case that needs a ranking.
    """
    method = _attribution_method(cfg)
    if cfg.groups is not None:
        players = _players_from_groups(cfg.groups, features)
    elif cfg.variable_order is not None:
        requested = [str(f) for f in cfg.variable_order]
        if set(requested) != set(features):
            raise ConfigError(
                "`variable_order` must be exactly the model's meteorological (non-time) "
                f"features, in any order. Missing: {sorted(set(features) - set(requested))}. "
                f"Not in model: {sorted(set(requested) - set(features))}."
            )
        twice = sorted({f for f in requested if requested.count(f) > 1})
        if twice:
            raise ConfigError(f"`variable_order` lists {twice} more than once.")
        players = [(f, [f]) for f in requested]
    else:
        order = default_order(features) if method == "sequential" else list(features)
        players = [(f, [f]) for f in order]

    k = len(players)
    if method == "shapley" and cfg.n_permutations is None and k > _SHAPLEY_EXACT_MAX:
        raise ConfigError(
            f"exact Shapley values over {k} features need 2**{k} = {2**k:,} normalisations. "
            "Pass `groups` to attribute to fewer, larger units, or `n_permutations` for a "
            "sampled estimate."
        )
    return players, method


def _attribute(
    players: list[_Player],
    value_of: Callable[[list[str]], np.ndarray],
    *,
    method: str,
    n_permutations: int | None,
    seed: int,
    verbose: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Split ``v(every player fixed) - v(none fixed)`` among ``players``.

    ``value_of(resample)`` returns the normalised series with the features in
    ``resample`` resampled and every other feature at its observed values; the
    value of a coalition (the set of players held at observed values) is that
    series. Each coalition is evaluated once. Returns ``emi_total`` -- nothing
    fixed -- and one contribution per player, which for every method add up to
    ``v(all fixed) - emi_total``.
    """
    k = len(players)
    values: dict[frozenset[int], np.ndarray] = {}
    if method == "shapley" and n_permutations is not None:
        n_orders = n_permutations + n_permutations % 2
        if n_orders * max(k - 1, 1) + 2 >= 2**k:
            (log.info if verbose else log.debug)(
                "%d sampled orders would evaluate as many coalitions as the exact Shapley "
                "values need (2**%d); computing them exactly.",
                n_orders,
                k,
            )
            n_permutations = None
    planned = (
        k + 1
        if method == "sequential"
        else 2**k
        if n_permutations is None
        else (n_permutations + n_permutations % 2) * (k - 1) + 2
    )
    start = time.time()

    def v(fixed: frozenset[int]) -> np.ndarray:
        if fixed not in values:
            if not fixed:
                label = "emi_total"
            elif method == "sequential":
                label = players[max(fixed)][0]
            else:
                label = "with fixed " + ", ".join(players[i][0] for i in sorted(fixed))
            _log_decomposition_progress(verbose, start, len(values) + 1, planned, label)
            resample = [f for i, (_, feats) in enumerate(players) if i not in fixed for f in feats]
            values[fixed] = np.asarray(value_of(resample), dtype=float)
        return values[fixed]

    none_fixed: frozenset[int] = frozenset()
    emi_total = v(none_fixed)
    totals = [np.zeros_like(emi_total) for _ in players]

    if method == "sequential":
        prev = none_fixed
        for i in range(k):
            cur = prev | {i}
            totals[i] = v(cur) - v(prev)
            prev = cur
    elif n_permutations is None:
        # Exact: phi_i = sum over S not containing i of |S|!(k-|S|-1)!/k! * (v(S+i) - v(S)).
        weight = [factorial(s) * factorial(k - s - 1) / factorial(k) for s in range(k)]
        coalitions = [frozenset(i for i in range(k) if mask >> i & 1) for mask in range(2**k)]
        coalitions.sort(key=lambda s: (len(s), sorted(s)))
        for s in coalitions:
            v(s)
        for s in coalitions:
            for i in range(k):
                if i not in s:
                    totals[i] += weight[len(s)] * (values[s | {i}] - values[s])
    else:
        # Monte Carlo over orders, in antithetic pairs: an order and its reverse.
        rng = np.random.default_rng(seed)
        n_pairs = (n_permutations + 1) // 2
        for _ in range(n_pairs):
            perm = [int(i) for i in rng.permutation(k)]
            for order in (perm, perm[::-1]):
                prev = none_fixed
                for i in order:
                    cur = prev | {i}
                    totals[i] += v(cur) - v(prev)
                    prev = cur
        totals = [t / (2 * n_pairs) for t in totals]

    return emi_total, {name: totals[i] for i, (name, _) in enumerate(players)}


def _met_result(
    result: pd.DataFrame, emi_total: np.ndarray, contributions: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Add ``emi_total``, the contributions and the ``met_*`` totals to ``result``."""
    result["emi_total"] = emi_total
    for name, contribution in contributions.items():
        result[name] = contribution
    result["met_total"] = result["observed"] - result["emi_total"]
    result["met_base"] = float(result["met_total"].mean())
    contrib_sum = result[list(contributions)].sum(axis=1) if contributions else 0.0
    result["met_noise"] = result["met_total"] - (result["met_base"] + contrib_sum)
    return result


def decompose(
    df: pd.DataFrame,
    model: object | None = None,
    *,
    config: DecomposeConfig | None = None,
    method: str = "emission",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    High-level wrapper for time series decomposition.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with datetime and target column.
    model : object, optional
        Pre-trained model. If None, a new model will be trained.
    config : DecomposeConfig, optional
        Consolidated config object. Individual keyword arguments (``target``,
        ``backend``, ``covariates``, …) override the corresponding field
        when provided.
    method : {"emission", "meteorology"}
        Decomposition strategy.

    Returns
    -------
    pandas.DataFrame
        Decomposed result.

    Examples
    --------
    >>> import pandas as pd
    >>> from normet import decompose
    >>> df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=24, freq="h"),
    ...                    "value": range(24), "t2m": 10.0, "blh": 500.0})
    >>> result = decompose(df, method="emission", covariates=["t2m", "blh"],
    ...                    n_samples=2)  # doctest: +SKIP
    """
    _cfg = _resolve_config(config=config, method=method, **kwargs)
    # n_samples is None-by-default so it can follow the backend; the AutoML
    # paths below hand it straight to normalise(), which needs a number.
    _cfg.n_samples = resolve_n_samples(_cfg.n_samples, _cfg.backend)

    if df is None:
        raise DataError("`df` must be provided.")
    if _cfg.target is None:
        raise DataError("`target` must be provided.")
    if model is None and _cfg.covariates is None:
        raise ConfigError("Either `model` or `covariates` must be provided.")
    if model is None and _cfg.backend is None:
        raise ConfigError("When training a model, `backend` must be specified.")

    if _cfg.backend == CHRONOS_BACKEND:
        if _cfg.method == "emission":
            raise ConfigError(
                "method='emission' is not available on the chronos-2 backend. The "
                "emission decomposition isolates the calendar components by resampling "
                "date_unix / day_julian / weekday / hour, which works because an AutoML "
                "model sees them as ordinary features. Chronos-2 conditions on the "
                "target's own history and deweather() never resamples history, so a "
                "repeating calendar signal survives every draw -- the model reads it off "
                "the past and takes nothing from the covariate. Measured covariate "
                "sensitivity, calendar encoders supplied as ordinary covariates: "
                "trend 0.53%, weekly 0.75%, diurnal 1.50%, all six together 2.47%, "
                "against 4.78-10.72% for meteorology in the same runs -- and the diurnal "
                "signal was the largest injected component of all, so attribution runs "
                "opposite to signal size. The components would come back near zero and "
                "read as 'no trend' rather than 'not separable'. Use "
                "method='meteorology', or an AutoML backend."
            )
        if _cfg.method == "meteorology":
            return _decom_met_zero_shot(df=df, model=model, cfg=_cfg)

    if _cfg.method == "emission":
        return decom_emi(df=df, model=model, config=_cfg)

    if _cfg.method == "meteorology":
        return decom_met(df=df, model=model, config=_cfg)

    raise ConfigError(
        f"Unsupported decomposition method: '{_cfg.method}'. "
        "Must be one of 'emission' or 'meteorology'."
    )


def _decom_met_zero_shot(
    df: pd.DataFrame, model: object | None, cfg: DecomposeConfig
) -> pd.DataFrame:
    """Meteorological decomposition on Chronos-2, one nested de-weathering per coalition.

    Structurally identical to :func:`decom_met` -- same ``groups`` /
    ``attribution`` options, same result columns. Only the inner call changes:
    :meth:`Chronos2Estimator.deweather` in place of :func:`normalise`, because
    both answer the same question, "what would this series be with these
    covariates averaged over". ``deweather`` draws the weather from the frame
    itself, so ``resample_df`` / ``resample_pools`` / ``conditional_on`` are
    refused.

    Ordering for the sequential attribution is the one real difference.
    :func:`decom_met` ranks features by fitted importance, which does not exist
    here. ``variable_order`` is used when given; otherwise features are ranked
    by their individual covariate sensitivity, which is the zero-shot analogue:
    how far the forecast moves when that one feature is shuffled. Ranking needs
    more rows than ``context_length + prediction_length``; below that the
    covariate order is kept as given and a warning is logged, since an
    arbitrary order still yields a valid decomposition, only a less
    interpretable one. Shapley attribution needs no ranking.
    """
    from ..foundation import Chronos2Estimator, to_indexed_frame

    if cfg.target is None:
        raise DataError("`target` must be provided.")
    refused = [
        n
        for n in ("resample_df", "resample_pools", "conditional_on")
        if getattr(cfg, n) is not None
    ]
    if refused:
        raise ConfigError(
            f"{', '.join(refused)} not available on the chronos-2 backend: "
            "Chronos2Estimator.deweather draws the weather from the frame it is given."
        )
    _attribution_method(cfg)  # before any weights are loaded
    work = df.copy()
    if "date" not in work.columns:
        work = process_date(work)
    work = work[work["date"].notna()].sort_values("date").reset_index(drop=True)
    if cfg.target not in work.columns:
        raise DataError(f"`df` does not contain the target column '{cfg.target}'.")
    observed = work[cfg.target].to_numpy()
    if cfg.target != "value":
        work = work.rename(columns={cfg.target: "value"})

    time_var_set = {"hour", "weekday", "day_julian", "date_unix"}
    met = [c for c in (cfg.covariates or []) if c not in time_var_set and c in work.columns]
    if not met:
        raise ConfigError(
            "the chronos-2 backend needs meteorological covariates to decompose: "
            "pass covariates, excluding the time variables it cannot use"
        )

    indexed = to_indexed_frame(work)
    est = model if isinstance(model, Chronos2Estimator) else None
    if est is None:
        est = Chronos2Estimator(met_covariates=met, **(cfg.model_config or {}))
        est._load_pipeline()
    est._check_index(indexed.index)

    players, method = _attribution_plan(
        met, cfg, default_order=lambda feats: _rank_by_sensitivity(est, indexed, feats, cfg)
    )

    n_samples = resolve_n_samples(cfg.n_samples, CHRONOS_BACKEND)
    result = pd.DataFrame({"observed": observed}, index=pd.DatetimeIndex(work["date"]))
    result.index.name = "date"

    def value_of(resample: list[str]) -> np.ndarray:
        out = est.deweather(
            indexed,
            "value",
            met_features=resample,
            n_samples=n_samples,
            quantiles=(0.5,),
            random_state=cfg.seed,
            schema="normet",
        )
        return out["normalised"].reindex(result.index).to_numpy()

    emi_total, contributions = _attribute(
        players,
        value_of,
        method=method,
        n_permutations=cfg.n_permutations,
        seed=cfg.seed,
        verbose=cfg.verbose,
    )
    return _met_result(result, emi_total, contributions)


def _rank_by_sensitivity(
    est: Any, indexed: pd.DataFrame, met: list[str], cfg: DecomposeConfig
) -> list[str]:
    """Order meteorological features by how far each one alone moves the forecast."""
    horizon = est.prediction_length
    if len(indexed) <= est.context_length + horizon:
        log.warning(
            "only %d rows: need more than %d to rank features by covariate sensitivity, "
            "so the given covariate order is kept. Pass variable_order to pin it explicitly.",
            len(indexed),
            est.context_length + horizon,
        )
        return met
    anchor = indexed.index[-horizon]
    scores: dict[str, float] = {}
    for feat in met:
        shift = est.covariate_sensitivity(
            indexed,
            "value",
            anchor=anchor,
            horizon=horizon,
            met_features=[feat],
            random_state=cfg.seed,
        )
        scores[feat] = float(shift["mean_abs_shift"])
    ordered = sorted(met, key=lambda f: scores[f], reverse=not cfg.importance_ascending)
    (log.info if cfg.verbose else log.debug)(
        "covariate sensitivity ranking: %s",
        ", ".join(f"{f}={scores[f]:.3f}" for f in ordered),
    )
    return ordered


def decom_emi(
    df: pd.DataFrame,
    model: object | None = None,
    *,
    config: DecomposeConfig | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Emission-based decomposition by nested normalisation.

    Sequentially fixes time variables, in the fixed order ``base`` ->
    ``date_unix`` -> ``day_julian`` -> ``weekday`` -> ``hour``, to isolate
    the marginal contribution of each temporal component to the predicted
    concentration. Each returned component is the difference between two
    consecutive nested predictions (previous variables already fixed at
    their observed values, current variable now also fixed, everything
    else still resampled).

    .. important::
        **This fixed order is not just bookkeeping -- it determines what
        each component can and cannot represent.** Because ``date_unix``
        is fixed *before* ``day_julian``, ``weekday``, and ``hour``, the
        returned ``date_unix`` ("trend") component is computed while every
        within-year calendar position is still being averaged over
        resampling -- it cannot carry a recurring, calendar-aligned signal
        (e.g. a Christmas/New Year dip that recurs every year), only a
        genuine long-term drift. Conversely, ``day_julian`` (nominally the
        "seasonal" component) is computed with ``date_unix`` already fixed
        at *each row's own observed value*, so it is NOT a pooled,
        climatological quantity the way a bottom-up seasonal factor would
        be -- it stays native to the specific year and can register a
        one-off, non-repeating event (e.g. a single year's holiday dip, or
        a structural break such as a lockdown) despite its "seasonal"
        label. If you need to examine a recurring calendar effect, use
        ``day_julian``, not ``date_unix``, even though "trend" sounds like
        the more natural place to look for it.

    .. note::
        Time variables are opt-in at the model level (see
        :func:`normet.build_model`'s ``covariates``), not mandatory --
        this function adapts automatically. Only whichever of
        ``date_unix``/``day_julian``/``weekday``/``hour`` actually ended up
        as a model feature get decomposed into their own component; the
        rest are simply absent from the result (no error). A model trained
        on none of the four (e.g. meteorology/traffic predictors only)
        still decomposes cleanly into ``base``/``emi_base``/``emi_noise``
        with no time-variable columns at all.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Input data with datetime index and target column.
    model : object, optional
        Pre-trained model.
    config : DecomposeConfig, optional
        Consolidated config object. ``resample_df``, ``resample_pools`` and
        ``conditional_on`` are forwarded to every :func:`normalise` call;
        ``groups``, ``n_permutations`` and ``attribution="shapley"`` belong to
        :func:`decom_met` and are refused here.
    **kwargs
        Supported shorthand for overriding individual :class:`DecomposeConfig`
        fields without constructing a config object. Any field passed both via
        ``config`` and as a keyword is resolved in favour of the keyword.

    Returns
    -------
    pandas.DataFrame
        Decomposition results.
    """
    _cfg = _resolve_config(config=config, **kwargs)
    # n_samples is None-by-default so it can follow the backend; the AutoML
    # paths below hand it straight to normalise(), which needs a number.
    _cfg.n_samples = resolve_n_samples(_cfg.n_samples, _cfg.backend)

    if df is None:
        raise DataError("`df` must be provided.")
    if _cfg.target is None:
        raise DataError("`target` (target column name) must be provided.")
    if model is None and _cfg.covariates is None:
        raise ConfigError("Either `model` or `covariates` must be provided.")
    if model is None and _cfg.backend is None:
        raise ConfigError("When training a model, `backend` must be specified.")
    if _cfg.groups is not None or _cfg.n_permutations is not None or _cfg.attribution == "shapley":
        raise ConfigError(
            "`groups`, `n_permutations` and attribution='shapley' apply to the meteorological "
            "decomposition (decom_met); decom_emi fixes the time variables in its own "
            "calendar order."
        )

    df_work = process_date(df.copy()) if "date" not in df.columns else df.copy()
    if "date" not in df_work.columns:
        raise DataError("Could not find or create a 'date' column.")

    if _cfg.target not in df_work.columns:
        raise DataError(f"`df` does not contain the target column '{_cfg.target}'.")

    observed_series = df_work[_cfg.target].copy()
    if _cfg.target != "value":
        df_work = df_work.rename(columns={_cfg.target: "value"})

    mask_valid = df_work["date"].notna() & df_work["value"].notna()
    df_work = df_work.loc[mask_valid].sort_values("date").reset_index(drop=True)
    observed_series = observed_series.loc[mask_valid].reset_index(drop=True)

    if _cfg.covariates:
        missing_time_vars = [
            v
            for v in ["date_unix", "day_julian", "weekday", "hour"]
            if v in _cfg.covariates and v not in df_work.columns
        ]
        if missing_time_vars:
            try:
                df_work = add_date_variables(df_work)
                (log.info if _cfg.verbose else log.debug)(
                    "Generated time variables: %s", missing_time_vars
                )
            except Exception:
                log.warning(
                    "Could not generate some time features: %s", missing_time_vars, exc_info=False
                )

    if model is None:
        if _cfg.covariates is None:
            raise ValueError("covariates must be provided")
        (log.info if _cfg.verbose else log.debug)(
            "Training model via backend='%s' with features=%d...",
            _cfg.backend or "flaml",
            len(_cfg.covariates),
        )
        df_work, model = build_model(
            df=df_work,
            target="value",
            backend=_cfg.backend or "flaml",
            covariates=_cfg.covariates,
            split_method=_cfg.split_method,
            train_fraction=_cfg.train_fraction,
            model_config=_cfg.model_config,
            seed=_cfg.seed,
            n_cores=_cfg.n_cores,
            verbose=_cfg.verbose,
            cache=_cfg.cache,
        )

    try:
        model_feats = [str(c) for c in extract_features(model)]
    except Exception as exc:
        if not _cfg.covariates:
            raise ModelError("Cannot infer model features; please provide `covariates`.") from exc
        model_feats = [str(c) for c in _cfg.covariates]

    # The missing-time-vars generation above only runs when the caller
    # passes `covariates` explicitly. Repeat it here against model_feats
    # (covers the case where covariates is instead auto-derived from
    # `model` via extract_features()) so date_unix/day_julian/weekday/hour
    # get generated whenever the model needs them, regardless of which
    # path supplied the feature list. add_date_variables() is idempotent,
    # so this is a no-op whenever the first pass above already handled it.
    missing_time_vars = [
        v
        for v in ["date_unix", "day_julian", "weekday", "hour"]
        if v in model_feats and v not in df_work.columns
    ]
    if missing_time_vars:
        try:
            df_work = add_date_variables(df_work)
            (log.info if _cfg.verbose else log.debug)(
                "Generated time variables: %s", missing_time_vars
            )
        except Exception:
            log.warning(
                "Could not generate some time features: %s", missing_time_vars, exc_info=False
            )

    model_feats = [c for c in model_feats if c in df_work.columns]
    if not model_feats:
        raise DataError("No valid model features found in the provided `df` for decomposition.")

    result = (
        pd.DataFrame({"date": df_work["date"].to_numpy(), "observed": observed_series.to_numpy()})
        .set_index("date")
        .sort_index()
    )

    time_vars_order = ["base", "date_unix", "day_julian", "weekday", "hour"]
    present_time_vars = ["base"] + [
        v for v in time_vars_order[1:] if v in model_feats and v in df_work.columns
    ]

    n_cores_eff = _effective_cores(_cfg.n_cores)
    start = time.time()

    resample_vars = [v for v in model_feats if v != "value"]

    for i, var_to_fix in enumerate(present_time_vars, start=1):
        if var_to_fix != "base":
            resample_vars = [v for v in resample_vars if v != var_to_fix]

        _log_decomposition_progress(_cfg.verbose, start, i, len(present_time_vars), var_to_fix)

        df_norm = normalise(
            df=df_work,
            model=model,
            covariates=model_feats,
            variables_resample=resample_vars,
            n_samples=_cfg.n_samples,
            replace=True,
            aggregate=True,
            seed=_cfg.seed,
            n_cores=n_cores_eff,
            resample_df=_cfg.resample_df,
            resample_pools=_cfg.resample_pools,
            conditional_on=_cfg.conditional_on,
            memory_save=_cfg.memory_save,
            cache=_cfg.cache,
        )
        if "normalised" not in df_norm.columns:
            log.exception("`normalise` did not return 'normalised' column (aggregate=True).")
            raise ModelError(
                "`normalise` must return a DataFrame with column 'normalised' when aggregate=True."
            )

        result[var_to_fix] = df_norm.reindex(result.index)["normalised"].to_numpy()

    result["emi_total"] = result.get("hour", result["observed"])

    for higher_freq, lower_freq, target_col in [
        ("hour", "weekday", "hour"),
        ("weekday", "day_julian", "weekday"),
        ("day_julian", "date_unix", "day_julian"),
        ("date_unix", "base", "date_unix"),
    ]:
        if higher_freq in result.columns and lower_freq in result.columns:
            result[target_col] = result[higher_freq] - result[lower_freq]

    base_mean = float(result["base"].mean())
    result["emi_noise"] = result["base"] - base_mean
    result["emi_base"] = base_mean
    del result["base"]

    return result


def decom_met(
    df: pd.DataFrame,
    model: object | None = None,
    *,
    config: DecomposeConfig | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Meteorological decomposition by nested normalisation.

    ``emi_total`` is the normalised series with every meteorological (non-time)
    feature resampled; the time variables stay at their observed values
    throughout. The model's prediction minus ``emi_total`` -- the part the
    meteorology accounts for -- is split into one contribution per feature, or
    per group of features (``groups``), by re-running :func:`normalise` with
    some of them held at their observed values instead of resampled:

    - ``attribution="sequential"`` fixes them one at a time and reports each
      step's change. This is cumulative fixing, not leave-one-out: each
      contribution is conditional on everything fixed before it, so the split
      depends on the order -- ``variable_order`` if given, else fitted
      importance (``importance_ascending``), which can reorder when the model
      is refitted; with ``groups``, the order they are listed in.
    - ``attribution="shapley"`` averages each feature's (or group's) marginal
      effect over every order it could be fixed in: the Shapley value of the
      game whose value for a set ``S`` is the normalised series with ``S`` at
      observed values. No order is privileged, so the split does not move when
      features are listed differently or importance reshuffles.

    Either way the contributions add up exactly to ``prediction - emi_total``,
    and every :func:`normalise` call uses the same seed, so the differences
    between calls are paired (common random numbers).

    Separating transport from local effects is what ``groups`` is for::

        decom_met(df, model, groups={"local": met_cols, "transport": traj_cols})

    gives one ``local`` and one ``transport`` column (Shapley by default). Both
    are measured against ``emi_total``, which averages over the air masses in
    the resample pool, so over the record they are anomalies with a mean near
    zero. To measure transport against a reference air mass instead, give the
    trajectory features a pool of their own --
    ``resample_pools={"transport": clean_hours[traj_cols]}`` makes
    ``emi_total`` the level under that air mass and the ``transport`` column
    the change from it to the air that actually arrived.

    Note the asymmetry with :func:`decom_emi`, which fixes the time variables
    in a hardcoded calendar order chosen so each component has a specific
    temporal-frequency meaning.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Input data with datetime and target column.
    model : object, optional
        Pre-trained model. If None, a new model will be trained.
    config : DecomposeConfig, optional
        Consolidated config object. Individual keyword arguments (``target``,
        ``backend``, ``covariates``, ``groups``, ``attribution``,
        ``resample_pools``, …) override the corresponding field when provided.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``date``: ``observed``; ``emi_total``; one contribution
        column per feature, or per group (named after it); ``met_total``
        (``observed - emi_total``); ``met_base``, its mean (a constant); and
        ``met_noise`` = ``met_total - met_base - sum of contributions``. That
        last one equals the model residual ``observed - prediction`` shifted
        by the constant ``met_base``: it is what the model does not explain,
        not a meteorological term. When the model is trained here, rows with a
        missing target are dropped, as :func:`build_model` drops them.

    Raises
    ------
    DataError, ConfigError
        If required arguments are missing, columns are not found, or
        ``groups`` / ``attribution`` / ``variable_order`` /
        ``n_permutations`` are inconsistent with the model's features.
    ModelError
        If ``normalise`` does not return a ``normalised`` column.
    """
    _cfg = _resolve_config(config=config, **kwargs)
    # n_samples is None-by-default so it can follow the backend; the AutoML
    # paths below hand it straight to normalise(), which needs a number.
    _cfg.n_samples = resolve_n_samples(_cfg.n_samples, _cfg.backend)

    if df is None:
        raise DataError("`df` must be provided.")
    if _cfg.target is None:
        raise DataError("`target` (target column name) must be provided.")
    if model is None and _cfg.covariates is None:
        raise ConfigError("Either `model` or `covariates` must be provided.")
    if model is None and _cfg.backend is None:
        raise ConfigError("When training a model, `backend` must be specified.")
    _attribution_method(_cfg)  # before any model is trained

    df = df.copy()
    if "date" not in df.columns:
        df = process_date(df)
    df = df[df["date"].notna()].sort_values("date").reset_index(drop=True)

    if _cfg.target not in df.columns:
        raise DataError(f"`df` does not contain the target column '{_cfg.target}'.")

    df_work = df.copy()
    if _cfg.target != "value":
        df_work = df_work.rename(columns={_cfg.target: "value"})

    if _cfg.covariates:
        missing_time_vars = [
            v for v in _TIME_VARS if v in _cfg.covariates and v not in df_work.columns
        ]
        if missing_time_vars:
            try:
                df_work = add_date_variables(df_work)
                (log.info if _cfg.verbose else log.debug)(
                    "Generated time variables: %s", missing_time_vars
                )
            except Exception:
                log.warning(
                    "Missing time features not generated: %s", missing_time_vars, exc_info=False
                )

    if model is None:
        if _cfg.covariates is None:
            raise ValueError("covariates must be provided")
        (log.info if _cfg.verbose else log.debug)(
            "Training model via backend='%s' with features=%d...",
            _cfg.backend or "flaml",
            len(_cfg.covariates),
        )
        df_work, model = build_model(
            df=df_work,
            target="value",
            backend=_cfg.backend or "flaml",
            covariates=_cfg.covariates,
            split_method=_cfg.split_method,
            train_fraction=_cfg.train_fraction,
            model_config=_cfg.model_config,
            seed=_cfg.seed,
            verbose=_cfg.verbose,
            cache=_cfg.cache,
        )

    # Observed values come from the frame actually decomposed: a model trained
    # here drops the rows with a missing target, and taking `observed` from the
    # input instead left it longer than the dates it was paired with.
    observed_series = df_work["value"]

    try:
        feat_sorted = extract_features(model, importance_ascending=_cfg.importance_ascending)
    except Exception as exc:
        if not _cfg.covariates:
            raise ModelError("Cannot infer model features; please provide `covariates`.") from exc
        feat_sorted = list(_cfg.covariates)

    feat_sorted = [f for f in feat_sorted if f in df_work.columns]
    if not feat_sorted:
        raise DataError("No valid model features found in `df`.")

    # Already in importance order, which is the default sequential order.
    contrib_candidates = [f for f in feat_sorted if f not in _TIME_VARS]
    players, method = _attribution_plan(contrib_candidates, _cfg, default_order=list)

    result = (
        pd.DataFrame({"date": df_work["date"].to_numpy(), "observed": observed_series.to_numpy()})
        .set_index("date")
        .sort_index()
    )

    n_cores_eff = _effective_cores(_cfg.n_cores)

    def value_of(resample: list[str]) -> np.ndarray:
        df_norm = normalise(
            df=df_work,
            model=model,
            covariates=feat_sorted,
            variables_resample=resample,
            n_samples=_cfg.n_samples,
            replace=True,
            aggregate=True,
            seed=_cfg.seed,
            n_cores=n_cores_eff,
            resample_df=_cfg.resample_df,
            resample_pools=_cfg.resample_pools,
            conditional_on=_cfg.conditional_on,
            memory_save=_cfg.memory_save,
            cache=_cfg.cache,
        )
        if "normalised" not in df_norm.columns:
            log.exception("`normalise` did not return 'normalised' column (aggregate=True).")
            raise ModelError(
                "`normalise` must return a DataFrame with column 'normalised' when aggregate=True."
            )
        return df_norm.reindex(result.index)["normalised"].to_numpy()

    emi_total, contributions = _attribute(
        players,
        value_of,
        method=method,
        n_permutations=_cfg.n_permutations,
        seed=_cfg.seed,
        verbose=_cfg.verbose,
    )
    return _met_result(result, emi_total, contributions)
