# src/normet/analysis/decomposition.py
"""Split a normalised series into emission- and meteorology-driven components.

Provides :func:`decompose` (and the convenience wrappers :func:`decom_emi`,
:func:`decom_met`) plus :class:`DecomposeConfig`.
"""

from __future__ import annotations

import numbers
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import factorial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import ConfigError, DataError, ModelError
from ..foundation.estimator import CHRONOS_BACKEND, resolve_n_samples
from ..io.trajectory import is_trajectory_qc_column
from ..model.predict import ml_predict
from ..model.train import build_model, train_model
from ..utils import progress
from ..utils._config import DEFAULT_SEED, resolve_config, resolve_workers
from ..utils.features import extract_features
from ..utils.logging import get_logger
from ..utils.prepare import (
    add_date_variables,
    log1p_target,
    process_date,
)
from .normalise import (
    _apply_conditional_filter,
    _conditional_mask,
    _draw_sample_indices,
    _extract_strata,
    _parse_stratify_by,
    _split_across_pools,
    normalise,
)

_DEFAULT_NORMALISE = normalise
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
    stratify_by: str | Sequence[str] | None = None
    """Optional stratum column(s) forwarded to every :func:`normalise` call
    (e.g. ``"month"``, ``"season"``, ``"hour"``, or ``"traj_cluster"``) to
    prevent cross-season or cross-regime OOD extrapolation. In :func:`decom_emi`
    / :func:`decom_emi_shapley`, calendar strata (``"month"``/``"season"``,
    ``"weekday"``, ``"hour"``) are activated once the corresponding time
    variable is fixed. Not on chronos-2."""
    conditional_k: int | None = None
    """Optional positive integer ``k`` for kNN on-manifold conditional resampling
    forwarded to :func:`normalise` (Conditional Shapley over fixed meteorological
    features). Not on chronos-2."""
    exact_base: bool = False
    """``decom_emi`` / ``decom_emi_shapley`` only. If True, evaluate the unfixed
    ``base`` baseline analytically as the single-pass pool mean (when no extra
    ``resample_pools`` or non-calendar strata require Monte Carlo draws), making
    ``emi_noise == 0`` and preventing MC sampling noise from leaking into
    ``date_unix``."""
    include_diagnostics: bool = False
    """``decom_met`` only. If True, include explicit model-vs-observation columns
    ``predicted``, ``met_explained`` (sum of meteorological player contributions),
    and ``residual`` (``observed - predicted``) alongside the legacy ``met_*``
    columns."""
    return_uncertainty: bool = False
    """``decom_met`` only. If True, attach standard error columns ``emi_total_se``
    and ``{player}_se`` computed from paired per-seed Monte Carlo draws (and
    permutation spread when ``n_permutations`` is active) without extra model
    passes."""
    scale: str = "additive"
    """Attribution scale: ``"additive"`` (default) or ``"multiplicative"``. When
    ``"multiplicative"``, differences are evaluated on the ``log1p`` scale and
    mapped back to exact concentration-unit contributions via the Logarithmic
    Mean Divisia Index (LMDI), plus ``{player}_factor`` exponential ratios."""
    log_transform: bool = False
    """Whether ``df`` and ``model`` already live on the ``log1p`` scale (for
    example from :func:`normet.utils.prepare.log1p_target`)."""
    n_sensitivity_anchors: int = 1
    """Chronos-2 sequential attribution only. Number of valid forecast anchors
    to average covariate sensitivity over when ordering meteorological features
    (``1`` = latest accepted anchor, matching legacy single-anchor behaviour;
    ``> 1`` evenly spans valid anchors across the record)."""


@dataclass
class ShapleyInteractionResult:
    """Output of second-order Shapley interaction analysis (:func:`shapley_interactions`)."""

    decomposition: pd.DataFrame
    main_effects: pd.DataFrame
    pairwise: pd.DataFrame
    mean_matrix: pd.DataFrame
    std_matrix: pd.DataFrame
    summary: pd.DataFrame


_TIME_VARS = ("date_unix", "day_julian", "weekday", "hour")
# decom_met's own result columns, which a group may not be named after.
_MET_RESULT_COLUMNS = frozenset(
    {
        "date",
        "observed",
        "predicted",
        "emi_total",
        "met_total",
        "met_explained",
        "met_base",
        "met_noise",
        "residual",
    }
)
# Exact Shapley values need 2**k normalisations for k features or groups.
_SHAPLEY_EXACT_MAX = 10
#: Standard columns of :attr:`ShapleyInteractionResult.summary`, also emitted
#: when the model has no meteorological feature and the table is empty (an
#: empty frame without columns cannot be sorted by ``std``).
_INTERACTION_SUMMARY_COLUMNS = ("term", "kind", "mean", "std", "mean_abs", "max_abs")

#: Atmospheric-physics grouping taxonomy used by :func:`suggest_met_groups`.
MET_PHYSICAL_GROUPS: dict[str, tuple[str, ...]] = {
    "dynamics": (
        "ws",
        "wd",
        "u10",
        "v10",
        "blh",
        "pblh",
        "mixdepth",
        "wind_speed",
        "wind_dir",
    ),
    "thermodynamics": (
        "t2m",
        "temp",
        "air_temp",
        "d2m",
        "rh",
        "rh2m",
        "ssrd",
        "ssr",
        "srad",
        "solar",
        "tcc",
        "cloud",
    ),
    "synoptic_removal": (
        "sp",
        "msl",
        "pres",
        "pressure",
        "tp",
        "rain",
        "rainfall",
        "precip",
        "precipitation",
    ),
}


def _resolve_config(config: DecomposeConfig | None = None, **kwargs: Any) -> DecomposeConfig:
    return resolve_config(DecomposeConfig, config, **kwargs)


def _effective_cores(n_cores: int | None) -> int:
    """Resolve parallel worker count (>=1)."""
    return resolve_workers(n_cores)


def _log_decomposition_progress(
    verbose: bool, start: float, i: int, total: int, var_to_fix: str
) -> None:
    """Log a "Decomposing <var>" progress line with an ETA, shared by decom_emi/decom_met,
    and report the steps done (:mod:`normet.utils.progress`)."""
    progress.report(i - 1, total)
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
        if isinstance(members, str):
            feats = [members]
        elif isinstance(members, Iterable):
            feats = [str(f) for f in members]
        else:
            raise ConfigError(
                f"group {name!r} must be a feature name or a list of feature names, "
                f"got {type(members).__name__}."
            )
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


def suggest_met_groups(
    features: Sequence[str],
    *,
    custom_groups: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    """Group non-time meteorological and transport predictors into physical regimes.

    Organises predictors into atmospheric-physics groups suitable for
    :func:`decom_met` ``groups=...``: ``dynamics`` (wind / boundary layer),
    ``thermodynamics`` (temperature / humidity / solar radiation / clouds),
    ``synoptic_removal`` (pressure / precipitation), ``transport``
    (``traj_*`` trajectory descriptors, automatically excluding QC bookkeeping
    columns via :func:`normet.io.trajectory.is_trajectory_qc_column`), and
    ``other`` for remaining non-time predictors. Empty groups are omitted.
    """
    seen: set[str] = set()
    candidates: list[str] = []
    time_set = set(_TIME_VARS)
    for raw in features:
        col = str(raw)
        if col in seen or col in time_set or is_trajectory_qc_column(col):
            continue
        seen.add(col)
        candidates.append(col)

    out: dict[str, list[str]] = {}
    claimed: set[str] = set()
    cand_set = set(candidates)

    if custom_groups is not None:
        for g_name, members in custom_groups.items():
            m_list = [members] if isinstance(members, str) else [str(m) for m in members]
            kept = [m for m in m_list if m in cand_set and m not in claimed]
            if kept:
                out[str(g_name)] = kept
                claimed.update(kept)

    dyn_set = set(MET_PHYSICAL_GROUPS["dynamics"])
    thm_set = set(MET_PHYSICAL_GROUPS["thermodynamics"])
    syn_set = set(MET_PHYSICAL_GROUPS["synoptic_removal"])

    for col in candidates:
        if col in claimed:
            continue
        low = col.lower()
        if low in dyn_set:
            grp = "dynamics"
        elif low in thm_set:
            grp = "thermodynamics"
        elif low in syn_set:
            grp = "synoptic_removal"
        elif low.startswith("traj_"):
            grp = "transport"
        else:
            grp = "other"
        out.setdefault(grp, []).append(col)
        claimed.add(col)

    return out


def _validate_shared_options(cfg: DecomposeConfig) -> None:
    """Validate backend-independent config fields shared by decom_emi and decom_met."""
    if cfg.scale not in ("additive", "multiplicative"):
        raise ConfigError(f"`scale` must be 'additive' or 'multiplicative', got {cfg.scale!r}.")
    if cfg.conditional_k is not None:
        cond_k: object = cfg.conditional_k
        if isinstance(cond_k, bool) or not isinstance(cond_k, numbers.Integral):
            raise ConfigError(
                f"`conditional_k` must be a positive integer or None, got {cond_k!r}."
            )
        if int(cond_k) < 1:
            raise ConfigError(f"`conditional_k` must be at least 1, got {cond_k!r}.")
    n_anc: object = cfg.n_sensitivity_anchors
    if isinstance(n_anc, bool) or not isinstance(n_anc, numbers.Integral) or int(n_anc) < 1:
        raise ConfigError(f"`n_sensitivity_anchors` must be a positive integer, got {n_anc!r}.")


def _attribution_method(cfg: DecomposeConfig) -> str:
    """Resolve and validate the attribution options that do not depend on the
    model's features, so a bad call fails before a model is trained or loaded."""
    _validate_shared_options(cfg)
    method = (
        cfg.attribution
        if cfg.attribution is not None
        else ("shapley" if cfg.groups is not None else "sequential")
    )
    if method not in ("sequential", "shapley"):
        raise ConfigError(
            f"`attribution` must be 'sequential' or 'shapley', got {cfg.attribution!r}."
        )
    if cfg.n_permutations is not None:
        n: object = cfg.n_permutations  # annotated int, but nothing enforces it
        if method != "shapley":
            raise ConfigError("`n_permutations` only applies to attribution='shapley'.")
        if isinstance(n, bool) or not isinstance(n, numbers.Integral):
            raise ConfigError(f"`n_permutations` must be a whole number, got {n!r}.")
        if n < 1:
            raise ConfigError(f"`n_permutations` must be at least 1, got {n}.")
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


def _mean_series(arr: np.ndarray) -> np.ndarray:
    """Collapse optional MC draws axis ``(n_rows, M) -> (n_rows,)``."""
    return arr.mean(axis=1) if arr.ndim == 2 else arr


def _mc_se(arr: np.ndarray) -> np.ndarray:
    """Paired Monte Carlo standard error across draws on axis 1."""
    if arr.ndim < 2 or arr.shape[1] <= 1:
        n_rows = arr.shape[0] if arr.ndim >= 1 else 0
        return np.zeros(n_rows, dtype=float)
    return np.std(arr, axis=1, ddof=1) / np.sqrt(float(arr.shape[1]))


def _lmdi_weight(conc_end: np.ndarray, conc_start: np.ndarray) -> np.ndarray:
    """Logarithmic Mean Divisia Index weight ``(c1 - c0) / (ln(1+c1) - ln(1+c0))``."""
    c1 = np.asarray(conc_end, dtype=float)
    c0 = np.asarray(conc_start, dtype=float)
    z1 = np.log1p(np.maximum(c1, 0.0))
    z0 = np.log1p(np.maximum(c0, 0.0))
    dz = z1 - z0
    limit = 1.0 + 0.5 * (np.maximum(c1, 0.0) + np.maximum(c0, 0.0))
    safe_dz = np.where(np.abs(dz) > 1e-8, dz, 1.0)
    return np.where(np.abs(dz) > 1e-8, (c1 - c0) / safe_dz, limit)


def _step_stratify_by(
    stratify_by: str | Sequence[str] | None, resample_vars: Sequence[str]
) -> list[str] | None:
    """Drop calendar strata whose defining time variable is still resampled in this step."""
    cols = _parse_stratify_by(stratify_by)
    if not cols:
        return None
    res_set = set(resample_vars)
    active: list[str] = []
    for col in cols:
        if col in ("month", "season") and "day_julian" in res_set:
            continue
        if col == "weekday" and "weekday" in res_set:
            continue
        if col == "hour" and "hour" in res_set:
            continue
        active.append(col)
    return active or None


def _compute_interaction_indices(
    player_names: Sequence[str],
    values: Mapping[frozenset[int], np.ndarray],
    shapley_phi: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute exact 2nd-order Grabisch-Roubens Shapley interactions & main effects.

    Uses already-evaluated coalition values ``values[S]`` (no extra normalisations).
    Returns ``(main_effects, pairwise)`` satisfying exact closure:
    ``sum(main_effects) + sum(pairwise) == sum(shapley_phi)``.
    """
    k = len(player_names)
    first_arr = next(iter(values.values()))
    n_rows = len(first_arr)
    main_effects: dict[str, np.ndarray] = {
        name: np.asarray(shapley_phi[name], dtype=float).copy() for name in player_names
    }
    pairwise: dict[str, np.ndarray] = {}
    if k < 2:
        return main_effects, pairwise

    weights = [factorial(s) * factorial(k - s - 2) / factorial(k - 1) for s in range(k - 1)]
    for i in range(k):
        for j in range(i + 1, k):
            others = [m for m in range(k) if m not in (i, j)]
            acc = np.zeros(n_rows, dtype=float)
            for mask in range(1 << len(others)):
                s_set = frozenset(others[b] for b in range(len(others)) if mask >> b & 1)
                w = weights[len(s_set)]
                diff = (
                    values[s_set | {i, j}]
                    - values[s_set | {i}]
                    - values[s_set | {j}]
                    + values[s_set]
                )
                acc += w * diff
            pair_name = f"{player_names[i]}:{player_names[j]}"
            pairwise[pair_name] = acc
            main_effects[player_names[i]] = main_effects[player_names[i]] - 0.5 * acc
            main_effects[player_names[j]] = main_effects[player_names[j]] - 0.5 * acc
    return main_effects, pairwise


def _attribute_core(
    players: list[_Player],
    value_of: Callable[[list[str]], np.ndarray],
    *,
    method: str,
    n_permutations: int | None,
    seed: int,
    verbose: bool,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[frozenset[int], np.ndarray],
    dict[str, np.ndarray],
]:
    """Split ``v(every player fixed) - v(none fixed)`` among ``players``.

    Supports both 1D coalition values ``(n_rows,)`` and 2D paired-draw matrices
    ``(n_rows, n_draws)``. Returns ``(emi_total_raw, totals_raw, coalition_values,
    perm_se)``, where ``perm_se`` gives permutation-sampling standard errors for
    sampled Shapley (zeros otherwise).
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
            with progress.part(len(values), planned):
                values[fixed] = np.asarray(value_of(resample), dtype=float)
        return values[fixed]

    none_fixed: frozenset[int] = frozenset()
    emi_total = v(none_fixed)
    totals = [np.zeros_like(emi_total, dtype=float) for _ in players]
    perm_se = {name: np.zeros(emi_total.shape[0], dtype=float) for name, _ in players}

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
        pair_means: list[list[np.ndarray]] = [[] for _ in players]
        for _ in range(n_pairs):
            perm = [int(i) for i in rng.permutation(k)]
            pair_accum = [np.zeros_like(emi_total, dtype=float) for _ in players]
            for order in (perm, perm[::-1]):
                prev = none_fixed
                for i in order:
                    cur = prev | {i}
                    delta = v(cur) - v(prev)
                    totals[i] += delta
                    pair_accum[i] += delta
                    prev = cur
            for i in range(k):
                pair_means[i].append(_mean_series(pair_accum[i] / 2.0))
        totals = [t / (2 * n_pairs) for t in totals]
        if n_pairs >= 2:
            for idx_p, (name, _) in enumerate(players):
                stacked_pairs = np.stack(pair_means[idx_p], axis=0)
                perm_se[name] = np.std(stacked_pairs, axis=0, ddof=1) / np.sqrt(float(n_pairs))

    return emi_total, {name: totals[i] for i, (name, _) in enumerate(players)}, values, perm_se


def _attribute(
    players: list[_Player],
    value_of: Callable[[list[str]], np.ndarray],
    *,
    method: str,
    n_permutations: int | None,
    seed: int,
    verbose: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Split ``v(every player fixed) - v(none fixed)`` among ``players``."""
    emi_total, contribs, _values, _perm_se = _attribute_core(
        players,
        value_of,
        method=method,
        n_permutations=n_permutations,
        seed=seed,
        verbose=verbose,
    )
    return emi_total, contribs


def _met_result(
    result: pd.DataFrame,
    emi_total: np.ndarray,
    contributions: dict[str, np.ndarray],
    *,
    include_diagnostics: bool = False,
    uncertainty: Mapping[str, np.ndarray] | None = None,
    factors: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Add ``emi_total``, the contributions, optional diagnostics/SE/factors, and ``met_*`` totals."""
    result["emi_total"] = emi_total
    for name, contribution in contributions.items():
        result[name] = contribution
    contrib_sum = result[list(contributions)].sum(axis=1) if contributions else 0.0
    if include_diagnostics:
        result["predicted"] = result["emi_total"] + contrib_sum
        result["met_explained"] = contrib_sum
        result["residual"] = result["observed"] - result["predicted"]
    result["met_total"] = result["observed"] - result["emi_total"]
    result["met_base"] = float(result["met_total"].mean())
    result["met_noise"] = result["met_total"] - (result["met_base"] + contrib_sum)
    if uncertainty:
        for col, se_arr in uncertainty.items():
            result[col] = se_arr
    if factors:
        for col, fac_arr in factors.items():
            result[col] = fac_arr
    return result


def decompose(
    df: pd.DataFrame,
    model: object | None = None,
    *,
    config: DecomposeConfig | None = None,
    method: str | None = None,
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
    method : {"emission", "meteorology"}, optional
        Decomposition strategy. ``None`` (the default) keeps the config's
        value, or the dataclass default ``"emission"`` -- a caller who set
        ``method`` on the config used to have it overwritten here by this
        keyword's old non-``None`` default.

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
    if config is None:
        # Without a config there is nothing to protect, so the old unfiltered
        # construction is kept: an explicit ``target=None`` must still reach
        # the DataError below rather than be dropped as "not given". ``method``
        # is the one keyword whose default is now ``None``; applying it only
        # when given keeps the dataclass default otherwise.
        _cfg = _resolve_config(**kwargs)
        if method is not None:
            _cfg.method = method
    else:
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
    itself, so ``resample_df`` / ``resample_pools`` / ``conditional_on`` /
    ``stratify_by`` / ``conditional_k`` are refused.
    """
    from ..foundation import (
        Chronos2Estimator,
        ICNDesign,
        MetPool,
        add_calendar_covariates,
        to_indexed_frame,
    )

    if cfg.target is None:
        raise DataError("`target` must be provided.")
    mc = dict(cfg.model_config or {})
    method_mode = str(mc.pop("method", "deweather")).lower()
    estimate = str(mc.pop("estimate", "level")).lower()
    model_path = mc.pop("model_path", None)
    icn_design = mc.pop("icn_design", None)
    if method_mode not in ("deweather", "icn"):
        raise ConfigError(
            f"model_config['method'] must be 'deweather' or 'icn', got {method_mode!r}"
        )
    if estimate not in ("level", "paired"):
        raise ConfigError(f"model_config['estimate'] must be 'level' or 'paired', got {estimate!r}")

    refuse_names = (
        ("resample_pools", "conditional_on", "stratify_by", "conditional_k")
        if method_mode == "icn"
        else (
            "resample_df",
            "resample_pools",
            "conditional_on",
            "stratify_by",
            "conditional_k",
        )
    )
    refused = [n for n in refuse_names if getattr(cfg, n) is not None]
    if cfg.return_uncertainty:
        refused.append("return_uncertainty")
    if cfg.scale != "additive":
        refused.append(f"scale={cfg.scale!r}")
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
        if model_path is not None:
            est = Chronos2Estimator.load(model_path, device=mc.get("device"))
            if est.met_covariates is None:
                est.met_covariates = list(met)
        else:
            est = Chronos2Estimator(met_covariates=met, **mc)
            est._load_pipeline()
    est._check_index(indexed.index)

    players, method = _attribution_plan(
        met, cfg, default_order=lambda feats: _rank_by_sensitivity(est, indexed, feats, cfg)
    )

    n_samples = resolve_n_samples(cfg.n_samples, CHRONOS_BACKEND)
    result = pd.DataFrame({"observed": observed}, index=pd.DatetimeIndex(work["date"]))
    result.index.name = "date"

    ext_pool: MetPool | None = None
    indexed_icn: pd.DataFrame | None = None
    if method_mode == "icn":
        indexed_icn = add_calendar_covariates(indexed)
        if cfg.resample_df is not None:
            rw = cfg.resample_df.copy()
            if not isinstance(rw.index, pd.DatetimeIndex) and "date" not in rw.columns:
                rw = process_date(rw)
            pool_indexed = to_indexed_frame(rw)
            ext_pool = MetPool.from_frame(pool_indexed, met)

    def value_of(resample: list[str]) -> np.ndarray:
        if method_mode == "icn":
            assert indexed_icn is not None
            if isinstance(icn_design, ICNDesign):
                design = icn_design
            else:
                overrides: dict[str, Any] = {
                    "context_length": est.context_length,
                    "n_resamples": n_samples,
                    "seed": cfg.seed,
                }
                if "prediction_length" in mc:
                    bh = int(mc["prediction_length"])
                    overrides["block_h"] = bh
                    overrides["stride"] = max(1, bh // 2)
                if isinstance(icn_design, Mapping):
                    overrides.update(icn_design)
                design = ICNDesign.v1(**overrides)
            out_icn = est.normalise_in_context(
                indexed_icn,
                "value",
                design=design,
                met_features=resample,
                pool=ext_pool,
            )
            return (
                out_icn.to_normet_frame(estimate=estimate)["normalised"]
                .reindex(result.index)
                .to_numpy()
            )
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
    return _met_result(
        result,
        emi_total,
        contributions,
        include_diagnostics=cfg.include_diagnostics,
    )


def _rank_by_sensitivity(
    est: Any, indexed: pd.DataFrame, met: list[str], cfg: DecomposeConfig
) -> list[str]:
    """Order meteorological features by mean sensitivity across evenly spaced anchors."""
    horizon = est.prediction_length
    if len(indexed) <= est.context_length + horizon:
        log.warning(
            "only %d rows: need more than %d to rank features by covariate sensitivity, "
            "so the given covariate order is kept. Pass variable_order to pin it explicitly.",
            len(indexed),
            est.context_length + horizon,
        )
        return met
    target = indexed["value"].to_numpy(dtype=float)
    n_ctx = est.context_length
    valid_anchors = [
        indexed.index[i]
        for i in range(n_ctx, len(indexed) - horizon + 1)
        if est._context_coverage(target[i - n_ctx : i]) >= est.min_context_coverage
    ]
    if not valid_anchors:
        log.warning(
            "no %d-h context has %.0f%% of its target observed to rank features by "
            "covariate sensitivity from, so the given covariate order is kept. Pass "
            "variable_order to pin it explicitly.",
            n_ctx,
            100 * est.min_context_coverage,
        )
        return met
    k_anc = max(1, int(cfg.n_sensitivity_anchors))
    if k_anc == 1 or len(valid_anchors) == 1:
        anchors = [valid_anchors[-1]]
    else:
        pick_idx = np.linspace(0, len(valid_anchors) - 1, min(k_anc, len(valid_anchors)), dtype=int)
        anchors = [valid_anchors[int(idx)] for idx in pick_idx]
    scores: dict[str, float] = {}
    for feat in met:
        vals = [
            float(
                est.covariate_sensitivity(
                    indexed,
                    "value",
                    anchor=anc,
                    horizon=horizon,
                    met_features=[feat],
                    random_state=cfg.seed,
                )["mean_abs_shift"]
            )
            for anc in anchors
        ]
        scores[feat] = float(np.mean(vals))
    ordered = sorted(met, key=lambda f: scores[f], reverse=not cfg.importance_ascending)
    (log.info if cfg.verbose else log.debug)(
        "covariate sensitivity ranking (%d anchor(s)): %s",
        len(anchors),
        ", ".join(f"{f}={scores[f]:.3f}" for f in ordered),
    )
    return ordered


def _prepare_decomposition_frame(
    df: pd.DataFrame | None,
    cfg: DecomposeConfig,
    *,
    model: object | None,
    drop_na_target: bool,
    sort_by_importance: bool,
) -> tuple[pd.DataFrame, object, list[str], np.ndarray, bool]:
    """Shared input cleaning, optional model training, and calendar feature generation.

    Returns ``(df_work, model, features, observed_conc, is_log_space)``.
    """
    if df is None:
        raise DataError("`df` must be provided.")
    if cfg.target is None:
        raise DataError("`target` (target column name) must be provided.")
    if model is None and cfg.covariates is None:
        raise ConfigError("Either `model` or `covariates` must be provided.")
    if model is None and cfg.backend is None:
        raise ConfigError("When training a model, `backend` must be specified.")

    work = process_date(df.copy()) if "date" not in df.columns else df.copy()
    if "date" not in work.columns:
        raise DataError("Could not find or create a 'date' column.")
    if cfg.target not in work.columns:
        raise DataError(f"`df` does not contain the target column '{cfg.target}'.")
    if cfg.target != "value":
        work = work.rename(columns={cfg.target: "value"})

    mask = work["date"].notna() & work["value"].notna() if drop_na_target else work["date"].notna()
    work = work.loc[mask].sort_values("date").reset_index(drop=True)

    is_log_space = bool(cfg.log_transform)
    observed_raw = work["value"].to_numpy(dtype=float)
    observed_conc = np.expm1(observed_raw) if cfg.log_transform else observed_raw.copy()

    if cfg.covariates:
        missing_time = [v for v in _TIME_VARS if v in cfg.covariates and v not in work.columns]
        if missing_time:
            try:
                work = add_date_variables(work)
                (log.info if cfg.verbose else log.debug)(
                    "Generated time variables: %s", missing_time
                )
            except Exception:
                log.warning(
                    "Could not generate some time features: %s", missing_time, exc_info=False
                )

    if model is None:
        if cfg.covariates is None:
            raise ValueError("covariates must be provided")
        if cfg.scale == "multiplicative" and not is_log_space:
            if (work["value"].to_numpy(dtype=float) < 0.0).any():
                raise DataError("Cannot apply scale='multiplicative' with negative target values.")
            work = log1p_target(work, "value")
            is_log_space = True
        (log.info if cfg.verbose else log.debug)(
            "Training model via backend='%s' with features=%d...",
            cfg.backend or "flaml",
            len(cfg.covariates),
        )
        work, model = build_model(
            df=work,
            target="value",
            backend=cfg.backend or "flaml",
            covariates=cfg.covariates,
            split_method=cfg.split_method,
            train_fraction=cfg.train_fraction,
            model_config=cfg.model_config,
            seed=cfg.seed,
            n_cores=cfg.n_cores,
            verbose=cfg.verbose,
            cache=cfg.cache,
        )
        obs_after = work["value"].to_numpy(dtype=float)
        observed_conc = np.expm1(obs_after) if is_log_space else obs_after.copy()

    try:
        if sort_by_importance:
            feats = [
                str(c)
                for c in extract_features(model, importance_ascending=cfg.importance_ascending)
            ]
        else:
            feats = [str(c) for c in extract_features(model)]
    except Exception as exc:
        if not cfg.covariates:
            raise ModelError("Cannot infer model features; please provide `covariates`.") from exc
        feats = [str(c) for c in cfg.covariates]

    missing_time = [v for v in _TIME_VARS if v in feats and v not in work.columns]
    if missing_time:
        try:
            work = add_date_variables(work)
            (log.info if cfg.verbose else log.debug)("Generated time variables: %s", missing_time)
        except Exception:
            log.warning("Could not generate some time features: %s", missing_time, exc_info=False)

    feats = [f for f in feats if f in work.columns]
    if not feats:
        raise DataError("No valid model features found in `df`.")
    return work, model, feats, observed_conc, is_log_space


def _can_use_analytic_base(
    model: object,
    cfg: DecomposeConfig,
    model_feats: Sequence[str],
    resample_vars: Sequence[str],
) -> bool:
    """Whether ``base`` can be computed analytically via one pool mean pass."""
    if not callable(getattr(model, "predict", None)):
        return False
    if _step_stratify_by(cfg.stratify_by, resample_vars) is not None:
        return False
    if cfg.resample_pools:
        feat_set = set(model_feats)
        for pool in cfg.resample_pools.values():
            if isinstance(pool, pd.DataFrame) and (set(pool.columns) & feat_set):
                return False
    return True


def _analytic_base_value(
    df_work: pd.DataFrame,
    model: object,
    cfg: DecomposeConfig,
    *,
    is_log_space: bool,
    n_cores_eff: int,
    n_rows: int,
) -> np.ndarray:
    """Evaluate expected prediction over the reference pool in a single predict pass."""
    base_pool = df_work if cfg.resample_df is None else cfg.resample_df.copy()
    if cfg.conditional_on:
        base_pool = _apply_conditional_filter(base_pool, cfg.conditional_on)
        if base_pool.empty:
            raise DataError("`conditional_on` filter left no rows in the resample pool.")
    preds_base = ml_predict(model, base_pool, num_threads=n_cores_eff)
    if cfg.scale == "multiplicative":
        # The Monte Carlo path aggregates the draws first and only then maps
        # the aggregate to the log1p scale (decom_emi's
        # ``np.log1p(np.maximum(vals, 0.0))``); so the analytic limit of the
        # linear-space branch is ``log1p(mean(preds))``, not
        # ``mean(log1p(preds))``. The latter is a different estimand with a
        # Jensen gap that does not shrink with n_samples. A log-space model
        # already predicts the transformed scale, so its mean is unchanged.
        scalar = (
            float(np.nanmean(preds_base))
            if is_log_space
            else float(np.log1p(np.maximum(np.nanmean(preds_base), 0.0)))
        )
    elif cfg.log_transform:
        scalar = float(np.expm1(float(np.nanmean(preds_base))))
    else:
        scalar = float(np.nanmean(preds_base))
    return np.full(n_rows, scalar, dtype=float)


def _normet_draws_array(
    df_norm: pd.DataFrame,
    target_index: pd.Index,
    *,
    want_draws: bool,
) -> np.ndarray:
    """Align normalised Series or per-seed draw columns to ``target_index``."""
    aligned = df_norm.reindex(target_index)
    if not want_draws:
        if "normalised" not in aligned.columns:
            log.exception("`normalise` did not return 'normalised' column (aggregate=True).")
            raise ModelError(
                "`normalise` must return a DataFrame with column 'normalised' when aggregate=True."
            )
        return aligned["normalised"].to_numpy(dtype=float)

    seed_cols = [c for c in aligned.columns if c != "observed"]
    if not seed_cols:
        raise ModelError("`normalise` with aggregate=False returned no sample columns.")
    arr = aligned[seed_cols].to_numpy(dtype=float)
    return arr.reshape(len(target_index), -1)


def _finalise_emission_scale(
    result: pd.DataFrame,
    component_cols: Sequence[str],
    z_or_c_series: Mapping[str, np.ndarray],
    *,
    base_key: str,
    total_key: str,
    scale: str,
    is_log_space: bool,
    main_effects: Mapping[str, np.ndarray] | None = None,
    pairwise: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Convert raw ``z`` (multiplicative) or concentration increments into output columns."""
    raw_base = np.asarray(z_or_c_series[base_key], dtype=float)
    raw_total = np.asarray(z_or_c_series[total_key], dtype=float)

    if scale == "additive":
        base_mean = float(np.mean(raw_base))
        # Components first, then the totals, matching the pre-scale-option
        # decom_emi column order (``observed``, calendar components,
        # ``emi_total``, ``emi_noise``, ``emi_base``). Positional consumers
        # of the default emission output -- e.g. ``out.columns[1:-3]`` for the
        # components -- rely on it.
        for col in component_cols:
            result[col] = np.asarray(z_or_c_series[col], dtype=float)
        if main_effects is not None:
            for k, arr in main_effects.items():
                result[f"{k}_main"] = np.asarray(arr, dtype=float)
        if pairwise is not None:
            for k, arr in pairwise.items():
                result[k] = np.asarray(arr, dtype=float)
        result["emi_total"] = raw_total
        result["emi_noise"] = raw_base - base_mean
        result["emi_base"] = base_mean
        return result

    def _to_z(arr: np.ndarray) -> np.ndarray:
        return arr if is_log_space else np.log1p(np.maximum(arr, 0.0))

    z_base = _to_z(raw_base)
    z_total = _to_z(raw_total)
    z_base_mean = float(np.mean(z_base))
    emi_total_c = np.expm1(z_total)
    emi_base_c = float(np.expm1(z_base_mean))
    w = _lmdi_weight(emi_total_c, np.full_like(emi_total_c, emi_base_c))
    dz_noise = z_base - z_base_mean

    result["emi_total"] = emi_total_c
    result["emi_base"] = emi_base_c
    result["emi_noise"] = w * dz_noise
    for col in component_cols:
        dz = _to_z(raw_total) * 0.0  # shape only
        if col in z_or_c_series:
            dz = np.asarray(z_or_c_series[col], dtype=float)
        result[col] = w * dz
        result[f"{col}_factor"] = np.exp(dz)
    if main_effects is not None:
        for k, arr in main_effects.items():
            result[f"{k}_main"] = w * np.asarray(arr, dtype=float)
    if pairwise is not None:
        for k, arr in pairwise.items():
            result[k] = w * np.asarray(arr, dtype=float)
    return result


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
        Consolidated config object. ``resample_df``, ``resample_pools``,
        ``conditional_on`` and ``stratify_by`` are forwarded to every
        :func:`normalise` call; ``groups``, ``n_permutations`` and
        ``attribution`` belong to :func:`decom_met` and are refused here.
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
    _cfg.n_samples = resolve_n_samples(_cfg.n_samples, _cfg.backend)
    _validate_shared_options(_cfg)

    if _cfg.groups is not None or _cfg.n_permutations is not None or _cfg.attribution is not None:
        raise ConfigError(
            "`groups`, `n_permutations` and `attribution` apply to the meteorological "
            "decomposition (decom_met); decom_emi fixes the time variables in its own "
            "calendar order."
        )

    df_work, model, model_feats, observed_conc, is_log_space = _prepare_decomposition_frame(
        df, _cfg, model=model, drop_na_target=True, sort_by_importance=False
    )

    result = (
        pd.DataFrame({"date": df_work["date"].to_numpy(), "observed": observed_conc})
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
    norm_log_flag = bool(is_log_space and _cfg.scale != "multiplicative")

    fits: dict[str, np.ndarray] = {}
    for i, var_to_fix in enumerate(present_time_vars, start=1):
        if var_to_fix != "base":
            resample_vars = [v for v in resample_vars if v != var_to_fix]

        _log_decomposition_progress(_cfg.verbose, start, i, len(present_time_vars), var_to_fix)
        if (
            var_to_fix == "base"
            and _cfg.exact_base
            and _can_use_analytic_base(model, _cfg, model_feats, resample_vars)
        ):
            fits["base"] = _analytic_base_value(
                df_work,
                model,
                _cfg,
                is_log_space=is_log_space,
                n_cores_eff=n_cores_eff,
                n_rows=len(result),
            )
            continue

        step_strat = _step_stratify_by(_cfg.stratify_by, resample_vars)
        with progress.part(i - 1, len(present_time_vars)):
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
                stratify_by=step_strat,
                conditional_k=_cfg.conditional_k,
                log_transform=norm_log_flag,
                memory_save=_cfg.memory_save,
                cache=_cfg.cache,
            )
        vals = _normet_draws_array(df_norm, result.index, want_draws=False)
        if _cfg.scale == "multiplicative" and not is_log_space:
            vals = np.log1p(np.maximum(vals, 0.0))
        fits[var_to_fix] = vals

    series_map: dict[str, np.ndarray] = {
        "__base__": fits["base"],
        "__total__": fits[present_time_vars[-1]],
    }
    lower_fit = fits["base"]
    for var_to_fix in present_time_vars[1:]:
        fit = fits[var_to_fix]
        series_map[var_to_fix] = fit - lower_fit
        lower_fit = fit

    return _finalise_emission_scale(
        result,
        present_time_vars[1:],
        series_map,
        base_key="__base__",
        total_key="__total__",
        scale=_cfg.scale,
        is_log_space=True if _cfg.scale == "multiplicative" else is_log_space,
    )


def decom_emi_shapley(
    df: pd.DataFrame,
    model: object | None = None,
    *,
    config: DecomposeConfig | None = None,
    include_interactions: bool = True,
    exact_base: bool | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Order-independent Shapley decomposition over time-scale components.

    Evaluates exact Shapley values over whichever of ``date_unix``,
    ``day_julian``, ``weekday``, ``hour`` are present in the model (at most
    ``2**4 = 16`` coalitions), resolving order sensitivity between trend and
    seasonality. When ``include_interactions=True``, also exposes main effects
    (``{var}_main``) and second-order interaction terms (e.g.
    ``date_unix:day_julian`` capturing non-repeating annual anomalies).
    """
    _cfg = _resolve_config(config=config, **kwargs)
    _cfg.n_samples = resolve_n_samples(_cfg.n_samples, _cfg.backend)
    _validate_shared_options(_cfg)
    if _cfg.groups is not None or _cfg.n_permutations is not None or _cfg.attribution is not None:
        raise ConfigError(
            "`groups`, `n_permutations` and `attribution` do not apply to `decom_emi_shapley`."
        )
    use_exact_base = bool(_cfg.exact_base) if exact_base is None else bool(exact_base)

    df_work, model, model_feats, observed_conc, is_log_space = _prepare_decomposition_frame(
        df, _cfg, model=model, drop_na_target=True, sort_by_importance=False
    )

    result = (
        pd.DataFrame({"date": df_work["date"].to_numpy(), "observed": observed_conc})
        .set_index("date")
        .sort_index()
    )
    time_players = [t for t in _TIME_VARS if t in model_feats and t in df_work.columns]
    n_cores_eff = _effective_cores(_cfg.n_cores)
    norm_log_flag = bool(is_log_space and _cfg.scale != "multiplicative")

    # A model with no time feature is a valid input (``time_players == []``)
    # and is handled by the ordinary zero-player attribution below: the one
    # coalition value its empty player set needs is enough. Do NOT delegate
    # to decom_emi here: the frame was already log1p-transformed above when
    # ``scale='multiplicative'`` was requested and the model is in log space,
    # while a fresh decom_emi call would re-derive ``is_log_space`` from
    # ``log_transform`` (False) and log1p the predictions a second time,
    # reporting observations in log1p units.

    players: list[_Player] = [(t, [t]) for t in time_players]
    all_model_vars = [v for v in model_feats if v != "value"]
    met_vars = [v for v in all_model_vars if v not in time_players]

    def value_of(resample_time: list[str]) -> np.ndarray:
        full_resample = [
            v for v in model_feats if v != "value" and (v in met_vars or v in resample_time)
        ]
        if (
            set(resample_time) == set(time_players)
            and use_exact_base
            and _can_use_analytic_base(model, _cfg, model_feats, full_resample)
        ):
            return _analytic_base_value(
                df_work,
                model,
                _cfg,
                is_log_space=is_log_space,
                n_cores_eff=n_cores_eff,
                n_rows=len(result),
            )
        step_strat = _step_stratify_by(_cfg.stratify_by, full_resample)
        df_norm = normalise(
            df=df_work,
            model=model,
            covariates=model_feats,
            variables_resample=full_resample,
            n_samples=_cfg.n_samples,
            replace=True,
            aggregate=True,
            seed=_cfg.seed,
            n_cores=n_cores_eff,
            resample_df=_cfg.resample_df,
            resample_pools=_cfg.resample_pools,
            conditional_on=_cfg.conditional_on,
            stratify_by=step_strat,
            conditional_k=_cfg.conditional_k,
            log_transform=norm_log_flag,
            memory_save=_cfg.memory_save,
            cache=_cfg.cache,
        )
        vals = _normet_draws_array(df_norm, result.index, want_draws=False)
        if _cfg.scale == "multiplicative" and not is_log_space:
            vals = np.log1p(np.maximum(vals, 0.0))
        return vals

    raw_base, phi, coal_vals, _ = _attribute_core(
        players,
        value_of,
        method="shapley",
        n_permutations=None,
        seed=_cfg.seed,
        verbose=_cfg.verbose,
    )
    all_fixed = frozenset(range(len(players)))
    raw_total = coal_vals[all_fixed]

    main_eff: dict[str, np.ndarray] | None = None
    pair_eff: dict[str, np.ndarray] | None = None
    if include_interactions and len(time_players) >= 2:
        main_eff, pair_eff = _compute_interaction_indices(time_players, coal_vals, phi)

    series_map: dict[str, np.ndarray] = {"__base__": raw_base, "__total__": raw_total}
    for name in time_players:
        series_map[name] = phi[name]

    return _finalise_emission_scale(
        result,
        time_players,
        series_map,
        base_key="__base__",
        total_key="__total__",
        scale=_cfg.scale,
        is_log_space=True if _cfg.scale == "multiplicative" else is_log_space,
        main_effects=main_eff,
        pairwise=pair_eff,
    )


def _build_fused_coalition_evaluator(
    df_work: pd.DataFrame,
    model: object,
    feat_sorted: list[str],
    all_resample_vars: list[str],
    cfg: DecomposeConfig,
    target_index: pd.Index,
    *,
    norm_log_flag: bool,
    is_mult: bool,
    is_log_space: bool,
    n_cores_eff: int,
) -> Callable[[list[str]], np.ndarray]:
    """Precompute pool indices and base-column tiles once across all coalitions.

    Because Common Random Numbers depend only on ``seed`` and pool stream IDs
    (not on which subset of variables is resampled), precomputing the batch
    indices and base column tiles eliminates redundant ``process_date``,
    ``check_data``, stratification, and RNG sampling across ``2**k`` coalitions
    while producing predictions bit-identical to :func:`normalise`.
    """
    from ..utils.prepare import check_data

    df_dated = process_date(df_work)
    strat_cols = _parse_stratify_by(cfg.stratify_by)
    df_strata = _extract_strata(df_dated, strat_cols, label="df") if strat_cols else None
    df_checked = check_data(df_dated, feat_sorted, "value")
    if "date" not in df_checked.columns:
        raise DataError("`df` must contain a 'date' column.")

    resample_df = df_checked if cfg.resample_df is None else cfg.resample_df
    base_strata_source = df_dated if cfg.resample_df is None else resample_df
    if cfg.conditional_on:
        before = len(resample_df)
        cond_mask = np.asarray(_conditional_mask(resample_df, cfg.conditional_on), dtype=bool)
        resample_df = resample_df.loc[cond_mask]
        base_strata_source = (
            base_strata_source.loc[cond_mask] if cfg.resample_df is None else resample_df
        )
        if resample_df.empty:
            raise DataError(
                f"`conditional_on` filter left no rows in the resample pool (was {before}). "
                "Loosen the condition or pass a wider resample_df."
            )

    for name, pool in (cfg.resample_pools or {}).items():
        if isinstance(pool, pd.DataFrame) and not set(pool.columns) & set(feat_sorted):
            raise ConfigError(
                f"resample pool {name!r} has no covariate among its columns "
                f"({list(pool.columns)[:6]}); a pool's columns name the variables drawn from it."
            )

    import zlib

    draws_all = _split_across_pools(all_resample_vars, resample_df, cfg.resample_pools)
    base_cols = draws_all[0][1] if draws_all and draws_all[0][0] == 0 else []
    missing = [c for c in base_cols if c not in resample_df.columns]
    if missing:
        raise DataError(f"`resample_df` is missing columns required for resampling: {missing}")

    pool_strata_map: dict[int, np.ndarray] = {}
    if strat_cols:
        pool_strata_map[0] = _extract_strata(base_strata_source, strat_cols, label="resample_df")
        for name, pool in (cfg.resample_pools or {}).items():
            if isinstance(pool, pd.DataFrame):
                stream_id = zlib.crc32(str(name).encode("utf-8")) or 1
                pool_strata_map[stream_id] = _extract_strata(
                    pool, strat_cols, label=f"resample_pools[{name!r}]"
                )

    n_samples = int(cfg.n_samples or 300)
    rng = np.random.default_rng(cfg.seed)
    random_seeds = rng.choice(1_000_000, size=n_samples, replace=False)

    n_rows = len(df_checked)
    one_copy_bytes = int(df_checked.memory_usage(deep=True).sum())
    eff_batch_size = int(max(1, min((400 * 1024**2) // max(one_copy_bytes, 1), n_samples)))
    seeds_batched = [
        random_seeds[i : i + eff_batch_size] for i in range(0, n_samples, eff_batch_size)
    ]

    dates_arr = df_checked["date"].to_numpy()
    dates_unique = bool(pd.Index(dates_arr).is_unique)
    order_dates = np.argsort(dates_arr, kind="stable") if dates_unique else None
    sorted_idx = pd.Index(dates_arr[order_dates], name="date") if order_dates is not None else None

    # Precompute only the draw plan: for each batch and resampling stream, the
    # pool-row index of every (draw, row). Building the resampled columns here
    # instead meant retaining one full panel tile per batch for the lifetime of
    # the evaluator -- O(n_samples * n_rows * n_cols) up to 2x the batch budget
    # -- while the indices that define the draws are O(n_samples * n_rows).
    stream_specs: list[tuple[list[str], np.ndarray, int]] = [
        (list(cols), pool[cols].to_numpy(), stream) for stream, cols, pool in draws_all
    ]
    batch_draw_idx: list[list[np.ndarray]] = []
    sample_offset = 0
    for batch_seeds in seeds_batched:
        b_size = len(batch_seeds)
        batch_positions = list(range(sample_offset, sample_offset + b_size))
        sample_offset += b_size

        plans: list[np.ndarray] = []
        cycle_cache: dict[tuple[Any, ...], Any] = {}
        for _cols, values, stream in stream_specs:
            p_strata = pool_strata_map.get(stream)
            idx = np.empty((b_size, n_rows), dtype=np.int64)
            for i, (s, pos) in enumerate(zip(batch_seeds, batch_positions, strict=True)):
                idx[i] = _draw_sample_indices(
                    stream,
                    int(s),
                    base_seed=cfg.seed,
                    sample_pos=int(pos),
                    n_pool=len(values),
                    n_rows=n_rows,
                    replace=True,
                    df_strata=df_strata,
                    p_strata=p_strata,
                    cycle_cache=cycle_cache,
                )
            plans.append(idx)
        batch_draw_idx.append(plans)

    zero_pass_cache: np.ndarray | None = None

    def _eval_coalition(resample: list[str]) -> np.ndarray:
        nonlocal zero_pass_cache
        if not resample:
            if zero_pass_cache is None:
                preds = ml_predict(model, df_checked, num_threads=n_cores_eff)
                s_zero = (
                    pd.Series(preds, index=pd.Index(dates_arr, name="date")).groupby(level=0).mean()
                )
                vals_zero = s_zero.reindex(target_index).to_numpy(dtype=float)
                if norm_log_flag:
                    vals_zero = np.expm1(vals_zero)
                if is_mult and not is_log_space:
                    vals_zero = np.log1p(np.maximum(vals_zero, 0.0))
                zero_pass_cache = vals_zero
            return zero_pass_cache

        res_set = set(resample)
        sum_norm = np.zeros(n_rows, dtype=np.float64)
        n_completed = 0
        for b_idx, batch_seeds in enumerate(seeds_batched):
            b_size = len(batch_seeds)
            parts: dict[str, np.ndarray] = {
                c: np.tile(df_checked[c].to_numpy(), b_size)
                for c in df_checked.columns
                if c not in res_set
            }
            for s_idx, (cols, values, _stream) in enumerate(stream_specs):
                needed = [j for j, c in enumerate(cols) if c in res_set]
                if not needed:
                    continue
                picked = values[batch_draw_idx[b_idx][s_idx].ravel()]
                for j in needed:
                    parts[cols[j]] = picked[:, j]
            # A requested column with no stream keeps its factual values, as
            # before; build in the frame's column order so positional
            # backends see the same feature order they were trained on.
            for c in df_checked.columns:
                if c not in parts:
                    parts[c] = np.tile(df_checked[c].to_numpy(), b_size)
            df_batch = pd.DataFrame({c: parts[c] for c in df_checked.columns})
            batch_preds = ml_predict(model, df_batch, num_threads=n_cores_eff)
            sum_norm += batch_preds.reshape(b_size, n_rows).sum(axis=0)
            n_completed += b_size

        norm_raw_mean = sum_norm / float(n_completed)
        if dates_unique:
            assert order_dates is not None and sorted_idx is not None
            norm_out = norm_raw_mean[order_dates]
            if norm_log_flag:
                norm_out = np.expm1(norm_out)
            arr = pd.Series(norm_out, index=sorted_idx).reindex(target_index).to_numpy(dtype=float)
        else:
            s_out = (
                pd.Series(norm_raw_mean, index=pd.Index(dates_arr, name="date"))
                .groupby(level=0)
                .mean()
            )
            if norm_log_flag:
                s_out = pd.Series(np.expm1(s_out.to_numpy(dtype=float)), index=s_out.index)
            arr = s_out.reindex(target_index).to_numpy(dtype=float)
        if is_mult and not is_log_space:
            arr = np.log1p(np.maximum(arr, 0.0))
        return arr

    return _eval_coalition


def _run_met_attribution(
    df_work: pd.DataFrame,
    model: object,
    feat_sorted: list[str],
    players: list[_Player],
    method: str,
    cfg: DecomposeConfig,
    observed_conc: np.ndarray,
    is_log_space: bool,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[frozenset[int], np.ndarray]]:
    """Shared meteorological attribution engine for ``decom_met`` and ``shapley_interactions``."""
    result = (
        pd.DataFrame({"date": df_work["date"].to_numpy(), "observed": observed_conc})
        .set_index("date")
        .sort_index()
    )
    n_cores_eff = _effective_cores(cfg.n_cores)
    want_draws = bool(cfg.return_uncertainty)
    is_mult = cfg.scale == "multiplicative"
    norm_log_flag = bool(is_log_space and not is_mult)

    def _output_scale(values: np.ndarray) -> np.ndarray:
        """Map raw model outputs to the space the aggregate path reports."""
        if norm_log_flag:
            return np.expm1(values)
        if is_mult and not is_log_space:
            return np.log1p(np.maximum(values, 0.0))
        return values

    use_fused = (
        cfg.cache is None
        and cfg.conditional_k is None
        and not want_draws
        and not cfg.memory_save
        and int(cfg.n_samples or 0) > 0
        and normalise is _DEFAULT_NORMALISE
        and callable(getattr(model, "predict", None))
    )

    if use_fused:
        all_player_vars = [f for _, feats in players for f in feats]
        value_of = _build_fused_coalition_evaluator(
            df_work,
            model,
            feat_sorted,
            all_player_vars,
            cfg,
            result.index,
            norm_log_flag=norm_log_flag,
            is_mult=is_mult,
            is_log_space=is_log_space,
            n_cores_eff=n_cores_eff,
        )
    else:

        def value_of(resample: list[str]) -> np.ndarray:
            df_norm = normalise(
                df=df_work,
                model=model,
                covariates=feat_sorted,
                variables_resample=resample,
                n_samples=cfg.n_samples,
                replace=True,
                aggregate=not want_draws,
                seed=cfg.seed,
                n_cores=n_cores_eff,
                resample_df=cfg.resample_df,
                resample_pools=cfg.resample_pools,
                conditional_on=cfg.conditional_on,
                stratify_by=cfg.stratify_by,
                conditional_k=cfg.conditional_k,
                # The wide table is needed for per-seed draws; keep it on the
                # model's native scale so the point estimate can be aggregated
                # before the back-transform (see below), exactly as the
                # aggregate=True path does.
                log_transform=False if want_draws else norm_log_flag,
                memory_save=cfg.memory_save,
                cache=cfg.cache,
            )
            arr = _normet_draws_array(df_norm, result.index, want_draws=want_draws)
            if not want_draws:
                if is_mult and not is_log_space:
                    arr = np.log1p(np.maximum(arr, 0.0))
                return arr
            # ``return_uncertainty`` promises only to attach SE columns, so it
            # must not move the point estimates. The aggregate path maps the mean
            # of the raw draws to the output scale once; a wide table
            # back-transformed per seed would instead average expm1(draw), a
            # Jensen gap that survives more draws (measured ~7 % on emi_total).
            # Shift every coalition's draw matrix by a constant so its mean is
            # the aggregate-path point value; the spread across draws, and hence
            # every MC/permutation SE, is unchanged.
            raw = np.asarray(arr, dtype=float)
            point = _output_scale(raw.mean(axis=1))
            draws = _output_scale(raw)
            return draws + (point - draws.mean(axis=1))[:, None]

    raw_emi, raw_contribs, raw_values, perm_se = _attribute_core(
        players,
        value_of,
        method=method,
        n_permutations=cfg.n_permutations,
        seed=cfg.seed,
        verbose=cfg.verbose,
    )
    all_fixed = frozenset(range(len(players)))
    mean_values = {s: _mean_series(v) for s, v in raw_values.items()}
    mean_contribs = {name: _mean_series(arr) for name, arr in raw_contribs.items()}

    uncertainty_cols: dict[str, np.ndarray] | None = None
    factor_cols: dict[str, np.ndarray] | None = None

    if not is_mult:
        emi_out = mean_values[frozenset()]
        contribs_out = mean_contribs
        out_values = mean_values
        if want_draws:
            uncertainty_cols = {"emi_total_se": _mc_se(raw_emi)}
            for name, arr in raw_contribs.items():
                uncertainty_cols[f"{name}_se"] = np.hypot(_mc_se(arr), perm_se[name])
    else:
        z_emi = mean_values[frozenset()]
        z_pred = mean_values[all_fixed]
        emi_out = np.expm1(z_emi)
        pred_c = np.expm1(z_pred)
        w = _lmdi_weight(pred_c, emi_out)
        contribs_out = {name: w * dz for name, dz in mean_contribs.items()}
        factor_cols = {f"{name}_factor": np.exp(dz) for name, dz in mean_contribs.items()}
        out_values = {s: w * z_s for s, z_s in mean_values.items()}
        if want_draws:
            uncertainty_cols = {"emi_total_se": (1.0 + np.maximum(emi_out, 0.0)) * _mc_se(raw_emi)}
            for name, arr in raw_contribs.items():
                se_z = np.hypot(_mc_se(arr), perm_se[name])
                uncertainty_cols[f"{name}_se"] = np.abs(w) * se_z

    frame = _met_result(
        result,
        emi_out,
        contribs_out,
        include_diagnostics=cfg.include_diagnostics,
        uncertainty=uncertainty_cols,
        factors=factor_cols,
    )
    return frame, contribs_out, out_values


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
      step's change.
    - ``attribution="shapley"`` averages each feature's (or group's) marginal
      effect over every order it could be fixed in.

    Supports ``include_diagnostics=True`` (explicit ``predicted``,
    ``met_explained``, ``residual``), ``return_uncertainty=True`` (paired MC
    and permutation SEs), ``scale="multiplicative"`` (LMDI log-scale closure +
    ``_factor`` columns), ``stratify_by``, and ``conditional_k``.
    """
    _cfg = _resolve_config(config=config, **kwargs)
    _cfg.n_samples = resolve_n_samples(_cfg.n_samples, _cfg.backend)
    _attribution_method(_cfg)

    df_work, model, feat_sorted, observed_conc, is_log_space = _prepare_decomposition_frame(
        df, _cfg, model=model, drop_na_target=False, sort_by_importance=True
    )
    contrib_candidates = [f for f in feat_sorted if f not in _TIME_VARS]
    players, method = _attribution_plan(contrib_candidates, _cfg, default_order=list)

    frame, _contribs, _vals = _run_met_attribution(
        df_work,
        model,
        feat_sorted,
        players,
        method,
        _cfg,
        observed_conc,
        is_log_space,
    )
    return frame


def shapley_interactions(
    df: pd.DataFrame,
    model: object | None = None,
    *,
    config: DecomposeConfig | None = None,
    **kwargs: Any,
) -> ShapleyInteractionResult:
    """Decompose meteorological attribution into pure main effects and pairwise interactions.

    Computes exact Grabisch–Roubens second-order Shapley interaction indices
    ``I_ij(t)`` and adjusted main effects ``I_ii(t) = phi_i(t) - 0.5 * sum_{j!=i} I_ij(t)``
    from the ``2**k`` coalitions evaluated during exact Shapley attribution --
    requiring **zero** additional model predictions.

    Exact closure holds row-wise:
    ``main_effects.sum(axis=1) + pairwise.sum(axis=1) == decomposition["met_explained"]``.

    A model with only calendar features (no meteorological player) is valid:
    ``main_effects`` and ``pairwise`` then have zero columns and ``summary``
    is empty but still carries its standard columns (``term``, ``kind``,
    ``mean``, ``std``, ``mean_abs``, ``max_abs``).
    """
    _cfg = _resolve_config(config=config, **kwargs)
    _cfg.n_samples = resolve_n_samples(_cfg.n_samples, _cfg.backend)
    if _cfg.attribution not in (None, "shapley"):
        raise ConfigError("`shapley_interactions` requires attribution='shapley'.")
    if _cfg.n_permutations is not None:
        raise ConfigError(
            "`shapley_interactions` requires exact Shapley over all 2**k coalitions (`n_permutations=None`)."
        )
    _cfg.attribution = "shapley"
    _cfg.include_diagnostics = True
    _attribution_method(_cfg)

    df_work, model, feat_sorted, observed_conc, is_log_space = _prepare_decomposition_frame(
        df, _cfg, model=model, drop_na_target=False, sort_by_importance=True
    )
    contrib_candidates = [f for f in feat_sorted if f not in _TIME_VARS]
    players, method = _attribution_plan(contrib_candidates, _cfg, default_order=list)
    player_names = [p[0] for p in players]

    frame, contribs, coal_values = _run_met_attribution(
        df_work,
        model,
        feat_sorted,
        players,
        method,
        _cfg,
        observed_conc,
        is_log_space,
    )
    main_dict, pair_dict = _compute_interaction_indices(player_names, coal_values, contribs)
    main_df = pd.DataFrame(main_dict, index=frame.index)
    pair_df = pd.DataFrame(pair_dict, index=frame.index)

    k = len(player_names)
    mean_mat = pd.DataFrame(np.zeros((k, k), dtype=float), index=player_names, columns=player_names)
    std_mat = pd.DataFrame(np.zeros((k, k), dtype=float), index=player_names, columns=player_names)
    summary_rows: list[dict[str, Any]] = []

    for i, name in enumerate(player_names):
        arr = main_df[name].to_numpy(dtype=float)
        m_val, s_val = float(np.mean(arr)), float(np.std(arr))
        mean_mat.iloc[i, i] = m_val
        std_mat.iloc[i, i] = s_val
        summary_rows.append(
            {
                "term": name,
                "kind": "main",
                "mean": m_val,
                "std": s_val,
                "mean_abs": float(np.mean(np.abs(arr))),
                "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
            }
        )

    for i in range(k):
        for j in range(i + 1, k):
            pname = f"{player_names[i]}:{player_names[j]}"
            arr = pair_df[pname].to_numpy(dtype=float)
            m_val, s_val = float(np.mean(arr)), float(np.std(arr))
            mean_mat.iloc[i, j] = mean_mat.iloc[j, i] = m_val
            std_mat.iloc[i, j] = std_mat.iloc[j, i] = s_val
            summary_rows.append(
                {
                    "term": pname,
                    "kind": "interaction",
                    "mean": m_val,
                    "std": s_val,
                    "mean_abs": float(np.mean(np.abs(arr))),
                    "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
                }
            )

    summary_df = pd.DataFrame(summary_rows, columns=_INTERACTION_SUMMARY_COLUMNS)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["std", "mean_abs"], ascending=[False, False]
        ).reset_index(drop=True)
    return ShapleyInteractionResult(
        decomposition=frame,
        main_effects=main_df,
        pairwise=pair_df,
        mean_matrix=mean_mat,
        std_matrix=std_mat,
        summary=summary_df,
    )


def _impute_from_train(train: pd.DataFrame, frames: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
    """Fill predictor NaNs in ``frames`` with statistics learned on ``train``.

    :func:`normet.utils.prepare.impute_values` computes every median (and
    categorical mode) over the whole frame, so a temporally blocked
    cross-fit fold's training rows were filled with a value computed from its
    held-out block -- future information the split is meant to exclude, and
    the test block's own NaNs were filled from statistics that included
    themselves. Deriving the statistics from the training rows and applying
    them to every frame of the fold closes both leaks. A column that is all
    NaN in ``train`` keeps whatever fill pandas can do with a NaN median,
    matching ``impute_values``'s behaviour for an all-NaN column.
    """
    medians: dict[str, float] = {}
    for col in train.select_dtypes(include=[np.number]).columns:
        if train[col].isna().any():
            medians[str(col)] = float(train[col].median())
    modes: dict[str, Any] = {}
    for col in train.select_dtypes(include=["object", "category"]).columns:
        if train[col].isna().any():
            mode_series = train[col].mode(dropna=True)
            if not mode_series.empty:
                modes[str(col)] = mode_series.iloc[0]

    filled: list[pd.DataFrame] = []
    for frame in frames:
        out = frame.copy()
        for col, median in medians.items():
            if col not in out.columns:
                continue
            try:
                out[col] = out[col].fillna(median)
            except (TypeError, ValueError):
                # A nullable Int64 column only accepts a whole-number fill;
                # the column has to become float to carry a fractional
                # median (same handling as impute_values).
                out[col] = out[col].astype(float).fillna(median)
        for col, mode in modes.items():
            if col in out.columns:
                out[col] = out[col].fillna(mode)
        filled.append(out)
    return filled


def cross_fit_decompose(
    df: pd.DataFrame,
    *,
    target: str = "value",
    covariates: Sequence[str],
    backend: str = "lightgbm",
    method: str = "meteorology",
    n_folds: int = 5,
    block_freq: str | None = None,
    embargo_hours: float = 0.0,
    resample_pool: str = "full",
    config: DecomposeConfig | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Out-of-fold temporally blocked decomposition to prevent emission absorption.

    Trains one model per contiguous temporal block (excluding that block plus an
    optional ``embargo_hours`` buffer on either side) and runs ``decom_met``,
    ``decom_emi``, or ``decom_emi_shapley`` strictly out-of-sample on each held-out
    block before stitching the full record back together.
    """
    _cfg = _resolve_config(
        config=config, target=target, covariates=list(covariates), backend=backend, **kwargs
    )
    _cfg.n_samples = resolve_n_samples(_cfg.n_samples, _cfg.backend)
    _validate_shared_options(_cfg)

    if _cfg.backend == CHRONOS_BACKEND:
        raise ConfigError(
            "`cross_fit_decompose` applies to trained AutoML backends, not chronos-2."
        )
    if method not in ("emission", "meteorology", "emission_shapley"):
        raise ConfigError(
            f"Unsupported `method` {method!r}: expected 'meteorology', 'emission', or 'emission_shapley'."
        )
    if resample_pool not in ("full", "train"):
        raise ConfigError(f"`resample_pool` must be 'full' or 'train', got {resample_pool!r}.")
    if float(embargo_hours) < 0.0:
        raise ConfigError(f"`embargo_hours` must be non-negative, got {embargo_hours!r}.")
    if not _cfg.covariates:
        raise ConfigError("`covariates` must be non-empty.")
    if df is None:
        raise DataError("`df` must be provided.")
    if method == "meteorology":
        _attribution_method(_cfg)

    work = process_date(df.copy()) if "date" not in df.columns else df.copy()
    if "date" not in work.columns or _cfg.target not in work.columns:
        raise DataError(f"`df` must contain 'date' and target column {_cfg.target!r}.")
    if _cfg.target != "value":
        work = work.rename(columns={_cfg.target: "value"})
    work = (
        work.loc[work["date"].notna() & work["value"].notna()]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if work.empty:
        raise DataError("No valid dated observations available for `cross_fit_decompose`.")

    use_covs = [str(c) for c in _cfg.covariates]
    if any(v in use_covs and v not in work.columns for v in _TIME_VARS):
        work = add_date_variables(work)
    use_covs = [c for c in use_covs if c in work.columns]
    if not use_covs:
        raise DataError("No requested covariates found in `df`.")
    # Imputation happens per fold from the training rows only (see
    # _impute_from_train); a whole-record median would leak each held-out
    # block into every training fold.

    is_log_work = bool(_cfg.log_transform)
    work_fit = work.copy()
    if _cfg.scale == "multiplicative" and not is_log_work:
        if (work_fit["value"].to_numpy(dtype=float) < 0.0).any():
            raise DataError("Cannot apply scale='multiplicative' with negative target values.")
        work_fit = log1p_target(work_fit, "value")
        is_log_work = True

    folds: list[np.ndarray[Any, Any]]
    if block_freq is not None:
        folds = [
            grp.index.to_numpy(dtype=int)
            for _, grp in work_fit.groupby(pd.Grouper(key="date", freq=block_freq), sort=True)
            if not grp.empty
        ]
    else:
        if int(n_folds) < 2 or int(n_folds) > len(work_fit):
            raise ConfigError(
                f"`n_folds` must be between 2 and len(df) ({len(work_fit)}), got {n_folds}."
            )
        folds = [
            np.asarray(idx, dtype=int)
            for idx in np.array_split(np.arange(len(work_fit)), int(n_folds))
        ]

    if len(folds) < 2:
        raise DataError("Temporal partitioning produced fewer than 2 non-empty folds.")

    dates = pd.DatetimeIndex(work_fit["date"])
    delta = pd.Timedelta(hours=float(embargo_hours))
    n_cores_eff = _cfg.n_cores
    parts: list[pd.DataFrame] = []

    for f_idx, test_idx in enumerate(folds):
        t_lo, t_hi = dates[test_idx].min(), dates[test_idx].max()
        if float(embargo_hours) > 0.0:
            train_mask = np.asarray((dates < (t_lo - delta)) | (dates > (t_hi + delta)), dtype=bool)
        else:
            train_mask = np.ones(len(work_fit), dtype=bool)
            train_mask[test_idx] = False
        if not train_mask.any():
            raise DataError(
                f"Fold {f_idx} has zero training rows after embargo ({embargo_hours} h); "
                "reduce embargo_hours or increase data span."
            )

        df_train_raw = work_fit.iloc[train_mask].reset_index(drop=True)
        df_test_raw = work_fit.iloc[test_idx].reset_index(drop=True)
        if _cfg.resample_df is not None:
            # An external pool is used as given (the other decompositions do
            # not impute it either); only the fold frames need filling.
            df_train, df_test = _impute_from_train(df_train_raw, [df_train_raw, df_test_raw])
            fold_pool = _cfg.resample_df
        else:
            pool_raw = work_fit if resample_pool == "full" else df_train_raw
            df_train, df_test, fold_pool = _impute_from_train(
                df_train_raw, [df_train_raw, df_test_raw, pool_raw]
            )
        fold_model = train_model(
            df=df_train,
            target="value",
            backend=_cfg.backend or "lightgbm",
            covariates=use_covs,
            model_config=_cfg.model_config,
            seed=_cfg.seed + f_idx,
            verbose=_cfg.verbose,
            n_cores=n_cores_eff,
            cache=_cfg.cache,
        )
        fold_cfg = _resolve_config(
            config=_cfg,
            target="value",
            covariates=use_covs,
            resample_df=fold_pool,
            log_transform=is_log_work,
        )
        if method == "meteorology":
            part = decom_met(df_test, model=fold_model, config=fold_cfg)
        elif method == "emission":
            part = decom_emi(df_test, model=fold_model, config=fold_cfg)
        else:
            part = decom_emi_shapley(df_test, model=fold_model, config=fold_cfg)
        parts.append(part)

    stitched = pd.concat(parts, axis=0).sort_index()
    if method == "meteorology":
        stitched["met_total"] = stitched["observed"] - stitched["emi_total"]
        stitched["met_base"] = float(stitched["met_total"].mean())
        excluded = set(_MET_RESULT_COLUMNS) | set(_TIME_VARS)
        c_cols = [
            c
            for c in stitched.columns
            if c not in excluded and not str(c).endswith(("_se", "_factor"))
        ]
        c_sum = stitched[c_cols].sum(axis=1) if c_cols else 0.0
        stitched["met_noise"] = stitched["met_total"] - (stitched["met_base"] + c_sum)
        if _cfg.include_diagnostics:
            stitched["predicted"] = stitched["emi_total"] + c_sum
            stitched["met_explained"] = c_sum
            stitched["residual"] = stitched["observed"] - stitched["predicted"]
    elif "emi_base" in stitched.columns and "emi_noise" in stitched.columns:
        g_base = float(stitched["emi_base"].mean())
        stitched["emi_noise"] = stitched["emi_noise"] + (stitched["emi_base"] - g_base)
        stitched["emi_base"] = g_base
    return stitched


def _predict_tree_contrib(
    model: object, X: pd.DataFrame, feat_names: list[str], n_cores: int | None
) -> np.ndarray:
    """Extract raw TreeSHAP contributions ``(n_rows, n_features + 1)`` from supported models."""
    pred_fn = getattr(model, "predict_contrib", None)
    if callable(pred_fn):
        try:
            arr = np.asarray(pred_fn(X, num_threads=n_cores), dtype=float)
        except TypeError:
            arr = np.asarray(pred_fn(X), dtype=float)
    else:
        inner = (
            getattr(getattr(model, "model", None), "estimator", None)
            or getattr(model, "model", None)
            or model
        )
        booster = getattr(inner, "booster_", None) or getattr(inner, "_Booster", None)
        if booster is not None and callable(getattr(booster, "predict", None)):
            arr = np.asarray(
                booster.predict(X.to_numpy(dtype=float), pred_contrib=True), dtype=float
            )
        elif hasattr(inner, "get_booster"):
            import xgboost as xgb

            dm = xgb.DMatrix(X.to_numpy(dtype=float), feature_names=feat_names)
            arr = np.asarray(inner.get_booster().predict(dm, pred_contribs=True), dtype=float)
        else:
            inner_pred = getattr(inner, "predict", None)
            if inner.__class__.__name__.startswith("LGBM") and callable(inner_pred):
                arr = np.asarray(inner_pred(X, pred_contrib=True), dtype=float)
            else:
                raise ModelError(
                    f"Model type {type(model).__name__!r} does not expose TreeSHAP pred_contrib."
                )

    expected_shape = (len(X), len(feat_names) + 1)
    if arr.ndim != 2 or arr.shape != expected_shape:
        raise ModelError(
            f"TreeSHAP contribution shape mismatch: expected {expected_shape}, got {arr.shape}."
        )
    return arr


def decom_met_treeshap(
    df: pd.DataFrame,
    model: object,
    *,
    target: str = "value",
    groups: Mapping[str, Sequence[str]] | None = None,
    include_diagnostics: bool = True,
    n_cores: int | None = None,
) -> pd.DataFrame:
    """Observational path-dependent TreeSHAP meteorological attribution for tree models.

    Complements interventional Monte Carlo Shapley (:func:`decom_met`) by
    conditioning each tree split on the training joint path distribution, so
    highly collinear meteorological predictors receive on-manifold observational
    attributions. Use :func:`compare_attributions` to inspect discrepancies caused
    by collinear extrapolation.
    """
    if df is None:
        raise DataError("`df` must be provided.")
    work = process_date(df.copy()) if "date" not in df.columns else df.copy()
    if "date" not in work.columns or target not in work.columns:
        raise DataError(f"`df` must contain 'date' and target column {target!r}.")
    work = work.loc[work["date"].notna()].sort_values("date").reset_index(drop=True)

    try:
        feat_names = [str(c) for c in extract_features(model)]
    except Exception as exc:
        raise ModelError("Could not extract feature names from `model` for TreeSHAP.") from exc

    if any(v in feat_names and v not in work.columns for v in _TIME_VARS):
        work = add_date_variables(work)
    missing = [f for f in feat_names if f not in work.columns]
    if missing:
        raise DataError(f"`df` is missing model features required for TreeSHAP: {missing}")

    X = work[feat_names]
    arr = _predict_tree_contrib(model, X, feat_names, _effective_cores(n_cores))
    shap_by_feat = {f: arr[:, i] for i, f in enumerate(feat_names)}
    bias = arr[:, -1]

    emi_total = bias.copy()
    for t_var in feat_names:
        if t_var in _TIME_VARS:
            emi_total = emi_total + shap_by_feat[t_var]

    met_feats = [f for f in feat_names if f not in _TIME_VARS]
    if groups is not None:
        players = _players_from_groups(groups, met_feats)
        contributions = {
            name: np.sum([shap_by_feat[f] for f in g_feats], axis=0) for name, g_feats in players
        }
    else:
        contributions = {f: shap_by_feat[f] for f in met_feats}

    result = (
        pd.DataFrame(
            {"date": work["date"].to_numpy(), "observed": work[target].to_numpy(dtype=float)}
        )
        .set_index("date")
        .sort_index()
    )
    return _met_result(result, emi_total, contributions, include_diagnostics=include_diagnostics)


def compare_attributions(
    interventional: pd.DataFrame,
    conditional: pd.DataFrame,
    *,
    cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compare interventional vs. conditional/TreeSHAP attribution tables.

    Aligns both DataFrames on shared timestamps and computes per-component
    Pearson correlation, RMSE, MAE, mean difference, spread ratio, and normalised
    OOD discrepancy ``rmse / std(interventional)``.
    """
    a = interventional.set_index("date") if "date" in interventional.columns else interventional
    b = conditional.set_index("date") if "date" in conditional.columns else conditional
    shared_idx = a.index.intersection(b.index)
    if shared_idx.empty:
        raise DataError("`interventional` and `conditional` share no overlapping timestamps.")

    if cols is not None:
        comp_cols = [str(c) for c in cols]
    else:
        skip = set(_MET_RESULT_COLUMNS) | set(_TIME_VARS) | {"emi_base", "emi_noise"}
        comp_cols = [
            c
            for c in a.columns
            if c in b.columns
            and c not in skip
            and not str(c).endswith(("_se", "_factor", "_ci_low", "_ci_high"))
            and pd.api.types.is_numeric_dtype(a[c])
            and pd.api.types.is_numeric_dtype(b[c])
        ]
    if not comp_cols:
        raise DataError("No shared numeric contribution columns to compare.")

    rows: list[dict[str, Any]] = []
    for col in comp_cols:
        va = pd.to_numeric(a.loc[shared_idx, col], errors="coerce").to_numpy(dtype=float)
        vb = pd.to_numeric(b.loc[shared_idx, col], errors="coerce").to_numpy(dtype=float)
        fin = np.isfinite(va) & np.isfinite(vb)
        va_f, vb_f = va[fin], vb[fin]
        diff = va_f - vb_f
        std_a = float(np.std(va_f)) if va_f.size else float("nan")
        std_b = float(np.std(vb_f)) if vb_f.size else float("nan")
        rmse = float(np.sqrt(np.mean(diff**2))) if diff.size else float("nan")
        if va_f.size >= 2 and std_a > 1e-12 and std_b > 1e-12:
            corr = float(np.corrcoef(va_f, vb_f)[0, 1])
        else:
            corr = float("nan")
        rows.append(
            {
                "component": col,
                "pearson_r": corr,
                "rmse": rmse,
                "mae": float(np.mean(np.abs(diff))) if diff.size else float("nan"),
                "mean_diff": float(np.mean(diff)) if diff.size else float("nan"),
                "std_interventional": std_a,
                "std_conditional": std_b,
                "std_ratio": std_b / std_a
                if np.isfinite(std_a) and std_a > 1e-12
                else float("nan"),
                "ood_discrepancy": rmse / std_a
                if np.isfinite(std_a) and std_a > 1e-12
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("component")
