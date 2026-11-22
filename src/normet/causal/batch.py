# src/normet/causal/batch.py
"""Batch runner: :func:`scm_all` runs multiple SCM specs in parallel, and
:func:`run_scm_staggered` fits a staggered (multi-unit, multi-cutoff) design
with the same worker pool pattern."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd

from ..utils._config import dispatch_batch_size, resolve_workers
from ..utils._time import to_datetime_coerced
from ..utils.logging import get_logger
from .run_scm import run_scm

log = get_logger(__name__)

__all__ = ["run_scm_staggered", "scm_all"]


def scm_all(
    df: pd.DataFrame,
    *,
    date_col: str,
    outcome_col: str,
    unit_col: str,
    donors: list[str] | None,
    cutoff_date: str,
    scm_backend: str = "scm",
    n_cores: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Run synthetic-control for many candidate treated units (in parallel).

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel with at least {date_col, unit_col, outcome_col}.
    date_col : str
        Column name for dates (convertible to datetime).
    outcome_col : str
        Name of the outcome variable.
    unit_col : str
        Column containing unit identifiers.
    donors : Optional[List[str]]
        Donor pool. If None, use all units except the treated unit.
    cutoff_date : str
        Treatment cutoff in "YYYY-MM-DD" format.
    scm_backend : {"scm","mlscm"}, optional
        Synthetic-control backend to use:
          - "scm"   : Augmented SCM.
          - "mlscm" : ML-augmented SCM (requires ML backend via kwargs).
    n_cores : int | None, optional
        Number of parallel workers. Defaults to all cores - 1.
    **kwargs
        Forwarded to `run_scm`. Examples:
          - If `scm_backend="mlscm"`:
              * backend="flaml"         (ML backend; default "flaml")
              * model_config={...}      (AutoML config)
              * seed=123                (random seed)
          - If `scm_backend="scm"`:
              * pre_covariates=[...]
              * allow_negative_weights=True

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with columns:
          - date_col
          - observed, synthetic, effect
          - unit_col (treated unit ID)
    """
    units = sorted(pd.unique(df[unit_col]))
    n_cores_eff = resolve_workers(n_cores)

    ml_backend_for_log = kwargs.get("backend") if scm_backend == "mlscm" else None
    log.info(
        "scm_all: scm_backend=%s | ml_backend=%s | cutoff=%s | units=%d | n_cores=%d",
        scm_backend,
        ml_backend_for_log,
        cutoff_date,
        len(units),
        n_cores_eff,
    )

    def _one(code: str) -> pd.DataFrame | None:
        try:
            donors_u = donors if donors is not None else [u for u in units if u != code]
            syn = run_scm(
                df=df,
                date_col=date_col,
                outcome_col=outcome_col,
                unit_col=unit_col,
                treated_unit=code,
                donors=donors_u,
                cutoff_date=cutoff_date,
                scm_backend=scm_backend,
                **kwargs,  # may include backend="flaml" if mlscm
            )
            out = syn.copy()
            out[unit_col] = code
            out = out.reset_index().rename(columns={syn.index.name or "index": date_col})
            return out
        except Exception as e:
            log.warning("%s failed for unit %s: %s", scm_backend, code, e)
            return None

    # joblib is ~100 ms of `import normet`; the parallel branches pay it.
    from joblib import Parallel, delayed

    pieces = Parallel(n_jobs=n_cores_eff, batch_size=dispatch_batch_size(len(units), n_cores_eff))(
        delayed(_one)(code) for code in units
    )
    pieces = [p for p in pieces if p is not None]
    if not pieces:
        raise RuntimeError("All synthetic-control runs failed.")

    return pd.concat(pieces, ignore_index=True)


def _staggered_cutoffs(
    treated_units: list[str], cutoff_dates: list[str] | dict[str, str]
) -> list[tuple[str, pd.Timestamp]]:
    """Normalise ``cutoff_dates`` to the panel's units: a dict keyed by unit,
    or a list aligned with ``treated_units``. Raises with the offending units."""
    if isinstance(cutoff_dates, dict):
        missing = [u for u in treated_units if u not in cutoff_dates]
        if missing:
            raise ValueError(f"cutoff_dates is missing entries for treated units: {missing}")
        pairs = [(u, pd.to_datetime(cutoff_dates[u])) for u in treated_units]
    elif isinstance(cutoff_dates, (list, tuple)):
        if len(cutoff_dates) != len(treated_units):
            raise ValueError(
                f"cutoff_dates as a list must align with treated_units "
                f"({len(cutoff_dates)} != {len(treated_units)})."
            )
        pairs = [(u, pd.to_datetime(c)) for u, c in zip(treated_units, cutoff_dates, strict=True)]
    else:
        raise ValueError(
            "cutoff_dates must be a list aligned with treated_units, "
            "or a dict mapping unit to cutoff date."
        )
    return pairs


_NS_PER_DAY = 86_400_000_000_000.0


def _panel_step_days(dates: pd.Series | pd.Index) -> float:
    """Median spacing between distinct panel dates, in days.

    Event time is measured in panel periods, not dataframe rows: a shared
    date gap must add its length to the event index instead of consuming one
    row, or units whose cutoff sits at a different distance from the gap are
    pooled at different calendar lags. The median spacing is robust both to an
    occasional gap and to the ragged lengths of a monthly panel.
    """
    index = pd.DatetimeIndex(pd.unique(pd.to_datetime(dates)))
    ns = np.asarray(index, dtype="datetime64[ns]").astype(np.int64)
    uniq = np.unique(np.sort(ns))
    if uniq.size < 2:
        return 1.0
    diffs = np.diff(uniq).astype(float) / _NS_PER_DAY
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return 1.0
    step = float(np.median(positive))
    return step if step > 0.0 else 1.0


def _period_positions(index: pd.DatetimeIndex, step_days: float) -> np.ndarray:
    """Integer panel-period position of each sorted date.

    Consecutive dates advance by ``round(gap / step_days)`` periods, so a
    daily panel with an eleven-day gap advances eleven periods. Accumulating
    whole periods (rather than dividing each date's distance from an origin)
    keeps monthly panels aligned despite their 28-31 day spacings.
    """
    ns = np.diff(np.asarray(index, dtype="datetime64[ns]").astype(np.int64)).astype(float)
    counts = np.rint(ns / (_NS_PER_DAY * float(step_days))).astype(int)
    counts = np.maximum(counts, 0)
    return np.concatenate(([0], np.cumsum(counts)))


def _rel_event_series(
    syn: pd.DataFrame, cutoff_ts: pd.Timestamp, step_days: float = 1.0
) -> pd.Series:
    """Re-index an effect path onto event time in panel periods (``0 = anchor``).

    The anchor is the first observed post-cutoff date, as before; from there
    dates are labelled by calendar/period distance rather than row count, so a
    shared date gap does not shift the labels of later events.
    """
    s = syn.sort_index()
    idx = pd.to_datetime(s.index)
    post_pos = np.nonzero(np.asarray(idx >= cutoff_ts, dtype=bool))[0]
    if post_pos.size == 0:
        return pd.Series(dtype=float, name="effect")
    t0 = int(post_pos[0])
    pos = _period_positions(idx, step_days)
    rel_idx = pd.Index((pos - pos[t0]).astype(int), name="event_time")
    return pd.Series(s["effect"].to_numpy(dtype=float), index=rel_idx, name="effect")


def _donor_perm_tuples(
    donor_pool: list[str],
    n_slots: int,
    *,
    max_permutations: int,
    random_state: int,
) -> list[tuple[str, ...]]:
    """Build distinct ``n_slots``-tuples of donors for the cross-unit permutation test."""
    n_donors = len(donor_pool)
    if n_slots == 1:
        return [(u,) for u in donor_pool]
    if n_donors >= n_slots:
        total = math.perm(n_donors, n_slots)
        if total <= int(max_permutations):
            return list(itertools.permutations(donor_pool, n_slots))
        rng = np.random.default_rng(random_state)
        arr = np.asarray(donor_pool, dtype=object)
        seen: set[tuple[str, ...]] = set()
        tuples_out: list[tuple[str, ...]] = []
        max_tries = int(max_permutations) * 4
        tries = 0
        while len(tuples_out) < int(max_permutations) and tries < max_tries:
            tries += 1
            cand = tuple(str(x) for x in arr[rng.choice(n_donors, size=n_slots, replace=False)])
            if cand not in seen:
                seen.add(cand)
                tuples_out.append(cand)
        return tuples_out

    log.warning(
        "run_scm_staggered: fewer donors (%d) than treated units (%d); "
        "sampling cross-unit permutation tuples with replacement.",
        n_donors,
        n_slots,
    )
    if n_donors == 1:
        return [(donor_pool[0],) * n_slots]
    total_rep = n_donors**n_slots
    if total_rep <= int(max_permutations):
        return list(itertools.product(donor_pool, repeat=n_slots))
    rng = np.random.default_rng(random_state)
    arr = np.asarray(donor_pool, dtype=object)
    return [
        tuple(str(x) for x in arr[rng.integers(0, n_donors, size=n_slots)])
        for _ in range(int(max_permutations))
    ]


def run_scm_staggered(
    df: pd.DataFrame,
    *,
    treated_units: list[str],
    cutoff_dates: list[str] | dict[str, str],
    date_col: str,
    unit_col: str,
    outcome_col: str,
    donors: list[str] | None = None,
    scm_backend: str = "scm",
    post_agg: str = "mean",
    include_not_yet_treated: bool = False,
    horizon: int | None = None,
    max_permutations: int = 2000,
    random_state: int = 7_654_321,
    n_cores: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Fit a staggered design: several treated units, each with its own cutoff.

    Every treated unit gets its own :func:`run_scm` fit on its own pre- and
    post-period with later-treated units either excluded (default) or safely
    admitted only when still untreated throughout the focal evaluation window
    (``include_not_yet_treated=True`` with optional ``horizon``).

    The ``mean`` summary row's ``p_value`` uses the cross-unit ``G``-tuple
    permutation test (drawing ``G`` distinct donors without replacement across
    the ``G`` treated cutoffs rather than reusing a single donor ``G`` times,
    which previously over-estimated the null variance by ~sqrt(G)). A dynamic
    event-study trajectory ATT(e) with cross-unit permutation CIs and
    p-values is attached at ``result.attrs["event_study"]`` (and per-unit fit
    frames at ``result.attrs["unit_effects"]``).

    Event time is measured in panel periods from the first observed
    post-cutoff date, not in dataframe rows, and ``horizon`` likewise counts
    periods. On a regular grid this is the familiar row alignment; on a panel
    with a shared date gap it keeps event ``e`` at the same calendar lag for
    every unit (otherwise the pooled ATT(e) mixes lags across units). When
    ``include_not_yet_treated=True``, a placebo fit uses the same donor pool
    as the real fit for its slot, minus the placebo unit itself, so the
    permutation p-values and bands are a randomization test of the estimator
    that produced the reported effect.
    """
    if not treated_units:
        raise ValueError("treated_units must not be empty.")
    if len(set(treated_units)) != len(treated_units):
        raise ValueError("treated_units must be unique.")
    if horizon is not None and int(horizon) < 1:
        raise ValueError(f"horizon must be >= 1 or None, got {horizon}")
    if int(max_permutations) < 1:
        raise ValueError(f"max_permutations must be >= 1, got {max_permutations}")

    d = df.copy()
    d[date_col] = to_datetime_coerced(d[date_col])
    all_units = [u for u in sorted(pd.unique(d[unit_col]))]
    missing = [u for u in treated_units if u not in all_units]
    if missing:
        raise ValueError(f"treated units not in the panel: {missing}")
    pairs = _staggered_cutoffs(treated_units, cutoff_dates)

    treated_set = set(treated_units)
    if donors is not None:
        overlap = sorted(set(donors) & treated_set)
        if overlap:
            raise ValueError(f"donors must not include treated units: {overlap}")
        donor_pool = [u for u in sorted(set(donors)) if u in all_units]
    else:
        donor_pool = [u for u in all_units if u not in treated_set]
    if not donor_pool:
        raise ValueError("No donor units left after excluding the treated units.")

    post_agg = (post_agg or "mean").lower()
    if post_agg not in {"mean", "sum"}:
        log.warning("Invalid post_agg=%s; falling back to 'mean'.", post_agg)
        post_agg = "mean"

    unit_dates = {u: pd.to_datetime(sub).sort_values() for u, sub in d.groupby(unit_col)[date_col]}
    panel_step_days = _panel_step_days(d[date_col])
    no_post = [u for u, c in pairs if not (unit_dates[u] >= c).any()]
    if no_post:
        raise ValueError(
            "No post-cutoff observations at/after the cutoff for treated units: "
            f"{[(u, str(pd.to_datetime(c).date())) for u, c in pairs if u in no_post]}"
        )

    n_cores_eff = resolve_workers(n_cores)
    log.info(
        "run_scm_staggered: scm_backend=%s | treated=%d | donors=%d | post_agg=%s | n_cores=%d",
        scm_backend,
        len(pairs),
        len(donor_pool),
        post_agg,
        n_cores_eff,
    )

    slot_specs: list[tuple[str, pd.Timestamp, str, pd.Timestamp | None, list[str]]] = []
    use_slice = bool(horizon is not None or include_not_yet_treated)
    for unit, c_ts in pairs:
        u_post = unit_dates[unit][unit_dates[unit] >= c_ts]
        if horizon is not None:
            # The horizon counts panel periods (events 0..H-1), matching the
            # event-study labels; on a regular grid this is the H-th row.
            post_pos = _period_positions(pd.DatetimeIndex(u_post), panel_step_days)
            last = int(np.searchsorted(post_pos, int(horizon) - 1, side="right") - 1)
            u_end = pd.Timestamp(u_post.iloc[max(last, 0)])
        else:
            u_end = pd.Timestamp(unit_dates[unit].iloc[-1])
        pool_u = list(donor_pool)
        if include_not_yet_treated:
            nyt = [v for v, c_v in pairs if v != unit and c_v > u_end]
            pool_u.extend(nyt)
        slot_specs.append(
            (unit, c_ts, c_ts.strftime("%Y-%m-%d"), u_end if use_slice else None, pool_u)
        )

    def _read(
        syn: pd.DataFrame, c_ts: pd.Timestamp, end_ts: pd.Timestamp | None
    ) -> tuple[float, int, int, pd.Series]:
        syn_sub = syn[syn.index.to_series() <= end_ts] if end_ts is not None else syn
        post = syn_sub[syn_sub.index.to_series() >= c_ts]
        pre = syn_sub[syn_sub.index.to_series() < c_ts]
        stat = float(post["effect"].mean() if post_agg == "mean" else post["effect"].sum())
        rel = _rel_event_series(syn_sub, c_ts, panel_step_days)
        return stat, len(pre), len(post), rel

    def _one(
        unit: str,
        cutoff: str,
        end_ts: pd.Timestamp | None,
        pool: list[str],
        kind: str,
        key_tag: Any,
    ) -> tuple[str, Any, float, float, int, int, pd.Series, pd.DataFrame | None] | None:
        df_sub = d.loc[d[date_col] <= end_ts] if end_ts is not None else df
        try:
            syn = run_scm(
                df=df_sub,
                date_col=date_col,
                unit_col=unit_col,
                outcome_col=outcome_col,
                treated_unit=unit,
                cutoff_date=cutoff,
                donors=pool,
                scm_backend=scm_backend,
                **kwargs,
            )
        except Exception as e:
            log.warning(
                "Staggered %s run failed for unit %s (cutoff %s): %s", kind, unit, cutoff, e
            )
            return None
        c_ts = pd.Timestamp(cutoff)
        stat, n_pre, n_post, rel = _read(syn, c_ts, end_ts)
        if kind != "real":
            return (kind, key_tag, stat, float("nan"), n_pre, n_post, rel, None)
        syn_eval = syn[syn.index.to_series() <= end_ts] if end_ts is not None else syn
        pre = syn_eval[syn_eval.index.to_series() < c_ts]
        pre_r2 = float("nan")
        if len(pre) >= 2:
            y = pre["observed"].to_numpy(dtype=float)
            x = pre["synthetic"].to_numpy(dtype=float)
            ss_res = float(np.sum((y - x) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            pre_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return (kind, key_tag, stat, pre_r2, n_pre, n_post, rel, syn_eval)

    from joblib import Parallel, delayed

    jobs = [
        delayed(_one)(u, c_str, end_ts, pool_u, "real", u)
        for u, _c_ts, c_str, end_ts, pool_u in slot_specs
    ]
    # Deduplicate placebo tasks that share identical (unit, cutoff_str, end_ts,
    # donor pool). The pool is part of the key because include_not_yet_treated
    # gives each slot its own pool.
    unique_plc_keys: dict[tuple[str, str, pd.Timestamp | None, tuple[str, ...]], list[int]] = {}
    for s_idx, (_u, _c_ts, c_str, end_ts, pool_u) in enumerate(slot_specs):
        for pu in donor_pool:
            unique_plc_keys.setdefault((pu, c_str, end_ts, tuple(pool_u)), []).append(s_idx)

    plc_specs = list(unique_plc_keys.items())
    for idx_k, ((pu, c_str, end_ts, pool_key), _s_indices) in enumerate(plc_specs):
        # The placebo must refit the same design as the real slot, excluding
        # only the unit now playing the treated role; otherwise the permutation
        # null is for a different (smaller) donor pool.
        jobs.append(
            delayed(_one)(pu, c_str, end_ts, [x for x in pool_key if x != pu], "placebo", idx_k)
        )

    results = Parallel(n_jobs=n_cores_eff, batch_size=dispatch_batch_size(len(jobs), n_cores_eff))(
        jobs
    )

    real: dict[str, tuple[float, float, int, int, pd.Series, pd.DataFrame]] = {}
    placebo: dict[tuple[str, int], float] = {}
    plc_rel: dict[tuple[str, int], pd.Series] = {}
    for r in results:
        if r is None:
            continue
        kind, tag, stat, pre_r2, n_pre_r, n_post_r, rel, syn_eval = r
        if kind == "real" and syn_eval is not None:
            real[str(tag)] = (stat, pre_r2, n_pre_r, n_post_r, rel, syn_eval)
        elif kind == "placebo":
            (pu, _c_str, _end_ts, _pool_key), s_indices = plc_specs[int(tag)]
            for s_idx in s_indices:
                placebo[(pu, s_idx)] = stat
                plc_rel[(pu, s_idx)] = rel

    if len(real) != len(pairs):
        failed = [u for u, _ in pairs if u not in real]
        raise RuntimeError(f"Staggered fits failed for treated units: {failed}")

    def _perm_p(obs: float, refs: list[float]) -> float:
        refs_f = [r for r in refs if np.isfinite(r)]
        if not refs_f or not np.isfinite(obs):
            return float("nan")
        return float((np.sum(np.abs(np.asarray(refs_f)) >= np.abs(obs)) + 1) / (len(refs_f) + 1))

    rows = []
    unit_stats: list[float] = []
    unit_effect_frames: dict[str, pd.DataFrame] = {}
    for s_idx, (unit, c_ts, c_str, _end_ts, _pool_u) in enumerate(slot_specs):
        stat, pre_r2, n_pre_u, n_post_u, _rel, syn_eval = real[unit]
        unit_effect_frames[unit] = syn_eval
        refs = [placebo[(pu, s_idx)] for pu in donor_pool if (pu, s_idx) in placebo]
        rows.append(
            {
                "unit": unit,
                "cutoff_date": c_str,
                "n_pre": n_pre_u,
                "n_post": n_post_u,
                "pre_r2": pre_r2,
                "effect": stat,
                "p_value": _perm_p(stat, refs),
            }
        )
        unit_stats.append(stat)

    stat_obs = float(np.mean(unit_stats) if post_agg == "mean" else np.sum(unit_stats))
    perm_tuples = _donor_perm_tuples(
        donor_pool, len(pairs), max_permutations=max_permutations, random_state=random_state
    )
    pooled: list[float] = []
    for tup in perm_tuples:
        vals_t = [placebo[(tup[k], k)] for k in range(len(pairs)) if (tup[k], k) in placebo]
        if len(vals_t) == len(pairs):
            pooled.append(float(np.mean(vals_t) if post_agg == "mean" else np.sum(vals_t)))

    rows.append(
        {
            "unit": "mean",
            "cutoff_date": None,
            "n_pre": None,
            "n_post": None,
            "pre_r2": None,
            "effect": stat_obs,
            "p_value": _perm_p(stat_obs, pooled),
        }
    )
    rows.append(
        {
            "unit": "median",
            "cutoff_date": None,
            "n_pre": None,
            "n_post": None,
            "pre_r2": None,
            "effect": float(np.median(unit_stats)),
            "p_value": float("nan"),
        }
    )

    out_df = pd.DataFrame(
        rows, columns=["unit", "cutoff_date", "n_pre", "n_post", "pre_r2", "effect", "p_value"]
    )

    # Dynamic Event-Study ATT(e) aligned around e = 0.
    real_rel_list = [real[u][4] for u, _ in pairs]
    all_e = sorted({int(e) for s in real_rel_list for e in s.index})
    if all_e:
        e_index = pd.Index(all_e, name="event_time")
        real_mat = np.column_stack(
            [s.reindex(e_index).to_numpy(dtype=float) for s in real_rel_list]
        )
        u_to_col = {u: i for i, u in enumerate(donor_pool)}
        n_slots = len(pairs)
        m_plc = np.full((n_slots, len(all_e), len(donor_pool)), np.nan, dtype=float)
        for (pu, s_idx), rel_s in plc_rel.items():
            if pu in u_to_col and not rel_s.empty:
                m_plc[s_idx, :, u_to_col[pu]] = rel_s.reindex(e_index).to_numpy(dtype=float)

        perm_idx = (
            np.array([[u_to_col[u] for u in tup] for tup in perm_tuples], dtype=int)
            if perm_tuples
            else np.empty((0, n_slots), dtype=int)
        )
        es_rows: list[dict[str, Any]] = []
        for ei, e_val in enumerate(all_e):
            r_row = real_mat[ei]
            active = np.flatnonzero(np.isfinite(r_row))
            if active.size == 0:
                continue
            r_act = r_row[active]
            att_e = float(np.mean(r_act))
            n_u = int(active.size)
            se_e = float(np.std(r_act, ddof=1) / np.sqrt(n_u)) if n_u >= 2 else float("nan")

            if perm_idx.shape[0] > 0:
                p_vals_act = np.column_stack([m_plc[k, ei, perm_idx[:, k]] for k in active])
                valid_p = np.all(np.isfinite(p_vals_act), axis=1)
                perm_e = p_vals_act[valid_p].mean(axis=1)
            else:
                perm_e = np.empty(0, dtype=float)

            if perm_e.size > 0:
                p_e = float((np.sum(np.abs(perm_e) >= abs(att_e)) + 1.0) / (perm_e.size + 1.0))
                ci_lo = float(att_e - np.quantile(perm_e, 0.975))
                ci_hi = float(att_e - np.quantile(perm_e, 0.025))
            else:
                p_e = float("nan")
                ci_lo = att_e - 1.96 * se_e if np.isfinite(se_e) else float("nan")
                ci_hi = att_e + 1.96 * se_e if np.isfinite(se_e) else float("nan")

            es_rows.append(
                {
                    "event_time": int(e_val),
                    "att": att_e,
                    "se": se_e,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "p_value": p_e,
                    "n_units": n_u,
                }
            )
        es_df = pd.DataFrame(es_rows).set_index("event_time")
    else:
        es_df = pd.DataFrame(columns=["att", "se", "ci_low", "ci_high", "p_value", "n_units"])
        es_df.index.name = "event_time"

    out_df.attrs["event_study"] = es_df
    out_df.attrs["unit_effects"] = unit_effect_frames
    return out_df
