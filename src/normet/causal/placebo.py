# src/normet/causal/placebo.py
"""Placebo tests for synthetic-control results: :func:`placebo_in_space` and :func:`placebo_in_time`."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..utils.logging import get_logger
from .run_scm import run_scm

log = get_logger(__name__)

__all__ = ["placebo_in_space", "placebo_in_time"]


def _eff_cores(n_cores: int | None) -> int:
    """Resolve effective worker count (>=1)."""
    return max(1, int(n_cores) if n_cores is not None else (os.cpu_count() or 2) - 1)


def placebo_in_space(
    df: pd.DataFrame,
    *,
    date_col: str,
    unit_col: str,
    outcome_col: str,
    treated_unit: str,
    cutoff_date: str,
    donors: list[str] | None = None,
    scm_backend: str = "scm",
    post_agg: str = "mean",  # {'mean','sum'}
    n_cores: int | None = None,
    **kwargs,
) -> dict:
    """Placebo-in-space analysis for a synthetic-control backend (SCM or ML-SCM).

    Each donor in turn is treated as a pseudo-treated unit and given a synthetic
    control built from the *remaining donors only* — the real ``treated_unit`` is
    excluded from every placebo's donor pool so its post-cutoff intervention
    cannot leak into the placebo fits (standard Abadie placebo-in-space). The
    resulting placebo effects form the reference distribution the true effect is
    ranked against for the permutation ``p_value``.

    Each donor's placebo run is independent of the others, so they are
    dispatched in parallel via :mod:`joblib` (see ``n_cores``), the same
    pattern used by :func:`placebo_in_time` and :func:`normet.causal.batch.scm_all`.
    """
    cutoff_ts = pd.to_datetime(cutoff_date)

    post_agg = (post_agg or "mean").lower()
    if post_agg not in {"mean", "sum"}:
        log.warning("Invalid post_agg=%s; falling back to 'mean'.", post_agg)
        post_agg = "mean"

    ml_backend = kwargs.get("backend") if scm_backend == "mlscm" else None
    log.info(
        "Placebo-in-space: scm_backend=%s | backend=%s | treated=%s | cutoff=%s",
        scm_backend,
        ml_backend,
        treated_unit,
        cutoff_date,
    )

    # True treated run
    df_true = run_scm(
        df=df,
        date_col=date_col,
        unit_col=unit_col,
        outcome_col=outcome_col,
        treated_unit=treated_unit,
        cutoff_date=cutoff_ts.strftime("%Y-%m-%d"),
        donors=donors,
        scm_backend=scm_backend,
        **kwargs,
    )

    all_units = sorted(pd.unique(df[unit_col]))
    if donors is None:
        donor_pool = [u for u in all_units if u != treated_unit]
    else:
        donor_pool = [u for u in sorted(set(donors)) if u in all_units and u != treated_unit]

    if not donor_pool:
        log.warning("No donor units available for placebo-in-space.")
        empty_band = pd.DataFrame(
            index=df_true.index,
            columns=["p10", "p90", "p2_5", "p97_5", "mean", "std", "band_low_1sd", "band_high_1sd"],
        )
        return {"treated": df_true, "placebos": {}, "p_value": float("nan"), "ref_band": empty_band}

    # Run placebos in parallel — each donor-as-treated run is independent.
    n_cores_eff = _eff_cores(n_cores)

    def _one_donor_placebo(u: str) -> tuple[str, pd.DataFrame] | None:
        try:
            # Build each placebo's synthetic from the *other donors only*: the
            # real treated unit is excluded so its post-cutoff intervention
            # cannot leak into the placebo fits and inflate their effects
            # (standard Abadie placebo-in-space). ``u`` itself is also excluded.
            syn_u = run_scm(
                df=df,
                date_col=date_col,
                unit_col=unit_col,
                outcome_col=outcome_col,
                treated_unit=u,
                cutoff_date=cutoff_ts.strftime("%Y-%m-%d"),
                donors=[d for d in all_units if d != u and d != treated_unit],
                scm_backend=scm_backend,
                **kwargs,
            )
            return u, syn_u[["effect"]].rename(columns={"effect": u})
        except Exception as e:
            log.warning("Placebo failed for unit %s: %s", u, e)
            return None

    results = Parallel(n_jobs=n_cores_eff)(delayed(_one_donor_placebo)(u) for u in donor_pool)
    placebo_effects: dict[str, pd.DataFrame] = {
        u: eff for r in results if r is not None for u, eff in [r]
    }

    if not placebo_effects:
        log.warning("All placebo runs failed.")
        empty_band = pd.DataFrame(
            index=df_true.index,
            columns=["p10", "p90", "p2_5", "p97_5", "mean", "std", "band_low_1sd", "band_high_1sd"],
        )
        return {"treated": df_true, "placebos": {}, "p_value": float("nan"), "ref_band": empty_band}

    placebo_mat = pd.concat(placebo_effects.values(), axis=1).reindex(df_true.index)

    ref_band = pd.DataFrame(
        {
            "p10": placebo_mat.quantile(0.10, axis=1),
            "p90": placebo_mat.quantile(0.90, axis=1),
            "p2_5": placebo_mat.quantile(0.025, axis=1),
            "p97_5": placebo_mat.quantile(0.975, axis=1),
            "mean": placebo_mat.mean(axis=1),
            "std": placebo_mat.std(axis=1, ddof=1),
        },
        index=placebo_mat.index,
    )
    ref_band["band_low_1sd"] = ref_band["mean"] - ref_band["std"]
    ref_band["band_high_1sd"] = ref_band["mean"] + ref_band["std"]

    post_mask = df_true.index.to_series() >= cutoff_ts
    if not post_mask.any():
        p_value = float("nan")
    else:
        if post_agg == "sum":
            obs_stat = float(df_true["effect"][post_mask].sum())
            plc_stats = placebo_mat[post_mask].sum(axis=0, skipna=True)
        else:
            obs_stat = float(df_true["effect"][post_mask].mean())
            plc_stats = placebo_mat[post_mask].mean(axis=0, skipna=True)

        p_value = (np.sum(np.abs(plc_stats.values) >= np.abs(obs_stat)) + 1) / (len(plc_stats) + 1)

    return {
        "treated": df_true,
        "placebos": placebo_effects,
        "p_value": float(p_value),
        "ref_band": ref_band,
    }


def placebo_in_time(
    df: pd.DataFrame,
    *,
    date_col: str,
    unit_col: str,
    outcome_col: str,
    treated_unit: str,
    cutoff_date: str,
    donors: list[str] | None = None,
    scm_backend: str = "scm",
    post_agg: str = "mean",  # {'mean','sum'}
    min_pre_period: int = 30,
    placebo_every: int = 7,
    n_cores: int | None = None,
    **kwargs,
) -> dict:
    """Placebo-in-time analysis for a synthetic-control backend (SCM or ML-SCM)."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    if d[date_col].isna().any():
        raise ValueError("Some rows have invalid dates after coercion.")
    cutoff_dt = pd.to_datetime(cutoff_date)

    if treated_unit not in pd.unique(d[unit_col]):
        raise ValueError(f"Treated unit '{treated_unit}' not found in data.")

    post_agg = (post_agg or "mean").lower()
    if post_agg not in {"mean", "sum"}:
        log.warning("Invalid post_agg=%s; falling back to 'mean'.", post_agg)
        post_agg = "mean"

    all_units = sorted(pd.unique(d[unit_col]))
    if donors is None:
        donors = [u for u in all_units if u != treated_unit]
    else:
        donors = [u for u in sorted(set(donors)) if u in all_units and u != treated_unit]
    if not donors:
        raise ValueError("No valid donors after excluding the treated unit.")

    df_true = run_scm(
        df=d,
        date_col=date_col,
        unit_col=unit_col,
        outcome_col=outcome_col,
        treated_unit=treated_unit,
        cutoff_date=cutoff_dt.strftime("%Y-%m-%d"),
        donors=donors,
        scm_backend=scm_backend,
        **kwargs,
    )
    dates_all = df_true.index.sort_values()
    post_dates_true = dates_all[(dates_all >= cutoff_dt).tolist()]
    post_len = len(post_dates_true)
    if post_len == 0:
        raise ValueError("No post-period observations at/after the true cutoff date.")

    treated_dates = d.loc[d[unit_col] == treated_unit, date_col].sort_values().unique()

    placebo_candidates = []
    for i in range(min_pre_period, len(treated_dates) - post_len):
        pc = pd.to_datetime(treated_dates[i])
        if pc >= cutoff_dt:
            break
        idx_end = i + post_len - 1
        last_date = pd.to_datetime(treated_dates[idx_end])
        if last_date < cutoff_dt:
            placebo_candidates.append(pc)

    if placebo_every > 1 and placebo_candidates:
        placebo_candidates = placebo_candidates[::placebo_every]

    if not placebo_candidates:
        log.warning("No valid placebo cutoffs found before the true cutoff.")
        return {
            "treated": df_true,
            "placebos": {},
            "p_value": float("nan"),
            "ref_band_event_time": None,
            "placebo_stats": pd.Series(dtype=float),
        }

    def _one_placebo(pc_date: pd.Timestamp):
        try:
            syn_pc = run_scm(
                df=d,
                date_col=date_col,
                unit_col=unit_col,
                outcome_col=outcome_col,
                treated_unit=treated_unit,
                cutoff_date=pc_date.strftime("%Y-%m-%d"),
                donors=donors,
                scm_backend=scm_backend,
                **kwargs,
            )
            eff = syn_pc["effect"]

            treated_dates_ts = pd.to_datetime(treated_dates.tolist())
            eff_aligned = eff.reindex(treated_dates_ts)
            idx = np.where(treated_dates_ts == pc_date)[0]
            if len(idx) == 0:
                return None
            start = int(idx[0])
            seg = eff_aligned.iloc[start : start + post_len]
            if len(seg) != post_len:
                return None

            stat = float(seg.mean() if post_agg == "mean" else seg.sum())
            seg_df = pd.DataFrame(
                {"effect": seg.values},
                index=pd.to_datetime(treated_dates[start : start + post_len]),
            )
            return pc_date, seg_df, stat
        except Exception as e:
            log.debug("Placebo run failed at %s: %s", pc_date, e)
            return None

    n_cores_eff = _eff_cores(n_cores)
    jobs = Parallel(n_jobs=n_cores_eff)(delayed(_one_placebo)(pc) for pc in placebo_candidates)
    jobs = [j for j in jobs if j is not None]
    if not jobs:
        log.warning("All placebo-in-time runs failed.")
        return {
            "treated": df_true,
            "placebos": {},
            "p_value": float("nan"),
            "ref_band_event_time": None,
            "placebo_stats": pd.Series(dtype=float),
        }

    placebo_dict = {pc.strftime("%Y-%m-%d"): seg for (pc, seg, _) in jobs}
    placebo_stats = pd.Series(
        {pc.strftime("%Y-%m-%d"): stat for (pc, _, stat) in jobs}
    ).sort_index()

    obs_series = df_true["effect"][df_true.index.to_series() >= cutoff_dt]
    obs_stat = float(obs_series.mean() if post_agg == "mean" else obs_series.sum())
    p_value = (np.sum(np.abs(placebo_stats.values) >= np.abs(obs_stat)) + 1) / (
        len(placebo_stats) + 1
    )

    M = np.vstack([seg["effect"].to_numpy() for seg in placebo_dict.values()])
    k_index = pd.RangeIndex(start=0, stop=post_len, step=1, name="event_time")

    ref_band_event_time = pd.DataFrame(
        {
            "p10": np.nanpercentile(M, 10, axis=0),
            "p90": np.nanpercentile(M, 90, axis=0),
            "ci_lo": np.nanpercentile(M, 2.5, axis=0),
            "ci_hi": np.nanpercentile(M, 97.5, axis=0),
            "std": np.nanstd(M, axis=0, ddof=1),
        },
        index=k_index,
    )

    log.info(
        "Placebo-in-time: scm_backend=%s | treated=%s | donors=%d | cutoffs=%d | post_agg=%s",
        scm_backend,
        treated_unit,
        len(donors),
        len(placebo_candidates),
        post_agg,
    )

    return {
        "treated": df_true,
        "placebos": placebo_dict,
        "p_value": float(p_value),
        "ref_band_event_time": ref_band_event_time,
        "placebo_stats": placebo_stats,
    }
