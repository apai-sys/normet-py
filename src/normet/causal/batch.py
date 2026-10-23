# src/normet/causal/batch.py
"""Batch runner: :func:`scm_all` runs multiple SCM specs in parallel, and
:func:`run_scm_staggered` fits a staggered (multi-unit, multi-cutoff) design
with the same worker pool pattern."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

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
    n_cores_eff = max(1, n_cores if n_cores is not None else (os.cpu_count() or 2) - 1)

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

    pieces = Parallel(n_jobs=n_cores_eff)(delayed(_one)(code) for code in units)
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
    n_cores: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Fit a staggered design: several treated units, each with its own cutoff.

    Every treated unit gets its own :func:`run_scm` fit — on its own pre-period
    and its own post-period — with the *other treated units excluded from every
    run's donor pool*, so one unit's intervention cannot leak into another
    unit's synthetic control (the same exclusion :func:`placebo_in_space`
    applies to the single treated unit). The runs are dispatched in parallel
    with the :mod:`joblib` pool :func:`scm_all` uses.

    Each unit's effect is the post-cutoff mean (or ``sum``) of its effect path,
    and each unit is ranked against its own placebo-in-space distribution:
    every donor in turn takes the unit's role at the unit's own cutoff, and the
    permutation ``p_value`` is the usual ``(count + 1) / (n + 1)``. Each
    placebo fit needs a valid pool of its own — with the default ``scm``
    backend that is at least three donors (its ridge-augmented path), so a
    smaller pool yields NaN p-values while the effects are still read.

    The summary rows aggregate the units' effects (mean and median). The
    ``mean`` row's ``p_value`` is the cross-unit permutation test: each donor
    in turn is assigned every treated unit's role (the full set of cutoffs) and
    its pseudo-effects are averaged the same way the real units' are; a donor
    with no intervention is a clean draw from the null, while the real treated
    units are deliberately *not* permuted into the reference — their data
    already carries the intervention, and giving one of them a later cutoff
    would pour its real effect into its own "pre-period". The reference is
    therefore donor-only, which keeps the null honest in a staggered design
    where permuting treated identities is contaminated.

    Typical use: a scheme that rolls out city by city (the UK ULEZ 2021
    expansion, staggered driving restrictions across cities) — one row per
    adopting city, one pooled reading for the programme as a whole.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel with at least {date_col, unit_col, outcome_col}.
    treated_units : list of str
        The units that received the intervention, in the order the summary
        rows are read.
    cutoff_dates : list of str, or dict
        The cutoff of each treated unit: a list aligned with ``treated_units``
        or a dict ``{unit: "YYYY-MM-DD"}``.
    date_col, unit_col, outcome_col : str
        Column names in the panel.
    donors : list of str, optional
        The donor pool. If None, every unit that is not treated. Treated units
        in an explicit list are an error; units absent from the panel are
        dropped (as in :func:`placebo_in_space`).
    scm_backend : str, optional
        Backend forwarded to :func:`run_scm` (default "scm").
    post_agg : {"mean", "sum"}, optional
        How a unit's post-cutoff effect path is read into its effect (and how
        the units' effects are aggregated in the summary rows).
    n_cores : int | None, optional
        Parallel workers for the combined pool of real and placebo runs;
        defaults to all cores - 1, as in :func:`scm_all`.
    **kwargs
        Forwarded to :func:`run_scm`.

    Returns
    -------
    pandas.DataFrame
        One row per treated unit — ``unit``, ``cutoff_date``, ``n_pre``,
        ``n_post``, ``pre_r2`` (the pre-period fit of that unit), ``effect``
        (post-cutoff ``post_agg`` of the effect path) and ``p_value`` (that
        unit's placebo-in-space permutation) — followed by two summary rows,
        ``mean`` and ``median``, of the units' effects. The ``mean`` row
        carries the cross-unit permutation ``p_value``; the ``median`` row's
        is NaN, as the test is stated on the mean.

    Raises
    ------
    ValueError
        For empty or duplicate treated lists, units absent from the panel,
        a misaligned ``cutoff_dates`` list, a treated unit inside ``donors``,
        no donor units left, or a treated unit with no post-cutoff
        observations.
    """
    if not treated_units:
        raise ValueError("treated_units must not be empty.")
    if len(set(treated_units)) != len(treated_units):
        raise ValueError("treated_units must be unique.")
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

    # A treated unit whose cutoff falls after its last observation has no
    # post-period to measure: say so before any fit is spent.
    unit_dates = {u: pd.to_datetime(sub).sort_values() for u, sub in d.groupby(unit_col)[date_col]}
    no_post = [u for u, c in pairs if not (unit_dates[u] >= c).any()]
    if no_post:
        raise ValueError(
            "No post-cutoff observations at/after the cutoff for treated units: "
            f"{[(u, str(pd.to_datetime(c).date())) for u, c in pairs if u in no_post]}"
        )

    n_cores_eff = max(1, int(n_cores) if n_cores is not None else (os.cpu_count() or 2) - 1)
    log.info(
        "run_scm_staggered: scm_backend=%s | treated=%d | donors=%d | post_agg=%s | n_cores=%d",
        scm_backend,
        len(pairs),
        len(donor_pool),
        post_agg,
        n_cores_eff,
    )

    def _read(syn: pd.DataFrame, c_ts: pd.Timestamp) -> tuple[float, int, int]:
        post = syn[syn.index.to_series() >= c_ts]
        pre = syn[syn.index.to_series() < c_ts]
        stat = float(post["effect"].mean() if post_agg == "mean" else post["effect"].sum())
        return stat, len(pre), len(post)

    def _one(
        unit: str, cutoff: str, pool: list[str], kind: str
    ) -> tuple[str, str, float, float, int, int] | None:
        try:
            syn = run_scm(
                df=df,
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
        stat, n_pre, n_post = _read(syn, pd.Timestamp(cutoff))
        if kind != "real":
            return (unit, cutoff, stat, float("nan"), n_pre, n_post)
        pre = syn[syn.index.to_series() < pd.Timestamp(cutoff)]
        pre_r2 = float("nan")
        if len(pre) >= 2:
            y = pre["observed"].to_numpy(dtype=float)
            x = pre["synthetic"].to_numpy(dtype=float)
            ss_res = float(np.sum((y - x) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            pre_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return (unit, cutoff, stat, pre_r2, n_pre, n_post)

    # One pool for the real runs and for every donor taking every treated
    # unit's role — the cross-unit reference, and the material of each unit's
    # own placebo-in-space test.
    jobs = [delayed(_one)(u, c.strftime("%Y-%m-%d"), donor_pool, "real") for u, c in pairs]
    jobs += [
        delayed(_one)(u, c.strftime("%Y-%m-%d"), [x for x in donor_pool if x != u], "placebo")
        for u in donor_pool
        for _, c in pairs
    ]
    results = Parallel(n_jobs=n_cores_eff)(jobs)

    real: dict[str, tuple[float, float]] = {}
    placebo: dict[tuple[str, str], float] = {}
    for r in results:
        if r is None:
            continue
        unit, cutoff, stat, pre_r2, _, _ = r
        key = (unit, cutoff)
        if unit in treated_set:
            real[unit] = (stat, pre_r2)
        else:
            placebo[key] = stat
    if len(real) != len(pairs):
        failed = [u for u, _ in pairs if u not in real]
        raise RuntimeError(f"Staggered fits failed for treated units: {failed}")

    slot_dates = [c.strftime("%Y-%m-%d") for _, c in pairs]

    def _perm_p(obs: float, refs: list[float]) -> float:
        refs = [r for r in refs if np.isfinite(r)]
        if not refs or not np.isfinite(obs):
            return float("nan")
        return float((np.sum(np.abs(np.asarray(refs)) >= np.abs(obs)) + 1) / (len(refs) + 1))

    rows = []
    unit_stats: list[float] = []
    for unit, c in pairs:
        stat, pre_r2 = real[unit]
        # n_pre/n_post read from the panel: the runs agree on the dates a unit
        # has, so the cheaper reading is the panel's own.
        dates = unit_dates[unit]
        n_pre = int((dates < c).sum())
        n_post = int((dates >= c).sum())
        refs = [
            placebo[(u, c.strftime("%Y-%m-%d"))]
            for u in donor_pool
            if (u, c.strftime("%Y-%m-%d")) in placebo
        ]
        rows.append(
            {
                "unit": unit,
                "cutoff_date": c.strftime("%Y-%m-%d"),
                "n_pre": n_pre,
                "n_post": n_post,
                "pre_r2": pre_r2,
                "effect": stat,
                "p_value": _perm_p(stat, refs),
            }
        )
        unit_stats.append(stat)
    stat_obs = float(np.mean(unit_stats) if post_agg == "mean" else np.sum(unit_stats))
    pooled = []
    for u in donor_pool:
        refs_u = [placebo[(u, s)] for s in slot_dates if (u, s) in placebo]
        if refs_u:
            pooled.append(float(np.mean(refs_u) if post_agg == "mean" else np.sum(refs_u)))
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
    return pd.DataFrame(
        rows, columns=["unit", "cutoff_date", "n_pre", "n_post", "pre_r2", "effect", "p_value"]
    )
