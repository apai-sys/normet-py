"""First-principles scientific operators built on Chronos-2.

Combines Chronos-2's covariate-conditioned probabilistic kernel with normet's
causal and atmospheric workflows:

- :func:`chronos_scm` — zero-shot, covariate-conditioned synthetic control
  using untreated donor series as known-future covariates alongside optional
  local meteorology and past-only co-pollutants.
- :func:`chronos_anomaly_scores` and :func:`detect_chronos_events` —
  meteorology-conditioned probabilistic event detection scored by PIT and
  two-sided tail surprise against the model's predictive quantile fan.
- :func:`chronos_attention_attribution` — white-box inspection of group and
  temporal self-attention weights across target, past-only, and known-future
  covariates.
- :func:`chronos_pdp` — future-only in-context partial dependence curves and
  predictive quantile intervals.
- :func:`chronos_impute` — covariate-conditioned bidirectional gap imputation
  with native predictive uncertainty bounds.
- :func:`chronos_backtest` — rolling multi-horizon probabilistic verification
  (CRPS, pinball loss, PIT uniformity, interval coverage) with causal
  prefix-to-holdout conformal calibration.
- :class:`CovariateSensitivityCallback` — HuggingFace ``TrainerCallback``
  monitoring meteorological sensitivity during Chronos-2 fine-tuning.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .calibration import (
    AdaptiveConformalCalibrator,
    CalibratedEstimator,
    QuantileCalibration,
    calibrate_quantiles,
)
from .chronos import ChronosEmbedder, _to_numpy_f32
from .estimator import (
    Chronos2Estimator,
    CounterfactualResult,
    InsufficientContextError,
    to_regular_index,
)

log = logging.getLogger(__name__)

__all__ = [
    "chronos_scm",
    "select_donors_by_embedding",
    "chronos_anomaly_scores",
    "detect_chronos_events",
    "chronos_attention_attribution",
    "chronos_pdp",
    "chronos_impute",
    "chronos_backtest",
    "CovariateSensitivityCallback",
]


def _align_timestamp(value: str | pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Align a cutoff timestamp to ``index``'s timezone (or strip tz if naive)."""
    ts = pd.Timestamp(value)
    tz = index.tz
    if tz is None:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)


def _pivot_panel(
    df: pd.DataFrame,
    *,
    date_col: str,
    unit_col: str,
    outcome_col: str,
    treated_unit: str,
    donors: Sequence[str] | None,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Validate and pivot a long multi-unit panel into wide form ``[treated, *donors]``."""
    missing = [c for c in (date_col, unit_col, outcome_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required panel columns: {missing}")
    if not treated_unit:
        raise ValueError("`treated_unit` must be non-empty")

    work = df[[date_col, unit_col, outcome_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work[unit_col] = work[unit_col].astype(str)
    treated_key = str(treated_unit)

    panel = work.pivot_table(
        index=date_col, columns=unit_col, values=outcome_col, aggfunc="mean"
    ).sort_index()
    panel.columns = [str(c) for c in panel.columns]

    if treated_key not in panel.columns:
        raise ValueError(f"Treated unit {treated_unit!r} not found in panel")

    if donors is None:
        donor_pool = [c for c in panel.columns if c != treated_key]
    else:
        donor_pool = [str(u) for u in dict.fromkeys(donors) if str(u) != treated_key]
        absent = [u for u in donor_pool if u not in panel.columns]
        if absent:
            raise ValueError(f"Donor unit(s) not found in panel: {absent}")

    if not donor_pool:
        raise ValueError("No valid donor units available after excluding treated unit")

    return panel[[treated_key, *donor_pool]].copy(), treated_key, donor_pool


def chronos_scm(
    estimator: Chronos2Estimator,
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    unit_col: str = "ID",
    outcome_col: str = "value",
    treated_unit: str,
    cutoff_date: str | pd.Timestamp,
    donors: Sequence[str] | None = None,
    met_df: pd.DataFrame | None = None,
    met_covariates: Sequence[str] | None = None,
    past_only_covariates: Sequence[str] | None = None,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    validation_steps: int | None = None,
    max_horizon: int | None = None,
    calibrated: bool = False,
    rollout: str = "quantile",
    freq: str | None = None,
) -> CounterfactualResult:
    """Zero-shot covariate-conditioned Synthetic Control with Chronos-2.

    Pivots a long-format multi-unit panel into wide form and conditions the
    treated unit's pre- and post-cutoff trajectory on untreated donor outcomes
    (and optional meteorology) supplied as known-future covariates. Unlike
    :func:`normet.mlscm`, no parametric or tree mapping is fitted on the
    pre-period, avoiding small-sample donor overfitting while emitting native
    predictive uncertainty bands (optionally conformal-calibrated).

    Parameters
    ----------
    estimator : Chronos2Estimator
        Zero-shot or fine-tuned Chronos-2 estimator.
    df : pandas.DataFrame
        Long-format panel containing ``[date_col, unit_col, outcome_col]``.
    date_col, unit_col, outcome_col : str
        Panel column names.
    treated_unit : str
        Identifier of the treated unit.
    cutoff_date : str or Timestamp
        Intervention timestamp; pre-period is strictly before this timestamp.
    donors : sequence of str, optional
        Donor unit identifiers. Defaults to all units other than ``treated_unit``.
    met_df : pandas.DataFrame, optional
        Optional meteorology frame (datetime-indexed or carrying ``date_col`` /
        ``"date"``) aligned onto the panel index as additional covariates.
    met_covariates : sequence of str, optional
        Known-future columns to take from ``met_df``. Defaults to all columns
        in ``met_df`` except any listed in ``past_only_covariates``.
    past_only_covariates : sequence of str, optional
        Historical columns in ``met_df`` supplied only before ``cutoff_date``.
    quantiles : sequence of float, default ``(0.1, 0.5, 0.9)``
        Quantile levels requested from the counterfactual projection.
    validation_steps : int, optional
        Number of pre-intervention steps held out for the placebo bias check.
        Defaults to ``min(estimator.prediction_length, pre_len - context_length)``
        when room permits.
    max_horizon : int, optional
        Truncate post-cutoff projection after this many steps.
    calibrated : bool, default False
        Apply :func:`~normet.foundation.calibration.calibrate_quantiles` on the
        pre-cutoff history before projecting.
    rollout : {"quantile", "median"}, default "quantile"
        Long-horizon rollout strategy forwarded to
        :meth:`Chronos2Estimator.counterfactual`.
    freq : str, optional
        If provided, regularise the pivoted panel onto a uniform grid via
        :func:`~normet.foundation.estimator.to_regular_index`.

    Returns
    -------
    CounterfactualResult
        Counterfactual series, impact, pre-cutoff placebo bias, and summary;
        call ``.to_scm_frame()`` for the ``observed/synthetic/effect`` schema.
    """
    panel, treated_key, donor_pool = _pivot_panel(
        df,
        date_col=date_col,
        unit_col=unit_col,
        outcome_col=outcome_col,
        treated_unit=treated_unit,
        donors=donors,
    )
    if freq is not None:
        panel = to_regular_index(panel, freq=freq)

    if not isinstance(panel.index, pd.DatetimeIndex):
        raise TypeError("Panel index must be a DatetimeIndex")
    cutoff_ts = _align_timestamp(cutoff_date, panel.index)

    pre_mask = panel.index < cutoff_ts
    empty_donors = [
        d for d in donor_pool if not np.isfinite(panel.loc[pre_mask, d].to_numpy(dtype=float)).any()
    ]
    if empty_donors:
        raise ValueError(f"Donor unit(s) have no finite pre-cutoff data: {empty_donors}")

    met_cols: list[str] = []
    past_only_cols = [str(c) for c in dict.fromkeys(past_only_covariates or ())]
    if met_df is not None:
        m = met_df.copy()
        if not isinstance(m.index, pd.DatetimeIndex):
            if date_col in m.columns:
                m = m.set_index(date_col)
            elif "date" in m.columns:
                m = m.set_index("date")
            m.index = pd.to_datetime(m.index)
        m = m.sort_index()
        if met_covariates is not None:
            met_cols = [str(c) for c in met_covariates]
            missing_met = [c for c in met_cols if c not in m.columns]
            if missing_met:
                raise KeyError(f"met_covariates absent from met_df: {missing_met}")
        else:
            met_cols = [str(c) for c in m.columns if str(c) not in set(past_only_cols)]
        missing_po = [c for c in past_only_cols if c not in m.columns]
        if missing_po:
            raise KeyError(f"past_only_covariates absent from met_df: {missing_po}")
        all_met = list(dict.fromkeys([*met_cols, *past_only_cols]))
        collisions = set(panel.columns) & set(all_met)
        if collisions:
            raise ValueError(f"met_df columns collide with panel unit names: {sorted(collisions)}")
        panel = panel.join(m[all_met].reindex(panel.index))
    else:
        if met_covariates is not None:
            raise ValueError("`met_covariates` requires `met_df`")
        if past_only_cols:
            raise ValueError("`past_only_covariates` requires `met_df`")

    known_future_cols = [*donor_pool, *met_cols]
    pre_len = int(pre_mask.sum())
    if validation_steps is None:
        if pre_len > estimator.context_length:
            val_steps = min(estimator.prediction_length, pre_len - estimator.context_length)
        else:
            val_steps = estimator.prediction_length
    else:
        val_steps = int(validation_steps)

    res = estimator.counterfactual(
        panel,
        treated_key,
        cutoff_ts,
        features=known_future_cols,
        quantiles=quantiles,
        validation_hours=val_steps,
        max_horizon=max_horizon,
        calibrated=calibrated,
        past_only_covariates=past_only_cols,
        rollout=rollout,
    )
    res.summary.update(
        {
            "method": "chronos_scm",
            "treated_unit": treated_key,
            "n_donors": len(donor_pool),
            "donors": list(donor_pool),
            "met_covariates": list(met_cols),
            "rollout": rollout,
        }
    )
    return res


def select_donors_by_embedding(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    unit_col: str = "ID",
    outcome_col: str = "value",
    treated_unit: str,
    cutoff_date: str | pd.Timestamp,
    donors: Sequence[str] | None = None,
    embedder: ChronosEmbedder | None = None,
    context_length: int = 2048,
    top_k: int | None = None,
    min_similarity: float | None = None,
) -> pd.DataFrame:
    """Rank and filter synthetic-control donors using pre-cutoff Chronos-2 embeddings.

    Restricts strictly to timestamps before ``cutoff_date`` so post-intervention
    outcomes never leak into donor selection.
    """
    if top_k is not None and int(top_k) < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    panel, treated_key, donor_pool = _pivot_panel(
        df,
        date_col=date_col,
        unit_col=unit_col,
        outcome_col=outcome_col,
        treated_unit=treated_unit,
        donors=donors,
    )
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise TypeError("Panel index must be a DatetimeIndex")
    cutoff_ts = _align_timestamp(cutoff_date, panel.index)
    pre = panel.loc[panel.index < cutoff_ts]
    if pre.empty:
        raise ValueError("No pre-cutoff observations available for donor selection")

    emb = embedder if embedder is not None else ChronosEmbedder()
    eff_ctx = min(int(context_length), len(pre))
    min_cov = float(getattr(emb, "min_coverage", 0.1))

    treated_tail = pre[treated_key].to_numpy(dtype=float)[-eff_ctx:]
    if float(np.isfinite(treated_tail).mean()) < min_cov:
        raise InsufficientContextError(
            f"Treated unit {treated_key!r} has insufficient pre-cutoff coverage (< {min_cov:.0%})"
        )

    valid_donors = [
        d
        for d in donor_pool
        if float(np.isfinite(pre[d].to_numpy(dtype=float)[-eff_ctx:]).mean()) >= min_cov
    ]
    if not valid_donors:
        raise ValueError("No candidate donors meet minimum pre-cutoff coverage")

    vecs = emb.embed_stations(pre[[treated_key, *valid_donors]], context_length=eff_ctx)
    t_vec = np.asarray(vecs[treated_key], dtype=float)
    t_norm = float(np.linalg.norm(t_vec))

    scored: list[tuple[str, float]] = []
    for d in valid_donors:
        d_vec = np.asarray(vecs[d], dtype=float)
        denom = t_norm * float(np.linalg.norm(d_vec))
        sim = float(np.dot(t_vec, d_vec) / denom) if denom > 0.0 else 0.0
        scored.append((d, sim))

    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    rows: list[dict[str, Any]] = []
    for rank, (unit, sim) in enumerate(scored, start=1):
        sel = True
        if min_similarity is not None and sim < float(min_similarity):
            sel = False
        if top_k is not None and rank > int(top_k):
            sel = False
        rows.append(
            {
                "unit": unit,
                "similarity": sim,
                "rank": rank,
                "selected": sel,
            }
        )
    return pd.DataFrame(rows, columns=["unit", "similarity", "rank", "selected"])


def _resolve_cal_and_base(
    estimator: Any,
    calibration: QuantileCalibration | CalibratedEstimator | None,
) -> tuple[Any, QuantileCalibration | None]:
    """Unwrap an optional :class:`CalibratedEstimator` / :class:`QuantileCalibration`."""
    if isinstance(estimator, CalibratedEstimator):
        base = estimator.base
        cal: QuantileCalibration | CalibratedEstimator | None = (
            calibration if calibration is not None else estimator.calibration
        )
    else:
        base = estimator
        cal = calibration
    if isinstance(cal, CalibratedEstimator):
        resolved_cal: QuantileCalibration | None = cal.calibration
        return base, resolved_cal
    return base, cal


def _empirical_pit(y: float, q_vals: np.ndarray, q_levels: np.ndarray) -> float:
    """Compute PIT by interpolating observed ``y`` against sorted quantile levels."""
    qv = np.asarray(q_vals, dtype=float).copy()
    mid = int(np.argmin(np.abs(q_levels - 0.5)))
    # Enforce monotonicity outward from the median so partial-band calibration
    # widens lower and upper tails symmetrically rather than clamping the lower tail.
    qv[: mid + 1] = np.minimum.accumulate(qv[: mid + 1][::-1])[::-1]
    qv[mid:] = np.maximum.accumulate(qv[mid:])
    if y < qv[0]:
        return 0.0
    if y > qv[-1]:
        return 1.0
    eq = np.flatnonzero(qv == y)
    if eq.size > 1:
        return float(0.5 * (q_levels[eq[0]] + q_levels[eq[-1]]))
    return float(np.interp(y, qv, q_levels, left=0.0, right=1.0))


def chronos_anomaly_scores(
    estimator: Any,
    df: pd.DataFrame,
    target: str,
    *,
    covariates: Sequence[str] | None = None,
    past_only_covariates: Sequence[str] | None = None,
    stride: int | None = None,
    interval: tuple[float, float] = (0.1, 0.9),
    calibration: QuantileCalibration | CalibratedEstimator | None = None,
    adaptive: bool = False,
    aci_gamma: float = 0.05,
    aci_window: int = 336,
) -> pd.DataFrame:
    """Score per-timestamp anomalies conditioned on meteorology via Chronos-2.

    Rolls non-overlapping (or ``stride``-stepped) blocks across ``df`` and scores
    each observed value against the model's meteorology-conditioned quantile
    distribution (optionally rescaled by a :class:`QuantileCalibration` fit or
    causal :class:`AdaptiveConformalCalibrator` when ``adaptive=True``).
    Unlike univariate IQR or STL anomaly detectors, high concentrations driven
    by stagnant weather are expected and receive low surprise.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``df``, with columns ``["observed", "expected", "lower",
        "upper", "pit", "tail_p", "surprise", "z_score"]`` (plus ``aci_scale``
        and ``aci_alpha`` when ``adaptive=True``). Leading ``context_length``
        rows and sparse-context blocks are ``NaN``.
    """
    lo, hi = float(interval[0]), float(interval[1])
    if not (0.0 < lo < 0.5 < hi < 1.0):
        raise ValueError(f"`interval` must satisfy 0 < low < 0.5 < high < 1, got {interval!r}")
    if target not in df.columns:
        raise KeyError(f"target column absent from frame: {target!r}")

    base, cal = _resolve_cal_and_base(estimator, calibration)
    base._check_index(df.index)

    horizon = int(base.prediction_length)
    step = int(stride) if stride is not None else horizon
    if step <= 0:
        raise ValueError(f"stride must be positive, got {step}")

    cols = (
        list(covariates)
        if covariates is not None
        else base._resolve_covariates(df, target, exclude=past_only_covariates or ())
    )
    past_only_cols = base._resolve_past_only_covariates(df, target, cols, past_only_covariates)

    n = len(df)
    spans = [(s, min(s + horizon, n)) for s in range(base.context_length, n, step)]
    if not spans:
        raise InsufficientContextError(
            f"need more than context_length ({base.context_length}) rows to score "
            f"anomalies, got {n}"
        )

    obs = df[target].to_numpy(dtype=float)
    obs_f32 = obs.astype(np.float32)
    skipped = base._blocks_without_context(obs, spans, "anomaly scoring")
    active = [sp for sp in spans if sp not in skipped]

    q_levels = np.asarray(base.quantile_levels, dtype=float)
    q_sum = np.zeros((len(q_levels), n), dtype=float)
    q_cnt = np.zeros(n, dtype=int)

    by_len: dict[int, list[tuple[int, int]]] = {}
    for s, e in active:
        by_len.setdefault(e - s, []).append((s, e))

    for block_len, group in by_len.items():
        items = [
            (
                obs_f32[max(0, s - base.context_length) : s],
                df.iloc[max(0, s - base.context_length) : s],
                df.iloc[s:e],
            )
            for s, e in group
        ]
        if past_only_cols:
            blocks = base._forecast_blocks(items, cols, block_len, past_only_cols=past_only_cols)
        else:
            blocks = base._forecast_blocks(items, cols, block_len)
        for (s, e), raw_block in zip(group, blocks, strict=True):
            blk = np.asarray(raw_block, dtype=float)
            if cal is not None:
                blk = cal.rescale_block(blk, base._q_index)
            q_sum[:, s:e] += blk
            q_cnt[s:e] += 1

    expected = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    pit = np.full(n, np.nan)
    tail_p = np.full(n, np.nan)
    surprise = np.full(n, np.nan)
    z_score = np.full(n, np.nan)
    aci_scale: np.ndarray | None = None
    aci_alpha: np.ndarray | None = None

    i50 = base._q_index(0.5)
    i_lo = base._q_index(lo)
    i_hi = base._q_index(hi)

    covered = q_cnt > 0
    if covered.any():
        q_mean = q_sum[:, covered] / q_cnt[covered][None, :]
        expected[covered] = q_mean[i50]
        lower[covered] = q_mean[i_lo]
        upper[covered] = q_mean[i_hi]

        lower_cal = lower
        upper_cal = upper
        if adaptive:
            cal_df = AdaptiveConformalCalibrator(
                alpha=1.0 - (hi - lo),
                gamma=aci_gamma,
                window=aci_window,
            ).calibrate_series(
                pd.Series(obs), pd.Series(expected), pd.Series(lower), pd.Series(upper)
            )
            aci_scale = cal_df["scale"].to_numpy(dtype=float)
            aci_alpha = cal_df["alpha_t"].to_numpy(dtype=float)
            lower_cal = cal_df["lower_cal"].to_numpy(dtype=float)
            upper_cal = cal_df["upper_cal"].to_numpy(dtype=float)

        covered_idx = np.flatnonzero(covered)
        for local_j, t in enumerate(covered_idx):
            y = obs[t]
            if not np.isfinite(y):
                continue
            qv = q_mean[:, local_j]
            if not np.all(np.isfinite(qv)):
                continue
            if (
                adaptive
                and aci_scale is not None
                and np.isfinite(aci_scale[t])
                and aci_scale[t] > 0.0
            ):
                qv = expected[t] + aci_scale[t] * (qv - expected[t])
                lower[t] = lower_cal[t]
                upper[t] = upper_cal[t]
            p = _empirical_pit(float(y), qv, q_levels)
            tp = float(2.0 * min(p, 1.0 - p))
            pit[t] = p
            tail_p[t] = tp
            surprise[t] = float(-np.log10(max(tp, 1e-6)))
            half = 0.5 * (upper[t] - lower[t])
            if half > 0.0 and np.isfinite(half):
                z_score[t] = float((y - expected[t]) / half)

    out_cols: dict[str, np.ndarray] = {
        "observed": obs,
        "expected": expected,
        "lower": lower,
        "upper": upper,
        "pit": pit,
        "tail_p": tail_p,
        "surprise": surprise,
        "z_score": z_score,
    }
    if adaptive:
        out_cols["aci_scale"] = aci_scale if aci_scale is not None else np.full(n, np.nan)
        out_cols["aci_alpha"] = aci_alpha if aci_alpha is not None else np.full(n, np.nan)
    return pd.DataFrame(out_cols, index=df.index)


def detect_chronos_events(
    estimator: Any,
    df: pd.DataFrame,
    target: str,
    *,
    alpha: float = 0.05,
    min_length: int = 1,
    direction: str = "both",
    covariates: Sequence[str] | None = None,
    past_only_covariates: Sequence[str] | None = None,
    stride: int | None = None,
    interval: tuple[float, float] = (0.1, 0.9),
    calibration: QuantileCalibration | CalibratedEstimator | None = None,
    adaptive: bool = False,
    aci_gamma: float = 0.05,
    aci_window: int = 336,
) -> pd.DataFrame:
    """Detect consecutive weather-conditioned anomalous intervals via Chronos-2.

    Flags timestamps whose two-sided tail probability under the meteorology-
    conditioned forecast distribution satisfies ``tail_p <= alpha``, collapses
    contiguous runs of at least ``min_length`` steps, and ranks intervals by
    ``max_surprise`` descending.
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if int(min_length) < 1:
        raise ValueError(f"min_length must be >= 1, got {min_length}")
    if direction not in ("both", "high", "low"):
        raise ValueError(f"direction must be 'both', 'high', or 'low', got {direction!r}")

    scores = chronos_anomaly_scores(
        estimator,
        df,
        target,
        covariates=covariates,
        past_only_covariates=past_only_covariates,
        stride=stride,
        interval=interval,
        calibration=calibration,
        adaptive=adaptive,
        aci_gamma=aci_gamma,
        aci_window=aci_window,
    )

    tp = scores["tail_p"].to_numpy(dtype=float)
    pit = scores["pit"].to_numpy(dtype=float)
    surp = scores["surprise"].to_numpy(dtype=float)
    resid = (scores["observed"] - scores["expected"]).to_numpy(dtype=float)

    mask = np.isfinite(tp) & np.isfinite(pit) & (tp <= float(alpha))
    if direction == "high":
        mask &= pit > 0.5
    elif direction == "low":
        mask &= pit < 0.5

    cols_out = [
        "start",
        "end",
        "n",
        "direction",
        "max_surprise",
        "mean_surprise",
        "mean_residual",
    ]
    if not mask.any():
        return pd.DataFrame(columns=cols_out)

    idx = pd.DatetimeIndex(scores.index)
    rows: list[dict[str, Any]] = []
    in_run = False
    run_start = 0
    for i in range(len(mask) + 1):
        flagged = bool(mask[i]) if i < len(mask) else False
        if flagged and not in_run:
            run_start = i
            in_run = True
        elif (not flagged) and in_run:
            length = i - run_start
            if length >= int(min_length):
                seg_s = surp[run_start:i]
                seg_r = resid[run_start:i]
                mean_r = float(np.nanmean(seg_r))
                rows.append(
                    {
                        "start": idx[run_start],
                        "end": idx[i - 1],
                        "n": int(length),
                        "direction": "high" if mean_r >= 0.0 else "low",
                        "max_surprise": float(np.nanmax(seg_s)),
                        "mean_surprise": float(np.nanmean(seg_s)),
                        "mean_residual": mean_r,
                    }
                )
            in_run = False
    if not rows:
        return pd.DataFrame(columns=cols_out)
    return (
        pd.DataFrame(rows, columns=cols_out)
        .sort_values("max_surprise", ascending=False)
        .reset_index(drop=True)
    )


def _resolve_q_triple(
    estimator: Chronos2Estimator, quantiles: Sequence[float]
) -> tuple[int, int, int]:
    """Return ``(lower_idx, median_idx, upper_idx)`` for ``quantiles``."""
    qs = sorted(float(q) for q in quantiles)
    if not qs:
        raise ValueError("quantiles must not be empty")
    q_lo = qs[0]
    q_hi = qs[-1]
    q_mid = 0.5 if 0.5 in qs else qs[len(qs) // 2]
    return estimator._q_index(q_lo), estimator._q_index(q_mid), estimator._q_index(q_hi)


def _resolve_layers(layers: str | int | Sequence[int], n_layers: int) -> list[int]:
    """Resolve layer selector (``'all'``, ``'last'``, int, or sequence) to indices."""
    if n_layers < 1:
        raise RuntimeError(f"Expected at least 1 transformer layer, got {n_layers}")
    if isinstance(layers, str):
        key = layers.lower()
        if key == "all":
            return list(range(n_layers))
        if key == "last":
            return [n_layers - 1]
        raise ValueError(f"layers string must be 'all' or 'last', got {layers!r}")
    if isinstance(layers, bool):
        raise ValueError(f"layers must not be bool, got {layers!r}")
    if isinstance(layers, int):
        idx = n_layers + layers if layers < 0 else layers
        if not (0 <= idx < n_layers):
            raise ValueError(f"layer index {layers} out of range for {n_layers} layers")
        return [idx]
    if isinstance(layers, Sequence):
        resolved: list[int] = []
        for entry in layers:
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise ValueError(f"layer indices must be integers, got {entry!r}")
            idx = n_layers + entry if entry < 0 else entry
            if not (0 <= idx < n_layers):
                raise ValueError(f"layer index {entry} out of range for {n_layers} layers")
            resolved.append(idx)
        out = list(dict.fromkeys(resolved))
        if not out:
            raise ValueError("layers sequence must not be empty")
        return out
    raise ValueError(f"Unsupported layers selector: {layers!r}")


def _extract_attentions(
    pipe: Any,
    item: dict[str, Any],
    context_length: int,
    horizon: int,
    n_series_expected: int,
) -> tuple[list[np.ndarray], list[np.ndarray], int, int, int]:
    """Run one single-task forward pass with ``output_attentions=True``."""
    if hasattr(pipe, "inspect_attentions"):
        raw_g, raw_t, n_ctx_patches, n_out_patches = pipe.inspect_attentions(item, horizon)
        g_list = [_to_numpy_f32(w).astype(np.float64) for w in raw_g]
        t_list = [_to_numpy_f32(w).astype(np.float64) for w in raw_t]
        input_patch_size = int(getattr(pipe, "input_patch_size", 16))
        return g_list, t_list, int(n_ctx_patches), int(n_out_patches), input_patch_size

    from ..utils._omp import import_torch

    torch = import_torch()
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

    patch_out = int(pipe.model_output_patch_size)
    cfg = pipe.model.chronos_config
    input_patch_size = int(cfg.input_patch_size)
    n_out_patches = min(math.ceil(horizon / patch_out), int(pipe.max_output_patches))
    ds = Chronos2Dataset(
        [item],
        context_length=context_length,
        prediction_length=horizon,
        batch_size=max(1, n_series_expected + 1),
        output_patch_size=patch_out,
        mode=DatasetMode.TEST,
    )
    batch = next(iter(ds))
    device = pipe.model.device
    context = batch["context"].to(device=device, dtype=torch.float32)
    group_ids = batch["group_ids"].to(device=device)
    future_cov = batch["future_covariates"]
    if future_cov is not None:
        future_cov = future_cov.to(device=device, dtype=torch.float32)
        output_size = n_out_patches * patch_out
        if output_size > future_cov.shape[1]:
            pad = torch.full(
                (future_cov.shape[0], output_size - future_cov.shape[1]),
                fill_value=float("nan"),
                device=future_cov.device,
                dtype=future_cov.dtype,
            )
            future_cov = torch.cat([future_cov, pad], dim=1)
        elif output_size < future_cov.shape[1]:
            future_cov = future_cov[..., :output_size]

    with torch.no_grad():
        out = pipe.model(
            context=context,
            group_ids=group_ids,
            future_covariates=future_cov,
            num_output_patches=n_out_patches,
            output_attentions=True,
        )
    if out.enc_group_self_attn_weights is None or out.enc_time_self_attn_weights is None:
        raise RuntimeError("Chronos-2 model did not return encoder attention weights")

    g_list = [_to_numpy_f32(w).astype(np.float64) for w in out.enc_group_self_attn_weights]
    t_list = [_to_numpy_f32(w).astype(np.float64) for w in out.enc_time_self_attn_weights]
    use_reg = bool(getattr(cfg, "use_reg_token", True))
    seq_len = int(g_list[0].shape[0])
    n_ctx_patches = seq_len - n_out_patches - (1 if use_reg else 0)
    return g_list, t_list, n_ctx_patches, n_out_patches, input_patch_size


def chronos_attention_attribution(
    estimator: Chronos2Estimator,
    df: pd.DataFrame,
    target: str,
    anchor: pd.Timestamp | str,
    *,
    horizon: int | None = None,
    covariates: Sequence[str] | None = None,
    past_only_covariates: Sequence[str] | None = None,
    layers: str | int | Sequence[int] = "all",
) -> dict[str, Any]:
    """Inspect Chronos-2 white-box group and temporal attention weights at ``anchor``.

    Runs one forward pass with ``output_attentions=True`` (upstream automatically
    enables eager attention without reloading weights) and aggregates how strongly
    the target's forecast output patch tokens attend across variables (autoregressive
    target self-attention vs past-only and known-future covariates) and across historical
    context patches.

    Returns
    -------
    dict
        Keys: ``group_weights`` (DataFrame with columns ``["variable", "role",
        "attention", "share", "context_attention"]``), ``time_weights`` (DataFrame
        with columns ``["patch_idx", "date", "attention", "share"]``),
        ``target_self_share``, ``covariate_share``, ``n_layers``, ``layers``,
        ``num_context_patches``, and ``num_output_patches``.
    """
    if target not in df.columns:
        raise KeyError(f"target column absent from frame: {target!r}")
    estimator._check_index(df.index)

    cols = (
        list(covariates)
        if covariates is not None
        else estimator._resolve_covariates(df, target, exclude=past_only_covariates or ())
    )
    past_only_cols = estimator._resolve_past_only_covariates(df, target, cols, past_only_covariates)
    h = int(horizon or estimator.prediction_length)
    past, future = estimator._window(df, pd.Timestamp(anchor), h)
    hist = past[target].to_numpy(dtype=np.float32)
    estimator._check_context(hist)

    item = estimator._make_input(hist, past, future, cols, past_only_cols=past_only_cols)
    # Match chronos.chronos2.preprocess.from_list_of_dicts exact row stacking order:
    # target first, then sorted past-only covariates, then sorted known-future covariates.
    past_only_sorted = sorted(
        c for c in item["past_covariates"] if c not in item["future_covariates"]
    )
    future_sorted = sorted(item["future_covariates"].keys())
    row_meta: list[tuple[str, str]] = (
        [(target, "target")]
        + [(c, "past_only") for c in past_only_sorted]
        + [(c, "known_future") for c in future_sorted]
    )

    pipe = estimator._load_pipeline()
    g_list, t_list, n_ctx_patches, n_out_patches, patch_in = _extract_attentions(
        pipe, item, estimator.context_length, len(future), len(row_meta)
    )
    n_layers = len(g_list)
    layer_ids = _resolve_layers(layers, n_layers)

    if int(g_list[0].shape[-1]) != len(row_meta):
        raise RuntimeError(
            f"Group attention variate count {g_list[0].shape[-1]} does not match "
            f"expected series count {len(row_meta)}"
        )

    g_stack = np.stack([g_list[i] for i in layer_ids], axis=0)
    # Forecast-patch query weights on target (row 0) across all variates in the group.
    g_out = g_stack[:, -n_out_patches:, :, 0, :].mean(axis=(0, 1, 2))
    g_ctx = (
        g_stack[:, :n_ctx_patches, :, 0, :].mean(axis=(0, 1, 2))
        if n_ctx_patches > 0
        else np.zeros_like(g_out)
    )
    g_total = float(g_out.sum())
    shares = g_out / g_total if g_total > 0.0 else np.zeros_like(g_out)

    group_rows = [
        {
            "variable": var,
            "role": role,
            "attention": float(att),
            "share": float(sh),
            "context_attention": float(catt),
        }
        for (var, role), att, sh, catt in zip(row_meta, g_out, shares, g_ctx, strict=True)
    ]
    group_weights = (
        pd.DataFrame(
            group_rows,
            columns=["variable", "role", "attention", "share", "context_attention"],
        )
        .sort_values("share", ascending=False)
        .reset_index(drop=True)
    )

    target_self_share = float(shares[0])
    covariate_share = float(np.clip(1.0 - target_self_share, 0.0, 1.0))

    t_stack = np.stack([t_list[i] for i in layer_ids], axis=0)
    if n_ctx_patches > 0:
        t_ctx = t_stack[:, 0, :, -n_out_patches:, :n_ctx_patches].mean(axis=(0, 1, 2))
        t_sum = float(t_ctx.sum())
        t_share = t_ctx / t_sum if t_sum > 0.0 else np.zeros_like(t_ctx)
        hist_len = len(past)
        left_pad = max(0, n_ctx_patches * patch_in - hist_len)
        time_rows = []
        for p in range(n_ctx_patches):
            pos = min(hist_len - 1, max(0, (p + 1) * patch_in - left_pad - 1))
            time_rows.append(
                {
                    "patch_idx": p,
                    "date": past.index[pos],
                    "attention": float(t_ctx[p]),
                    "share": float(t_share[p]),
                }
            )
        time_weights = pd.DataFrame(time_rows, columns=["patch_idx", "date", "attention", "share"])
    else:
        time_weights = pd.DataFrame(columns=["patch_idx", "date", "attention", "share"])

    return {
        "group_weights": group_weights,
        "time_weights": time_weights,
        "target_self_share": target_self_share,
        "covariate_share": covariate_share,
        "n_layers": n_layers,
        "layers": layer_ids,
        "num_context_patches": int(n_ctx_patches),
        "num_output_patches": int(n_out_patches),
    }


def _numeric_grid(
    series: pd.Series, quantile_range: tuple[float, float], grid_points: int
) -> np.ndarray | None:
    """Build an evenly spaced evaluation grid across ``quantile_range``."""
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    lo, hi = np.quantile(vals, quantile_range)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        return np.array([float(lo)], dtype=float)
    return np.linspace(float(lo), float(hi), int(grid_points))


def chronos_pdp(
    estimator: Chronos2Estimator,
    df: pd.DataFrame,
    target: str,
    *,
    var_list: Sequence[str] | None = None,
    grid_points: int = 25,
    quantile_range: tuple[float, float] = (0.01, 0.99),
    n_anchors: int = 8,
    horizon: int | None = None,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    past_only_covariates: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute future-only in-context Partial Dependence curves with Chronos-2.

    Selects up to ``n_anchors`` evenly spaced forecast origins across ``df``, holds
    historical context fixed, intervenes on one known-future covariate across its
    ``quantile_range`` grid (batched across anchors and grid points), and summarises
    the predicted median alongside the lower/upper quantile envelope.
    """
    if target not in df.columns:
        raise KeyError(f"target column absent from frame: {target!r}")
    estimator._check_index(df.index)
    if int(grid_points) < 2:
        raise ValueError(f"grid_points must be >= 2, got {grid_points}")
    if int(n_anchors) < 1:
        raise ValueError(f"n_anchors must be >= 1, got {n_anchors}")
    q_lo, q_hi = float(quantile_range[0]), float(quantile_range[1])
    if not (0.0 <= q_lo < q_hi <= 1.0):
        raise ValueError(f"quantile_range must satisfy 0 <= lo < hi <= 1, got {quantile_range!r}")

    cols = estimator._resolve_covariates(df, target, exclude=past_only_covariates or ())
    past_only_cols = estimator._resolve_past_only_covariates(df, target, cols, past_only_covariates)

    if var_list is not None:
        req = [str(v) for v in var_list]
        vars_for_pdp = [
            v
            for v in req
            if v in df.columns and v != target and pd.api.types.is_numeric_dtype(df[v])
        ]
        # Ensure explicitly queried variables enter via known-future covariates
        # rather than being left out or masked as past-only.
        for v in vars_for_pdp:
            if v not in cols:
                cols.append(v)
        past_only_set = set(vars_for_pdp)
        past_only_cols = [c for c in past_only_cols if c not in past_only_set]
    else:
        vars_for_pdp = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    if not vars_for_pdp:
        raise ValueError("No valid numeric covariates available for chronos_pdp.")

    h = int(horizon or estimator.prediction_length)
    n = len(df)
    if n <= h:
        raise InsufficientContextError(
            f"need more than horizon ({h}) rows to compute chronos_pdp, got {n}"
        )

    obs = df[target].to_numpy(dtype=np.float32)
    ctx = int(estimator.context_length)
    start_a = ctx if n >= ctx + h else max(1, (n - h) // 2)
    valid_anchors = [
        a
        for a in range(start_a, n - h + 1)
        if estimator._context_coverage(obs[max(0, a - ctx) : a]) >= estimator.min_context_coverage
    ]
    if not valid_anchors:
        raise InsufficientContextError(
            f"no anchor window has {100 * estimator.min_context_coverage:.0f}% observed context "
            "for chronos_pdp"
        )

    k = min(int(n_anchors), len(valid_anchors))
    pick_idx = np.unique(np.linspace(0, len(valid_anchors) - 1, k, dtype=int))
    anchors = [valid_anchors[int(i)] for i in pick_idx]
    anchor_windows = [
        (
            obs[max(0, a - ctx) : a],
            df.iloc[max(0, a - ctx) : a],
            df.iloc[a : a + h],
        )
        for a in anchors
    ]

    rows: list[dict[str, Any]] = []
    q_indices: tuple[int, int, int] | None = None
    for var in vars_for_pdp:
        grid = _numeric_grid(df[var], (q_lo, q_hi), int(grid_points))
        if grid is None or len(grid) == 0:
            continue
        items: list[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]] = []
        for gval in grid:
            for hist, past, future in anchor_windows:
                fut = future.copy()
                fut[var] = float(gval)
                items.append((hist, past, fut))

        if past_only_cols:
            blocks = estimator._forecast_blocks(items, cols, h, past_only_cols=past_only_cols)
        else:
            blocks = estimator._forecast_blocks(items, cols, h)
        if q_indices is None:
            q_indices = _resolve_q_triple(estimator, quantiles)
        qi_lo, qi_mid, qi_hi = q_indices

        n_a = len(anchor_windows)
        reshaped = blocks.reshape(len(grid), n_a, blocks.shape[1], blocks.shape[2])
        for g_idx, gval in enumerate(grid):
            med = reshaped[g_idx, :, qi_mid, :]
            rows.append(
                {
                    "variable": var,
                    "value": float(gval),
                    "pdp_mean": float(np.nanmean(med)),
                    "pdp_std": float(np.nanstd(med)),
                    "pdp_lower": float(np.nanmean(reshaped[g_idx, :, qi_lo, :])),
                    "pdp_upper": float(np.nanmean(reshaped[g_idx, :, qi_hi, :])),
                }
            )

    cols_out = ["variable", "value", "pdp_mean", "pdp_std", "pdp_lower", "pdp_upper"]
    return pd.DataFrame(rows, columns=cols_out)


def _find_nan_runs(values: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous half-open ``[start, end)`` index spans of non-finite entries."""
    bad = ~np.isfinite(values)
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, flag in enumerate(bad):
        if flag and not in_run:
            start = i
            in_run = True
        elif not flag and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(bad)))
    return runs


def chronos_impute(
    estimator: Any,
    df: pd.DataFrame,
    target: str,
    *,
    covariates: Sequence[str] | None = None,
    past_only_covariates: Sequence[str] | None = None,
    direction: str = "bidirectional",
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    max_gap: int | None = None,
    calibration: QuantileCalibration | CalibratedEstimator | None = None,
) -> pd.DataFrame:
    """Impute contiguous ``NaN`` gaps in ``target`` conditioned on meteorology.

    Rolls forward and/or backward across each missing span using observed context
    outside the gap and observed covariates across the gap, then smoothly blends
    both passes weighted by proximity to each gap boundary (``bidirectional``), or
    falls back cleanly to whichever direction has sufficient context. Optionally
    applies a :class:`QuantileCalibration` fit (or unwraps a
    :class:`CalibratedEstimator`) to rescale the prediction interval bounds while
    keeping the median imputation unchanged.
    """
    if target not in df.columns:
        raise KeyError(f"target column absent from frame: {target!r}")
    if direction not in ("forward", "backward", "bidirectional"):
        raise ValueError(
            f"direction must be 'forward', 'backward', or 'bidirectional', got {direction!r}"
        )
    if max_gap is not None and int(max_gap) < 1:
        raise ValueError(f"max_gap must be >= 1, got {max_gap}")

    base, cal = _resolve_cal_and_base(estimator, calibration)
    base._check_index(df.index)
    cols = (
        list(covariates)
        if covariates is not None
        else base._resolve_covariates(df, target, exclude=past_only_covariates or ())
    )
    past_only_cols = base._resolve_past_only_covariates(df, target, cols, past_only_covariates)

    obs = df[target].to_numpy(dtype=float)
    n = len(obs)
    imputed = obs.copy()
    is_imputed = np.zeros(n, dtype=bool)
    lower = np.full(n, np.nan, dtype=float)
    upper = np.full(n, np.nan, dtype=float)

    runs = [(s, e) for s, e in _find_nan_runs(obs) if max_gap is None or (e - s) <= int(max_gap)]
    out_cols = ["observed", "imputed", "is_imputed", "lower", "upper"]
    if not runs:
        return pd.DataFrame(
            {
                "observed": obs,
                "imputed": imputed,
                "is_imputed": is_imputed,
                "lower": lower,
                "upper": upper,
            },
            index=df.index,
            columns=out_cols,
        )

    ctx = int(base.context_length)
    step = int(base.prediction_length)
    q_indices: list[tuple[int, int, int]] = []

    def _roll_gap(
        frame: pd.DataFrame, target_arr: np.ndarray, s: int, e: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if s <= 0:
            return None
        ctx_start = max(0, s - ctx)
        init_hist = target_arr[ctx_start:s]
        if base._context_coverage(init_hist) < base.min_context_coverage:
            return None

        hist = init_hist.astype(np.float32, copy=True)
        past_df = frame.iloc[ctx_start:s]
        med_chunks: list[np.ndarray] = []
        lo_chunks: list[np.ndarray] = []
        hi_chunks: list[np.ndarray] = []

        cur = s
        while cur < e:
            nxt = min(cur + step, e)
            fut_df = frame.iloc[cur:nxt]
            if past_only_cols:
                block = base._forecast_block(
                    hist, past_df, fut_df, cols, past_only_cols=past_only_cols
                )
            else:
                block = base._forecast_block(hist, past_df, fut_df, cols)
            block_arr = np.asarray(block, dtype=float)
            if cal is not None:
                block_arr = cal.rescale_block(block_arr, base._q_index)
            if not q_indices:
                q_indices.append(_resolve_q_triple(base, quantiles))
            qi_lo, qi_mid, qi_hi = q_indices[0]
            med_c = np.asarray(block_arr[qi_mid], dtype=float)
            med_chunks.append(med_c)
            lo_chunks.append(np.asarray(block_arr[qi_lo], dtype=float))
            hi_chunks.append(np.asarray(block_arr[qi_hi], dtype=float))
            hist = np.concatenate([hist, med_c.astype(np.float32)])[-ctx:]
            past_df = pd.concat([past_df, fut_df]).iloc[-ctx:]
            cur = nxt

        return (
            np.concatenate(med_chunks),
            np.concatenate(lo_chunks),
            np.concatenate(hi_chunks),
        )

    use_fwd = direction in ("forward", "bidirectional")
    use_bwd = direction in ("backward", "bidirectional")
    rev_df = df.iloc[::-1].reset_index(drop=True) if use_bwd else None
    rev_obs = obs[::-1] if use_bwd else None

    for s, e in runs:
        length = e - s
        fwd = _roll_gap(df, obs, s, e) if use_fwd else None
        bwd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if use_bwd and rev_df is not None and rev_obs is not None:
            bwd_raw = _roll_gap(rev_df, rev_obs, n - e, n - s)
            if bwd_raw is not None:
                bwd = (bwd_raw[0][::-1], bwd_raw[1][::-1], bwd_raw[2][::-1])

        if fwd is not None and bwd is not None:
            k = np.arange(length, dtype=float)
            w_fwd = (length - k) / (length + 1.0)
            w_bwd = (k + 1.0) / (length + 1.0)
            med_fill = w_fwd * fwd[0] + w_bwd * bwd[0]
            lo_fill = w_fwd * fwd[1] + w_bwd * bwd[1]
            hi_fill = w_fwd * fwd[2] + w_bwd * bwd[2]
        elif fwd is not None:
            med_fill, lo_fill, hi_fill = fwd
        elif bwd is not None:
            med_fill, lo_fill, hi_fill = bwd
        else:
            log.warning(
                "Skipping imputation of gap [%d:%d) (%d rows): insufficient observed context.",
                s,
                e,
                length,
            )
            continue

        imputed[s:e] = med_fill
        lower[s:e] = lo_fill
        upper[s:e] = hi_fill
        is_imputed[s:e] = True

    return pd.DataFrame(
        {
            "observed": obs,
            "imputed": imputed,
            "is_imputed": is_imputed,
            "lower": lower,
            "upper": upper,
        },
        index=df.index,
        columns=out_cols,
    )


def _outward_monotone(q_vals: np.ndarray, q_levels: np.ndarray) -> np.ndarray:
    """Enforce non-crossing outward from the median for partial-band calibration."""
    qv = np.asarray(q_vals, dtype=float).copy()
    mid = int(np.argmin(np.abs(q_levels - 0.5)))
    qv[: mid + 1] = np.minimum.accumulate(qv[: mid + 1][::-1])[::-1]
    qv[mid:] = np.maximum.accumulate(qv[mid:])
    return qv


def _summary_from_rows(
    y: np.ndarray,
    q_mat: np.ndarray,
    pits: np.ndarray,
    q_levels: np.ndarray,
    intervals: tuple[float, ...],
    crps_fn: Any,
    crps_obs_fn: Any,
    pinball_fn: Any,
) -> dict[str, Any]:
    """Compute probabilistic verification metrics for one set of backtest rows."""
    from ..core.exceptions import DataError

    mid_i = int(np.argmin(np.abs(q_levels - 0.5)))
    finite = np.isfinite(y) & np.isfinite(q_mat).all(axis=1)
    if not finite.any():
        raise DataError("No finite forecast-observation pairs in chronos_backtest")

    yf = y[finite]
    qf = q_mat[finite]
    med = qf[:, mid_i]
    pf = pits[finite & np.isfinite(pits)]

    crps_val = float(crps_fn(qf, yf, quantiles=list(q_levels)))
    crps_ref = float(crps_obs_fn(med, yf))
    pinball_50 = float(pinball_fn(med, yf, float(q_levels[mid_i])))
    err = med - yf
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((yf - np.mean(yf)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    crps_skill = float(1.0 - crps_val / crps_ref) if crps_ref > 0.0 else float("nan")

    if pf.size > 0:
        ps = np.sort(pf)
        m = ps.size
        ks = float(
            max(
                float(np.max(np.abs(np.arange(1, m + 1) / m - ps))),
                float(np.max(np.abs(np.arange(0, m) / m - ps))),
            )
        )
        pit_mean = float(np.mean(pf))
        pit_std = float(np.std(pf))
    else:
        ks = float("nan")
        pit_mean = float("nan")
        pit_std = float("nan")

    out: dict[str, Any] = {
        "n_obs": int(yf.size),
        "crps": crps_val,
        "crps_reference": crps_ref,
        "crps_skill": crps_skill,
        "pinball_50": pinball_50,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "r2": r2,
        "pit_mean": pit_mean,
        "pit_std": pit_std,
        "pit_ks_stat": ks,
    }
    for iv in intervals:
        lo_lvl = 0.5 * (1.0 - iv)
        hi_lvl = 1.0 - lo_lvl
        lo_vals = np.array([float(np.interp(lo_lvl, q_levels, row)) for row in qf], dtype=float)
        hi_vals = np.array([float(np.interp(hi_lvl, q_levels, row)) for row in qf], dtype=float)
        key = f"coverage_{int(round(100.0 * iv))}"
        out[key] = float(np.mean((yf >= lo_vals) & (yf <= hi_vals)))
    return out


def chronos_backtest(
    estimator: Any,
    df: pd.DataFrame,
    target: str,
    *,
    covariates: Sequence[str] | None = None,
    past_only_covariates: Sequence[str] | None = None,
    horizon: int | None = None,
    stride: int | None = None,
    quantiles: Sequence[float] | None = None,
    intervals: Sequence[float] = (0.5, 0.8, 0.9),
    calibrate: bool = False,
    calibration_frac: float = 0.5,
) -> dict[str, Any]:
    """Rolling multi-horizon probabilistic verification and conformal calibration closure.

    Evaluates ``estimator`` over rolling forecast origins across ``df`` conditioned on
    observed covariates, scoring exact Continuous Ranked Probability Score (CRPS),
    pinball loss, PIT uniformity, and empirical coverage overall and by forecast lead.
    When ``calibrate=True``, fits split-conformal :class:`QuantileCalibration` on
    the earlier block prefix (reusing cached forecasts when ``stride == horizon``)
    and scores the held-out suffix both before and after calibration.
    """
    from ..analysis.verification import crps_from_quantiles, crps_observed, pinball_loss

    if target not in df.columns:
        raise KeyError(f"target column absent from frame: {target!r}")
    if not (0.0 < float(calibration_frac) < 1.0):
        raise ValueError(f"calibration_frac must be in (0, 1), got {calibration_frac}")
    iv_tuple = tuple(sorted({float(iv) for iv in intervals}))
    if not iv_tuple or any(not (0.0 < iv < 1.0) for iv in iv_tuple):
        raise ValueError(f"intervals must be non-empty values in (0, 1), got {intervals!r}")

    base, existing_cal = _resolve_cal_and_base(estimator, None)
    base._check_index(df.index)

    h = int(horizon or base.prediction_length)
    step = int(stride) if stride is not None else h
    if h < 1:
        raise ValueError(f"horizon must be >= 1, got {h}")
    if step < 1:
        raise ValueError(f"stride must be >= 1, got {step}")

    cols = (
        list(covariates)
        if covariates is not None
        else base._resolve_covariates(df, target, exclude=past_only_covariates or ())
    )
    past_only_cols = base._resolve_past_only_covariates(df, target, cols, past_only_covariates)

    n = len(df)
    spans = [(s, min(s + h, n)) for s in range(int(base.context_length), n, step)]
    if not spans:
        raise InsufficientContextError(
            f"need more than context_length ({base.context_length}) rows for chronos_backtest, got {n}"
        )

    obs = df[target].to_numpy(dtype=float)
    obs_f32 = obs.astype(np.float32)
    skipped = base._blocks_without_context(obs, spans, "backtest")
    active = [sp for sp in spans if sp not in skipped]

    raw_by_span: dict[tuple[int, int], np.ndarray] = {}
    by_len: dict[int, list[tuple[int, int]]] = {}
    for s, e in active:
        by_len.setdefault(e - s, []).append((s, e))

    for block_len, group in by_len.items():
        items = [
            (
                obs_f32[max(0, s - int(base.context_length)) : s],
                df.iloc[max(0, s - int(base.context_length)) : s],
                df.iloc[s:e],
            )
            for s, e in group
        ]
        if past_only_cols:
            blocks = base._forecast_blocks(items, cols, block_len, past_only_cols=past_only_cols)
        else:
            blocks = base._forecast_blocks(items, cols, block_len)
        for sp, raw_block in zip(group, blocks, strict=True):
            raw_by_span[sp] = np.asarray(raw_block, dtype=float)

    native_levels = np.asarray(base.quantile_levels, dtype=float)
    if quantiles is None:
        q_indices = list(range(len(native_levels)))
    else:
        q_indices = sorted({int(base._q_index(float(q))) for q in quantiles})
        if len(q_indices) < 2:
            raise ValueError("quantiles must map to at least 2 distinct quantile levels")
    q_levels = np.asarray([float(native_levels[i]) for i in q_indices], dtype=float)
    mid_local = int(np.argmin(np.abs(q_levels - 0.5)))

    def _score_subset(
        subset_spans: Sequence[tuple[int, int]],
        cal_obj: QuantileCalibration | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        pred_rows: list[dict[str, Any]] = []
        q_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        pits: list[float] = []
        leads: list[int] = []

        for anchor_idx, (s, e) in enumerate(subset_spans):
            blk = raw_by_span[(s, e)]
            if cal_obj is not None:
                blk = cal_obj.rescale_block(blk, base._q_index)
            q_sub = blk[q_indices]  # (len(q_levels), block_len)
            anchor_ts = df.index[s]
            for offset in range(e - s):
                t_idx = s + offset
                y_val = float(obs[t_idx])
                qv_raw = q_sub[:, offset]
                qv = (
                    _outward_monotone(qv_raw, q_levels)
                    if (cal_obj is not None and np.all(np.isfinite(qv_raw)))
                    else np.asarray(qv_raw, dtype=float)
                )
                p = (
                    _empirical_pit(y_val, qv, q_levels)
                    if (np.isfinite(y_val) and np.all(np.isfinite(qv)))
                    else float("nan")
                )
                lo80 = (
                    float(np.interp(0.1, q_levels, qv)) if np.all(np.isfinite(qv)) else float("nan")
                )
                hi80 = (
                    float(np.interp(0.9, q_levels, qv)) if np.all(np.isfinite(qv)) else float("nan")
                )
                row_d: dict[str, Any] = {
                    "anchor_idx": anchor_idx,
                    "anchor": anchor_ts,
                    "date": df.index[t_idx],
                    "lead": offset + 1,
                    "observed": y_val,
                    "q10": lo80,
                    "q50": float(qv[mid_local]),
                    "q90": hi80,
                    "pit": p,
                }
                pred_rows.append(row_d)
                q_rows.append(qv)
                y_rows.append(y_val)
                pits.append(p)
                leads.append(offset + 1)

        preds_df = pd.DataFrame(
            pred_rows,
            columns=[
                "anchor_idx",
                "anchor",
                "date",
                "lead",
                "observed",
                "q10",
                "q50",
                "q90",
                "pit",
            ],
        )
        y_arr = np.asarray(y_rows, dtype=float)
        q_mat = np.vstack(q_rows)
        pit_arr = np.asarray(pits, dtype=float)
        lead_arr = np.asarray(leads, dtype=int)

        summary = _summary_from_rows(
            y_arr,
            q_mat,
            pit_arr,
            q_levels,
            iv_tuple,
            crps_from_quantiles,
            crps_observed,
            pinball_loss,
        )

        lead_summaries: list[dict[str, Any]] = []
        for ld in sorted(set(leads)):
            sel = lead_arr == ld
            row_sum = _summary_from_rows(
                y_arr[sel],
                q_mat[sel],
                pit_arr[sel],
                q_levels,
                iv_tuple,
                crps_from_quantiles,
                crps_observed,
                pinball_loss,
            )
            row_sum["lead"] = int(ld)
            lead_summaries.append(row_sum)
        by_lead_df = pd.DataFrame(lead_summaries)
        lead_cols = ["lead"] + [c for c in by_lead_df.columns if c != "lead"]
        by_lead_df = by_lead_df[lead_cols]
        return preds_df, by_lead_df, summary

    preds_df, by_lead_df, summary = _score_subset(active, existing_cal)

    fitted_cal: QuantileCalibration | None = existing_cal
    cal_summary: dict[str, Any] | None = None
    holdout_raw_summary: dict[str, Any] | None = None

    if calibrate:
        if len(active) < 2:
            raise ValueError("calibrate=True requires at least 2 active backtest blocks")
        n_cal = max(1, min(len(active) - 1, int(round(len(active) * float(calibration_frac)))))
        cached_blocks = {
            (pd.Timestamp(df.index[s]), e - s): raw_by_span[(s, e)] for s, e in active[:n_cal]
        }

        class _CachedQuantileEstimator:
            def __init__(self) -> None:
                self.prediction_length = h
                self.context_length = int(base.context_length)

            def predict_quantiles(
                self,
                frame: pd.DataFrame,
                tgt: str,
                anchor: pd.Timestamp | str,
                horizon: int | None = None,
                quantiles: Sequence[float] = (0.1, 0.5, 0.9),
                covariates: Sequence[str] | None = None,
            ) -> pd.DataFrame:
                ts = pd.Timestamp(anchor)
                hor = int(horizon or h)
                blk = cached_blocks.get((ts, hor))
                if blk is not None:
                    fut_idx = frame.loc[frame.index >= ts].iloc[:hor].index
                    return pd.DataFrame(
                        {f"q{q}": blk[base._q_index(float(q))] for q in quantiles},
                        index=fut_idx,
                    )
                return base.predict_quantiles(
                    frame,
                    tgt,
                    ts,
                    horizon=hor,
                    quantiles=quantiles,
                    covariates=covariates,
                    past_only_covariates=past_only_cols or None,
                )

        cand_levels: set[float] = {0.5, 0.05, 0.95}
        for q in q_levels:
            qf = float(f"{float(q):.6g}")
            if abs(float(native_levels[base._q_index(qf)]) - 0.5) > 1e-6:
                cand_levels.add(qf)
                cand_levels.add(float(f"{1.0 - qf:.6g}"))
        for iv in iv_tuple:
            hi_iv = float(f"{0.5 + 0.5 * iv:.6g}")
            if abs(float(native_levels[base._q_index(hi_iv)]) - 0.5) > 1e-6:
                cand_levels.add(hi_iv)
                cand_levels.add(float(f"{1.0 - hi_iv:.6g}"))
        cal_qs = sorted(q for q in cand_levels if 0.0 < q < 1.0)

        cal_anchor = df.index[active[n_cal][0]]
        cal_res = calibrate_quantiles(
            _CachedQuantileEstimator(),
            df,
            target,
            cal_anchor,
            quantiles=cal_qs,
            n_blocks=n_cal,
            covariates=cols,
        )
        fitted_cal = cal_res.calibration
        holdout_spans = active[n_cal:]
        _, _, holdout_raw_summary = _score_subset(holdout_spans, None)
        _, _, cal_summary = _score_subset(holdout_spans, fitted_cal)

    return {
        "predictions": preds_df,
        "by_lead": by_lead_df,
        "summary": summary,
        "calibration": fitted_cal,
        "calibrated_summary": cal_summary,
        "holdout_uncalibrated_summary": holdout_raw_summary,
    }


class CovariateSensitivityCallback:
    """Trainer callback monitoring meteorological sensitivity during Chronos-2 fine-tuning.

    Periodically evaluates :func:`normet.foundation.estimator.covariate_sensitivity` on a
    held-out context window so fine-tuning can detect (and optionally early-stop) if
    the adapted weights collapse toward pure target autoregression. Compatible with
    HuggingFace ``Trainer(callbacks=[...])`` while requiring zero ``torch`` or
    ``transformers`` imports at module import time.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        anchor: pd.Timestamp | str,
        *,
        horizon: int | None = None,
        met_features: Sequence[str] | None = None,
        context_length: int = 2048,
        prediction_length: int = 168,
        use_calendar: bool = True,
        past_only_covariates: Sequence[str] | None = None,
        eval_steps: int = 50,
        min_pct_shift: float = 1.0,
        patience: int = 1,
        stop_on_collapse: bool = False,
        random_state: int = 0,
    ) -> None:
        if int(eval_steps) < 1:
            raise ValueError(f"eval_steps must be >= 1, got {eval_steps}")
        if float(min_pct_shift) < 0.0:
            raise ValueError(f"min_pct_shift must be >= 0, got {min_pct_shift}")
        if int(patience) < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if target not in df.columns:
            raise KeyError(f"target column absent from frame: {target!r}")

        self.df = df
        self.target = target
        self.anchor = pd.Timestamp(anchor)
        self.horizon = int(horizon or prediction_length)
        self.met_features = list(met_features) if met_features is not None else None
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.use_calendar = bool(use_calendar)
        self.past_only_covariates = (
            list(past_only_covariates) if past_only_covariates is not None else None
        )
        self.eval_steps = int(eval_steps)
        self.min_pct_shift = float(min_pct_shift)
        self.patience = int(patience)
        self.stop_on_collapse = bool(stop_on_collapse)
        self.random_state = int(random_state)
        self.history: list[dict[str, float]] = []
        self._below_count = 0

    def _evaluate_model(self, model: Any, step: int) -> dict[str, float]:
        probe = Chronos2Estimator(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            met_covariates=self.met_features,
            use_calendar=self.use_calendar,
            device="cpu",
            past_only_covariates=self.past_only_covariates,
        )
        if hasattr(model, "predict"):
            probe._pipeline = model
            if hasattr(model, "quantiles"):
                probe._quantiles = np.asarray(model.quantiles, dtype=float)
            shift = probe.covariate_sensitivity(
                self.df,
                self.target,
                self.anchor,
                horizon=self.horizon,
                met_features=self.met_features,
                random_state=self.random_state,
            )
        else:
            from chronos.chronos2 import Chronos2Pipeline

            was_training = bool(getattr(model, "training", False))
            if hasattr(model, "eval"):
                model.eval()
            try:
                probe._pipeline = Chronos2Pipeline(model=model)
                probe._quantiles = np.asarray(probe._pipeline.quantiles, dtype=float)
                shift = probe.covariate_sensitivity(
                    self.df,
                    self.target,
                    self.anchor,
                    horizon=self.horizon,
                    met_features=self.met_features,
                    random_state=self.random_state,
                )
            finally:
                if was_training and hasattr(model, "train"):
                    model.train()

        entry: dict[str, float] = {"step": float(step), **{k: float(v) for k, v in shift.items()}}
        self.history.append(entry)
        return entry

    def on_step_end(
        self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any
    ) -> Any:
        step = int(getattr(state, "global_step", 0))
        if step <= 0 or step % self.eval_steps != 0 or model is None:
            return control

        entry = self._evaluate_model(model, step)
        pct = entry["pct_of_prediction"]
        if pct < self.min_pct_shift:
            self._below_count += 1
            log.warning(
                "Chronos-2 fine-tuning step %d: covariate sensitivity dropped to %.2f%% "
                "(threshold %.2f%%; streak %d/%d).",
                step,
                pct,
                self.min_pct_shift,
                self._below_count,
                self.patience,
            )
            if self.stop_on_collapse and self._below_count >= self.patience:
                control.should_training_stop = True
        else:
            self._below_count = 0
        return control

    # Explicit HuggingFace TrainerCallback lifecycle no-ops for CallbackHandler parity.
    def on_init_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return control

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return control

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return control

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return control

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return control

    def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return control

    def on_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return control

    def on_substep_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return control

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return control

    def on_predict(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return control

    def on_log(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return control

    def on_prediction_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return control

    def on_push_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return control
