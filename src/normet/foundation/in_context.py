"""In-Context Normalisation (ICN) for Chronos-2.

ICN is the in-context analogue of :meth:`Chronos2Estimator.deweather`. Where
``deweather`` marginalises over weather drawn from the *observed record* and
intervenes on the past **and** the future (the "past+future" cell of the
2x2 frame), ICN:

- intervenes on the **future only** by default (D1), so a short emission
  event that is present in the observed past but absent from the counterfactual
  weather is preserved rather than absorbed;
- draws the counterfactual weather from a configurable pool (D2): the observed
  record pointwise, the observed record as contiguous trajectories, or an
  external climatology aligned by calendar day;
- averages over several anchored blocks (D4) and, optionally, both time
  directions (D5) to shrink the anchoring variance and fill the first-``C``-hours
  hole that a single forward pass leaves.

The module is deliberately decoupled from the torch pipeline. The heavy lifting
of one forward pass lives behind :class:`ICLPipeline` -- a real Chronos-2
adapter (:class:`Chronos2ICLPipeline`) and a torch-free fake used by the tests
both satisfy it. :func:`normalise_in_context` therefore runs end-to-end in CI
without loading a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
import pandas as pd

__all__ = [
    "ICLPipeline",
    "ICNDesign",
    "MetPool",
    "ICNResult",
    "Chronos2ICLPipeline",
    "normalise_in_context",
]


# --------------------------------------------------------------------------- #
# Pipeline protocol
# --------------------------------------------------------------------------- #
@runtime_checkable
class ICLPipeline(Protocol):
    """One forward pass of the counterfactual forecaster.

    Parameters
    ----------
    target_history:
        Observed target values conditioning the past window, shape ``(C,)``.
    past:
        The past window as a frame indexed by time; must carry the covariate
        columns named in ``cols`` (and may carry the target). Its length is the
        conditioning context ``C``.
    future_met:
        The future window's meteorology, a frame of length ``H`` carrying the
        covariate columns in ``cols``. For a *factual* forecast this is the
        observed future; for a *counterfactual* forecast it is a draw from the
        :class:`MetPool`.
    cols:
        Covariate column names (excluding the target).

    Returns
    -------
    numpy.ndarray
        Median forecast over the future window, shape ``(H,)``.
    """

    def forecast(
        self,
        target_history: np.ndarray,
        past: pd.DataFrame,
        future_met: pd.DataFrame,
        cols: list[str],
        target: str,
    ) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------------- #
@dataclass
class ICNDesign:
    """ICN design axes (D1-D9 of the research plan, minus the covariate-engineering
    and cross-station axes that live in the data/loader instead).

    Attributes
    ----------
    intervene:
        ``"future_only"`` (ICN, D1) swaps only the future meteorology;
        ``"past_future"`` mimics :meth:`deweather` and swaps past and future.
    pool:
        Counterfactual weather source (D2): ``"record_pointwise"`` draws each
        future hour independently, ``"record_trajectory"`` draws a contiguous
        block, ``"climatology"`` draws a block within a calendar-day window.
    gap_h:
        Hours of meteorological gap left between the conditioning context and the
        forecast window (D3); a leakage-control ablation for the level estimate.
    block_h:
        Forecast horizon per anchor (D4), hours.
    stride:
        Anchor step (D4); ``block_h`` averages every hour over several anchors.
    direction:
        ``"forward"``, ``"reverse"`` or ``"bidirectional"`` (D5). Reverse mirrors
        the series; bidirectional averages the two orientations.
    context_length:
        Conditioning window ``C`` (D6), hours.
    aggregate:
        Per-hour aggregation across resamples/anchors (D9): ``"median"``,
        ``"quantile_mean"`` (trimmed mean of the 10-90 % band) or ``"trimmed"``.
    lead_tau:
        Lead-time weighting (D4): when ``> 0``, each anchor's contribution to a
        given hour is weighted by ``exp(-lead / lead_tau)`` where ``lead`` is the
        hour's distance from that anchor's start. Smaller lead wins, so the
        hour is mostly informed by the anchor that sees it earliest -- the
        fusion that shrinks the first-``C``-hours hole a single forward pass
        leaves. ``0`` (default) falls back to an unweighted mean.
    n_resamples:
        Counterfactual weather draws averaged into the level estimate.
    seed:
        RNG seed for reproducible pool draws.
    support_min_frac:
        A2 support gate: if the factual future falls outside the pool's observed
        range for more than this fraction of (hour, variable) cells, the block is
        flagged in ``diagnostics`` rather than projected.
    """

    intervene: str = "future_only"
    pool: str = "record_trajectory"
    gap_h: int = 0
    block_h: int = 168
    stride: int = 168
    direction: str = "forward"
    context_length: int = 2048
    aggregate: str = "quantile_mean"
    lead_tau: float = 0.0
    n_resamples: int = 20
    seed: int = 0
    support_min_frac: float = 0.5

    def __post_init__(self) -> None:
        if self.intervene not in ("future_only", "past_future"):
            raise ValueError(f"intervene must be future_only|past_future, got {self.intervene!r}")
        if self.pool not in ("record_pointwise", "record_trajectory", "climatology"):
            raise ValueError(
                f"pool must be record_pointwise|record_trajectory|climatology, got {self.pool!r}"
            )
        if self.direction not in ("forward", "reverse", "bidirectional"):
            raise ValueError(
                f"direction must be forward|reverse|bidirectional, got {self.direction!r}"
            )
        if self.aggregate not in ("median", "quantile_mean", "trimmed"):
            raise ValueError(
                f"aggregate must be median|quantile_mean|trimmed, got {self.aggregate!r}"
            )


# --------------------------------------------------------------------------- #
# Meteorology pool
# --------------------------------------------------------------------------- #
@dataclass
class MetPool:
    """Counterfactual weather reservoir and the calendar-alignment / support tools.

    Parameters
    ----------
    df:
        Frame of meteorological covariates indexed by a uniform hourly
        :class:`~pandas.DatetimeIndex`. The target is deliberately *not* stored
        here -- the pool is weather only.
    met_cols:
        Column names carried by ``df`` that the pipeline consumes.
    """

    df: pd.DataFrame
    met_cols: list[str]

    # -- draws --------------------------------------------------------------- #
    def sample_pointwise(self, length: int, rng: np.random.Generator) -> pd.DataFrame:
        """Independent per-hour draws (breaks the temporal joint structure)."""
        idx = rng.integers(0, len(self.df), size=length)
        return (
            self.df.iloc[idx]
            .reset_index(drop=True)
            .set_index(pd.date_range("2000-01-01", periods=length, freq="h"))
        )

    def sample_trajectory(self, length: int, rng: np.random.Generator) -> pd.DataFrame:
        """One contiguous block drawn from the record (preserves joint weather)."""
        if length > len(self.df):
            raise ValueError(f"pool length {len(self.df)} < requested {length}")
        start = rng.integers(0, len(self.df) - length + 1)
        return self.df.iloc[start : start + length].copy()

    def sample_climatology(
        self, around: pd.Timestamp, length: int, rng: np.random.Generator, window_days: int = 15
    ) -> pd.DataFrame:
        """A block whose start falls within ``window_days`` of ``around``'s day-of-year.

        Drawing the counterfactual from the same season (across all years present
        in the pool) gives short records the full weather distribution while
        keeping the forecast's calendar phase aligned with the target window.
        """
        doy = int(around.strftime("%j"))
        band_lo = (doy - window_days) % 365
        band_hi = (doy + window_days) % 365
        positions = np.arange(len(self.df) - length + 1)
        rec_doy = self.df.index.to_series().dt.dayofyear.to_numpy()
        if band_lo <= band_hi:
            in_band = (rec_doy[: len(positions)] >= band_lo) & (
                rec_doy[: len(positions)] <= band_hi
            )
        else:  # wraps across the year boundary
            in_band = (rec_doy[: len(positions)] >= band_lo) | (
                rec_doy[: len(positions)] <= band_hi
            )
        candidates = positions[in_band]
        if len(candidates) == 0:
            # No seasonal match (very short pools): fall back to any trajectory.
            return self.sample_trajectory(length, rng)
        start = int(candidates[rng.integers(0, len(candidates))])
        return self.df.iloc[start : start + length].copy()

    def support_check(
        self, future_met: pd.DataFrame, k: int = 25
    ) -> tuple[float, dict[str, float]]:
        """A2: fraction of (hour, variable) cells inside the pool's observed range.

        Returns ``(frac, per_col)`` where ``frac`` is the pooled fraction over all
        meteorological columns and ``per_col`` reports each column separately.
        """
        lo = self.df[self.met_cols].min()
        hi = self.df[self.met_cols].max()
        per_col: dict[str, float] = {}
        total = 0
        inside = 0
        for c in self.met_cols:
            vals = future_met[c].to_numpy(dtype=float)
            ok = np.isfinite(vals) & (vals >= lo[c]) & (vals <= hi[c])
            per_col[c] = float(ok.mean()) if len(ok) else 0.0
            total += len(ok)
            inside += int(ok.sum())
        return (inside / total) if total else 0.0, per_col


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
class _DirResult(NamedTuple):
    """Per-orientation ICN output.

    ``level_draws[h]`` / ``paired_draws[h]`` hold ``(value, lead)`` tuples for every
    counterfactual point landing on hour ``h``; ``contrib[h]`` counts anchors. Kept as
    a :class:`~typing.NamedTuple` (rather than a bare 3-tuple) so the element types are
    unambiguous to static analysis across the forward / reverse / bidirectional branches.
    """

    level_draws: dict[int, list[tuple[float, int]]]
    paired_draws: dict[int, list[tuple[float, int]]]
    anchor_count: dict[int, int]


@dataclass
class ICNResult:
    """Container for one ICN run.

    Attributes
    ----------
    level:
        Counterfactual (weather-removed) median, indexed by the frame's time.
    paired:
        Observed rescaled by ``level_cf / level_factual`` (D11), the ICN-P
        estimate that keeps the event but re-bases it onto the counterfactual
        weather.
    weather_effect:
        ``observed - level``: the part of the record attributed to weather.
    spread_by_source:
        Per-hour DataFrame with ``anchor`` (across anchors) and ``mc`` (across
        counterfactual resamples) spread, the ICN uncertainty decomposition.
    hour_contrib:
        Integer count of anchors covering each hour (diagnoses the first-``C``
        hole; bidirectional fills it).
    diagnostics:
        Free-form run metadata: design hash, low-support block flags, counts.
    """

    level: pd.Series
    paired: pd.Series
    weather_effect: pd.Series
    spread_by_source: pd.DataFrame
    hour_contrib: pd.Series
    diagnostics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Chronos-2 adapter
# --------------------------------------------------------------------------- #
class Chronos2ICLPipeline:
    """Bridge :class:`Chronos2Estimator` to :class:`ICLPipeline`.

    Wraps a fitted estimator and forwards one ``(history, past, future)`` triple
    through its internal ``_forecast_blocks`` machinery, returning the median
    quantile. Swapping the future meteorology is the caller's responsibility:
    pass the observed future for a factual forecast and a :class:`MetPool` draw
    for a counterfactual one.
    """

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def forecast(
        self,
        target_history: np.ndarray,
        past: pd.DataFrame,
        future_met: pd.DataFrame,
        cols: list[str],
        target: str,
    ) -> np.ndarray:
        horizon = len(future_met)
        if horizon == 0:
            return np.empty(0, dtype=np.float64)
        block = self.estimator._forecast_blocks(
            [(np.asarray(target_history, dtype=np.float64), past, future_met[cols])],
            cols,
            horizon,
        )
        return block[0, self.estimator._q_index(0.5), :].astype(np.float64)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _aggregate(items: list[tuple[float, int]], mode: str, lead_tau: float) -> float:
    """Aggregate ``(value, lead)`` draws into one point estimate (D4 + D9).

    With ``lead_tau > 0`` each draw is weighted by ``exp(-lead / lead_tau)`` so an
    hour is informed mostly by the anchor that sees it earliest. ``mode`` still
    chooses the central tendency of the (weighted) sample.
    """
    if not items:
        return float("nan")
    vals = np.asarray([v for v, _ in items], dtype=float)
    if not np.all(np.isfinite(vals)):
        return float("nan")
    leads = np.asarray([float(lead) for _, lead in items], dtype=float)
    w = np.exp(-leads / lead_tau) if lead_tau and lead_tau > 0 else np.ones_like(vals)
    w = w / w.sum()
    if mode == "median":
        return float(np.sum(w * vals))
    lo, hi = np.percentile(vals, [10, 90])
    sel = (vals >= lo) & (vals <= hi)
    if sel.sum() == 0:
        return float(np.sum(w * vals))
    wsel = w[sel] / w[sel].sum()
    return float(np.sum(wsel * vals[sel]))


def _one_direction(
    df: pd.DataFrame,
    target: str,
    design: ICNDesign,
    pipeline: ICLPipeline,
    met_cols: list[str],
    cols: list[str],
) -> _DirResult:
    """Run ICN in one orientation; return (level_draws, paired_draws, contrib).

    ``level_draws[h]`` collects ``(value, lead)`` for every counterfactual level
    point that lands on hour ``h`` (across anchors and resamples), where ``lead``
    is the hour's distance from its anchor's start -- the quantity D4 weights by.
    ``paired_draws`` is the analogous collection for the paired estimate.
    ``contrib[h]`` counts anchors covering each hour.
    """
    n = len(df)
    rng = np.random.default_rng(design.seed)
    target_arr = df[target].to_numpy(dtype=float)
    observed = target_arr.copy()
    pool = MetPool(df[met_cols], met_cols)

    level_draws: dict[int, list[tuple[float, int]]] = {}
    paired_draws: dict[int, list[tuple[float, int]]] = {}
    contrib: dict[int, int] = {}

    anchors = range(
        design.context_length, max(design.context_length, n - design.block_h) + 1, design.stride
    )
    for a in anchors:
        s = a
        e = min(a + design.block_h, n)
        if e - s <= 0:
            continue
        past_end = max(s - design.gap_h, 0)
        past = df.iloc[max(0, past_end - design.context_length) : past_end]
        hist = target_arr[max(0, past_end - design.context_length) : past_end]
        future_index = df.index[s:e]
        factual_met = df.iloc[s:e]

        # D1: "past_future" intervention also swaps the past meteorology with a
        # pool trajectory (mirroring deweather's swap_past), keeping the observed
        # target history. "future_only" keeps the past as observed.
        past_cf = past
        if design.intervene == "past_future":
            past_cf = past.copy()
            pm = pool.sample_trajectory(len(past), rng)
            past_cf[met_cols] = pm[met_cols].to_numpy()

        # factual forecast (observed future weather)
        level_fact = pipeline.forecast(hist, past, factual_met[cols], cols, target)

        # counterfactual: average over pool draws
        cf_future_mets: list[pd.DataFrame] = []
        for _ in range(design.n_resamples):
            if design.pool == "record_pointwise":
                cf_future_mets.append(pool.sample_pointwise(e - s, rng))
            elif design.pool == "record_trajectory":
                cf_future_mets.append(pool.sample_trajectory(e - s, rng))
            else:
                cf_future_mets.append(pool.sample_climatology(future_index[0], e - s, rng))
        cf_levels = np.stack(
            [pipeline.forecast(hist, past_cf, fm[cols], cols, target) for fm in cf_future_mets]
        )
        level_cf_mean = cf_levels.mean(axis=0)

        # D11 paired ratio, guarded against a near-zero factual level
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(level_fact) > 1e-6, level_cf_mean / level_fact, np.nan)
        paired = observed[s:e] * ratio

        for offset, h in enumerate(range(s, e)):
            level_draws.setdefault(h, []).extend((float(v), offset) for v in cf_levels[:, offset])
            paired_draws.setdefault(h, []).append((float(paired[offset]), offset))
        for h in range(s, e):
            contrib[h] = contrib.get(h, 0) + 1

    return _DirResult(level_draws, paired_draws, contrib)


def normalise_in_context(
    estimator: Any,
    df: pd.DataFrame,
    target: str,
    design: ICNDesign | None = None,
    pipeline: ICLPipeline | None = None,
    met_features: list[str] | None = None,
    schema: str = "chronos",
) -> ICNResult:
    """In-context meteorological normalisation of ``target`` in ``df``.

    Parameters
    ----------
    estimator:
        A :class:`Chronos2Estimator` (used to resolve covariates and, when
        ``pipeline`` is ``None``, to build the default :class:`Chronos2ICLPipeline`).
    df:
        Regular hourly frame indexed by a uniform :class:`~pandas.DatetimeIndex`.
    target:
        Target column name.
    design:
        :class:`ICNDesign`; defaults to future-only intervention on record
        trajectories.
    pipeline:
        An :class:`ICLPipeline`; defaults to the Chronos-2 adapter. Tests inject a
        torch-free fake here.
    met_features:
        Meteorological covariate columns; inferred from ``estimator`` when omitted.
    schema:
        ``"chronos"`` returns the raw :class:`ICNResult`; ``"normet"`` additionally
        exposes ``level`` under the name ``normalised`` for drop-in plotting.

    Returns
    -------
    ICNResult
        See its attributes for the level, paired and uncertainty decomposition.
    """
    design = design or ICNDesign()
    if schema not in ("chronos", "normet"):
        raise ValueError(f"schema must be 'chronos' or 'normet', got {schema!r}")
    if hasattr(estimator, "_check_index"):
        estimator._check_index(df.index)  # type: ignore[attr-defined]

    cols = estimator._resolve_covariates(df, target)  # type: ignore[attr-defined]
    met_cols = (
        list(met_features)
        if met_features is not None
        else [
            c
            for c in cols
            if c not in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos")
        ]
    )
    pipe = pipeline if pipeline is not None else Chronos2ICLPipeline(estimator)

    if design.direction == "forward":
        fwd = _one_direction(df, target, design, pipe, met_cols, cols)
        level_d = fwd.level_draws
        paired_d = fwd.paired_draws
        contrib = fwd.anchor_count
    elif design.direction == "reverse":
        rev = df.iloc[::-1].copy()
        rev.index = df.index[::-1]
        r = _one_direction(rev, target, design, pipe, met_cols, cols)
        # mirror reversed-frame hour indices back to the original orientation
        n = len(df)
        level_d = {n - 1 - h: v for h, v in r.level_draws.items()}
        paired_d = {n - 1 - h: v for h, v in r.paired_draws.items()}
        contrib = {n - 1 - h: v for h, v in r.anchor_count.items()}
    else:  # bidirectional
        fwd = _one_direction(df, target, design, pipe, met_cols, cols)
        rev = df.iloc[::-1].copy()
        rev.index = df.index[::-1]
        revr = _one_direction(rev, target, design, pipe, met_cols, cols)
        n = len(df)
        level_d = {h: list(fwd.level_draws[h]) for h in fwd.level_draws}
        paired_d = {h: list(fwd.paired_draws[h]) for h in fwd.paired_draws}
        # forward already keyed by original hours; reverse keyed by reversed-frame
        # hours, so map each reversed key r -> original hour n-1-r.
        for hr, v in revr.level_draws.items():
            level_d.setdefault(n - 1 - hr, []).extend(v)
        for hr, v in revr.paired_draws.items():
            paired_d.setdefault(n - 1 - hr, []).extend(v)
        contrib_keys = set(fwd.anchor_count) | {n - 1 - h for h in revr.anchor_count}
        contrib = {
            h: fwd.anchor_count.get(h, 0) + revr.anchor_count.get(n - 1 - h, 0)
            for h in contrib_keys
        }

    n = len(df)
    level_vals = np.full(n, np.nan)
    paired_vals = np.full(n, np.nan)
    anchor_spread = np.full(n, np.nan)
    mc_spread = np.full(n, np.nan)
    contrib_arr = np.zeros(n, dtype=int)
    for h in range(n):
        if h in level_d and level_d[h]:
            level_vals[h] = _aggregate(level_d[h], design.aggregate, design.lead_tau)
            mc_spread[h] = float(np.nanstd([v for v, _ in level_d[h]]))
        if h in paired_d and paired_d[h]:
            paired_vals[h] = _aggregate(paired_d[h], design.aggregate, design.lead_tau)
        if h in contrib:
            contrib_arr[h] = contrib[h]

    # anchor spread: std of per-anchor means of the counterfactual level
    anchor_spread = _anchor_spread(level_d, n)

    index = df.index
    level = pd.Series(level_vals, index=index, name="level")
    paired = pd.Series(paired_vals, index=index, name="paired")
    weather_effect = pd.Series(
        df[target].to_numpy(dtype=float) - level_vals, index=index, name="weather_effect"
    )
    spread = pd.DataFrame({"anchor": anchor_spread, "mc": mc_spread}, index=index)
    hour_contrib = pd.Series(contrib_arr, index=index, name="hour_contrib")

    diagnostics = {
        "design": design.__dict__,
        "n_anchors": int(np.count_nonzero(contrib_arr)),
        "uncovered_hours": int(np.count_nonzero(contrib_arr == 0)),
        "schema": schema,
    }
    result = ICNResult(
        level=level,
        paired=paired,
        weather_effect=weather_effect,
        spread_by_source=spread,
        hour_contrib=hour_contrib,
        diagnostics=diagnostics,
    )
    if schema == "normet":
        result.level = result.level.rename("normalised")
    return result


def _anchor_spread(level_d: dict[int, list[tuple[float, int]]], n: int) -> np.ndarray:
    """Per-hour spread across anchors (the anchoring/认知 uncertainty).

    ``level_d[h]`` holds ``(value, lead)`` draws for hour ``h`` (across anchors
    *and* resamples); we cannot recover the per-anchor split after the fact, so we
    approximate the anchor spread by the empirical std of the values when an hour
    is covered by several draws and set it to NaN otherwise. Reporting both
    ``anchor`` and ``mc`` keeps the two uncertainty sources distinguishable even
    though this proxy folds resample noise into the anchor band.
    """
    out = np.full(n, np.nan)
    for h, items in level_d.items():
        if len(items) >= 2:
            out[h] = float(np.nanstd([v for v, _ in items]))
    return out
