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

import gzip
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .estimator import _qname

__all__ = [
    "ICLPipeline",
    "BatchICLPipeline",
    "MultivariateICLPipeline",
    "ICNDesign",
    "MetPool",
    "ICNResult",
    "ICNDrawBank",
    "Chronos2ICLPipeline",
    "Chronos2MultivariateICLPipeline",
    "build_icn_draw_bank",
    "decompose_icn_uncertainty",
    "icn_decompose",
    "normalise_in_context",
    "normalise_in_context_multivariate",
    "normalise_in_context_multisite",
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


@runtime_checkable
class BatchICLPipeline(ICLPipeline, Protocol):
    """Batched extension of :class:`ICLPipeline` for GPU/accelerated passes."""

    def forecast_batch(
        self,
        items: Sequence[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]],
        cols: list[str],
        target: str,
    ) -> np.ndarray: ...


@runtime_checkable
class MultivariateICLPipeline(Protocol):
    """Joint multivariate batched extension for shared-frame multi-target ICN."""

    def forecast_multivariate_batch(
        self,
        items: Sequence[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]],
        cols: list[str],
        targets: Sequence[str],
        quantiles: Sequence[float] | None = None,
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
        Per-hour aggregation across resamples/anchors (D9): ``"mean"``
        (the default; weighted by ``lead_tau`` when it is set), ``"median"``
        (weighted median) or ``"trimmed"`` (weighted mean of the 10-90 %
        percentile band).  Until 2026-11-11 ``"median"`` was a mean and
        ``"quantile_mean"`` was the same code path as ``"trimmed"``, so a
        scan that compared the three names compared two estimators.
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
    min_context:
        Shortest conditioning window an anchor may use, hours.  ``None`` keeps
        the old behaviour -- an anchor only exists once ``context_length``
        hours precede it, so a record shorter than ``context_length`` produced
        **no output at all** (under 85.3 days at C=2048), and a three-month
        record yielded one partial anchor over its last 136 hours (6 % of the
        record).  With a value, anchors may start at ``min_context`` and the
        first one uses the whole record up to its block; the context then
        shrinks instead of disappearing.  The shortest usable value is set by
        the forecaster's own minimum (Chronos-2 needs a few hundred hours).
    support_min_frac:
        A2 support gate: an anchor whose factual future falls outside the
        pool's observed range for more than this fraction of (hour, variable)
        cells is dropped, and the anchor index and fraction are recorded in
        ``diagnostics["low_support"]``.  A counterfactual drawn from a pool
        that never saw such weather is an extrapolation, not an intervention.
    """

    intervene: str = "future_only"
    pool: str = "record_trajectory"
    gap_h: int = 0
    block_h: int = 168
    stride: int = 168
    direction: str = "forward"
    context_length: int = 2048
    aggregate: str = "trimmed"
    lead_tau: float = 0.0
    n_resamples: int = 20
    seed: int = 0
    min_context: int | None = None
    support_min_frac: float = 0.5
    log_transform: bool = False
    surprise_tau: float = 0.0

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
        if self.aggregate not in ("mean", "median", "trimmed", "quantile_mean"):
            raise ValueError(f"aggregate must be mean|median|trimmed, got {self.aggregate!r}")
        if self.min_context is not None and self.min_context <= 0:
            raise ValueError(f"min_context must be positive or None, got {self.min_context}")
        if not 0.0 <= self.support_min_frac <= 1.0:
            raise ValueError(f"support_min_frac must be in [0, 1], got {self.support_min_frac}")
        if float(self.surprise_tau) < 0.0:
            raise ValueError(f"surprise_tau must be >= 0, got {self.surprise_tau}")

    @classmethod
    def v1(cls, **overrides: Any) -> ICNDesign:
        """Frozen ICN-v1 preset (2026-11-16), with optional field overrides."""
        params: dict[str, Any] = {
            "intervene": "future_only",
            "pool": "record_trajectory",
            "gap_h": 0,
            "block_h": 336,
            "stride": 168,
            "direction": "bidirectional",
            "context_length": 2048,
            "aggregate": "trimmed",
            "lead_tau": 0.0,
            "n_resamples": 5,
            "seed": 0,
            "min_context": None,
            "support_min_frac": 0.5,
            "log_transform": False,
        }
        params.update(overrides)
        return cls(**params)


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

    @classmethod
    def from_frame(cls, df: pd.DataFrame, met_cols: Sequence[str] | None = None) -> MetPool:
        """Construct a :class:`MetPool` from a datetime-indexed or ``date``-column frame."""
        work = df.copy()
        if not isinstance(work.index, pd.DatetimeIndex):
            if "date" in work.columns:
                work = work.set_index("date")
            work.index = pd.to_datetime(work.index)
        work = work.sort_index()
        if met_cols is None:
            cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
        else:
            cols = list(met_cols)
            missing = [c for c in cols if c not in work.columns]
            if missing:
                raise KeyError(f"pool columns absent from frame: {missing}")
        return cls(df=work[cols].copy(), met_cols=cols)

    def reversed(self) -> MetPool:
        """Return a time-reversed copy so reverse-direction trajectories align with context."""
        rev = self.df.iloc[::-1].copy()
        rev.index = self.df.index[::-1]
        return MetPool(df=rev, met_cols=list(self.met_cols))

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
            # A categorical covariate has categories, not an interval: there
            # is no "inside the pool's range" to ask of it.
            if not (
                pd.api.types.is_numeric_dtype(self.df[c])
                and pd.api.types.is_numeric_dtype(future_met[c])
            ):
                continue
            vals = future_met[c].to_numpy(dtype=float)
            ok = np.isfinite(vals) & (vals >= lo[c]) & (vals <= hi[c])
            per_col[c] = float(ok.mean()) if len(ok) else 0.0
            total += len(ok)
            inside += int(ok.sum())
        # Nothing numeric to check is vacuously in support, not out of it.
        return (inside / total) if total else 1.0, per_col


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
class _DirResult(NamedTuple):
    """Per-orientation ICN output.

    ``level_draws[h]`` / ``paired_draws[h]`` hold ``(value, lead)`` tuples for every
    counterfactual point landing on hour ``h``; ``contrib[h]`` counts anchors. Kept as
    a :class:`~typing.NamedTuple` (rather than a bare 3-tuple) so the element types are
    unambiguous to static analysis across the forward / reverse / bidirectional branches.

    ``anchor_values[h]`` keeps the *per-anchor* mean of the counterfactual level, one
    entry per anchor covering ``h``, so the anchoring spread can be measured across
    anchors; ``mc_vars[h]`` keeps each anchor's variance across its resamples, so the
    resample spread is measured *within* anchors.  Before 2026-11-11 both columns were
    the std over every draw of the hour, i.e. they measured the same dispersion.
    ``low_support`` lists ``(anchor_start, support_fraction)`` for the blocks the A2
    gate dropped.
    """

    level_draws: dict[int, list[tuple[float, int]]]
    paired_draws: dict[int, list[tuple[float, int]]]
    anchor_count: dict[int, int]
    anchor_values: dict[int, list[float]]
    mc_vars: dict[int, list[float]]
    low_support: list[tuple[int, float]]
    quantile_draws: dict[float, dict[int, list[tuple[float, int]]]]
    paired_quantile_draws: dict[float, dict[int, list[tuple[float, int]]]]
    factual_errors: dict[int, list[float]]


@dataclass
class ICNResult:
    """Container for one ICN run.

    Attributes
    ----------
    level:
        Counterfactual (weather-removed) median, indexed by the frame's time.
    paired:
        Observed rescaled by ``level_cf / level_factual`` (D11), or additive in
        ``log1p`` space when ``log_transform=True``, the ICN-P estimate that keeps
        the event but re-bases it onto the counterfactual weather.
    weather_effect:
        ``observed - level``: the part of the record attributed to weather.
    spread_by_source:
        Per-hour DataFrame with ``anchor`` (across anchors) and ``mc`` (across
        counterfactual resamples) spread, the ICN uncertainty decomposition.
        Both are standard deviations on the scale of ``df[target]``: when the
        caller passes a ``log1p`` frame, these are log-scale spreads. A
        caller returning the levels to original units must rescale them (the
        delta method for ``expm1`` multiplies by ``1 + level``), which is what
        ``do_all(log_transform=True, chronos_mode="icn")`` does.
    hour_contrib:
        Integer count of anchors covering each hour (diagnoses the first-``C``
        hole; bidirectional fills it).
    diagnostics:
        Free-form run metadata: design hash, low-support block flags, counts.
    observed:
        Optional observed target series aligned with ``level`` / ``paired``.
    quantiles:
        Optional DataFrame of aggregated counterfactual level quantiles (``qNNN``),
        populated when ``quantiles`` are requested.
    paired_quantiles:
        Optional DataFrame of aggregated paired counterfactual quantiles (``qNNN``),
        populated when ``quantiles`` are requested.
    """

    level: pd.Series
    paired: pd.Series
    weather_effect: pd.Series
    spread_by_source: pd.DataFrame
    hour_contrib: pd.Series
    diagnostics: dict[str, Any] = field(default_factory=dict)
    observed: pd.Series | None = None
    quantiles: pd.DataFrame | None = None
    paired_quantiles: pd.DataFrame | None = None

    def to_normet_frame(self, estimate: str = "level") -> pd.DataFrame:
        """Convert the result onto :func:`normet.normalise`'s ``observed/normalised/qNNN`` schema."""
        if estimate not in ("level", "paired"):
            raise ValueError(f"estimate must be 'level' or 'paired', got {estimate!r}")
        if self.observed is None:
            raise ValueError("ICNResult has no `observed` series to build a normet frame")
        series = self.level if estimate == "level" else self.paired
        out = pd.DataFrame(
            {
                "observed": self.observed.to_numpy(),
                "normalised": series.to_numpy(),
            },
            index=pd.Index(self.observed.index, name="date"),
        )
        q_src = self.quantiles if estimate == "level" else self.paired_quantiles
        if q_src is not None:
            for col in q_src.columns:
                out[col] = q_src[col].reindex(self.observed.index).to_numpy()
        return out


@dataclass
class ICNDrawBank:
    """Cached raw ICN draws decoupled from orientation/aggregation parameters.

    Enables zero-inference sensitivity sweeps across ``direction``, ``aggregate``,
    ``lead_tau``, and ``schema`` without re-running the foundation model.
    """

    index: pd.DatetimeIndex
    observed: np.ndarray
    target: str
    design: ICNDesign
    forward: _DirResult | None
    reverse: _DirResult | None
    quantiles: tuple[float, ...] | None = None
    external_pool: bool = False
    schema: str = "chronos"

    def _merged_dir(self, direction: str | None) -> tuple[str, _DirResult]:
        """Resolve and mirror/merge directional cached draws."""
        dir_eff = direction if direction is not None else self.design.direction
        if dir_eff not in ("forward", "reverse", "bidirectional"):
            raise ValueError(f"direction must be forward|reverse|bidirectional, got {dir_eff!r}")
        n = len(self.index)
        if dir_eff == "forward":
            if self.forward is None:
                raise ValueError("ICNDrawBank has no cached 'forward' pass")
            return dir_eff, self.forward
        if dir_eff == "reverse":
            if self.reverse is None:
                raise ValueError("ICNDrawBank has no cached 'reverse' pass")
            return dir_eff, _mirror_dir_result(self.reverse, n)
        if self.forward is None or self.reverse is None:
            raise ValueError("ICNDrawBank requires both forward and reverse passes")
        return dir_eff, _merge_dir_results(self.forward, self.reverse, n)

    def aggregate(
        self,
        *,
        direction: str | None = None,
        aggregate: str | None = None,
        lead_tau: float | None = None,
        surprise_tau: float | None = None,
        schema: str | None = None,
    ) -> ICNResult:
        """Materialise an :class:`ICNResult` from cached draws under updated aggregation."""
        agg_eff = aggregate if aggregate is not None else self.design.aggregate
        tau_eff = float(lead_tau) if lead_tau is not None else self.design.lead_tau
        surp_eff = float(surprise_tau) if surprise_tau is not None else self.design.surprise_tau
        schema_eff = schema if schema is not None else self.schema
        if agg_eff not in ("mean", "median", "trimmed", "quantile_mean"):
            raise ValueError(f"aggregate must be mean|median|trimmed, got {agg_eff!r}")
        if surp_eff < 0.0:
            raise ValueError(f"surprise_tau must be >= 0, got {surp_eff!r}")
        if schema_eff not in ("chronos", "normet"):
            raise ValueError(f"schema must be 'chronos' or 'normet', got {schema_eff!r}")

        dir_eff, merged = self._merged_dir(direction)
        eff_design = replace(
            self.design,
            direction=dir_eff,
            aggregate=agg_eff,
            lead_tau=tau_eff,
            surprise_tau=surp_eff,
        )
        return _assemble_icn_result(
            index=self.index,
            observed_arr=self.observed,
            merged=merged,
            design=eff_design,
            req_qs=self.quantiles,
            ext_pool=self.external_pool,
            schema=schema_eff,
        )

    def decompose_uncertainty(self, *, direction: str | None = None) -> Any:
        """Partition cached ICN variance into weather and anchor sources (:func:`decompose_icn_uncertainty`)."""
        return decompose_icn_uncertainty(self, direction=direction)

    def save(self, path: str | Path) -> Path:
        """Persist raw :class:`ICNDrawBank` draws to a gzip-compressed JSON file.

        Enables GPU-once / CPU-everywhere workflows: run Chronos-2 once, save
        the draw bank, and sweep ``direction``, ``aggregate``, ``lead_tau``,
        ``surprise_tau``, or uncertainty decompositions offline without torch.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "index_iso": [ts.isoformat() for ts in self.index],
            "index_tz": str(self.index.tz) if self.index.tz is not None else None,
            "index_freq": self.index.freqstr,
            "index_name": self.index.name,
            "observed": [None if not np.isfinite(v) else float(v) for v in self.observed],
            "target": self.target,
            "design": dict(self.design.__dict__),
            "forward": _dir_to_json(self.forward) if self.forward is not None else None,
            "reverse": _dir_to_json(self.reverse) if self.reverse is not None else None,
            "quantiles": list(self.quantiles) if self.quantiles is not None else None,
            "external_pool": bool(self.external_pool),
            "schema": self.schema,
        }
        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, allow_nan=False)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> ICNDrawBank:
        """Load an :class:`ICNDrawBank` previously written by :meth:`save`."""
        in_path = Path(path)
        with gzip.open(in_path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("index_freq") is not None:
            try:
                idx = pd.DatetimeIndex(
                    payload["index_iso"],
                    freq=payload["index_freq"],
                    name=payload.get("index_name"),
                )
            except Exception:
                idx = pd.DatetimeIndex(payload["index_iso"], name=payload.get("index_name"))
        else:
            idx = pd.DatetimeIndex(payload["index_iso"], name=payload.get("index_name"))
        obs = np.asarray(
            [float("nan") if v is None else float(v) for v in payload["observed"]],
            dtype=np.float64,
        )
        design_dict = dict(payload["design"])
        design = ICNDesign(**design_dict)
        qs = tuple(float(q) for q in payload["quantiles"]) if payload.get("quantiles") else None
        fwd = _dir_from_json(payload["forward"]) if payload.get("forward") is not None else None
        rev = _dir_from_json(payload["reverse"]) if payload.get("reverse") is not None else None
        return cls(
            index=idx,
            observed=obs,
            target=str(payload["target"]),
            design=design,
            forward=fwd,
            reverse=rev,
            quantiles=qs,
            external_pool=bool(payload.get("external_pool", False)),
            schema=str(payload.get("schema", "chronos")),
        )


def _sanitise_float(v: float) -> float | None:
    fv = float(v)
    return fv if np.isfinite(fv) else None


def _restore_float(v: Any) -> float:
    return float("nan") if v is None else float(v)


def _dir_to_json(r: _DirResult) -> dict[str, Any]:
    return {
        "level_draws": {
            str(h): [[_sanitise_float(v), int(lead)] for v, lead in items]
            for h, items in r.level_draws.items()
        },
        "paired_draws": {
            str(h): [[_sanitise_float(v), int(lead)] for v, lead in items]
            for h, items in r.paired_draws.items()
        },
        "anchor_count": {str(h): int(c) for h, c in r.anchor_count.items()},
        "anchor_values": {
            str(h): [_sanitise_float(v) for v in vals] for h, vals in r.anchor_values.items()
        },
        "mc_vars": {str(h): [_sanitise_float(v) for v in vals] for h, vals in r.mc_vars.items()},
        "low_support": [[int(a), _sanitise_float(x)] for a, x in r.low_support],
        "quantile_draws": {
            str(q): {
                str(h): [[_sanitise_float(v), int(lead)] for v, lead in items]
                for h, items in hmap.items()
            }
            for q, hmap in r.quantile_draws.items()
        },
        "paired_quantile_draws": {
            str(q): {
                str(h): [[_sanitise_float(v), int(lead)] for v, lead in items]
                for h, items in hmap.items()
            }
            for q, hmap in r.paired_quantile_draws.items()
        },
        "factual_errors": {
            str(h): [_sanitise_float(v) for v in vals] for h, vals in r.factual_errors.items()
        },
    }


def _dir_from_json(d: Mapping[str, Any]) -> _DirResult:
    return _DirResult(
        level_draws={
            int(h): [(_restore_float(v), int(lead)) for v, lead in items]
            for h, items in d["level_draws"].items()
        },
        paired_draws={
            int(h): [(_restore_float(v), int(lead)) for v, lead in items]
            for h, items in d["paired_draws"].items()
        },
        anchor_count={int(h): int(c) for h, c in d["anchor_count"].items()},
        anchor_values={
            int(h): [_restore_float(v) for v in vals] for h, vals in d["anchor_values"].items()
        },
        mc_vars={int(h): [_restore_float(v) for v in vals] for h, vals in d["mc_vars"].items()},
        low_support=[(int(a), _restore_float(x)) for a, x in d["low_support"]],
        quantile_draws={
            float(q): {
                int(h): [(_restore_float(v), int(lead)) for v, lead in items]
                for h, items in hmap.items()
            }
            for q, hmap in d["quantile_draws"].items()
        },
        paired_quantile_draws={
            float(q): {
                int(h): [(_restore_float(v), int(lead)) for v, lead in items]
                for h, items in hmap.items()
            }
            for q, hmap in d["paired_quantile_draws"].items()
        },
        factual_errors={
            int(h): [_restore_float(v) for v in vals]
            for h, vals in d.get("factual_errors", {}).items()
        },
    )


def decompose_icn_uncertainty(bank: ICNDrawBank, *, direction: str | None = None) -> Any:
    """Zero-inference Law-of-Total-Variance decomposition over cached :class:`ICNDrawBank` draws.

    Separates the counterfactual predictive spread at each timestamp into:

    - ``var_weather``: mean within-anchor meteorological resampling variance;
    - ``var_model``: across-anchor context-origin variance (of each anchor's conditional mean);
    - ``var_interaction``: identically zero (nested independent resamples per anchor).

    Satisfies ``var_weather + var_model + var_interaction == var_total`` identically and
    returns an :class:`~normet.analysis.uncertainty.UncDecomposition` ready for ``.summary()``,
    ``.frame()``, and ``.plot()``.
    """
    from ..analysis.uncertainty import UncDecomposition

    _, merged = bank._merged_dir(direction)
    n = len(bank.index)
    var_weather = np.full(n, np.nan, dtype=float)
    var_model = np.full(n, np.nan, dtype=float)
    var_interaction = np.full(n, np.nan, dtype=float)
    var_total = np.full(n, np.nan, dtype=float)
    max_anchors = 0

    for h in range(n):
        a_vals = merged.anchor_values.get(h)
        w_vals = merged.mc_vars.get(h)
        if not a_vals or not w_vals:
            continue
        a_arr = np.asarray(a_vals, dtype=float)
        w_arr = np.asarray(w_vals, dtype=float)
        finite = np.isfinite(a_arr) & np.isfinite(w_arr)
        if not finite.any():
            continue
        af = a_arr[finite]
        wf = w_arr[finite]
        max_anchors = max(max_anchors, int(af.size))
        vw = float(np.mean(wf))
        vm = float(np.var(af))
        var_weather[h] = vw
        var_model[h] = vm
        var_interaction[h] = 0.0
        var_total[h] = vw + vm

    return UncDecomposition(
        var_weather=var_weather,
        var_model=var_model,
        var_interaction=var_interaction,
        var_total=var_total,
        index=bank.index,
        target=bank.target,
        n_models=max(1, max_anchors),
        n_weather_draws=int(bank.design.n_resamples),
        seed=int(bank.design.seed),
        notes=[
            "Computed from cached ICNDrawBank draws via the exact Law of Total Variance "
            "(zero extra model forward passes).",
            "`model` measures across-anchor context-origin variance; `weather` measures "
            "mean within-anchor meteorological resampling variance.",
            "`interaction` is identically zero because each anchor draws nested resamples.",
        ],
    )


# --------------------------------------------------------------------------- #
# Chronos-2 adapter
# --------------------------------------------------------------------------- #
def _uniform_horizon(
    items: Sequence[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]],
    name: str,
) -> int:
    """Extract and validate a single uniform future horizon across batched items."""
    horizons = {len(future) for _, _, future in items}
    if len(horizons) != 1:
        raise ValueError(f"{name} requires uniform horizons, got {sorted(horizons)}")
    return int(horizons.pop())


class Chronos2ICLPipeline:
    """Bridge :class:`Chronos2Estimator` to :class:`ICLPipeline` / :class:`BatchICLPipeline`.

    Wraps a fitted estimator and forwards ``(history, past, future)`` triples
    through its internal ``_forecast_blocks`` machinery, returning the median
    (or requested quantile) trajectories. Swapping the future meteorology is the
    caller's responsibility: pass the observed future for a factual forecast and
    a :class:`MetPool` draw for a counterfactual one.
    """

    def __init__(self, estimator: Any, *, cross_learning: bool = False) -> None:
        self.estimator = estimator
        self.cross_learning = bool(cross_learning)

    def _run_blocks(
        self,
        items: Sequence[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]],
        cols: list[str],
        horizon: int,
    ) -> np.ndarray:
        prepared = [
            (np.asarray(hist, dtype=np.float64), past, future[cols]) for hist, past, future in items
        ]
        if self.cross_learning:
            return self.estimator._forecast_blocks(prepared, cols, horizon, cross_learning=True)
        return self.estimator._forecast_blocks(prepared, cols, horizon)

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
        block = self._run_blocks([(target_history, past, future_met)], cols, horizon)
        return block[0, self.estimator._q_index(0.5), :].astype(np.float64)

    def forecast_batch(
        self,
        items: Sequence[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]],
        cols: list[str],
        target: str,
    ) -> np.ndarray:
        """Batch-evaluate uniform-horizon ``(history, past, future)`` items, returning ``(n, H)``."""
        if not items:
            return np.empty((0, 0), dtype=np.float64)
        horizon = _uniform_horizon(items, "forecast_batch")
        if horizon == 0:
            return np.empty((len(items), 0), dtype=np.float64)
        blocks = self._run_blocks(items, cols, horizon)
        return blocks[:, self.estimator._q_index(0.5), :].astype(np.float64)

    def forecast_quantiles_batch(
        self,
        items: Sequence[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]],
        cols: list[str],
        target: str,
        quantiles: Sequence[float],
    ) -> np.ndarray:
        """Batch-evaluate uniform-horizon items for multiple quantile levels ``(n, Q, H)``."""
        if not items:
            return np.empty((0, len(quantiles), 0), dtype=np.float64)
        horizon = _uniform_horizon(items, "forecast_quantiles_batch")
        if horizon == 0:
            return np.empty((len(items), len(quantiles), 0), dtype=np.float64)
        blocks = self._run_blocks(items, cols, horizon)
        q_idx = [self.estimator._q_index(float(q)) for q in quantiles]
        return blocks[:, q_idx, :].astype(np.float64)


class Chronos2MultivariateICLPipeline:
    """Adapter executing joint multivariate Chronos-2 predictions for ICN."""

    def __init__(self, estimator: Any, *, cross_learning: bool = False) -> None:
        self.estimator = estimator
        self.cross_learning = bool(cross_learning)

    def forecast_multivariate_batch(
        self,
        items: Sequence[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]],
        cols: list[str],
        targets: Sequence[str],
        quantiles: Sequence[float] | None = None,
    ) -> np.ndarray:
        n_var = len(targets)
        if not items:
            if quantiles is None:
                return np.empty((0, n_var, 0), dtype=np.float64)
            return np.empty((0, n_var, len(quantiles), 0), dtype=np.float64)
        horizon = _uniform_horizon(items, "forecast_multivariate_batch")
        if horizon == 0:
            if quantiles is None:
                return np.empty((len(items), n_var, 0), dtype=np.float64)
            return np.empty((len(items), n_var, len(quantiles), 0), dtype=np.float64)

        inputs: list[dict[str, Any]] = []
        for hist, past, future in items:
            stacked = np.asarray(hist, dtype=np.float32)
            if stacked.ndim != 2 or stacked.shape[0] != n_var:
                raise ValueError(f"Expected target_history shape ({n_var}, T), got {stacked.shape}")
            for row in stacked:
                self.estimator._check_context(row)
            inputs.append(self.estimator._make_input(stacked, past, future[cols], cols))

        raw = (
            self.estimator._predict_raw(inputs, horizon, cross_learning=True)
            if self.cross_learning
            else self.estimator._predict_raw(inputs, horizon)
        )
        if len(raw) != len(items):
            raise RuntimeError(f"pipeline returned {len(raw)} forecasts for {len(items)} inputs")
        blocks = np.stack(raw, axis=0)
        if blocks.shape[1] != n_var:
            raise RuntimeError(f"Expected {n_var} variates from pipeline, got {blocks.shape[1]}")
        if quantiles is None:
            return blocks[:, :, self.estimator._q_index(0.5), :].astype(np.float64)
        q_idx = [self.estimator._q_index(float(q)) for q in quantiles]
        return blocks[:, :, q_idx, :].astype(np.float64)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _aggregate(
    items: list[tuple[float, int]],
    mode: str,
    lead_tau: float,
    *,
    surprise_errors: Sequence[float] | None = None,
    surprise_tau: float = 0.0,
) -> float:
    """Aggregate ``(value, lead)`` draws into one point estimate (D4 + D9).

    With ``lead_tau > 0`` each draw is weighted by ``exp(-lead / lead_tau)`` so an
    hour is informed mostly by the anchor that sees it earliest. When
    ``surprise_tau > 0`` and ``surprise_errors`` are supplied, each anchor's
    draws are additionally gated by ``exp(-(err / surprise_tau)**2)`` so anchors
    whose factual forecast jumped across a step-event boundary do not smear
    the pre/post-event edge. ``mode`` chooses the central tendency of the
    (weighted) sample:

    - ``"mean"`` -- the weighted mean;
    - ``"median"`` -- the weighted median (first value whose cumulative weight
      reaches one half), which tolerates a heavy right tail from anchors whose
      counterfactual landed on a near-zero weather hour;
    - ``"trimmed"`` -- the weighted mean of the draws inside the 10-90 %
      percentile band, falling back to the full mean when the band is empty;
    - ``"quantile_mean"`` -- deprecated alias of ``"trimmed"`` (it was a
      separate name for the same code path).
    """
    if not items:
        return float("nan")
    vals = np.asarray([v for v, _ in items], dtype=float)
    if not np.all(np.isfinite(vals)):
        return float("nan")
    leads = np.asarray([float(lead) for _, lead in items], dtype=float)
    w = np.exp(-leads / lead_tau) if lead_tau and lead_tau > 0 else np.ones_like(vals)
    if surprise_tau and surprise_tau > 0.0 and surprise_errors:
        n_err = len(surprise_errors)
        if n_err > 0 and len(items) % n_err == 0:
            k_rep = len(items) // n_err
            err_arr = np.repeat(np.asarray(surprise_errors, dtype=float), k_rep)
            err_arr = np.where(np.isfinite(err_arr), err_arr, 0.0)
            w = w * np.exp(-((err_arr / float(surprise_tau)) ** 2))
    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        w = np.ones_like(vals) / float(len(vals))
    else:
        w = w / w_sum
    if mode == "mean":
        return float(np.sum(w * vals))
    if mode == "median":
        order = np.argsort(vals, kind="stable")
        cum = np.cumsum(w[order])
        return float(vals[order][np.searchsorted(cum, 0.5 - 1e-12, side="left")])
    lo, hi = np.percentile(vals, [10, 90])
    sel = (vals >= lo) & (vals <= hi)
    if sel.sum() == 0 or float(w[sel].sum()) <= 0.0:
        return float(np.sum(w * vals))
    wsel = w[sel] / w[sel].sum()
    return float(np.sum(wsel * vals[sel]))


class _PendingAnchor(NamedTuple):
    """Prepared anchor awaiting chunked pipeline execution."""

    start: int
    end: int
    factual_item: tuple[np.ndarray, pd.DataFrame, pd.DataFrame]
    cf_items: list[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]]


class _PreparedSiteAnchor(NamedTuple):
    """Per-site anchor keyed for cross-site slot batching in multisite ICN."""

    site: str
    start: int
    end: int
    obs_seg: np.ndarray
    z_obs_seg: np.ndarray
    factual_item: tuple[np.ndarray, pd.DataFrame, pd.DataFrame]
    cf_items: list[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]]


def _design_hash(design: ICNDesign) -> str:
    d = dict(design.__dict__)
    # Preserve the frozen ICN-v1 hash when surprise_tau is at its default 0.0.
    if float(d.get("surprise_tau", 0.0)) == 0.0:
        d.pop("surprise_tau", None)
    payload = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _empty_dir_result(quantiles: tuple[float, ...] | None) -> _DirResult:
    qs = quantiles or ()
    return _DirResult(
        level_draws={},
        paired_draws={},
        anchor_count={},
        anchor_values={},
        mc_vars={},
        low_support=[],
        quantile_draws={q: {} for q in qs},
        paired_quantile_draws={q: {} for q in qs},
        factual_errors={},
    )


def _mirror_dir_result(r: _DirResult, n: int) -> _DirResult:
    """Mirror reversed-frame hour indices back onto the original orientation."""
    return _DirResult(
        level_draws={n - 1 - h: v for h, v in r.level_draws.items()},
        paired_draws={n - 1 - h: v for h, v in r.paired_draws.items()},
        anchor_count={n - 1 - h: v for h, v in r.anchor_count.items()},
        anchor_values={n - 1 - h: v for h, v in r.anchor_values.items()},
        mc_vars={n - 1 - h: v for h, v in r.mc_vars.items()},
        low_support=[(n - 1 - a, x) for a, x in r.low_support],
        quantile_draws={
            q: {n - 1 - h: v for h, v in hmap.items()} for q, hmap in r.quantile_draws.items()
        },
        paired_quantile_draws={
            q: {n - 1 - h: v for h, v in hmap.items()}
            for q, hmap in r.paired_quantile_draws.items()
        },
        factual_errors={n - 1 - h: v for h, v in r.factual_errors.items()},
    )


def _merge_dir_results(fwd: _DirResult, revr: _DirResult, n: int) -> _DirResult:
    """Combine forward and reversed orientation results onto original hour indices."""
    level_d = {h: list(fwd.level_draws[h]) for h in fwd.level_draws}
    paired_d = {h: list(fwd.paired_draws[h]) for h in fwd.paired_draws}
    anchor_v = {h: list(fwd.anchor_values[h]) for h in fwd.anchor_values}
    mc_v = {h: list(fwd.mc_vars[h]) for h in fwd.mc_vars}
    low_support = list(fwd.low_support)
    q_draws = {
        q: {h: list(vals) for h, vals in hmap.items()} for q, hmap in fwd.quantile_draws.items()
    }
    pq_draws = {
        q: {h: list(vals) for h, vals in hmap.items()}
        for q, hmap in fwd.paired_quantile_draws.items()
    }
    f_errs = {h: list(fwd.factual_errors[h]) for h in fwd.factual_errors}
    # forward already keyed by original hours; reverse keyed by reversed-frame
    # hours, so map each reversed key r -> original hour n-1-r.
    for hr, lvals in revr.level_draws.items():
        level_d.setdefault(n - 1 - hr, []).extend(lvals)
    for hr, pvals in revr.paired_draws.items():
        paired_d.setdefault(n - 1 - hr, []).extend(pvals)
    for hr, avals in revr.anchor_values.items():
        anchor_v.setdefault(n - 1 - hr, []).extend(avals)
    for hr, mvals in revr.mc_vars.items():
        mc_v.setdefault(n - 1 - hr, []).extend(mvals)
    for hr, evals in revr.factual_errors.items():
        f_errs.setdefault(n - 1 - hr, []).extend(evals)
    for q, hmap in revr.quantile_draws.items():
        dst = q_draws.setdefault(q, {})
        for hr, qvals in hmap.items():
            dst.setdefault(n - 1 - hr, []).extend(qvals)
    for q, hmap in revr.paired_quantile_draws.items():
        dst = pq_draws.setdefault(q, {})
        for hr, pqvals in hmap.items():
            dst.setdefault(n - 1 - hr, []).extend(pqvals)
    low_support.extend((n - 1 - a, x) for a, x in revr.low_support)
    contrib_keys = set(fwd.anchor_count) | {n - 1 - h for h in revr.anchor_count}
    contrib = {
        h: fwd.anchor_count.get(h, 0) + revr.anchor_count.get(n - 1 - h, 0) for h in contrib_keys
    }
    return _DirResult(
        level_draws=level_d,
        paired_draws=paired_d,
        anchor_count=contrib,
        anchor_values=anchor_v,
        mc_vars=mc_v,
        low_support=low_support,
        quantile_draws=q_draws,
        paired_quantile_draws=pq_draws,
        factual_errors=f_errs,
    )


def _aggregate_quantile_map(
    q_map: dict[float, dict[int, list[tuple[float, int]]]],
    req_qs: tuple[float, ...],
    n: int,
    index: pd.DatetimeIndex,
    aggregate_mode: str,
    lead_tau: float,
    *,
    factual_errors: dict[int, list[float]] | None = None,
    surprise_tau: float = 0.0,
) -> pd.DataFrame:
    q_cols_data: dict[str, np.ndarray] = {}
    for q in req_qs:
        q_arr = np.full(n, np.nan)
        qd = q_map.get(q, {})
        for h in range(n):
            if h in qd and qd[h]:
                errs = factual_errors.get(h) if factual_errors is not None else None
                q_arr[h] = _aggregate(
                    qd[h],
                    aggregate_mode,
                    lead_tau,
                    surprise_errors=errs,
                    surprise_tau=surprise_tau,
                )
        q_cols_data[_qname(q)] = q_arr
    return pd.DataFrame(q_cols_data, index=index)


def _assemble_icn_result(
    *,
    index: pd.DatetimeIndex,
    observed_arr: np.ndarray,
    merged: _DirResult,
    design: ICNDesign,
    req_qs: tuple[float, ...] | None,
    ext_pool: bool,
    schema: str,
) -> ICNResult:
    n = len(index)
    level_vals = np.full(n, np.nan)
    paired_vals = np.full(n, np.nan)
    mc_spread = np.full(n, np.nan)
    contrib_arr = np.zeros(n, dtype=int)
    for h in range(n):
        errs_h = merged.factual_errors.get(h)
        if h in merged.level_draws and merged.level_draws[h]:
            level_vals[h] = _aggregate(
                merged.level_draws[h],
                design.aggregate,
                design.lead_tau,
                surprise_errors=errs_h,
                surprise_tau=design.surprise_tau,
            )
            if h in merged.mc_vars and merged.mc_vars[h]:
                # within-anchor resample spread, pooled over the anchors that
                # cover the hour (their resamples are independent draws)
                mc_spread[h] = float(np.sqrt(np.mean(merged.mc_vars[h])))
        if h in merged.paired_draws and merged.paired_draws[h]:
            paired_vals[h] = _aggregate(
                merged.paired_draws[h],
                design.aggregate,
                design.lead_tau,
                surprise_errors=errs_h,
                surprise_tau=design.surprise_tau,
            )
        if h in merged.anchor_count:
            contrib_arr[h] = merged.anchor_count[h]

    q_frame = (
        _aggregate_quantile_map(
            merged.quantile_draws,
            req_qs,
            n,
            index,
            design.aggregate,
            design.lead_tau,
            factual_errors=merged.factual_errors,
            surprise_tau=design.surprise_tau,
        )
        if req_qs is not None
        else None
    )
    pq_frame = (
        _aggregate_quantile_map(
            merged.paired_quantile_draws,
            req_qs,
            n,
            index,
            design.aggregate,
            design.lead_tau,
            factual_errors=merged.factual_errors,
            surprise_tau=design.surprise_tau,
        )
        if req_qs is not None
        else None
    )

    anchor_spread = _anchor_spread(merged.anchor_values, n)
    obs_series = pd.Series(observed_arr, index=index, name="observed")
    level = pd.Series(level_vals, index=index, name="level")
    paired = pd.Series(paired_vals, index=index, name="paired")
    weather_effect = pd.Series(observed_arr - level_vals, index=index, name="weather_effect")
    spread = pd.DataFrame({"anchor": anchor_spread, "mc": mc_spread}, index=index)
    hour_contrib = pd.Series(contrib_arr, index=index, name="hour_contrib")

    diagnostics = {
        "design": dict(design.__dict__),
        "design_hash": _design_hash(design),
        "n_anchors": int(np.count_nonzero(contrib_arr)),
        "uncovered_hours": int(np.count_nonzero(contrib_arr == 0)),
        "schema": schema,
        "min_context": design.min_context,
        "low_support": list(merged.low_support),
        "external_pool": ext_pool,
    }
    result = ICNResult(
        level=level,
        paired=paired,
        weather_effect=weather_effect,
        spread_by_source=spread,
        hour_contrib=hour_contrib,
        diagnostics=diagnostics,
        observed=obs_series,
        quantiles=q_frame,
        paired_quantiles=pq_frame,
    )
    if schema == "normet":
        result.level = result.level.rename("normalised")
    return result


def _validate_quantiles_and_schema(
    schema: str,
    quantiles: Sequence[float] | None,
    pipe: Any,
    *,
    multivariate: bool = False,
) -> tuple[tuple[float, ...] | None, list[float], int]:
    if schema not in ("chronos", "normet"):
        raise ValueError(f"schema must be 'chronos' or 'normet', got {schema!r}")
    req_qs: tuple[float, ...] | None = None
    ask_qs: list[float] = []
    i50_ask = 0
    if quantiles is not None:
        req_qs = tuple(dict.fromkeys(float(q) for q in quantiles))
        if not req_qs:
            raise ValueError("quantiles must not be empty")
        for q in req_qs:
            if not (0.0 <= q <= 1.0):
                raise ValueError(f"Quantile must be in [0, 1]: got {q}")
        if not multivariate and not hasattr(pipe, "forecast_quantiles_batch"):
            raise TypeError(
                f"pipeline {type(pipe).__name__} does not implement "
                "forecast_quantiles_batch required for `quantiles`"
            )
        if 0.5 in req_qs:
            ask_qs = list(req_qs)
            i50_ask = ask_qs.index(0.5)
        else:
            ask_qs = [*req_qs, 0.5]
            i50_ask = len(req_qs)
    return req_qs, ask_qs, i50_ask


def _resolve_met_cols(
    estimator: Any,
    df: pd.DataFrame,
    target: str | Sequence[str],
    met_features: list[str] | None,
) -> tuple[list[str], list[str]]:
    cols: list[str] = estimator._resolve_covariates(df, target)  # type: ignore[attr-defined]
    met_cols = (
        list(met_features)
        if met_features is not None
        else [
            c
            for c in cols
            if c not in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos")
            # A categorical covariate (a site or season label Chronos accepts)
            # is not weather to resample and marginalise; the A2 check and the
            # pool's marginalisation both assume a numeric column.
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    )
    return cols, met_cols


def _resolve_ext_pool(pool: MetPool | pd.DataFrame | None, met_cols: list[str]) -> MetPool | None:
    if pool is None:
        return None
    if isinstance(pool, pd.DataFrame):
        return MetPool.from_frame(pool, met_cols)
    if isinstance(pool, MetPool):
        missing = [c for c in met_cols if c not in pool.df.columns]
        if missing:
            raise KeyError(f"pool columns absent from MetPool: {missing}")
        if list(pool.met_cols) != list(met_cols):
            return MetPool(df=pool.df[met_cols].copy(), met_cols=list(met_cols))
        return pool
    raise TypeError(f"pool must be a MetPool or DataFrame, got {type(pool).__name__}")


def _transform_target_values(arr: np.ndarray, log_transform: bool) -> np.ndarray:
    if not log_transform:
        return arr
    finite = arr[np.isfinite(arr)]
    if (finite < 0.0).any():
        raise ValueError("log_transform requires non-negative target values")
    return np.where(np.isfinite(arr), np.log1p(np.maximum(arr, 0.0)), np.nan)


def _anchor_range(n: int, design: ICNDesign) -> range:
    # With ``min_context`` the first anchor may use a shortened context (a
    # 6-month record at C=2048 otherwise yields a single anchor, and a 3-month
    # one none at all); without it the first anchor needs a full
    # ``context_length`` of history, as before.
    if design.min_context is None:
        first = design.context_length
    else:
        first = max(1, min(design.min_context, n - 1))
    return range(first, max(first, n - design.block_h) + 1, design.stride)


def _sample_cf_futures(
    hist: np.ndarray,
    past: pd.DataFrame,
    factual_met: pd.DataFrame,
    future_index: pd.Index,
    horizon: int,
    pool: MetPool,
    met_cols: list[str],
    cols: list[str],
    design: ICNDesign,
    rng: np.random.Generator,
    low_support: list[tuple[int, float]],
    anchor_start: int,
) -> list[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]] | None:
    """Apply past intervention + A2 gate and sample counterfactual futures."""
    past_cf = past
    if design.intervene == "past_future":
        past_cf = past.copy()
        if met_cols:
            pm = pool.sample_trajectory(len(past), rng)
            past_cf[met_cols] = pm[met_cols].to_numpy()

    if design.support_min_frac > 0 and met_cols:
        support, _ = pool.support_check(factual_met[met_cols])
        if support < design.support_min_frac:
            low_support.append((int(anchor_start), float(support)))
            return None

    if not met_cols:
        return []

    cf_future_mets: list[pd.DataFrame] = []
    for _ in range(design.n_resamples):
        if design.pool == "record_pointwise":
            cf_future_mets.append(pool.sample_pointwise(horizon, rng))
        elif design.pool == "record_trajectory":
            cf_future_mets.append(pool.sample_trajectory(horizon, rng))
        else:
            cf_future_mets.append(pool.sample_climatology(future_index[0], horizon, rng))

    cf_items: list[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]] = []
    for fm in cf_future_mets:
        ff = factual_met.copy()
        ff[met_cols] = fm[met_cols].to_numpy()
        cf_items.append((hist, past_cf, ff[cols]))
    return cf_items


def _record_anchor_draws(
    acc: _DirResult,
    s: int,
    e: int,
    obs_seg: np.ndarray,
    z_obs_seg: np.ndarray,
    pred_fact: np.ndarray,
    pred_cf: np.ndarray,
    preds_q_cf: np.ndarray | None,
    design: ICNDesign,
    has_met: bool,
    quantiles: tuple[float, ...] | None,
    ask_qs: list[float],
) -> None:
    """Project factual/counterfactual anchor outputs onto linear/log level & paired draws."""
    if design.log_transform:
        cf_levels = np.maximum(np.expm1(pred_cf), 0.0)
        if not has_met:
            paired = np.where(np.isfinite(obs_seg), np.maximum(obs_seg, 0.0), np.nan)
        else:
            paired = np.maximum(np.expm1(z_obs_seg + (pred_cf.mean(axis=0) - pred_fact)), 0.0)
    else:
        cf_levels = pred_cf
        if not has_met:
            paired = obs_seg.copy()
        else:
            level_cf_mean = cf_levels.mean(axis=0)
            # D11 paired ratio, guarded against a near-zero factual level
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(np.abs(pred_fact) > 1e-6, level_cf_mean / pred_fact, np.nan)
            paired = obs_seg * ratio

    if design.log_transform:
        err_seg = np.abs(pred_fact - z_obs_seg)
    else:
        fin_obs = obs_seg[np.isfinite(obs_seg)]
        obs_sd = float(np.std(fin_obs)) if fin_obs.size else 1.0
        scale_seg = np.maximum(np.maximum(np.abs(obs_seg), obs_sd), 1e-6)
        err_seg = np.abs(pred_fact - obs_seg) / scale_seg
    err_seg = np.where(np.isfinite(err_seg), err_seg, 0.0)

    for offset, h in enumerate(range(s, e)):
        acc.level_draws.setdefault(h, []).extend((float(v), offset) for v in cf_levels[:, offset])
        acc.paired_draws.setdefault(h, []).append((float(paired[offset]), offset))
        acc.anchor_values.setdefault(h, []).append(float(cf_levels[:, offset].mean()))
        acc.mc_vars.setdefault(h, []).append(float(cf_levels[:, offset].var()))
        acc.factual_errors.setdefault(h, []).append(float(err_seg[offset]))

    if preds_q_cf is not None and quantiles is not None:
        for q in quantiles:
            qi = ask_qs.index(q)
            raw_q = preds_q_cf[:, qi, :]
            if design.log_transform:
                cf_q = np.maximum(np.expm1(raw_q), 0.0)
                p_q = np.maximum(np.expm1(z_obs_seg[None, :] + (raw_q - pred_fact[None, :])), 0.0)
            else:
                cf_q = raw_q
                with np.errstate(divide="ignore", invalid="ignore"):
                    p_q = np.where(
                        (pred_fact[None, :] > 1e-6) & (obs_seg[None, :] >= 0.0),
                        obs_seg[None, :] * (cf_q / pred_fact[None, :]),
                        np.nan,
                    )
            qd = acc.quantile_draws[q]
            pqd = acc.paired_quantile_draws[q]
            for offset, h in enumerate(range(s, e)):
                qd.setdefault(h, []).extend((float(v), offset) for v in cf_q[:, offset])
                pqd.setdefault(h, []).extend((float(v), offset) for v in p_q[:, offset])

    for h in range(s, e):
        acc.anchor_count[h] = acc.anchor_count.get(h, 0) + 1


def _run_univariate_batch(
    pipeline: ICLPipeline,
    flat_items: Sequence[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]],
    cols: list[str],
    target: str,
    quantiles: tuple[float, ...] | None,
    ask_qs: list[float],
    i50_ask: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    preds_q: np.ndarray | None = None
    if quantiles is not None:
        q_pipe: Any = pipeline
        preds_q = np.asarray(
            q_pipe.forecast_quantiles_batch(flat_items, cols, target, ask_qs),
            dtype=float,
        )
        preds_med = preds_q[:, i50_ask, :]
    elif hasattr(pipeline, "forecast_batch"):
        b_pipe: Any = pipeline
        preds_med = np.asarray(b_pipe.forecast_batch(flat_items, cols, target), dtype=float)
    else:
        preds_med = np.stack(
            [
                np.asarray(pipeline.forecast(h_i, p_i, f_i, cols, target), dtype=float)
                for h_i, p_i, f_i in flat_items
            ]
        )
    return preds_med, preds_q


def _one_direction(
    df: pd.DataFrame,
    target: str,
    design: ICNDesign,
    pipeline: ICLPipeline,
    met_cols: list[str],
    cols: list[str],
    external_pool: MetPool | None = None,
    quantiles: tuple[float, ...] | None = None,
) -> _DirResult:
    """Run ICN in one orientation; return draws, counts, spreads and flags."""
    n = len(df)
    rng = np.random.default_rng(design.seed)
    target_arr = df[target].to_numpy(dtype=float)
    observed = target_arr.copy()
    work_target = _transform_target_values(target_arr, design.log_transform)
    pool = external_pool if external_pool is not None else MetPool(df[met_cols], met_cols)
    acc = _empty_dir_result(quantiles)
    has_met = bool(met_cols)

    can_batch = quantiles is not None or hasattr(pipeline, "forecast_batch")
    est_bs = getattr(getattr(pipeline, "estimator", None), "batch_size", None)
    base_bs = int(est_bs) if est_bs is not None else 32
    batch_limit = (
        max(base_bs, 1 + (max(1, design.n_resamples) if has_met else 0)) if can_batch else 1
    )

    _, ask_qs, i50_ask = _validate_quantiles_and_schema("chronos", quantiles, pipeline)

    pending: list[_PendingAnchor] = []
    pending_items = 0
    pending_horizon: int | None = None

    def _flush() -> None:
        nonlocal pending, pending_items, pending_horizon
        if not pending:
            return
        flat_items: list[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]] = []
        for entry in pending:
            flat_items.append(entry.factual_item)
            flat_items.extend(entry.cf_items)

        preds_med, preds_q = _run_univariate_batch(
            pipeline, flat_items, cols, target, quantiles, ask_qs, i50_ask
        )

        cursor = 0
        for entry in pending:
            s, e = entry.start, entry.end
            n_cf = len(entry.cf_items)
            pred_fact = preds_med[cursor]
            if has_met:
                pred_cf = preds_med[cursor + 1 : cursor + 1 + n_cf]
                preds_q_cf = (
                    preds_q[cursor + 1 : cursor + 1 + n_cf] if preds_q is not None else None
                )
            else:
                pred_cf = np.repeat(pred_fact[None, :], design.n_resamples, axis=0)
                preds_q_cf = (
                    np.repeat(preds_q[cursor][None, :, :], design.n_resamples, axis=0)
                    if preds_q is not None
                    else None
                )
            _record_anchor_draws(
                acc,
                s,
                e,
                observed[s:e],
                work_target[s:e],
                pred_fact,
                pred_cf,
                preds_q_cf,
                design,
                has_met,
                quantiles,
                ask_qs,
            )
            cursor += 1 + n_cf

        pending = []
        pending_items = 0
        pending_horizon = None

    for a in _anchor_range(n, design):
        s = a
        e = min(a + design.block_h, n)
        horizon = e - s
        if horizon <= 0:
            continue
        past_end = max(s - design.gap_h, 0)
        ctx_start = max(0, past_end - design.context_length)
        past = df.iloc[ctx_start:past_end]
        hist = work_target[ctx_start:past_end]
        future_index = df.index[s:e]
        factual_met = df.iloc[s:e]

        cf_items = _sample_cf_futures(
            hist,
            past,
            factual_met,
            future_index,
            horizon,
            pool,
            met_cols,
            cols,
            design,
            rng,
            acc.low_support,
            s,
        )
        if cf_items is None:
            continue

        entry = _PendingAnchor(
            start=s,
            end=e,
            factual_item=(hist, past, factual_met[cols]),
            cf_items=cf_items,
        )
        entry_items = 1 + len(cf_items)
        if pending and (pending_horizon != horizon or pending_items + entry_items > batch_limit):
            _flush()
        pending.append(entry)
        pending_items += entry_items
        pending_horizon = horizon
        if not can_batch:
            _flush()

    _flush()
    return acc


def build_icn_draw_bank(
    estimator: Any,
    df: pd.DataFrame,
    target: str,
    design: ICNDesign | None = None,
    pipeline: ICLPipeline | None = None,
    met_features: list[str] | None = None,
    schema: str = "chronos",
    pool: MetPool | pd.DataFrame | None = None,
    quantiles: Sequence[float] | None = None,
) -> ICNDrawBank:
    """Run ICN forward/reverse model passes once and return a reusable :class:`ICNDrawBank`."""
    design = design or ICNDesign()
    if hasattr(estimator, "_check_index"):
        estimator._check_index(df.index)  # type: ignore[attr-defined]

    cols, met_cols = _resolve_met_cols(estimator, df, target, met_features)
    pipe = pipeline if pipeline is not None else Chronos2ICLPipeline(estimator)
    req_qs, _, _ = _validate_quantiles_and_schema(schema, quantiles, pipe)
    ext_pool = _resolve_ext_pool(pool, met_cols)

    fwd_res: _DirResult | None = None
    rev_res: _DirResult | None = None
    if design.direction in ("forward", "bidirectional"):
        fwd_res = _one_direction(
            df, target, design, pipe, met_cols, cols, external_pool=ext_pool, quantiles=req_qs
        )
    if design.direction in ("reverse", "bidirectional"):
        rev = df.iloc[::-1].copy()
        rev.index = df.index[::-1]
        rev_pool = ext_pool.reversed() if ext_pool is not None else None
        rev_res = _one_direction(
            rev, target, design, pipe, met_cols, cols, external_pool=rev_pool, quantiles=req_qs
        )

    obs_arr = df[target].to_numpy(dtype=float)
    return ICNDrawBank(
        index=pd.DatetimeIndex(df.index),
        observed=obs_arr,
        target=target,
        design=design,
        forward=fwd_res,
        reverse=rev_res,
        quantiles=req_qs,
        external_pool=ext_pool is not None,
        schema=schema,
    )


def normalise_in_context(
    estimator: Any,
    df: pd.DataFrame,
    target: str,
    design: ICNDesign | None = None,
    pipeline: ICLPipeline | None = None,
    met_features: list[str] | None = None,
    schema: str = "chronos",
    pool: MetPool | pd.DataFrame | None = None,
    quantiles: Sequence[float] | None = None,
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
    pool:
        Optional external meteorology reservoir (:class:`MetPool` or
        :class:`~pandas.DataFrame`) supplying counterfactual weather trajectories
        when ``df`` itself is short or a fixed climatology baseline is desired.
        Time-reversed automatically for ``reverse`` / ``bidirectional`` passes.
    quantiles:
        Optional quantile levels in ``[0, 1]`` to aggregate across counterfactual
        draws (requires a pipeline exposing ``forecast_quantiles_batch``).

    Returns
    -------
    ICNResult
        See its attributes for the level, paired and uncertainty decomposition.
    """
    return build_icn_draw_bank(
        estimator,
        df,
        target,
        design=design,
        pipeline=pipeline,
        met_features=met_features,
        schema=schema,
        pool=pool,
        quantiles=quantiles,
    ).aggregate()


def _one_direction_multivariate(
    df: pd.DataFrame,
    targets: list[str],
    design: ICNDesign,
    pipeline: MultivariateICLPipeline,
    met_cols: list[str],
    cols: list[str],
    external_pool: MetPool | None = None,
    quantiles: tuple[float, ...] | None = None,
) -> dict[str, _DirResult]:
    """Run joint multivariate ICN in one orientation sharing counterfactual weather draws."""
    n = len(df)
    n_var = len(targets)
    rng = np.random.default_rng(design.seed)
    obs_mat = np.stack([df[t].to_numpy(dtype=float) for t in targets], axis=0)  # (V, n)
    work_mat = np.stack(
        [_transform_target_values(obs_mat[i], design.log_transform) for i in range(n_var)],
        axis=0,
    )
    pool = external_pool if external_pool is not None else MetPool(df[met_cols], met_cols)
    accs = {t: _empty_dir_result(quantiles) for t in targets}
    low_support: list[tuple[int, float]] = []
    has_met = bool(met_cols)

    est_bs = getattr(getattr(pipeline, "estimator", None), "batch_size", None)
    base_bs = int(est_bs) if est_bs is not None else 32
    batch_limit = max(base_bs, 1 + (max(1, design.n_resamples) if has_met else 0))

    _, ask_qs, i50_ask = _validate_quantiles_and_schema(
        "chronos", quantiles, pipeline, multivariate=True
    )

    pending: list[_PendingAnchor] = []
    pending_items = 0
    pending_horizon: int | None = None

    def _flush() -> None:
        nonlocal pending, pending_items, pending_horizon
        if not pending:
            return
        flat_items: list[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]] = []
        for entry in pending:
            flat_items.append(entry.factual_item)
            flat_items.extend(entry.cf_items)

        preds_q: np.ndarray | None = None
        if quantiles is not None:
            preds_q = np.asarray(
                pipeline.forecast_multivariate_batch(flat_items, cols, targets, ask_qs),
                dtype=float,
            )  # (n, V, Q, H)
            preds_med = preds_q[:, :, i50_ask, :]  # (n, V, H)
        else:
            preds_med = np.asarray(
                pipeline.forecast_multivariate_batch(flat_items, cols, targets, None),
                dtype=float,
            )  # (n, V, H)

        cursor = 0
        for entry in pending:
            s, e = entry.start, entry.end
            n_cf = len(entry.cf_items)
            for vi, tgt in enumerate(targets):
                pred_fact = preds_med[cursor, vi]
                if has_met:
                    pred_cf = preds_med[cursor + 1 : cursor + 1 + n_cf, vi]
                    preds_q_cf = (
                        preds_q[cursor + 1 : cursor + 1 + n_cf, vi] if preds_q is not None else None
                    )
                else:
                    pred_cf = np.repeat(pred_fact[None, :], design.n_resamples, axis=0)
                    preds_q_cf = (
                        np.repeat(preds_q[cursor, vi][None, :, :], design.n_resamples, axis=0)
                        if preds_q is not None
                        else None
                    )
                _record_anchor_draws(
                    accs[tgt],
                    s,
                    e,
                    obs_mat[vi, s:e],
                    work_mat[vi, s:e],
                    pred_fact,
                    pred_cf,
                    preds_q_cf,
                    design,
                    has_met,
                    quantiles,
                    ask_qs,
                )
            cursor += 1 + n_cf

        pending = []
        pending_items = 0
        pending_horizon = None

    for a in _anchor_range(n, design):
        s = a
        e = min(a + design.block_h, n)
        horizon = e - s
        if horizon <= 0:
            continue
        past_end = max(s - design.gap_h, 0)
        ctx_start = max(0, past_end - design.context_length)
        past = df.iloc[ctx_start:past_end]
        hist = work_mat[:, ctx_start:past_end]
        future_index = df.index[s:e]
        factual_met = df.iloc[s:e]

        cf_items = _sample_cf_futures(
            hist,
            past,
            factual_met,
            future_index,
            horizon,
            pool,
            met_cols,
            cols,
            design,
            rng,
            low_support,
            s,
        )
        if cf_items is None:
            continue

        entry = _PendingAnchor(
            start=s,
            end=e,
            factual_item=(hist, past, factual_met[cols]),
            cf_items=cf_items,
        )
        entry_items = 1 + len(cf_items)
        if pending and (pending_horizon != horizon or pending_items + entry_items > batch_limit):
            _flush()
        pending.append(entry)
        pending_items += entry_items
        pending_horizon = horizon

    _flush()
    for tgt in targets:
        accs[tgt].low_support.extend(low_support)
    return accs


def normalise_in_context_multivariate(
    estimator: Any,
    df: pd.DataFrame,
    targets: Sequence[str],
    design: ICNDesign | None = None,
    pipeline: MultivariateICLPipeline | None = None,
    met_features: list[str] | None = None,
    schema: str = "chronos",
    pool: MetPool | pd.DataFrame | None = None,
    quantiles: Sequence[float] | None = None,
) -> dict[str, ICNResult]:
    """Joint multivariate ICN over co-varying pollutants sharing weather draws."""
    target_list = list(dict.fromkeys(str(t) for t in targets))
    if not target_list:
        raise ValueError("targets must contain at least one target column")
    missing = [t for t in target_list if t not in df.columns]
    if missing:
        raise KeyError(f"target columns absent from frame: {missing}")

    design = design or ICNDesign()
    if hasattr(estimator, "_check_index"):
        estimator._check_index(df.index)  # type: ignore[attr-defined]

    cols, met_cols = _resolve_met_cols(estimator, df, target_list, met_features)
    pipe: MultivariateICLPipeline = (
        pipeline if pipeline is not None else Chronos2MultivariateICLPipeline(estimator)
    )
    if not hasattr(pipe, "forecast_multivariate_batch"):
        raise TypeError(
            f"pipeline {type(pipe).__name__} does not implement forecast_multivariate_batch"
        )
    req_qs, _, _ = _validate_quantiles_and_schema(schema, quantiles, pipe, multivariate=True)
    ext_pool = _resolve_ext_pool(pool, met_cols)

    fwd_map: dict[str, _DirResult] | None = None
    rev_map: dict[str, _DirResult] | None = None
    if design.direction in ("forward", "bidirectional"):
        fwd_map = _one_direction_multivariate(
            df, target_list, design, pipe, met_cols, cols, external_pool=ext_pool, quantiles=req_qs
        )
    if design.direction in ("reverse", "bidirectional"):
        rev = df.iloc[::-1].copy()
        rev.index = df.index[::-1]
        rev_pool = ext_pool.reversed() if ext_pool is not None else None
        rev_map = _one_direction_multivariate(
            rev, target_list, design, pipe, met_cols, cols, external_pool=rev_pool, quantiles=req_qs
        )

    idx = pd.DatetimeIndex(df.index)
    out: dict[str, ICNResult] = {}
    for tgt in target_list:
        bank = ICNDrawBank(
            index=idx,
            observed=df[tgt].to_numpy(dtype=float),
            target=tgt,
            design=design,
            forward=fwd_map[tgt] if fwd_map is not None else None,
            reverse=rev_map[tgt] if rev_map is not None else None,
            quantiles=req_qs,
            external_pool=ext_pool is not None,
            schema=schema,
        )
        out[tgt] = bank.aggregate()
    return out


def _one_direction_multisite(
    site_frames: dict[str, pd.DataFrame],
    target: str,
    design: ICNDesign,
    pipeline: ICLPipeline,
    met_cols: list[str],
    cols: list[str],
    external_pool: MetPool | None,
    quantiles: tuple[float, ...] | None,
) -> dict[str, _DirResult]:
    """Execute one orientation across sites, batching co-occurring anchors per slot."""
    accs = {st: _empty_dir_result(quantiles) for st in site_frames}
    has_met = bool(met_cols)
    _, ask_qs, i50_ask = _validate_quantiles_and_schema("chronos", quantiles, pipeline)

    grouped: dict[tuple[pd.Timestamp, int], list[_PreparedSiteAnchor]] = {}
    for st, df_st in site_frames.items():
        n = len(df_st)
        rng = np.random.default_rng(design.seed)
        obs_arr = df_st[target].to_numpy(dtype=float)
        work_arr = _transform_target_values(obs_arr, design.log_transform)
        pool = external_pool if external_pool is not None else MetPool(df_st[met_cols], met_cols)
        for a in _anchor_range(n, design):
            s = a
            e = min(a + design.block_h, n)
            horizon = e - s
            if horizon <= 0:
                continue
            past_end = max(s - design.gap_h, 0)
            ctx_start = max(0, past_end - design.context_length)
            past = df_st.iloc[ctx_start:past_end]
            hist = work_arr[ctx_start:past_end]
            future_index = df_st.index[s:e]
            factual_met = df_st.iloc[s:e]
            cf_items = _sample_cf_futures(
                hist,
                past,
                factual_met,
                future_index,
                horizon,
                pool,
                met_cols,
                cols,
                design,
                rng,
                accs[st].low_support,
                s,
            )
            if cf_items is None:
                continue
            slot = (pd.Timestamp(df_st.index[s]), horizon)
            grouped.setdefault(slot, []).append(
                _PreparedSiteAnchor(
                    site=st,
                    start=s,
                    end=e,
                    obs_seg=obs_arr[s:e],
                    z_obs_seg=work_arr[s:e],
                    factual_item=(hist, past, factual_met[cols]),
                    cf_items=cf_items,
                )
            )

    for group in grouped.values():
        fact_items = [g.factual_item for g in group]
        fact_med, fact_q = _run_univariate_batch(
            pipeline, fact_items, cols, target, quantiles, ask_qs, i50_ask
        )

        cf_med_by_k: list[np.ndarray] = []
        cf_q_by_k: list[np.ndarray] = []
        if has_met:
            for k in range(design.n_resamples):
                k_items = [g.cf_items[k] for g in group]
                k_med, k_q = _run_univariate_batch(
                    pipeline, k_items, cols, target, quantiles, ask_qs, i50_ask
                )
                cf_med_by_k.append(k_med)
                if k_q is not None:
                    cf_q_by_k.append(k_q)

        for gi, g in enumerate(group):
            pred_fact = fact_med[gi]
            if has_met:
                pred_cf = np.stack([m[gi] for m in cf_med_by_k], axis=0)
                preds_q_cf = (
                    np.stack([q_arr[gi] for q_arr in cf_q_by_k], axis=0)
                    if fact_q is not None
                    else None
                )
            else:
                pred_cf = np.repeat(pred_fact[None, :], design.n_resamples, axis=0)
                preds_q_cf = (
                    np.repeat(fact_q[gi][None, :, :], design.n_resamples, axis=0)
                    if fact_q is not None
                    else None
                )
            _record_anchor_draws(
                accs[g.site],
                g.start,
                g.end,
                g.obs_seg,
                g.z_obs_seg,
                pred_fact,
                pred_cf,
                preds_q_cf,
                design,
                has_met,
                quantiles,
                ask_qs,
            )

    return accs


def normalise_in_context_multisite(
    estimator: Any,
    frames: Mapping[str, pd.DataFrame],
    target: str,
    design: ICNDesign | None = None,
    pipeline: ICLPipeline | None = None,
    met_features: list[str] | None = None,
    schema: str = "chronos",
    pool: MetPool | pd.DataFrame | None = None,
    quantiles: Sequence[float] | None = None,
    cross_learning: bool = True,
) -> dict[str, ICNResult]:
    """Run ICN across a station network, optionally sharing cross-site context per slot."""
    if not frames:
        raise ValueError("frames is empty; nothing to normalise")
    names = [str(k) for k in frames]
    site_frames = {str(k): v for k, v in frames.items()}
    design = design or ICNDesign()

    first = site_frames[names[0]]
    if target not in first.columns:
        raise KeyError(f"site {names[0]!r} has no column {target!r}")
    if hasattr(estimator, "_check_index"):
        estimator._check_index(first.index)  # type: ignore[attr-defined]
    cols, met_cols = _resolve_met_cols(estimator, first, target, met_features)

    need_cols = list(dict.fromkeys([target, *cols, *met_cols]))
    for name in names:
        df_st = site_frames[name]
        if target not in df_st.columns:
            raise KeyError(f"site {name!r} has no column {target!r}")
        if hasattr(estimator, "_check_index"):
            estimator._check_index(df_st.index)  # type: ignore[attr-defined]
        missing = [c for c in need_cols if c not in df_st.columns]
        if missing:
            raise KeyError(f"site {name!r} is missing columns {missing}")

    if not cross_learning:
        return {
            name: normalise_in_context(
                estimator,
                site_frames[name],
                target,
                design=design,
                pipeline=pipeline,
                met_features=met_cols,
                schema=schema,
                pool=pool,
                quantiles=quantiles,
            )
            for name in names
        }

    pipe = pipeline if pipeline is not None else Chronos2ICLPipeline(estimator, cross_learning=True)
    req_qs, _, _ = _validate_quantiles_and_schema(schema, quantiles, pipe)
    ext_pool = _resolve_ext_pool(pool, met_cols)

    fwd_map: dict[str, _DirResult] | None = None
    rev_map: dict[str, _DirResult] | None = None
    if design.direction in ("forward", "bidirectional"):
        fwd_map = _one_direction_multisite(
            site_frames, target, design, pipe, met_cols, cols, ext_pool, req_qs
        )
    if design.direction in ("reverse", "bidirectional"):
        rev_frames: dict[str, pd.DataFrame] = {}
        for name, df_st in site_frames.items():
            rev = df_st.iloc[::-1].copy()
            rev.index = df_st.index[::-1]
            rev_frames[name] = rev
        rev_pool = ext_pool.reversed() if ext_pool is not None else None
        rev_map = _one_direction_multisite(
            rev_frames, target, design, pipe, met_cols, cols, rev_pool, req_qs
        )

    out: dict[str, ICNResult] = {}
    for name in names:
        df_st = site_frames[name]
        bank = ICNDrawBank(
            index=pd.DatetimeIndex(df_st.index),
            observed=df_st[target].to_numpy(dtype=float),
            target=target,
            design=design,
            forward=fwd_map[name] if fwd_map is not None else None,
            reverse=rev_map[name] if rev_map is not None else None,
            quantiles=req_qs,
            external_pool=ext_pool is not None,
            schema=schema,
        )
        out[name] = bank.aggregate()
    return out


def _anchor_spread(anchor_values: dict[int, list[float]], n: int) -> np.ndarray:
    """Per-hour spread across anchors (the anchoring uncertainty).

    ``anchor_values[h]`` holds one mean counterfactual level per anchor covering
    ``h``, so the std over it is the anchoring uncertainty alone.  Until
    2026-11-11 this was the std over all ``(value, lead)`` draws of the hour,
    which is the resample spread the ``mc`` column already reports (the two were
    the same number whenever an hour had one anchor).
    """
    out = np.full(n, np.nan)
    for h, vals in anchor_values.items():
        if len(vals) >= 2:
            out[h] = float(np.nanstd(vals))
    return out


def _eval_icn_coalitions_dir(
    df: pd.DataFrame,
    target: str,
    design: ICNDesign,
    pipeline: ICLPipeline,
    players: list[tuple[str, list[str]]],
    coalitions: Sequence[frozenset[int]],
    met_cols: list[str],
    cols: list[str],
    estimate: str,
    external_pool: MetPool | None = None,
) -> dict[frozenset[int], tuple[dict[int, list[tuple[float, int]]], dict[int, list[float]]]]:
    """Evaluate all requested Shapley/sequential coalitions in one direction sharing CRN weather."""
    n = len(df)
    k_players = len(players)
    all_fixed = frozenset(range(k_players))
    rng = np.random.default_rng(design.seed)
    obs_arr = df[target].to_numpy(dtype=float)
    work_target = _transform_target_values(obs_arr, design.log_transform)
    pool = external_pool if external_pool is not None else MetPool(df[met_cols], met_cols)

    draws_by_c: dict[frozenset[int], dict[int, list[tuple[float, int]]]] = {
        c: {} for c in coalitions
    }
    errs_by_c: dict[frozenset[int], dict[int, list[float]]] = {c: {} for c in coalitions}

    # Precompute resampled column subsets for every non-trivial coalition.
    coal_res_cols: list[tuple[frozenset[int], list[str]]] = []
    for c in coalitions:
        if c != all_fixed:
            res_c = [f for i, (_name, feats) in enumerate(players) if i not in c for f in feats]
            coal_res_cols.append((c, res_c))

    can_batch = hasattr(pipeline, "forecast_batch")
    est_bs = getattr(getattr(pipeline, "estimator", None), "batch_size", None)
    base_bs = int(est_bs) if est_bs is not None else 32
    items_per_anchor = 1 + len(coal_res_cols) * (design.n_resamples if met_cols else 0)
    batch_limit = max(base_bs, items_per_anchor) if can_batch else 1

    pending_anchors: list[
        tuple[
            int,
            int,
            tuple[np.ndarray, pd.DataFrame, pd.DataFrame],
            list[tuple[frozenset[int], list[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]]]],
        ]
    ] = []
    pending_items = 0
    pending_horizon: int | None = None

    def _flush() -> None:
        nonlocal pending_anchors, pending_items, pending_horizon
        if not pending_anchors:
            return
        flat_items: list[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]] = []
        for _s, _e, fact_item, coal_blocks in pending_anchors:
            flat_items.append(fact_item)
            for _c, c_items in coal_blocks:
                flat_items.extend(c_items)

        preds_med, _ = _run_univariate_batch(pipeline, flat_items, cols, target, None, [], 0)
        cursor = 0
        for s, e, _fact_item, coal_blocks in pending_anchors:
            pred_fact = preds_med[cursor]
            cursor += 1
            obs_seg = obs_arr[s:e]
            z_obs_seg = work_target[s:e]
            if design.log_transform:
                err_seg = np.abs(pred_fact - z_obs_seg)
                fact_level = np.maximum(np.expm1(pred_fact), 0.0)
            else:
                fin_obs = obs_seg[np.isfinite(obs_seg)]
                obs_sd = float(np.std(fin_obs)) if fin_obs.size else 1.0
                scale_seg = np.maximum(np.maximum(np.abs(obs_seg), obs_sd), 1e-6)
                err_seg = np.abs(pred_fact - obs_seg) / scale_seg
                fact_level = pred_fact
            err_seg = np.where(np.isfinite(err_seg), err_seg, 0.0)

            if all_fixed in draws_by_c:
                vals_f = (
                    np.where(np.isfinite(obs_seg), np.maximum(obs_seg, 0.0), np.nan)
                    if (estimate == "paired" and design.log_transform)
                    else (obs_seg.copy() if estimate == "paired" else fact_level)
                )
                for offset, h in enumerate(range(s, e)):
                    draws_by_c[all_fixed].setdefault(h, []).append((float(vals_f[offset]), offset))
                    errs_by_c[all_fixed].setdefault(h, []).append(float(err_seg[offset]))

            for c, c_items in coal_blocks:
                n_cf = len(c_items)
                if n_cf > 0:
                    pred_cf = preds_med[cursor : cursor + n_cf]
                    cursor += n_cf
                else:
                    pred_cf = np.repeat(pred_fact[None, :], design.n_resamples, axis=0)
                if design.log_transform:
                    cf_levels = np.maximum(np.expm1(pred_cf), 0.0)
                    paired = np.maximum(
                        np.expm1(z_obs_seg + (pred_cf.mean(axis=0) - pred_fact)), 0.0
                    )
                else:
                    cf_levels = pred_cf
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = np.where(
                            np.abs(pred_fact) > 1e-6, cf_levels.mean(axis=0) / pred_fact, np.nan
                        )
                    paired = obs_seg * ratio
                for offset, h in enumerate(range(s, e)):
                    if estimate == "paired":
                        draws_by_c[c].setdefault(h, []).append((float(paired[offset]), offset))
                    else:
                        draws_by_c[c].setdefault(h, []).extend(
                            (float(v), offset) for v in cf_levels[:, offset]
                        )
                    errs_by_c[c].setdefault(h, []).append(float(err_seg[offset]))

        pending_anchors = []
        pending_items = 0
        pending_horizon = None

    for a in _anchor_range(n, design):
        s = a
        e = min(a + design.block_h, n)
        horizon = e - s
        if horizon <= 0:
            continue
        past_end = max(s - design.gap_h, 0)
        ctx_start = max(0, past_end - design.context_length)
        past = df.iloc[ctx_start:past_end]
        hist = work_target[ctx_start:past_end]
        future_index = df.index[s:e]
        factual_met = df.iloc[s:e]

        if design.support_min_frac > 0 and met_cols:
            support, _ = pool.support_check(factual_met[met_cols])
            if support < design.support_min_frac:
                continue

        past_cf = past
        if design.intervene == "past_future":
            past_cf = past.copy()
            if met_cols:
                pm = pool.sample_trajectory(len(past), rng)
                past_cf[met_cols] = pm[met_cols].to_numpy()

        cf_future_mets: list[pd.DataFrame] = []
        if met_cols:
            for _ in range(design.n_resamples):
                if design.pool == "record_pointwise":
                    cf_future_mets.append(pool.sample_pointwise(horizon, rng))
                elif design.pool == "record_trajectory":
                    cf_future_mets.append(pool.sample_trajectory(horizon, rng))
                else:
                    cf_future_mets.append(pool.sample_climatology(future_index[0], horizon, rng))

        coal_blocks: list[
            tuple[frozenset[int], list[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]]]
        ] = []
        entry_items = 1
        for c, res_c in coal_res_cols:
            if not met_cols or not res_c:
                c_items = (
                    [(hist, past_cf, factual_met[cols])] * design.n_resamples if met_cols else []
                )
            else:
                c_items = []
                for fm in cf_future_mets:
                    ff = factual_met.copy()
                    ff[res_c] = fm[res_c].to_numpy()
                    c_items.append((hist, past_cf, ff[cols]))
            coal_blocks.append((c, c_items))
            entry_items += len(c_items)

        if pending_anchors and (
            pending_horizon != horizon or pending_items + entry_items > batch_limit
        ):
            _flush()
        pending_anchors.append((s, e, (hist, past, factual_met[cols]), coal_blocks))
        pending_items += entry_items
        pending_horizon = horizon
        if not can_batch:
            _flush()

    _flush()
    return {c: (draws_by_c[c], errs_by_c[c]) for c in coalitions}


def icn_decompose(
    estimator: Any,
    df: pd.DataFrame,
    target: str = "value",
    *,
    design: ICNDesign | None = None,
    pipeline: ICLPipeline | None = None,
    met_features: list[str] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    attribution: str | None = None,
    n_permutations: int | None = None,
    estimate: str = "level",
    pool: MetPool | pd.DataFrame | None = None,
    include_diagnostics: bool = False,
    return_interactions: bool = False,
) -> Any:
    """In-context meteorological attribution and Shapley decomposition for Chronos-2.

    Attributes ``predicted - emi_total`` across individual meteorological
    variables or physical ``groups`` (e.g. from :func:`suggest_met_groups`)
    under future-only intervention (:class:`ICNDesign`) using shared Common
    Random Number weather draws across coalitions batched onto the GPU/pipeline.

    Parameters
    ----------
    estimator : Chronos2Estimator
        Fitted or zero-shot estimator.
    df : pandas.DataFrame
        Hourly DataFrame (DatetimeIndex or ``date`` column).
    target : str, default "value"
        Target column name.
    design : ICNDesign, optional
        ICN design specification; defaults to :meth:`ICNDesign.v1`.
    pipeline : ICLPipeline, optional
        Optional custom / fake pipeline for torch-free execution.
    met_features : list of str, optional
        Meteorological features to attribute. Defaults to numeric non-calendar
        covariates resolved from ``estimator``.
    groups : mapping, optional
        Optional physical grouping ``{group_name: [features]}``.
    attribution : {"sequential", "shapley"}, optional
        Defaults to ``"shapley"`` when ``groups`` or ``return_interactions`` is
        provided, else ``"sequential"``.
    n_permutations : int, optional
        Number of antithetic permutations for sampled Shapley when ``k`` is large.
    estimate : {"level", "paired"}, default "level"
        Decompose counterfactual levels or D11 paired estimates.
    pool : MetPool or DataFrame, optional
        Optional external counterfactual weather reservoir.
    include_diagnostics : bool, default False
        Include ``predicted``, ``met_explained``, and ``residual`` columns.
    return_interactions : bool, default False
        If True, compute exact second-order Grabisch–Roubens interactions and
        return a :class:`~normet.analysis.decomposition.ShapleyInteractionResult`.
    """
    from math import factorial

    from ..analysis.decomposition import (
        ShapleyInteractionResult,
        _compute_interaction_indices,
        _met_result,
        _players_from_groups,
    )
    from .estimator import to_indexed_frame

    if estimate not in ("level", "paired"):
        raise ValueError(f"estimate must be 'level' or 'paired', got {estimate!r}")
    method = (
        attribution
        if attribution is not None
        else ("shapley" if (groups is not None or return_interactions) else "sequential")
    )
    if method not in ("sequential", "shapley"):
        raise ValueError(f"attribution must be 'sequential' or 'shapley', got {attribution!r}")
    if return_interactions and (method != "shapley" or n_permutations is not None):
        raise ValueError("`return_interactions=True` requires exact Shapley (n_permutations=None).")

    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work = to_indexed_frame(work)
    if target not in work.columns:
        raise KeyError(f"target {target!r} absent from df")

    design = design or ICNDesign.v1()
    if hasattr(estimator, "_check_index"):
        estimator._check_index(work.index)  # type: ignore[attr-defined]
    cols, met_cols = _resolve_met_cols(estimator, work, target, met_features)
    if not met_cols:
        raise ValueError("icn_decompose requires at least one meteorological feature.")

    players = (
        _players_from_groups(groups, met_cols)
        if groups is not None
        else [(f, [f]) for f in met_cols]
    )
    k_players = len(players)
    if method == "shapley" and n_permutations is None and k_players > 10:
        raise ValueError(
            f"exact Shapley over {k_players} features needs 2**{k_players} coalitions; "
            "pass `groups` or `n_permutations`."
        )

    all_fixed = frozenset(range(k_players))
    none_fixed: frozenset[int] = frozenset()
    orders_sampled: list[list[int]] = []
    if method == "sequential":
        needed_set = {frozenset(range(i)) for i in range(k_players + 1)}
    elif n_permutations is None:
        needed_set = {
            frozenset(i for i in range(k_players) if mask >> i & 1)
            for mask in range(1 << k_players)
        }
    else:
        rng_perm = np.random.default_rng(design.seed)
        n_pairs = (int(n_permutations) + 1) // 2
        needed_set = {none_fixed, all_fixed}
        for _ in range(n_pairs):
            perm = [int(i) for i in rng_perm.permutation(k_players)]
            for order in (perm, perm[::-1]):
                orders_sampled.append(order)
                cur: frozenset[int] = none_fixed
                for i in order:
                    cur = cur | {i}
                    needed_set.add(cur)

    coalitions = sorted(needed_set, key=lambda s: (len(s), sorted(s)))
    pipe = pipeline if pipeline is not None else Chronos2ICLPipeline(estimator)
    ext_pool = _resolve_ext_pool(pool, met_cols)

    fwd_map = None
    rev_map = None
    n = len(work)
    if design.direction in ("forward", "bidirectional"):
        fwd_map = _eval_icn_coalitions_dir(
            work, target, design, pipe, players, coalitions, met_cols, cols, estimate, ext_pool
        )
    if design.direction in ("reverse", "bidirectional"):
        rev_df = work.iloc[::-1].copy()
        rev_df.index = work.index[::-1]
        rev_pool = ext_pool.reversed() if ext_pool is not None else None
        rev_map = _eval_icn_coalitions_dir(
            rev_df, target, design, pipe, players, coalitions, met_cols, cols, estimate, rev_pool
        )

    coal_series: dict[frozenset[int], np.ndarray] = {}
    for c in coalitions:
        arr = np.full(n, np.nan, dtype=float)
        for h in range(n):
            draws_h: list[tuple[float, int]] = []
            errs_h: list[float] = []
            if fwd_map is not None:
                draws_h.extend(fwd_map[c][0].get(h, ()))
                errs_h.extend(fwd_map[c][1].get(h, ()))
            if rev_map is not None:
                draws_h.extend(rev_map[c][0].get(n - 1 - h, ()))
                errs_h.extend(rev_map[c][1].get(n - 1 - h, ()))
            if draws_h:
                arr[h] = _aggregate(
                    draws_h,
                    design.aggregate,
                    design.lead_tau,
                    surprise_errors=errs_h,
                    surprise_tau=design.surprise_tau,
                )
        coal_series[c] = arr

    emi_total = coal_series[none_fixed]
    contribs: dict[str, np.ndarray] = {name: np.zeros(n, dtype=float) for name, _ in players}
    if method == "sequential":
        prev = none_fixed
        for idx, (name, _) in enumerate(players):
            cur_s = prev | {idx}
            contribs[name] = coal_series[cur_s] - coal_series[prev]
            prev = cur_s
    elif n_permutations is None:
        weight = [
            factorial(s) * factorial(k_players - s - 1) / factorial(k_players)
            for s in range(k_players)
        ]
        for s_set in coalitions:
            for idx, (name, _) in enumerate(players):
                if idx not in s_set:
                    contribs[name] += weight[len(s_set)] * (
                        coal_series[s_set | {idx}] - coal_series[s_set]
                    )
    else:
        for order in orders_sampled:
            prev = none_fixed
            for idx in order:
                cur_s = prev | {idx}
                name = players[idx][0]
                contribs[name] += coal_series[cur_s] - coal_series[prev]
                prev = cur_s
        for name, _ in players:
            contribs[name] /= float(len(orders_sampled))

    obs_arr = work[target].to_numpy(dtype=float)
    res_df = pd.DataFrame({"observed": obs_arr}, index=pd.DatetimeIndex(work.index, name="date"))
    frame = _met_result(
        res_df,
        emi_total,
        contribs,
        include_diagnostics=bool(include_diagnostics or return_interactions),
    )
    if not return_interactions:
        return frame

    player_names = [p[0] for p in players]
    main_dict, pair_dict = _compute_interaction_indices(player_names, coal_series, contribs)
    main_df = pd.DataFrame(main_dict, index=frame.index)
    pair_df = pd.DataFrame(pair_dict, index=frame.index)
    mean_mat = pd.DataFrame(
        np.zeros((k_players, k_players), dtype=float), index=player_names, columns=player_names
    )
    std_mat = pd.DataFrame(
        np.zeros((k_players, k_players), dtype=float), index=player_names, columns=player_names
    )
    summary_rows: list[dict[str, Any]] = []
    for i, name in enumerate(player_names):
        arr = main_df[name].to_numpy(dtype=float)
        fin = arr[np.isfinite(arr)]
        m_val = float(np.mean(fin)) if fin.size else float("nan")
        s_val = float(np.std(fin)) if fin.size else float("nan")
        mean_mat.iloc[i, i] = m_val
        std_mat.iloc[i, i] = s_val
        summary_rows.append(
            {
                "term": name,
                "kind": "main",
                "mean": m_val,
                "std": s_val,
                "mean_abs": float(np.mean(np.abs(fin))) if fin.size else float("nan"),
                "max_abs": float(np.max(np.abs(fin))) if fin.size else 0.0,
            }
        )
    for i in range(k_players):
        for j in range(i + 1, k_players):
            pname = f"{player_names[i]}:{player_names[j]}"
            arr = pair_df[pname].to_numpy(dtype=float)
            fin = arr[np.isfinite(arr)]
            m_val = float(np.mean(fin)) if fin.size else float("nan")
            s_val = float(np.std(fin)) if fin.size else float("nan")
            mean_mat.iloc[i, j] = mean_mat.iloc[j, i] = m_val
            std_mat.iloc[i, j] = std_mat.iloc[j, i] = s_val
            summary_rows.append(
                {
                    "term": pname,
                    "kind": "interaction",
                    "mean": m_val,
                    "std": s_val,
                    "mean_abs": float(np.mean(np.abs(fin))) if fin.size else float("nan"),
                    "max_abs": float(np.max(np.abs(fin))) if fin.size else 0.0,
                }
            )
    summary_df = pd.DataFrame(
        summary_rows, columns=["term", "kind", "mean", "std", "mean_abs", "max_abs"]
    )
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
