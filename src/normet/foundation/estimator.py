"""Chronos-2 estimator for zero-shot de-weathering and counterfactual inference.

Implements :class:`Chronos2Estimator`, a covariate-conditioned wrapper around the
Chronos-2 time-series foundation model that conforms to the normet estimator API:

  - ``predict``            rolling covariate-conditioned point prediction (backend contract)
  - ``predict_quantiles``  native probabilistic forecast (Chronos-2 emits 21 quantile levels)
  - ``deweather``          meteorological normalisation by marginalising the covariate channel
  - ``counterfactual``     business-as-usual projection across an intervention, with a
                           pre-intervention hold-out that reports its own bias

Meteorology enters through Chronos-2's ``past_covariates`` / ``future_covariates``
channels. That is not optional book-keeping: on a 168 h hold-out at London
N. Kensington (2019-03-01, a placebo window with no intervention) the same model
scores RMSE 8.15 ug/m3 with covariates and 47.67 ug/m3 without them. A univariate
call is a forecast, not a de-weathering, and this class refuses to pretend otherwise.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_MODEL = "amazon/chronos-2"


def _import_foundation() -> tuple[Any, Any]:
    """Return ``(torch, Chronos2Pipeline)`` or explain which extra is missing."""
    try:
        import torch
        from chronos import Chronos2Pipeline
    except ImportError as err:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Chronos foundation dependencies not installed. "
            "Install with `pip install chronos-forecasting torch`."
        ) from err
    return torch, Chronos2Pipeline


def resolve_device(requested: str | None = None) -> str:
    """Pick a torch device, preferring an accelerator over the CPU.

    ``None`` means "choose for me": CUDA, then Apple Silicon's Metal backend
    (MPS), then the CPU. MPS matters because it is the difference between a
    usable and an unusable run on a Mac laptop -- de-weathering spends one full
    forward pass per Monte-Carlo sample -- and the earlier
    ``"cuda" if is_available() else "cpu"`` check never selected it, silently
    putting every Mac on the CPU path.

    An explicit string is returned unchanged, including a device this build of
    torch cannot serve: surfacing torch's own error beats second-guessing a
    caller who asked for something specific.
    """
    if requested is not None:
        return requested
    torch, _ = _import_foundation()

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _load_pipeline_on(model_name: str, device: str, auto: bool) -> tuple[Any, str]:
    """Load a Chronos-2 pipeline, returning it with the device it actually landed on.

    An auto-picked accelerator is a guess, and MPS is the one that goes wrong:
    it can refuse a dtype or an op that CUDA and the CPU both accept, and the
    refusal surfaces at load time. Falling back to the CPU is right when *we*
    chose the device; when the caller named one, torch's error is the answer
    they asked for.
    """
    _, Chronos2Pipeline = _import_foundation()

    try:
        return Chronos2Pipeline.from_pretrained(model_name, device_map=device), device
    except Exception:
        if not auto or device == "cpu":
            raise
        log.warning(
            "Could not load %s on the auto-selected device %r; falling back to the CPU. "
            "Pass device= explicitly to see the original error.",
            model_name,
            device,
            exc_info=True,
        )
        return Chronos2Pipeline.from_pretrained(model_name, device_map="cpu"), "cpu"


class InsufficientContextError(ValueError):
    """Raised when the conditioning window holds too little observed data to project from."""


class IrregularIndexError(ValueError):
    """Raised when the frame's timestamps are not uniformly spaced."""


#: Calendar covariates are known into the future by construction, so they are
#: always safe to pass as ``future_covariates``. Encoded as sin/cos pairs so the
#: model sees a continuous cycle rather than an integer discontinuity at midnight
#: / Sunday / New Year.
_CALENDAR_ENCODERS: dict[str, Any] = {
    "hour_sin": lambda i: np.sin(2 * np.pi * i.hour / 24.0),
    "hour_cos": lambda i: np.cos(2 * np.pi * i.hour / 24.0),
    "dow_sin": lambda i: np.sin(2 * np.pi * i.dayofweek / 7.0),
    "dow_cos": lambda i: np.cos(2 * np.pi * i.dayofweek / 7.0),
    "doy_sin": lambda i: np.sin(2 * np.pi * i.dayofyear / 365.25),
    "doy_cos": lambda i: np.cos(2 * np.pi * i.dayofyear / 365.25),
}


def to_regular_index(df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Reindex onto a gap-free grid, leaving the target missing where it was missing.

    Pipelines that *drop* incomplete rows rather than keeping them as NaN produce a
    frame whose row count no longer equals its time span. Chronos-2 reads a series as
    uniformly spaced, so those rows get treated as consecutive: the diurnal cycle is
    compressed against the calendar covariates and the projection inflates. Reindexing
    restores the correspondence; the target's gaps stay NaN, which the model masks.

    Covariates are interpolated over short gaps, since they must be finite — check the
    gap sizes before trusting a frame that needed much filling.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("to_regular_index requires a DatetimeIndex")
    full = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    out = df.reindex(full)
    out.index.name = df.index.name
    return out


def add_calendar_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with the six cyclical calendar covariates appended.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame with a :class:`~pandas.DatetimeIndex`.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with ``hour_sin``/``hour_cos``/``dow_sin``/``dow_cos``/
        ``doy_sin``/``doy_cos`` added.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("add_calendar_covariates requires a DatetimeIndex")
    out = df.copy()
    for name, fn in _CALENDAR_ENCODERS.items():
        out[name] = fn(out.index).astype(np.float32)
    return out


def _qname(q: float) -> str:
    """Format a quantile level the way ``normet.analysis.normalise`` does.

    Duplicated rather than imported so that ``normet.foundation`` does not pull
    in the AutoML backend registry just to name a column;
    ``test_quantile_column_names_match_normalise`` pins the two together.
    """
    if not (0.0 <= float(q) <= 1.0):
        raise ValueError(f"Quantile must be in [0,1]: got {q}")
    return f"q{int(round(float(q) * 1000)):03d}"


def to_normet_frame(
    df: pd.DataFrame,
    *,
    point_col: str,
    observed_col: str = "observed",
    quantile_cols: Mapping[float, str] | None = None,
) -> pd.DataFrame:
    """Rename a foundation-model result onto the schema :func:`normet.normalise` emits.

    ``normalise`` returns a frame indexed by ``date`` carrying ``observed`` and
    ``normalised``, plus one ``qNNN`` column per requested quantile. Everything
    downstream keys off that shape rather than off a type: :func:`normet.report`
    dispatches its automatic plot on ``{"observed", "normalised"}`` being present
    and picks the outermost ``qNNN`` columns as the shaded band, and
    :func:`normet.normalise_plot` defaults to the same two names. Chronos-2 calls
    its median ``dew_p50`` / ``counterfactual_p50``, so its output misses that
    dispatch entirely until it is renamed -- which is all this function does.

    Parameters
    ----------
    df : pandas.DataFrame
        Datetime-indexed foundation-model output.
    point_col : str
        Column holding the point estimate. Becomes ``normalised``.
    observed_col : str, default ``"observed"``
        Column holding the observations.
    quantile_cols : mapping of float to str, optional
        ``{level: column}`` pairs to carry across, renamed by level
        (``0.1`` becomes ``q100``).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``date``, with ``observed``, ``normalised``, and one ``qNNN``
        column per entry in *quantile_cols*.

    Examples
    --------
    >>> dew = est.deweather(df, "value")                      # doctest: +SKIP
    >>> normalise_plot(to_normet_frame(dew, point_col="dew_p50"))   # doctest: +SKIP
    """
    missing = [c for c in (observed_col, point_col) if c not in df.columns]
    if missing:
        raise KeyError(f"columns absent from frame: {missing}")
    out = pd.DataFrame(
        {
            "observed": df[observed_col].to_numpy(),
            "normalised": df[point_col].to_numpy(),
        },
        index=pd.Index(df.index, name="date"),
    )
    for q, col in (quantile_cols or {}).items():
        if col not in df.columns:
            raise KeyError(f"quantile column {col!r} absent from frame")
        out[_qname(q)] = df[col].to_numpy()
    return out


@dataclass
class CounterfactualResult:
    """Container for probabilistic counterfactual evaluation results.

    ``pre_intervention_bias_pct`` is the estimator's own error on a hold-out
    window immediately before the intervention, projected under identical
    conditions. It is the placebo check: an unbiased counterfactual should score
    near zero there, and any intervention effect smaller than this bias is not
    separable from counterfactual error.
    """

    observed: pd.Series
    counterfactual_p50: pd.Series
    counterfactual_p10: pd.Series
    counterfactual_p90: pd.Series
    absolute_impact_p50: pd.Series
    relative_impact_pct_p50: pd.Series
    pre_intervention_bias_pct: float
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert counterfactual time series to a unified DataFrame."""
        return pd.DataFrame(
            {
                "observed": self.observed,
                "counterfactual_p50": self.counterfactual_p50,
                "counterfactual_p10": self.counterfactual_p10,
                "counterfactual_p90": self.counterfactual_p90,
                "impact_p50": self.absolute_impact_p50,
                "impact_pct_p50": self.relative_impact_pct_p50,
            },
            index=self.observed.index,
        )

    def to_normet_frame(self) -> pd.DataFrame:
        """Return the projection on :func:`normet.normalise`'s schema.

        The business-as-usual median takes the ``normalised`` slot, so a
        counterfactual plots and reports through the same path as a de-weathered
        series -- observed against the projection, p10/p90 as the band. The frame
        spans the whole record and the projection columns are empty before the
        intervention, which renders as observed history running into the
        observed-versus-counterfactual comparison.
        """
        return to_normet_frame(
            self.to_dataframe(),
            point_col="counterfactual_p50",
            quantile_cols={0.1: "counterfactual_p10", 0.9: "counterfactual_p90"},
        )


class Chronos2Estimator:
    """Covariate-conditioned Chronos-2 estimator for de-weathering and counterfactuals.

    Parameters
    ----------
    model_name : str, default ``"amazon/chronos-2"``
        Hugging Face identifier. Chronos-2 is patch-based with instance
        normalisation; the earlier ``chronos-t5-*`` line tokenises values into
        discrete bins and saturates outside the range seen in context, which is
        the same failure mode that makes tree ensembles unusable for
        extrapolation. Pointing this at a ``chronos-t5`` checkpoint therefore
        disables covariate support and is rejected.
    context_length : int, default 2048
        Hours of history conditioned on. 2048 h is Chronos-2's native context.
    prediction_length : int, default 168
        Default horizon (hours) for a single forward pass, and the stride used
        when rolling across a long series.
    met_covariates : sequence of str, optional
        Meteorological columns passed through the covariate channel. When
        omitted, every numeric column other than the target is used.
    use_calendar : bool, default True
        Append the six cyclical calendar covariates.
    device : str, optional
        ``"cuda"``/``"cpu"``. Auto-detected when omitted.
    min_context_coverage : float, default 0.25
        Minimum fraction of the conditioning window that must carry observed
        target values. Chronos-2 masks missing values rather than failing, so a
        station whose record is empty across the context returns a plausible-looking
        forecast scaled to nothing; below this fraction an
        :class:`InsufficientContextError` is raised instead.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        prediction_length: int = 168,
        context_length: int = 2048,
        met_covariates: Sequence[str] | None = None,
        use_calendar: bool = True,
        device: str | None = None,
        min_context_coverage: float = 0.25,
    ) -> None:
        if "chronos-t5" in model_name or "chronos-bolt" in model_name:
            raise ValueError(
                f"{model_name!r} is a Chronos-1/Bolt checkpoint and does not accept "
                "covariates; meteorological conditioning would be silently dropped. "
                f"Use {DEFAULT_MODEL!r}."
            )
        self.model_name = model_name
        self.prediction_length = int(prediction_length)
        self.context_length = int(context_length)
        self.met_covariates = list(met_covariates) if met_covariates is not None else None
        self.use_calendar = bool(use_calendar)
        self.device = device
        self.min_context_coverage = float(min_context_coverage)
        self._pipeline: Any = None
        self._quantiles: np.ndarray | None = None
        self.target_col: str | None = None
        self.feature_cols: list[str] = []

    # ------------------------------------------------------------------ setup

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        auto = self.device is None
        self.device = resolve_device(self.device)
        log.info("Loading %s on %s", self.model_name, self.device)
        self._pipeline, self.device = _load_pipeline_on(self.model_name, self.device, auto)
        self._quantiles = np.asarray(self._pipeline.quantiles, dtype=float)
        return self._pipeline

    @property
    def quantile_levels(self) -> np.ndarray:
        """The quantile levels Chronos-2 emits natively."""
        self._load_pipeline()
        assert self._quantiles is not None
        return self._quantiles

    def _q_index(self, q: float) -> int:
        return int(np.argmin(np.abs(self.quantile_levels - q)))

    def _resolve_covariates(self, df: pd.DataFrame, target: str) -> list[str]:
        if self.met_covariates is not None:
            missing = [c for c in self.met_covariates if c not in df.columns]
            if missing:
                raise KeyError(f"covariate columns absent from frame: {missing}")
            cols = list(self.met_covariates)
        else:
            cols = [
                c
                for c in df.columns
                if c != target
                and pd.api.types.is_numeric_dtype(df[c])
                and c not in _CALENDAR_ENCODERS
            ]
        if self.use_calendar:
            cols = cols + [c for c in _CALENDAR_ENCODERS if c in df.columns]
        return cols

    # ------------------------------------------------------- low-level predict

    @staticmethod
    def _clean(a: np.ndarray, fill: float) -> np.ndarray:
        """Covariates must be finite; the target may contain NaN (Chronos-2 masks it)."""
        out = np.asarray(a, dtype=np.float32)
        return np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill)

    def _make_input(
        self,
        target_hist: np.ndarray,
        past: pd.DataFrame,
        future: pd.DataFrame,
        cols: Sequence[str],
    ) -> dict[str, Any]:
        fills = {c: float(np.nanmean(past[c])) if np.isfinite(past[c]).any() else 0.0 for c in cols}
        return {
            "target": np.asarray(target_hist, dtype=np.float32),
            "past_covariates": {c: self._clean(past[c].to_numpy(), fills[c]) for c in cols},
            "future_covariates": {c: self._clean(future[c].to_numpy(), fills[c]) for c in cols},
        }

    @staticmethod
    def _check_index(index: pd.Index) -> None:
        """Reject a non-uniform time index rather than silently distorting the series.

        Chronos-2 has no notion of timestamps -- it reads position as time. A frame
        whose rows skip missing hours is therefore read as if those hours never
        existed, which shifts every subsequent point earlier relative to its own
        calendar covariates. Measured on the UK network, feeding gap-dropped frames
        instead of reindexed ones raised the projected business-as-usual level by
        roughly 40% and turned a well-behaved counterfactual into a drifting one.
        """
        if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
            return
        steps = np.diff(index.to_numpy("datetime64[ns]").astype("int64"))
        if len(np.unique(steps)) > 1:
            gaps = int((steps != steps.min()).sum())
            worst = pd.Timedelta(int(steps.max())).total_seconds() / 3600.0
            raise IrregularIndexError(
                f"time index is not uniformly spaced ({gaps} irregular steps, largest "
                f"{worst:.0f} h); Chronos-2 reads position as time, so gaps shift the "
                "series against its calendar covariates. Pass the frame through "
                "normet.foundation.estimator.to_regular_index first."
            )

    def _check_context(self, target_hist: np.ndarray) -> None:
        """Refuse to project from a context that is mostly missing.

        Chronos-2 masks NaNs in the target, so a context window that is empty or
        nearly so does not raise — it returns a forecast scaled to nothing. In one
        real case (Belfast Centre, whose record has a gap spanning the whole
        conditioning window) that produced a business-as-usual level of
        0.25 ug/m3 against an observed 26 ug/m3, i.e. an apparent +10,000%
        "effect". Silence is the dangerous failure here, so it is made loud.
        """
        finite = float(np.isfinite(target_hist).mean()) if len(target_hist) else 0.0
        if finite < self.min_context_coverage:
            raise InsufficientContextError(
                f"context target is only {100 * finite:.1f}% finite "
                f"(minimum {100 * self.min_context_coverage:.0f}%); "
                "projecting from it would return a forecast scaled to missing data"
            )

    def _forecast_block(
        self,
        target_hist: np.ndarray,
        past: pd.DataFrame,
        future: pd.DataFrame,
        cols: Sequence[str],
    ) -> np.ndarray:
        """One forward pass. Returns ``(n_quantiles, horizon)``."""
        self._check_context(target_hist)
        pipe = self._load_pipeline()
        inp = self._make_input(target_hist, past, future, cols)
        out = pipe.predict([inp], prediction_length=len(future))
        return np.asarray(out[0])[0]

    # ------------------------------------------------------------ public API

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> Chronos2Estimator:
        """Record column metadata. Chronos-2 is zero-shot: no parameters are trained."""
        if isinstance(y, pd.Series) and isinstance(y.name, str):
            self.target_col = y.name
        if isinstance(X, pd.DataFrame):
            self.feature_cols = list(X.columns)
        return self

    def predict_quantiles(
        self,
        df: pd.DataFrame,
        target: str,
        anchor: pd.Timestamp | str,
        horizon: int | None = None,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        covariates: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Probabilistic forecast from ``anchor``, conditioned on observed covariates.

        Parameters
        ----------
        df : pandas.DataFrame
            Datetime-indexed frame holding ``target`` and the covariate columns.
        target : str
            Target column name.
        anchor : Timestamp or str
            Forecast origin. History strictly before it is used as context;
            covariates from it onward are supplied as known-future values.
        horizon : int, optional
            Hours to predict. Defaults to :attr:`prediction_length`.
        quantiles : sequence of float
            Levels to return. Snapped to the nearest natively emitted level.
        covariates : sequence of str, optional
            Overrides the resolved covariate set for this call.

        Returns
        -------
        pandas.DataFrame
            Indexed by forecast timestamp, one column per requested quantile
            (``q0.1``, ``q0.5``, ...).
        """
        self._check_index(df.index)
        anchor = pd.Timestamp(anchor)
        cols = list(covariates) if covariates is not None else self._resolve_covariates(df, target)
        horizon = int(horizon or self.prediction_length)

        past = df.loc[df.index < anchor].iloc[-self.context_length :]
        future = df.loc[df.index >= anchor].iloc[:horizon]
        if past.empty or future.empty:
            raise ValueError("anchor leaves no context or no forecast window")

        block = self._forecast_block(past[target].to_numpy(), past, future, cols)
        return pd.DataFrame(
            {f"q{q}": block[self._q_index(q)] for q in quantiles},
            index=future.index,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Rolling covariate-conditioned point prediction over a whole frame.

        Implements the normet backend contract so Chronos-2 can stand in for a
        tree backend. ``X`` must be datetime-indexed and carry the target column
        recorded by :meth:`fit` (or named ``value``); the first
        :attr:`context_length` rows are seeded with the observed values, as they
        have no history to condition on.

        Returns
        -------
        numpy.ndarray
            Median prediction, aligned with ``X``'s rows.
        """
        self._check_index(X.index)
        target = self.target_col or "value"
        if target not in X.columns:
            raise KeyError(f"target column {target!r} not in frame")
        cols = self._resolve_covariates(X, target)
        obs = X[target].to_numpy(dtype=np.float32)
        out = obs.astype(np.float32).copy()

        i50 = self._q_index(0.5)
        start = self.context_length
        step = self.prediction_length
        for s in range(start, len(X), step):
            e = min(s + step, len(X))
            past = X.iloc[max(0, s - self.context_length) : s]
            future = X.iloc[s:e]
            block = self._forecast_block(past[target].to_numpy(), past, future, cols)
            out[s:e] = block[i50]
        return out

    def deweather(
        self,
        df: pd.DataFrame,
        target: str,
        met_features: Sequence[str] | None = None,
        n_samples: int = 20,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        random_state: int | None = None,
        schema: str = "chronos",
    ) -> pd.DataFrame:
        """Meteorological normalisation by marginalising over resampled weather.

        Follows the same Monte Carlo logic as :func:`normet.normalise`: whole
        timestamps are drawn from the observed record so the joint structure of
        the meteorology is preserved, the drawn values replace the meteorological
        covariates while calendar covariates stay at their actual values, and the
        predictions are averaged. The expectation over weather is what
        ``dew_p50`` reports.

        One property is worth stating plainly rather than hiding: the target
        history conditioning each block is the *observed* series, which embeds
        the weather that actually occurred. Normalisation therefore acts on the
        covariate channel only, and the residual anchoring grows as
        :attr:`prediction_length` shrinks. :meth:`covariate_sensitivity`
        quantifies how much of a given configuration's prediction is driven by
        meteorology rather than by that anchoring.

        Parameters
        ----------
        schema : {"chronos", "normet"}, default ``"chronos"``
            ``"chronos"`` keeps this class's own names (``dew_p10`` ...), which
            carry the quantile level explicitly. ``"normet"`` renames the median
            to ``normalised`` and the levels to ``qNNN`` via
            :func:`to_normet_frame`, so the result drops straight into
            :func:`normet.normalise_plot` and :func:`normet.generate_html_report`.

        Returns
        -------
        pandas.DataFrame
            Indexed by the frame's timestamps, with ``observed`` and one column
            per quantile -- named ``dew_pNN`` or ``qNNN`` per *schema*.
        """
        if schema not in ("chronos", "normet"):
            raise ValueError(f"schema must be 'chronos' or 'normet', got {schema!r}")
        self._check_index(df.index)
        rng = np.random.default_rng(random_state)
        cols = self._resolve_covariates(df, target)
        met = (
            list(met_features)
            if met_features is not None
            else [c for c in cols if c not in _CALENDAR_ENCODERS]
        )

        i_q = {q: self._q_index(q) for q in quantiles}
        acc = {q: np.zeros(len(df), dtype=np.float64) for q in quantiles}
        seeded = slice(0, min(self.context_length, len(df)))
        for q in quantiles:
            acc[q][seeded] = df[target].to_numpy()[seeded]

        step = self.prediction_length
        for s in range(self.context_length, len(df), step):
            e = min(s + step, len(df))
            past = df.iloc[max(0, s - self.context_length) : s]
            future = df.iloc[s:e]
            draws = {q: np.zeros((n_samples, e - s)) for q in quantiles}
            for m in range(n_samples):
                idx = rng.integers(0, len(df), size=len(past) + len(future))
                swap_past = past.copy()
                swap_future = future.copy()
                swap_past[met] = df[met].to_numpy()[idx[: len(past)]]
                swap_future[met] = df[met].to_numpy()[idx[len(past) :]]
                block = self._forecast_block(past[target].to_numpy(), swap_past, swap_future, cols)
                for q in quantiles:
                    draws[q][m] = block[i_q[q]]
            for q in quantiles:
                acc[q][s:e] = draws[q].mean(axis=0)

        out = pd.DataFrame({"observed": df[target].to_numpy()}, index=df.index)
        for q in quantiles:
            out[f"dew_p{int(q * 100)}"] = acc[q]
        if schema == "chronos":
            return out
        median = min(quantiles, key=lambda q: abs(q - 0.5))
        return to_normet_frame(
            out,
            point_col=f"dew_p{int(median * 100)}",
            quantile_cols={q: f"dew_p{int(q * 100)}" for q in quantiles},
        )

    def covariate_sensitivity(
        self,
        df: pd.DataFrame,
        target: str,
        anchor: pd.Timestamp | str,
        horizon: int | None = None,
        met_features: Sequence[str] | None = None,
        random_state: int | None = None,
    ) -> dict[str, float]:
        """Diagnose whether the forecast responds to meteorology at all.

        Re-runs one forecast with the future meteorology randomly permuted. A
        model that is really doing autoregression will return an almost
        unchanged median, in which case any "de-weathered" series it produces is
        meaningless. Reported as the mean absolute median shift, both absolute
        and as a percentage of the prediction mean.
        """
        rng = np.random.default_rng(random_state)
        anchor = pd.Timestamp(anchor)
        cols = self._resolve_covariates(df, target)
        met = (
            list(met_features)
            if met_features is not None
            else [c for c in cols if c not in _CALENDAR_ENCODERS]
        )
        horizon = int(horizon or self.prediction_length)

        past = df.loc[df.index < anchor].iloc[-self.context_length :]
        future = df.loc[df.index >= anchor].iloc[:horizon]
        i50 = self._q_index(0.5)

        base = self._forecast_block(past[target].to_numpy(), past, future, cols)[i50]
        shuffled = future.copy()
        for c in met:
            shuffled[c] = rng.permutation(future[c].to_numpy())
        alt = self._forecast_block(past[target].to_numpy(), past, shuffled, cols)[i50]

        delta = np.abs(base - alt)
        denom = float(np.mean(np.abs(base))) or 1.0
        return {
            "mean_abs_shift": float(delta.mean()),
            "max_abs_shift": float(delta.max()),
            "pct_of_prediction": float(100.0 * delta.mean() / denom),
        }

    def counterfactual(
        self,
        df: pd.DataFrame,
        target: str,
        intervention_date: str | pd.Timestamp,
        features: Sequence[str] | None = None,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        validation_hours: int = 720,
        max_horizon: int | None = None,
    ) -> CounterfactualResult:
        """Business-as-usual projection across an intervention.

        The counterfactual is conditioned on pre-intervention history only and
        driven forward with the meteorology that actually occurred, so it answers
        "what would concentrations have been under the observed weather had the
        intervention not happened". Post-intervention observations are never fed
        back as context — doing so would leak the intervention into its own
        counterfactual — so long horizons roll forward on the model's own median.

        Before projecting, the same machinery is run on ``validation_hours`` of
        pre-intervention data held out from the context. The resulting
        ``pre_intervention_bias_pct`` is reported alongside the effect and bounds
        what size of effect is separable from counterfactual error.

        Parameters
        ----------
        intervention_date : str or Timestamp
            Start of the intervention. Choose it early enough to exclude
            anticipation effects; a pre-trend check on the hold-out is the way to
            confirm the cut is clean.
        validation_hours : int, default 720
            Length of the pre-intervention hold-out (default 30 days).
        max_horizon : int, optional
            Truncate the projection after this many hours. Counterfactual skill
            decays with horizon; use this to keep claims inside the range where
            the hold-out shows the projection is trustworthy.
        """
        self._check_index(df.index)
        interv = pd.to_datetime(intervention_date)
        cols = list(features) if features is not None else self._resolve_covariates(df, target)

        pre = df.loc[df.index < interv]
        post = df.loc[df.index >= interv]
        if post.empty:
            raise ValueError("no post-intervention data after intervention_date")
        if len(pre) < self.context_length + validation_hours:
            raise ValueError(
                f"need at least {self.context_length + validation_hours} pre-intervention hours, got {len(pre)}"
            )
        if max_horizon is not None:
            post = post.iloc[: int(max_horizon)]

        # --- placebo: project the pre-intervention hold-out under identical rules
        val = pre.iloc[-validation_hours:]
        val_ctx = pre.iloc[:-validation_hours]
        val_p50 = self._roll_forward(val_ctx, val, target, cols, {0.5: self._q_index(0.5)})[0.5]
        val_obs = val[target].to_numpy(dtype=float)
        ok = np.isfinite(val_obs)
        bias_pct = (
            float(100.0 * np.mean(val_p50[ok] - val_obs[ok]) / max(np.mean(val_obs[ok]), 1e-6))
            if ok.any()
            else float("nan")
        )

        # --- the counterfactual itself
        i_q = {q: self._q_index(q) for q in quantiles}
        proj = self._roll_forward(pre, post, target, cols, i_q)

        p50 = pd.Series(proj[0.5], index=post.index)
        p10 = pd.Series(proj[min(quantiles)], index=post.index)
        p90 = pd.Series(proj[max(quantiles)], index=post.index)

        obs_post = post[target]
        impact = obs_post - p50
        impact_pct = 100.0 * impact / p50.where(p50.abs() > 1e-6)

        mean_obs = float(np.nanmean(obs_post))
        mean_bau = float(np.nanmean(p50))
        outside = float(np.nanmean((obs_post < p10) | (obs_post > p90)))

        summary = {
            "intervention_date": str(interv),
            "post_intervention_hours": int(len(post)),
            "mean_observed": round(mean_obs, 3),
            "mean_counterfactual_bau": round(mean_bau, 3),
            "net_impact": round(mean_obs - mean_bau, 3),
            "net_impact_pct": round(100.0 * (mean_obs - mean_bau) / max(abs(mean_bau), 1e-6), 3),
            "pre_intervention_bias_pct": round(bias_pct, 3),
            "fraction_outside_interval": round(outside, 3),
            "validation_hours": int(validation_hours),
        }
        log.info(
            "counterfactual %s: effect %.2f%% against a pre-intervention bias of %.2f%%",
            interv.date(),
            summary["net_impact_pct"],
            bias_pct,
        )

        return CounterfactualResult(
            observed=df[target],
            counterfactual_p50=p50,
            counterfactual_p10=p10,
            counterfactual_p90=p90,
            absolute_impact_p50=impact,
            relative_impact_pct_p50=impact_pct,
            pre_intervention_bias_pct=bias_pct,
            summary=summary,
        )

    def _roll_forward(
        self,
        context: pd.DataFrame,
        horizon_frame: pd.DataFrame,
        target: str,
        cols: Sequence[str],
        q_index: dict[float, int],
    ) -> dict[float, np.ndarray]:
        """Project across ``horizon_frame`` without ever seeing its observations.

        Blocks longer than :attr:`prediction_length` are rolled forward on the
        model's own median, which is what makes the projection independent of the
        post-intervention record — and also why its error compounds with horizon.
        """
        self._check_index(context.index)
        self._check_index(horizon_frame.index)
        hist = context[target].to_numpy(dtype=np.float32)
        cov_hist = context[list(cols)].copy()
        out = {q: np.zeros(len(horizon_frame)) for q in q_index}

        step = self.prediction_length
        for s in range(0, len(horizon_frame), step):
            e = min(s + step, len(horizon_frame))
            future = horizon_frame.iloc[s:e]
            past_cov = cov_hist.iloc[-self.context_length :]
            block = self._forecast_block(hist[-self.context_length :], past_cov, future, cols)
            for q, qi in q_index.items():
                out[q][s:e] = block[qi]
            # roll the model's own median forward as pseudo-history
            hist = np.concatenate([hist, block[q_index[0.5]].astype(np.float32)])
            cov_hist = pd.concat([cov_hist, future[list(cols)]])
        return out
