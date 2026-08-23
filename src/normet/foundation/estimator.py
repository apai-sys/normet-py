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

import importlib.util
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_MODEL = "amazon/chronos-2"

#: ``backend`` value that switches the pipelines onto the zero-shot path.
#:
#: Deliberately *not* registered in ``normet.backends.backend_registry``: that
#: registry's contract is train/save/load, and Chronos-2 does none of the three.
CHRONOS_BACKEND = "chronos-2"

#: Monte-Carlo weather resamples for the zero-shot path.
#:
#: The AutoML default of 300 is a tree-ensemble budget. Here each sample is a
#: full context-length forward pass, so 300 would run for days on a CPU.
CHRONOS_DEFAULT_SAMPLES = 8


def resolve_n_samples(n_samples: int | None, backend: str | None) -> int:
    """Fill in the Monte-Carlo sample count for *backend* when left unset."""
    if n_samples is not None:
        return int(n_samples)
    return CHRONOS_DEFAULT_SAMPLES if backend == CHRONOS_BACKEND else 300


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


def to_indexed_frame(df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Put a normet-shaped frame on the gap-free DatetimeIndex Chronos-2 needs.

    :func:`normet.prepare_data` returns a ``date`` column, a ``value`` target and
    a ``set`` label, and it *drops* rows with missing covariates. Chronos-2 reads
    position as time, so those dropped hours are read as if they never happened
    and the series slides against its own calendar covariates. This moves ``date``
    to the index, rebuilds the grid via :func:`to_regular_index` so the holes come
    back as NaN for the model to mask, and drops ``set`` -- a train/test label
    means nothing zero-shot and would otherwise be picked up as a covariate.

    Safe to call on a frame that is already indexed; only the pieces that apply
    are done.
    """
    out = df.copy()
    if "date" in out.columns:
        out = out.set_index("date")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out.drop(columns=[c for c in ("set",) if c in out.columns])
    return to_regular_index(out, freq=freq)


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


def _is_categorical(s: pd.Series) -> bool:
    """True for a covariate Chronos-2 should encode as a category rather than a number.

    Anything pandas does not call numeric -- ``object``, ``category``,
    ``string`` -- is a category. Booleans fall on the numeric side, which is
    what we want: they already carry the ordering an encoder would have to
    rediscover, and upstream reads them as numeric too.
    """
    return not pd.api.types.is_numeric_dtype(s)


def _as_category_values(s: pd.Series) -> np.ndarray:
    """Categorical covariate values as an object array, NaN preserved.

    ``preprocess._stack_covariate`` concatenates these across the batch and asks
    numpy whether the result is numeric; an object array of strings is what makes
    it choose ``category`` dtype. Categories are not fixed here on purpose --
    upstream derives them from the past window and maps the future onto them, so
    a level that appears only in the forecast window is handled as unseen rather
    than silently renumbering the rest.
    """
    return np.asarray(s.astype(object).to_numpy(), dtype=object)


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
    batch_size : int, default 32
        How many resampled draws :meth:`deweather` packs into a single forward
        pass. Measured on an L40S this is worth ~3x from 8 draws upward; on a
        single-threaded CPU it is worth nothing, because the samples are
        compute-bound rather than dispatch-bound. Memory scales with it, so
        lower it if a long context and many covariates exhaust the device.
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
        batch_size: int = 32,
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
        self.batch_size = max(1, int(batch_size))
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

    def _resolve_covariates(self, df: pd.DataFrame, target: str | Sequence[str]) -> list[str]:
        """Covariate columns for ``target``, which may name several variates."""
        targets = {target} if isinstance(target, str) else set(target)
        if self.met_covariates is not None:
            missing = [c for c in self.met_covariates if c not in df.columns]
            if missing:
                raise KeyError(f"covariate columns absent from frame: {missing}")
            cols = list(self.met_covariates)
        else:
            cols = [
                c
                for c in df.columns
                if c not in targets
                and c not in _CALENDAR_ENCODERS
                and (pd.api.types.is_numeric_dtype(df[c]) or _is_categorical(df[c]))
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
        """One ``predict`` input dict: numeric covariates cleaned, categoricals left alone.

        Chronos-2 encodes categorical covariates itself -- ``predict`` routes a
        list of dicts through ``preprocess.from_list_of_dicts``, which reads a
        non-numeric column as a pandas ``category``, target-encodes it against
        the observed target, and maps the future values onto the categories seen
        in the past. Handing it an object array is therefore the whole
        integration; one-hotting first would spend a covariate slot per level and
        throw away the ordering the encoder recovers.

        NaN is passed through rather than filled, because the encoder gives it
        its own category slot -- a station with no recorded site type is a fact
        about the station, not a value to impute.
        """
        numeric = [c for c in cols if not _is_categorical(past[c])]
        categorical = [c for c in cols if _is_categorical(past[c])]
        fills = {
            c: float(np.nanmean(past[c])) if np.isfinite(past[c]).any() else 0.0 for c in numeric
        }
        past_cov: dict[str, np.ndarray] = {
            c: self._clean(past[c].to_numpy(), fills[c]) for c in numeric
        }
        future_cov: dict[str, np.ndarray] = {
            c: self._clean(future[c].to_numpy(), fills[c]) for c in numeric
        }
        for c in categorical:
            past_cov[c] = _as_category_values(past[c])
            future_cov[c] = _as_category_values(future[c])
        return {
            "target": np.asarray(target_hist, dtype=np.float32),
            "past_covariates": past_cov,
            "future_covariates": future_cov,
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
        return self._forecast_blocks([(target_hist, past, future)], cols, len(future))[0]

    def _forecast_blocks(
        self,
        items: Sequence[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]],
        cols: Sequence[str],
        horizon: int,
    ) -> np.ndarray:
        """Forward passes for several ``(history, past, future)`` triples.

        Chronos-2's ``predict`` takes a list of inputs sharing a horizon and a
        covariate schema, which is exactly what the resampled draws inside
        :meth:`deweather` are. Sending them together rather than one at a time is
        worth ~3x on a GPU and nothing on a single-threaded CPU, and leaves the
        numbers alone to float32 rounding (~1e-5) since ``predict`` is
        deterministic. Batches are capped at :attr:`batch_size`.

        Returns ``(len(items), n_quantiles, horizon)``.
        """
        inputs = []
        for hist, past, future in items:
            self._check_context(hist)
            inputs.append(self._make_input(hist, past, future, cols))
        raw = self._predict_raw(inputs, horizon)
        if len(raw) != len(items):
            raise RuntimeError(f"pipeline returned {len(raw)} forecasts for {len(items)} inputs")
        return np.stack([block[0] for block in raw])

    def _predict_raw(
        self,
        inputs: Sequence[dict[str, Any]],
        horizon: int,
        cross_learning: bool = False,
    ) -> list[np.ndarray]:
        """Every ``pipeline.predict`` call in this class goes through here.

        Returns one array per input, each ``(n_variates, n_quantiles, horizon)``
        -- the shape Chronos-2 emits, kept whole so the multivariate paths can
        read variates past the first.

        ``cross_learning`` puts every task in a batch into one group, so the
        model may share information across them. Two consequences follow from
        it being a *batch* property, and both are the caller's to manage:
        results depend on :attr:`batch_size`, and only tasks that land in the
        same chunk actually see each other. Upstream reports a group size of
        about 100 in the Chronos-2 technical report; far above that the group
        drifts from what the model saw in pretraining.
        """
        pipe = self._load_pipeline()
        out: list[np.ndarray] = []
        for s in range(0, len(inputs), self.batch_size):
            preds = pipe.predict(
                list(inputs[s : s + self.batch_size]),
                prediction_length=horizon,
                cross_learning=cross_learning,
            )
            out.extend(np.asarray(pred) for pred in preds)
        return out

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

    def _window(
        self, df: pd.DataFrame, anchor: pd.Timestamp, horizon: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Context and forecast windows around ``anchor``, sized by this estimator."""
        past = df.loc[df.index < anchor].iloc[-self.context_length :]
        future = df.loc[df.index >= anchor].iloc[:horizon]
        if past.empty or future.empty:
            raise ValueError("anchor leaves no context or no forecast window")
        return past, future

    def _quantile_frame(
        self, block: np.ndarray, index: pd.Index, quantiles: Sequence[float]
    ) -> pd.DataFrame:
        """One variate's ``(n_quantiles, horizon)`` block as a labelled frame."""
        return pd.DataFrame({f"q{q}": block[self._q_index(q)] for q in quantiles}, index=index)

    def predict_quantiles_multivariate(
        self,
        df: pd.DataFrame,
        targets: Sequence[str],
        anchor: pd.Timestamp | str,
        horizon: int | None = None,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        covariates: Sequence[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Forecast several target series that share a frame, as one joint task.

        Chronos-2 accepts a 2-D target of shape ``(n_variates, history)`` and
        attends across the variates, so the species measured at one site --
        or a site and its neighbours, once they are columns of one frame --
        are predicted together rather than one at a time. What this buys over
        looping is the cross-variate structure: NO2 and NOx at the same kerbside
        rise and fall together, and a joint call can use one to inform the other.

        The variates must share an index and a horizon, which is the same
        requirement :meth:`predict_quantiles` places on a single series.
        Covariates are shared across the variates -- they are properties of the
        site and its weather, not of the species.

        One upstream detail worth knowing: target encoding of categorical
        covariates is only defined against a single target, so on this path
        upstream falls back to ordinal encoding. A categorical covariate still
        reaches the model; it is simply encoded less informatively than it would
        be in a univariate call.

        Parameters
        ----------
        df : pandas.DataFrame
            Datetime-indexed frame holding every column in ``targets`` and the
            covariates.
        targets : sequence of str
            Target column names, in the order the variates should be stacked.
        anchor : Timestamp or str
            Forecast origin, as in :meth:`predict_quantiles`.
        horizon : int, optional
            Hours to predict. Defaults to :attr:`prediction_length`.
        quantiles : sequence of float
            Levels to return. Snapped to the nearest natively emitted level.
        covariates : sequence of str, optional
            Overrides the resolved covariate set for this call.

        Returns
        -------
        dict of str to pandas.DataFrame
            One frame per target, keyed by column name, each shaped like the
            return of :meth:`predict_quantiles`.
        """
        targets = list(targets)
        if len(targets) < 2:
            raise ValueError(
                "predict_quantiles_multivariate needs at least two targets; "
                "use predict_quantiles for a single series"
            )
        missing = [t for t in targets if t not in df.columns]
        if missing:
            raise KeyError(f"target columns absent from frame: {missing}")

        self._check_index(df.index)
        at = pd.Timestamp(anchor)
        horizon = int(horizon or self.prediction_length)
        cols = list(covariates) if covariates is not None else self._resolve_covariates(df, targets)
        past, future = self._window(df, at, horizon)

        stacked = np.stack([past[t].to_numpy(dtype=np.float32) for t in targets])
        for row in stacked:
            self._check_context(row)
        item = self._make_input(stacked, past, future, cols)

        block = self._predict_raw([item], horizon)[0]
        if block.shape[0] != len(targets):
            raise RuntimeError(
                f"pipeline returned {block.shape[0]} variates for {len(targets)} targets"
            )
        return {
            name: self._quantile_frame(block[i], future.index, quantiles)
            for i, name in enumerate(targets)
        }

    def predict_quantiles_multisite(
        self,
        frames: Mapping[str, pd.DataFrame],
        target: str,
        anchor: pd.Timestamp | str,
        horizon: int | None = None,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        covariates: Sequence[str] | None = None,
        cross_learning: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Forecast the same target at several sites in one call.

        Each site keeps its own frame, its own history and its own covariates --
        they are separate tasks, not variates of one task, so their records may
        differ in length and need not be aligned. What ``cross_learning=True``
        adds is that the model treats the batch as one group and may carry
        structure between the sites: upstream reports it helps most where an
        individual series has little history, which is exactly a newly
        commissioned station sitting next to twenty established ones.

        It is not a free improvement. Upstream is explicit that cross-learning
        does not always help and has to be tested per use case, and because the
        sharing happens within a batch, the answer for a site depends on which
        other sites were in the call and on :attr:`batch_size`. That is why it
        is a parameter and why ``cross_learning=False`` -- which reproduces a
        per-site loop, up to batching -- is one keyword away.

        Parameters
        ----------
        frames : mapping of str to pandas.DataFrame
            One datetime-indexed frame per site, keyed by site name.
        target : str
            Target column name, the same in every frame.
        anchor : Timestamp or str
            Forecast origin, applied to every site.
        horizon : int, optional
            Hours to predict. Defaults to :attr:`prediction_length`.
        quantiles : sequence of float
            Levels to return. Snapped to the nearest natively emitted level.
        covariates : sequence of str, optional
            Overrides the resolved covariate set. Resolved once from the first
            frame otherwise, because the batch shares one covariate schema.
        cross_learning : bool, default True
            Whether the sites are predicted jointly.

        Returns
        -------
        dict of str to pandas.DataFrame
            One frame per site, keyed as ``frames`` was.
        """
        if not frames:
            raise ValueError("frames is empty; nothing to forecast")
        names = list(frames)
        at = pd.Timestamp(anchor)
        horizon = int(horizon or self.prediction_length)

        first = frames[names[0]]
        cols = (
            list(covariates) if covariates is not None else self._resolve_covariates(first, target)
        )

        items: list[dict[str, Any]] = []
        indices: list[pd.Index] = []
        for name in names:
            df = frames[name]
            if target not in df.columns:
                raise KeyError(f"site {name!r} has no column {target!r}")
            self._check_index(df.index)
            past, future = self._window(df, at, horizon)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise KeyError(f"site {name!r} is missing covariates {missing}")
            hist = past[target].to_numpy(dtype=np.float32)
            self._check_context(hist)
            items.append(self._make_input(hist, past, future, cols))
            indices.append(future.index)

        raw = self._predict_raw(items, horizon, cross_learning=cross_learning)
        if len(raw) != len(names):
            raise RuntimeError(f"pipeline returned {len(raw)} forecasts for {len(names)} sites")
        return {
            name: self._quantile_frame(raw[i][0], indices[i], quantiles)
            for i, name in enumerate(names)
        }

    def finetune(
        self,
        df: pd.DataFrame,
        target: str,
        *,
        mode: str = "lora",
        prediction_length: int | None = None,
        covariates: Sequence[str] | None = None,
        validation_df: pd.DataFrame | None = None,
        learning_rate: float | None = None,
        num_steps: int = 1000,
        batch_size: int = 32,
        output_dir: str | Path | None = None,
        lora_config: Mapping[str, Any] | None = None,
        **trainer_kwargs: Any,
    ) -> Chronos2Estimator:
        """Adapt the checkpoint to one site's own record and return a new estimator.

        Everything else in this class is zero-shot: the pretrained weights are
        read and never written. This is the one method that trains, and it is
        deliberately not :meth:`fit`. ``fit`` is on the sklearn-shaped path that
        :func:`normet.do_all` walks, where a call costs nothing and is made
        freely; silently turning that into a thousand optimiser steps on a GPU
        would be a trap. Fine-tuning is something you ask for by name.

        The estimator returned is a new one wrapping a fine-tuned copy of the
        pipeline -- ``self`` is left on the pretrained weights, so a fine-tune
        can be compared against the baseline it came from without reloading.

        Parameters
        ----------
        df : pandas.DataFrame
            Datetime-indexed frame holding ``target`` and the covariates. The
            whole record is handed over as one series; the trainer samples its
            own windows from it, so no windowing is needed here.
        target : str
            Target column name.
        mode : {"lora", "full"}, default "lora"
            ``"lora"`` trains low-rank adapters and leaves the base weights
            alone; ``"full"`` updates every parameter. LoRA is the default
            because a single station's record is small next to a 119M-parameter
            model, which is the setting full fine-tuning overfits.
        prediction_length : int, optional
            Horizon to fine-tune for. Defaults to :attr:`prediction_length`, so
            the adapted model is trained for the horizon it will be asked about.
        covariates : sequence of str, optional
            Overrides the resolved covariate set.
        validation_df : pandas.DataFrame, optional
            Held-out frame for model selection. Same columns as ``df``.
        learning_rate : float, optional
            Defaults to upstream's recommendation for the mode: 1e-5 for LoRA,
            1e-6 for full.
        num_steps : int, default 1000
            Optimiser steps.
        batch_size : int, default 32
            Series per step, counting covariates. Upstream's default is 256,
            lowered here because normet's inputs carry a covariate per
            meteorological variable and the effective batch is correspondingly
            larger.
        output_dir : path-like, optional
            Where the HuggingFace ``Trainer`` writes checkpoints.
        lora_config : mapping, optional
            Overrides for ``peft.LoraConfig``. Ignored when ``mode="full"``.
        **trainer_kwargs
            Forwarded to ``TrainingArguments``.

        Returns
        -------
        Chronos2Estimator
            A new estimator holding the fine-tuned pipeline, configured
            identically to this one.

        Raises
        ------
        ImportError
            If ``mode="lora"`` and ``peft`` is not installed. Upstream warns and
            silently falls back to full fine-tuning in that case, which is a
            different and far more expensive thing than what was asked for.
        """
        if mode not in ("lora", "full"):
            raise ValueError(f"mode must be 'lora' or 'full', got {mode!r}")
        if target not in df.columns:
            raise KeyError(f"target column absent from frame: {target!r}")
        if mode == "lora" and importlib.util.find_spec("peft") is None:
            raise ImportError(
                "mode='lora' requires peft. Install it with `pip install peft`, or "
                "pass mode='full' if you meant to update every parameter."
            )

        horizon = int(prediction_length or self.prediction_length)
        cols = list(covariates) if covariates is not None else self._resolve_covariates(df, target)
        self._check_index(df.index)

        inputs = self._training_inputs(df, target, cols, horizon)
        validation = (
            self._training_inputs(validation_df, target, cols, horizon)
            if validation_df is not None
            else None
        )

        if learning_rate is None:
            learning_rate = 1e-5 if mode == "lora" else 1e-6

        log.info(
            "Fine-tuning %s (%s) on %d rows for horizon %d, %d steps at lr %g",
            self.model_name,
            mode,
            len(df),
            horizon,
            num_steps,
            learning_rate,
        )
        pipe = self._load_pipeline()
        tuned = pipe.fit(
            inputs,
            prediction_length=horizon,
            validation_inputs=validation,
            finetune_mode=mode,
            lora_config=dict(lora_config) if lora_config is not None else None,
            context_length=self.context_length,
            learning_rate=learning_rate,
            num_steps=num_steps,
            batch_size=batch_size,
            output_dir=output_dir,
            **trainer_kwargs,
        )

        out = Chronos2Estimator(
            model_name=self.model_name,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            met_covariates=cols,
            use_calendar=self.use_calendar,
            device=self.device,
            batch_size=self.batch_size,
            min_context_coverage=self.min_context_coverage,
        )
        out._pipeline = tuned
        out._quantiles = np.asarray(tuned.quantiles, dtype=float)
        out.target_col = target
        out.feature_cols = list(cols)
        return out

    def _training_inputs(
        self, df: pd.DataFrame, target: str, cols: Sequence[str], horizon: int
    ) -> list[Any]:
        """Prepare one series for ``Chronos2Pipeline.fit``.

        Built through upstream's ``from_list_of_dicts`` rather than handed over
        as a plain dict, because only that route takes
        ``known_covariates_names``. It matters: normet's covariates are
        meteorology, which is known across the forecast window -- that is what
        makes de-weathering possible at all -- and a training run that treated
        them as past-only would adapt the model to a problem it will never be
        asked to solve.
        """
        from chronos.chronos2 import preprocess

        past = {
            c: _as_category_values(df[c]) if _is_categorical(df[c]) else df[c].to_numpy(np.float32)
            for c in cols
        }
        item = {"target": df[target].to_numpy(np.float32), "past_covariates": past}
        return list(
            preprocess.from_list_of_dicts(
                [item],
                prediction_length=horizon,
                known_covariates_names=list(cols),
            )
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
        step = self.prediction_length
        # Each block conditions on observed history rather than on the previous
        # block's output, so the blocks are independent and can be batched. Only
        # the last one can be short, and a batch has to share one horizon, so it
        # is sent on its own.
        spans = [(s, min(s + step, len(X))) for s in range(self.context_length, len(X), step)]
        full = [sp for sp in spans if sp[1] - sp[0] == step]
        short = [sp for sp in spans if sp[1] - sp[0] != step]
        for group in (full, short):
            if not group:
                continue
            items = []
            for s, e in group:
                past = X.iloc[max(0, s - self.context_length) : s]
                items.append((past[target].to_numpy(), past, X.iloc[s:e]))
            blocks = self._forecast_blocks(items, cols, group[0][1] - group[0][0])
            for (s, e), block in zip(group, blocks, strict=True):
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
            hist = past[target].to_numpy()
            total = {q: np.zeros(e - s, dtype=np.float64) for q in quantiles}
            # Draws reach the pipeline batch_size at a time. Drawing them in the
            # same order as a per-sample loop keeps a given random_state on the
            # same weather; holding only one batch of resampled frames keeps
            # memory flat in n_samples.
            for done in range(0, n_samples, self.batch_size):
                items = []
                for _ in range(min(self.batch_size, n_samples - done)):
                    idx = rng.integers(0, len(df), size=len(past) + len(future))
                    swap_past = past.copy()
                    swap_future = future.copy()
                    swap_past[met] = df[met].to_numpy()[idx[: len(past)]]
                    swap_future[met] = df[met].to_numpy()[idx[len(past) :]]
                    items.append((hist, swap_past, swap_future))
                blocks = self._forecast_blocks(items, cols, e - s)
                for q in quantiles:
                    total[q] += blocks[:, i_q[q], :].sum(axis=0)
            for q in quantiles:
                acc[q][s:e] = total[q] / n_samples

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

        shuffled = future.copy()
        for c in met:
            shuffled[c] = rng.permutation(future[c].to_numpy())
        hist = past[target].to_numpy()
        blocks = self._forecast_blocks(
            [(hist, past, future), (hist, past, shuffled)], cols, len(future)
        )
        base, alt = blocks[0][i50], blocks[1][i50]

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
