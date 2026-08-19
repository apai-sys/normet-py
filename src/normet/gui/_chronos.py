"""Chronos-2 backend plumbing for the main window.

The AutoML backends (flaml, lightgbm) *train* a model and then hand it to
:func:`normet.normalise`. Chronos-2 is zero-shot: there is nothing to fit, no
train/test split, no hyperparameter budget and no feature importance, and the
de-weathering runs through :meth:`Chronos2Estimator.deweather` rather than
:func:`normet.normalise`. Rather than bend either side, the window branches on
the backend and calls into here.

Everything in this module is import-safe without the ``foundation`` extra:
:func:`chronos_availability` reports whether the extra is present so the window
can grey the option out instead of raising when it is picked.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

#: Value shown in the backend combo box.
BACKEND = "chronos-2"

#: Quantile levels requested from the model when the "quantiles" box is ticked.
#: Chronos-2 emits 21 levels natively and picks the nearest to each of these.
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)

#: Monte-Carlo resamples of the meteorology per de-weathering block.
#:
#: :meth:`Chronos2Estimator.deweather` defaults to 20, which is the wrong order
#: of magnitude behind a GUI: every sample is a full 2048-context forward pass,
#: so a 150-day hourly record (10 blocks) costs 200 of them -- close to two
#: hours on CPU, with no progress to show for it. Eight keeps the expectation
#: over weather stable enough to plot at roughly a third of that. The window
#: exposes this through the existing "Samples" spin box, so anyone with a GPU
#: or the patience for a tighter estimate can raise it.
DEFAULT_SAMPLES = 8


def chronos_availability() -> tuple[bool, str]:
    """Return ``(available, reason)`` for the Chronos-2 backend."""
    missing = [p for p in ("chronos", "torch") if importlib.util.find_spec(p) is None]
    if missing:
        return False, (
            f"Chronos-2 needs the 'foundation' extra ({', '.join(missing)} not installed).\n"
            "Install with:  pip install normet[foundation]"
        )
    return True, ""


def to_indexed_frame(df_prep: pd.DataFrame) -> pd.DataFrame:
    """Turn normet's ``date``-column frame into the gap-free DatetimeIndex Chronos-2 needs.

    Chronos-2 reads position as time, so a frame whose rows skip missing hours
    is read as if those hours never happened and the series slides against its
    own calendar covariates. ``prepare_data`` drops incomplete rows, which is
    exactly that situation, so the grid is rebuilt here and the gaps left as
    NaN for the model to mask.
    """
    from normet.foundation import to_regular_index

    out = df_prep.copy()
    if "date" in out.columns:
        out = out.set_index("date")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    # `set` is prepare_data's train/test label; it means nothing zero-shot and
    # would otherwise be picked up as a covariate.
    out = out.drop(columns=[c for c in ("set",) if c in out.columns])
    return to_regular_index(out)


def load_estimator(
    df_raw: pd.DataFrame,
    target: str,
    covariates: list[str],
    met_features: list[str],
    device: str | None = None,
) -> tuple[pd.DataFrame, Any]:
    """Prepare the data and bring up a Chronos-2 estimator. Runs on a worker thread.

    Downloads roughly 500 MB of weights the first time a checkpoint is used,
    which is the reason this must not run on the GUI thread.
    """
    from normet import prepare_data
    from normet.foundation import Chronos2Estimator

    df_prep = prepare_data(df_raw, target, covariates, split_method="random")

    estimator = Chronos2Estimator(met_covariates=met_features, device=device)
    log.info("Loading Chronos-2 checkpoint %s ...", estimator.model_name)
    estimator._load_pipeline()
    log.info(
        "Chronos-2 ready on %s (context %d h, horizon %d h, %d native quantile levels)",
        estimator.device,
        estimator.context_length,
        estimator.prediction_length,
        len(estimator.quantile_levels),
    )

    # Fail here, at "Load model", rather than deep inside the de-weathering run.
    indexed = to_indexed_frame(df_prep)
    estimator._check_index(indexed.index)

    return df_prep, estimator


def sensitivity(estimator: Any, df_prep: pd.DataFrame, met_features: list[str]) -> dict[str, float]:
    """Run the covariate-sensitivity diagnostic over the most recent horizon.

    This is the honest quality check for a zero-shot de-weathering: if the
    forecast barely moves when the future meteorology is shuffled, the model is
    autoregressing and the "normalised" series it produces means nothing. It
    stands in for the parity plot and feature importances the AutoML backends
    show, neither of which exists here.
    """
    indexed = to_indexed_frame(df_prep)
    horizon = estimator.prediction_length
    if len(indexed) <= estimator.context_length + horizon:
        raise ValueError(
            f"need more than {estimator.context_length + horizon} rows to run the "
            f"diagnostic (have {len(indexed)}); shorten the context or load a longer record"
        )
    anchor = indexed.index[-horizon]
    return estimator.covariate_sensitivity(
        indexed, "value", anchor=anchor, horizon=horizon, met_features=met_features, random_state=0
    )


def deweather(
    estimator: Any,
    df_prep: pd.DataFrame,
    met_features: list[str],
    with_quantiles: bool,
    n_samples: int = DEFAULT_SAMPLES,
) -> pd.DataFrame:
    """De-weather on normet's schema so the existing plot and report paths apply.

    ``n_samples`` is the dominant cost here -- see :data:`DEFAULT_SAMPLES`.
    """
    indexed = to_indexed_frame(df_prep)
    quantiles = QUANTILES if with_quantiles else (0.5,)
    out = estimator.deweather(
        indexed,
        "value",
        met_features=met_features,
        n_samples=n_samples,
        quantiles=quantiles,
        random_state=0,
        schema="normet",
    )
    out.index.name = "date"
    return out


def verdict_for(shift: dict[str, float]) -> tuple[str, str]:
    """Traffic-light verdict from the covariate-sensitivity numbers."""
    pct = shift["pct_of_prediction"]
    if pct >= 5.0:
        return (
            "ok",
            f"Meteorology drives the forecast — shuffling it moves the median by "
            f"{shift['mean_abs_shift']:.2f} ({pct:.1f}% of the prediction). "
            "De-weathering is meaningful here.",
        )
    if pct >= 1.0:
        return (
            "warn",
            f"Weak meteorological response — shuffling the weather moves the median by only "
            f"{shift['mean_abs_shift']:.2f} ({pct:.1f}%). The normalised series is mostly "
            "carried by the target's own history; treat it cautiously.",
        )
    return (
        "error",
        f"No meteorological response — shuffling the weather changes the median by "
        f"{shift['mean_abs_shift']:.3f} ({pct:.2f}%). The model is autoregressing, so a "
        "'de-weathered' series from it would be meaningless. Check the met columns.",
    )
