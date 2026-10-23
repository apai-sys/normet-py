"""Foundation model embeddings, zero-shot de-weathering, and counterfactual inference."""

from .calibration import CalibratedEstimator, QuantileCalibration, calibrate_quantiles
from .chronos import ChronosEmbedder
from .embedding_diagnostics import (
    embedding_projection,
    plot_embedding_diagnostics,
    silhouette_curve,
)
from .estimator import (
    CHRONOS_BACKEND,
    CHRONOS_DEFAULT_SAMPLES,
    Chronos2Estimator,
    CounterfactualResult,
    InsufficientContextError,
    IrregularIndexError,
    add_calendar_covariates,
    resolve_device,
    resolve_n_samples,
    to_indexed_frame,
    to_normet_frame,
    to_regular_index,
)

__all__ = [
    "CHRONOS_BACKEND",
    "CHRONOS_DEFAULT_SAMPLES",
    "CalibratedEstimator",
    "ChronosEmbedder",
    "QuantileCalibration",
    "add_calendar_covariates",
    "calibrate_quantiles",
    "embedding_projection",
    "plot_embedding_diagnostics",
    "silhouette_curve",
    "Chronos2Estimator",
    "CounterfactualResult",
    "InsufficientContextError",
    "IrregularIndexError",
    "resolve_device",
    "resolve_n_samples",
    "to_indexed_frame",
    "to_normet_frame",
    "to_regular_index",
]
