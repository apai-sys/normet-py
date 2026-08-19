"""Foundation model embeddings, zero-shot de-weathering, and counterfactual inference."""

from .chronos import ChronosEmbedder
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
    "ChronosEmbedder",
    "add_calendar_covariates",
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
