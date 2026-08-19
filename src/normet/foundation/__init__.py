"""Foundation model embeddings, zero-shot de-weathering, and counterfactual inference."""

from .chronos import ChronosEmbedder
from .estimator import (
    Chronos2Estimator,
    CounterfactualResult,
    InsufficientContextError,
    IrregularIndexError,
    add_calendar_covariates,
    resolve_device,
    to_normet_frame,
    to_regular_index,
)

__all__ = [
    "ChronosEmbedder",
    "add_calendar_covariates",
    "Chronos2Estimator",
    "CounterfactualResult",
    "InsufficientContextError",
    "IrregularIndexError",
    "resolve_device",
    "to_normet_frame",
    "to_regular_index",
]
