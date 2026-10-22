"""Core analysis routines: normalisation, decomposition, rolling, PDP, events."""

from .decomposition import DecomposeConfig, decom_emi, decom_met, decompose
from .events import anomaly_scores, detect_events
from .normalise import NormaliseConfig, normalise, normalise_auto
from .pdp import pdp
from .rolling import RollingConfig, rolling, window_columns
from .trend import (
    TrendBreaks,
    TrendResult,
    TrendSlope,
    TrendTest,
    detect_trend_breaks,
    mann_kendall,
    theil_sen,
    trend,
)
from .verification import crps_from_quantiles, crps_from_samples, crps_observed, pinball_loss

__all__ = [
    "TrendResult",
    "TrendSlope",
    "TrendTest",
    "TrendBreaks",
    "mann_kendall",
    "theil_sen",
    "trend",
    "detect_trend_breaks",
    "DecomposeConfig",
    "NormaliseConfig",
    "RollingConfig",
    "normalise",
    "normalise_auto",
    "decom_emi",
    "decom_met",
    "decompose",
    "pdp",
    "rolling",
    "window_columns",
    "detect_events",
    "anomaly_scores",
    "crps_from_quantiles",
    "crps_from_samples",
    "crps_observed",
    "pinball_loss",
]
