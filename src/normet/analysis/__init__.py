"""Core analysis routines: normalisation, decomposition, rolling, PDP, events."""

from .decomposition import DecomposeConfig, decom_emi, decom_met, decompose
from .events import anomaly_scores, detect_events
from .normalise import NormaliseConfig, normalise, normalise_auto
from .pdp import pdp
from .rolling import RollingConfig, rolling

__all__ = [
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
    "detect_events",
    "anomaly_scores",
]
