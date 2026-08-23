"""Single- and multi-site pipelines that chain prepare/train/normalise steps."""

from ..analysis.rolling import RollingConfig
from .do_all import CHRONOS_BACKEND, SingleConfig, UncConfig, do_all, do_all_unc
from .interface import run_workflow
from .multisite import (
    decompose_multisite,
    do_all_multisite,
    embed_multisite,
    multisite_apply,
)

__all__ = [
    "CHRONOS_BACKEND",
    "do_all",
    "do_all_unc",
    "run_workflow",
    "multisite_apply",
    "do_all_multisite",
    "decompose_multisite",
    "embed_multisite",
    "SingleConfig",
    "UncConfig",
    "RollingConfig",
]
