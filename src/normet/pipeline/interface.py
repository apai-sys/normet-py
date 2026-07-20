# src/normet/pipeline/interface.py
"""Generic :func:`run_workflow` dispatcher over the ``single``/``unc``/``rolling`` pipelines."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import pandas as pd

from ..analysis.rolling import RollingConfig, rolling

# --- Import your existing primitives ---
from .do_all import SingleConfig, UncConfig, do_all, do_all_unc

# =========================
# Config type aliases
# =========================
Backend = str
SplitMethod = Literal["random", "ts", "month_ts", "season_ts"]


# =========================
# Adapters (align with function signatures)
# =========================
def _single_adapter(cfg: SingleConfig) -> dict:
    return dict(
        config=cfg,
    )


def _unc_adapter(cfg: UncConfig) -> dict:
    return dict(
        config=cfg,
    )


def _rolling_adapter(cfg: RollingConfig) -> dict:
    return dict(
        config=cfg,
    )


Mode = Literal["single", "unc", "rolling"]

_REGISTRY: Mapping[Mode, tuple[Callable[..., Any], Callable[[Any], dict]]] = {
    "single": (lambda *, df, **kw: do_all(df=df, **kw), _single_adapter),
    "unc": (lambda *, df, **kw: do_all_unc(df=df, **kw), _unc_adapter),
    "rolling": (lambda *, df, **kw: rolling(df=df, **kw), _rolling_adapter),
}


# =========================
# Entry point
# =========================
def run_workflow(mode: Mode, df: pd.DataFrame, config: Any) -> Any:
    """
    Run a workflow by mode with its config dataclass.

    Parameters
    ----------
    mode : Mode
        One of ``"single"``, ``"unc"``, or ``"rolling"``.
    df : pandas.DataFrame
        Input data.
    config : object
        Configuration dataclass for the selected mode.

    Returns
    -------
    - mode == "single" : (out: DataFrame, model: object, mod_stats: DataFrame)
    - mode == "unc"    : (out: DataFrame, mod_stats: DataFrame)
    - mode == "rolling": out: DataFrame
    """
    if mode not in _REGISTRY:
        raise ValueError(f"Unknown mode '{mode}'. Available: {list(_REGISTRY)}")

    runner, adapter = _REGISTRY[mode]
    kwargs = adapter(config)
    return runner(df=df, **kwargs)
