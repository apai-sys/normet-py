# src/normet/causal/__init__.py
"""Synthetic-control / counterfactual estimators, diagnostics, and inference."""

# --- Core methods ------------------------------------------------------------
from .scm import scm

# mlscm may require optional deps; fail gracefully with a helpful error.
try:
    from .mlscm import mlscm

    _HAS_MLSCM = True
except Exception as _e:
    _HAS_MLSCM = False
    _MLSCM_IMPORT_ERR = _e

    def mlscm(*args, **kwargs):  # type: ignore[misc]
        """Raise ``ImportError`` because mlscm's optional dependencies are unavailable."""
        raise ImportError(
            "mlscm is unavailable because its optional dependencies failed to import. "
            f"Original error: {_MLSCM_IMPORT_ERR}\n"
            "Install an AutoML backend (e.g., flaml) and ensure it's importable."
        )


# --- Bands & uncertainty -----------------------------------------------------
from .bands import (
    effect_bands_space,
    effect_bands_time,
    plot_effect_with_bands,
    plot_uncertainty_bands,
    uncertainty_bands,
)

# --- Batch runner ------------------------------------------------------------
from .batch import scm_all
from .diagnostics import loo_weight_stability, scm_diagnostics
from .inference import conformal_effect_interval, rmspe_ratio_test

# Bayesian SCM is heavy (PyMC); graceful fallback if PyMC missing.
try:
    from .bayesian_scm import bayesian_scm  # noqa: F401

    _HAS_BAYESIAN = True
except Exception as _bay_err:
    _HAS_BAYESIAN = False
    _BAYESIAN_ERR = _bay_err

    def bayesian_scm(*args, **kwargs):  # type: ignore[misc]
        """Raise ``ImportError`` because ``pymc``/``arviz`` are not installed."""
        raise ImportError(
            f"bayesian_scm requires pymc + arviz. Original error: {_BAYESIAN_ERR}. "
            "Install via `pip install pymc arviz`."
        )


# --- Placebo tests -----------------------------------------------------------
# --- Panel preparation --------------------------------------------------------
from .panel import prepare_panel
from .placebo import placebo_in_space, placebo_in_time
from .run_scm import run_scm
from .variants import did_baseline, scm_abadie, scm_mcnnm, scm_robust

# --- Simple backend registry -------------------------------------------------
BACKENDS = {
    "scm": scm,
    "mlscm": mlscm,
    "abadie": scm_abadie,
    "did": did_baseline,
    "mcnnm": scm_mcnnm,
    "robust": scm_robust,
}

__all__ = [
    # run_scm
    "scm",
    "mlscm",
    "run_scm",
    # placebo
    "placebo_in_space",
    "placebo_in_time",
    # bands
    "effect_bands_space",
    "effect_bands_time",
    "uncertainty_bands",
    "plot_effect_with_bands",
    "plot_uncertainty_bands",
    # panel prep
    "prepare_panel",
    # batch
    "scm_all",
    # diagnostics
    "scm_diagnostics",
    "loo_weight_stability",
    # variants
    "scm_abadie",
    "did_baseline",
    "scm_mcnnm",
    "scm_robust",
    # inference
    "conformal_effect_interval",
    "rmspe_ratio_test",
    "bayesian_scm",
    # registry
    "BACKENDS",
]
