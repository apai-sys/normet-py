"""Shared utilities: config, data prep, features, metrics, CV, caching, provenance, validation."""

from ._config import DEFAULT_SEED, available_cores, default_workers, resolve_config
from .cache import config_hash, dataframe_hash, make_memory
from .cv import blocked_time_cv, cv_score, time_series_cv
from .featureeng import (
    DEFAULT_MET_BOUNDS,
    LagDiagnostics,
    add_lag_features,
    add_met_physics_features,
    add_rolling_features,
    analyze_lag,
    cyclical_encode,
    screen_aq_data,
    wind_to_uv,
)
from .features import extract_features, model_feature_names, permutation_importance
from .metrics import compare_models, exceedance_stats, modStats
from .prepare import (
    add_date_variables,
    check_data,
    compute_duan_smearing,
    impute_values,
    prepare_data,
    process_date,
    split_into_sets,
)
from .provenance import NormetRun, load_run, make_run, save_run
from .validate import require_column, require_no_duplicates, require_no_nan_in, require_not_empty

__all__ = [
    "resolve_config",
    "available_cores",
    "default_workers",
    "DEFAULT_SEED",
    "prepare_data",
    "process_date",
    "check_data",
    "impute_values",
    "compute_duan_smearing",
    "add_date_variables",
    "split_into_sets",
    "modStats",
    "exceedance_stats",
    "compare_models",
    "extract_features",
    "model_feature_names",
    "permutation_importance",
    "DEFAULT_MET_BOUNDS",
    "screen_aq_data",
    "add_met_physics_features",
    "add_lag_features",
    "add_rolling_features",
    "analyze_lag",
    "LagDiagnostics",
    "cyclical_encode",
    "wind_to_uv",
    "time_series_cv",
    "blocked_time_cv",
    "cv_score",
    "make_memory",
    "dataframe_hash",
    "config_hash",
    "NormetRun",
    "make_run",
    "save_run",
    "load_run",
    "require_column",
    "require_not_empty",
    "require_no_nan_in",
    "require_no_duplicates",
]
