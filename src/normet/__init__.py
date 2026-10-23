"""normet: Normalisation, Decomposition, and Counterfactual Modelling for Environmental Time-series.

High-level entry points for normalisation, decomposition, synthetic control,
modelling, and evaluation.
"""

from importlib.metadata import version as _version

__version__ = _version("normet")

# --- Exceptions ---
from . import datasets
from .exceptions import (
    CalibrationWarning,
    ConfigError,
    DataError,
    ExperimentalWarning,
    ModelError,
    NormetError,
)

# --- Modelling ---
from .model import (
    build_model,
    load_model,
    ml_predict,
    ml_predict_dask,
    save_model,
    train_model,
)

# --- Evaluation & utils ---
from .utils import (
    LagDiagnostics,
    NormetRun,
    add_date_variables,
    add_lag_features,
    add_rolling_features,
    analyze_lag,
    check_data,
    config_hash,
    cv_score,
    cyclical_encode,
    dataframe_hash,
    impute_values,
    load_run,
    make_memory,
    make_run,
    modStats,
    prepare_data,
    process_date,
    save_run,
    split_into_sets,
    time_series_cv,
    wind_to_uv,
)

# --- I/O adapters ---
try:
    from .io import (
        ALL_DIAGNOSTICS,
        ARL_GDAS1_BASE_URL,
        EEA_POLLUTANT_CODES,
        ERA5_AQ_VARIABLES_DEFAULT,
        UKAQ_SOURCES,
        build_trajectory_features,
        fetch_eea_data,
        fetch_era5_timeseries,
        fetch_gdas1,
        fetch_openaq_measurements,
        fetch_openmeteo_timeseries,
        fetch_ukaq_measurements,
        gdas1_filenames,
        join_trajectory_features,
        list_ukaq_stations,
        load_source_regions,
        openaq_locations,
        openaq_sensors,
        read_trajectory_tdump,
        run_back_trajectories,
        trajectory_features,
    )

    _HAS_IO = True
except Exception:  # requests / cdsapi missing
    _HAS_IO = False


# --- Analysis ---
from .analysis import (
    DecomposeConfig,
    NormaliseConfig,
    RollingConfig,
    TrendBreaks,
    TrendResult,
    TrendSlope,
    TrendTest,
    UncDecomposition,
    anomaly_scores,
    crps_from_quantiles,
    crps_from_samples,
    crps_observed,
    decom_emi,
    decom_met,
    decompose,
    decompose_uncertainty,
    detect_events,
    detect_trend_breaks,
    mann_kendall,
    normalise,
    normalise_auto,
    pdp,
    pinball_loss,
    rolling,
    theil_sen,
    trend,
    window_columns,
)

# --- Synthetic control ---
from .causal import (
    bayesian_scm,
    conformal_effect_interval,
    did_baseline,
    effect_bands_space,
    effect_bands_time,
    loo_weight_stability,
    mlscm,
    placebo_in_space,
    placebo_in_time,
    plot_effect_with_bands,
    plot_uncertainty_bands,
    pre_trend_test,
    prepare_panel,
    rmspe_ratio_test,
    run_scm,
    scm,
    scm_abadie,
    scm_all,
    scm_diagnostics,
    scm_mcnnm,
    scm_robust,
    uncertainty_bands,
)

# --- Counterfactual policy evaluation ---
from .counterfactual import HarmonicCounterfactual, HarmonicCounterfactualResult

# --- Foundation Models (Chronos) ---
from .foundation import (
    CalibratedEstimator,
    Chronos2Estimator,
    ChronosEmbedder,
    CounterfactualResult,
    InsufficientContextError,
    IrregularIndexError,
    QuantileCalibration,
    add_calendar_covariates,
    calibrate_quantiles,
    resolve_device,
    to_indexed_frame,
    to_normet_frame,
    to_regular_index,
)

# --- Pipelines ---
from .pipeline import (
    SingleConfig,
    UncConfig,
    decompose_multisite,
    do_all,
    do_all_multisite,
    do_all_unc,
    embed_multisite,
    multisite_apply,
    run_workflow,
)

__all__ = [
    # --- Example data ---
    "datasets",
    # --- Exceptions ---
    "NormetError",
    "DataError",
    "ConfigError",
    "ModelError",
    "CalibrationWarning",
    "ExperimentalWarning",
    # --- Config classes ---
    "DecomposeConfig",
    "NormaliseConfig",
    "RollingConfig",
    "SingleConfig",
    "UncConfig",
    # --- Pipelines ---
    "do_all",
    "do_all_unc",
    "run_workflow",
    "multisite_apply",
    "do_all_multisite",
    "embed_multisite",
    "decompose_multisite",
    # --- Analysis ---
    "normalise",
    "normalise_auto",
    "rolling",
    "window_columns",
    "pdp",
    "decom_emi",
    "decom_met",
    "decompose",
    "detect_events",
    "anomaly_scores",
    "trend",
    "mann_kendall",
    "theil_sen",
    "detect_trend_breaks",
    "TrendResult",
    "TrendSlope",
    "TrendTest",
    "TrendBreaks",
    "UncDecomposition",
    "decompose_uncertainty",
    # --- verification ---
    "crps_from_quantiles",
    "crps_from_samples",
    "crps_observed",
    "pinball_loss",
    # --- Synthetic control ---
    "scm",
    "mlscm",
    "run_scm",
    "prepare_panel",
    "placebo_in_space",
    "placebo_in_time",
    "effect_bands_space",
    "effect_bands_time",
    "uncertainty_bands",
    "plot_effect_with_bands",
    "plot_uncertainty_bands",
    "scm_all",
    "scm_diagnostics",
    "loo_weight_stability",
    "pre_trend_test",
    "scm_abadie",
    "did_baseline",
    "scm_mcnnm",
    "scm_robust",
    "conformal_effect_interval",
    "rmspe_ratio_test",
    "bayesian_scm",
    # --- Modelling ---
    "build_model",
    "train_model",
    "ml_predict",
    "ml_predict_dask",
    "load_model",
    "save_model",
    # --- Evaluation & utilities ---
    "modStats",
    "prepare_data",
    "process_date",
    "check_data",
    "impute_values",
    "add_date_variables",
    "split_into_sets",
    # --- Feature engineering ---
    "add_lag_features",
    "add_rolling_features",
    "analyze_lag",
    "LagDiagnostics",
    "cyclical_encode",
    "wind_to_uv",
    # --- Cross-validation ---
    "time_series_cv",
    "cv_score",
    # --- Caching ---
    "make_memory",
    "dataframe_hash",
    "config_hash",
    # --- Provenance ---
    "NormetRun",
    "make_run",
    "save_run",
    "load_run",
    # --- I/O ---
    "fetch_openaq_measurements",
    "openaq_locations",
    "openaq_sensors",
    "fetch_era5_timeseries",
    "ERA5_AQ_VARIABLES_DEFAULT",
    "fetch_openmeteo_timeseries",
    "read_trajectory_tdump",
    "trajectory_features",
    "build_trajectory_features",
    "join_trajectory_features",
    "run_back_trajectories",
    "load_source_regions",
    "ALL_DIAGNOSTICS",
    "ARL_GDAS1_BASE_URL",
    "gdas1_filenames",
    "fetch_gdas1",
    "EEA_POLLUTANT_CODES",
    "fetch_eea_data",
    "UKAQ_SOURCES",
    "fetch_ukaq_measurements",
    "list_ukaq_stations",
    # --- Plotting ---
    "polar_plot",
    "wind_rose",
    "pollution_rose",
    "pdp_grid",
    "decomposition_stack",
    "scm_dashboard",
    "normalise_plot",
    "plot_bayesian_scm",
    "time_series_plot",
    "cycle_profiles",
    # --- Foundation Models (Chronos) ---
    "ChronosEmbedder",
    "Chronos2Estimator",
    "CalibratedEstimator",
    "QuantileCalibration",
    "calibrate_quantiles",
    "CounterfactualResult",
    "InsufficientContextError",
    "IrregularIndexError",
    "add_calendar_covariates",
    "resolve_device",
    "to_indexed_frame",
    "to_normet_frame",
    "to_regular_index",
    # --- Counterfactual policy evaluation ---
    "HarmonicCounterfactual",
    "HarmonicCounterfactualResult",
    # --- Reporting ---
    "generate_html_report",
    "report_to_markdown",
]

from .plotting import (
    cycle_profiles,
    decomposition_stack,
    normalise_plot,
    pdp_grid,
    plot_bayesian_scm,
    polar_plot,
    pollution_rose,
    scm_dashboard,
    time_series_plot,
    wind_rose,
)
from .report import (
    generate_html as generate_html_report,
)
from .report import (
    report_to_markdown,
)
