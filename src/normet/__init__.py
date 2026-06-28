"""normet: Normalisation, Decomposition, and Counterfactual Modelling for Environmental Time-series.

High-level entry points for normalisation, decomposition, synthetic control,
modelling, and evaluation.
"""

from importlib.metadata import version as _version

__version__ = _version("normet")

# --- Exceptions ---
from .exceptions import ConfigError, DataError, ExperimentalWarning, ModelError, NormetError

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
        ARL_GDAS1_BASE_URL,
        AURN_POLLUTANT_CODES,
        EEA_POLLUTANT_CODES,
        ERA5_AQ_VARIABLES_DEFAULT,
        build_trajectory_features,
        fetch_aurn_measurements,
        fetch_eea_data,
        fetch_era5_timeseries,
        fetch_gdas1,
        fetch_openaq_measurements,
        gdas1_filenames,
        list_aurn_stations,
        openaq_locations,
        openaq_sensors,
        read_trajectory_tdump,
        run_back_trajectories,
        trajectory_features,
    )

    _HAS_IO = True
except Exception:  # requests / cdsapi missing
    _HAS_IO = False


# --- Pipelines ---
# --- Analysis ---
from .analysis import (
    DecomposeConfig,
    NormaliseConfig,
    RollingConfig,
    anomaly_scores,
    decom_emi,
    decom_met,
    decompose,
    detect_events,
    normalise,
    normalise_auto,
    pdp,
    rolling,
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
from .pipeline import (
    SingleConfig,
    UncConfig,
    decompose_multisite,
    do_all,
    do_all_multisite,
    do_all_unc,
    multisite_apply,
    run_workflow,
)

__all__ = [
    # --- Exceptions ---
    "NormetError",
    "DataError",
    "ConfigError",
    "ModelError",
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
    "decompose_multisite",
    # --- Analysis ---
    "normalise",
    "normalise_auto",
    "rolling",
    "pdp",
    "decom_emi",
    "decom_met",
    "decompose",
    "detect_events",
    "anomaly_scores",
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
    "read_trajectory_tdump",
    "trajectory_features",
    "build_trajectory_features",
    "run_back_trajectories",
    "ARL_GDAS1_BASE_URL",
    "gdas1_filenames",
    "fetch_gdas1",
    "EEA_POLLUTANT_CODES",
    "fetch_eea_data",
    "AURN_POLLUTANT_CODES",
    "fetch_aurn_measurements",
    "list_aurn_stations",
    # --- Plotting ---
    "polar_plot",
    "pdp_grid",
    "decomposition_stack",
    "scm_dashboard",
    "normalise_plot",
    "plot_bayesian_scm",
    "time_series_plot",
    # --- Reporting ---
    "generate_html_report",
    "report_to_markdown",
]

from .plotting import (
    decomposition_stack,
    normalise_plot,
    pdp_grid,
    plot_bayesian_scm,
    polar_plot,
    scm_dashboard,
    time_series_plot,
)
from .report import (
    generate_html as generate_html_report,
)
from .report import (
    report_to_markdown,
)
