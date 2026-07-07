import normet as nm


def test_top_level_symbols_present():
    must_have = {
        "do_all",
        "do_all_unc",
        "run_workflow",
        "normalise",
        "rolling",
        "pdp",
        "decompose",
        "scm",
        "mlscm",
        "run_scm",
        "placebo_in_space",
        "placebo_in_time",
        "uncertainty_bands",
        "effect_bands_space",
        "effect_bands_time",
        "build_model",
        "train_model",
        "ml_predict",
        "load_model",
        "save_model",
        "modStats",
        "prepare_data",
        "process_date",
        "add_lag_features",
        "add_rolling_features",
        "cyclical_encode",
        "wind_to_uv",
        "time_series_cv",
        "cv_score",
    }
    missing = must_have - set(nm.__all__)
    assert not missing, f"Missing top-level exports: {sorted(missing)}"
