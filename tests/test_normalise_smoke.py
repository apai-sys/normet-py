"""End-to-end smoke tests gated by optional AutoML backends."""

import numpy as np
import pandas as pd
import pytest


def _flaml_importable() -> bool:
    """True only when FLAML and its transitive estimator imports actually load.

    Some environments (e.g. mixed conda + pip XGBoost) raise on import even
    though `flaml` is technically installed. Skip those.
    """
    try:
        from flaml.automl import AutoML  # noqa: F401

        return True
    except Exception:
        return False


needs_flaml = pytest.mark.skipif(not _flaml_importable(), reason="flaml not importable")


@needs_flaml
def test_do_all_runs_with_flaml(synthetic_aq):
    import normet as nm

    out, model, df_prep = nm.do_all(
        df=synthetic_aq.copy(),
        target="PM2.5",
        backend="flaml",
        covariates=["t2m", "blh", "u10", "v10", "date_unix", "day_julian", "weekday", "hour"],
        variables_resample=["t2m", "blh", "u10", "v10"],
        n_samples=10,
        split_method="ts",
        train_fraction=0.8,
        model_config={"time_budget": 5, "metric": "r2", "estimator_list": ["lgbm"]},
        n_cores=1,
        verbose=False,
    )

    assert set(out.columns) >= {"observed", "normalised"}
    assert len(out) > 0
    assert getattr(model, "backend", None) == "flaml"
    assert "set" in df_prep.columns


@needs_flaml
def test_normalise_quantile_bands(synthetic_aq):
    import normet as nm
    from normet.analysis.normalise import normalise
    from normet.model.train import train_model
    from normet.utils.prepare import prepare_data

    df_prep = prepare_data(
        synthetic_aq.copy(),
        target="PM2.5",
        covariates=["t2m", "blh", "u10", "v10"],
        split_method="ts",
        train_fraction=0.8,
    )
    model = train_model(
        df_prep,
        target="value",
        backend="flaml",
        covariates=["t2m", "blh", "u10", "v10"],
        model_config={"time_budget": 5, "metric": "r2", "estimator_list": ["lgbm"]},
        verbose=False,
    )
    out = normalise(
        df=df_prep,
        model=model,
        covariates=["t2m", "blh", "u10", "v10"],
        variables_resample=["t2m", "blh", "u10", "v10"],
        n_samples=20,
        n_cores=1,
        return_quantiles=(0.1, 0.5, 0.9),
        verbose=False,
    )
    assert {"observed", "normalised", "q100", "q500", "q900"} <= set(out.columns)
    # Quantile ordering at each timestamp.
    assert (out["q100"] <= out["q500"] + 1e-9).all()
    assert (out["q500"] <= out["q900"] + 1e-9).all()


@needs_flaml
def test_cv_score_walk_forward(synthetic_aq):
    import normet as nm
    from normet.utils.prepare import prepare_data

    df_prep = prepare_data(
        synthetic_aq.copy(),
        target="PM2.5",
        covariates=["t2m", "blh", "u10", "v10"],
        split_method="ts",
        train_fraction=0.8,
    )
    scores = nm.cv_score(
        df_prep,
        target="value",
        covariates=["t2m", "blh", "u10", "v10"],
        backend="flaml",
        n_splits=3,
        statistic=["RMSE", "r"],
        model_config={"time_budget": 3, "metric": "r2", "estimator_list": ["lgbm"]},
        verbose=False,
    )
    assert len(scores) == 3
    assert {"fold", "RMSE", "r", "n_train", "n_test"} <= set(scores.columns)
