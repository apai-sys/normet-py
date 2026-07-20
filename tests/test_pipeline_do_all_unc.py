"""Tests for the ensemble uncertainty pipeline ``do_all_unc``.

Uses the LightGBM backend with a deliberately tiny model config so the
multi-model loop stays fast while still exercising the full assembly path
(per-seed normalise, weighting, quantile bands).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

needs_lgb = pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None, reason="lightgbm not installed"
)

FEATURES = ["t2m", "blh", "u10", "v10"]
# Minimal LightGBM tuning so each of the n_models fits in well under a second.
FAST_LGB = {"n_trials": 1, "cv_folds": 2, "nrounds": 10, "early_stopping_rounds": 5}


@needs_lgb
def test_do_all_unc_returns_bands_and_weights(synthetic_aq):
    from normet.pipeline.do_all import do_all_unc

    out, mod_stats = do_all_unc(
        df=synthetic_aq.copy(),
        target="PM2.5",
        backend="lightgbm",
        covariates=FEATURES,
        variables_resample=["t2m", "blh", "u10", "v10"],
        split_method="ts",
        train_fraction=0.8,
        model_config=FAST_LGB,
        n_samples=5,
        n_models=3,
        confidence_level=0.9,
        n_cores=1,
        verbose=False,
    )

    # One normalised_<seed> column per model, plus the summary statistics.
    pred_cols = [c for c in out.columns if c.startswith("normalised_")]
    assert len(pred_cols) == 3
    assert {"observed", "mean", "std", "median", "lower_bound", "upper_bound", "weighted"} <= set(
        out.columns
    )

    # Bands must bracket the ensemble mean and never invert.
    assert (out["lower_bound"] <= out["upper_bound"]).all()
    assert len(out) > 0

    # Weighted combination stays within the per-row span of member predictions.
    row_min = out[pred_cols].min(axis=1)
    row_max = out[pred_cols].max(axis=1)
    assert (out["weighted"] >= row_min - 1e-9).all()
    assert (out["weighted"] <= row_max + 1e-9).all()

    # Metrics carry a per-seed weight that forms a valid distribution.
    assert not mod_stats.empty
    assert "weight" in mod_stats.columns
    # modStats emits several rows per seed (subsets), so weights repeat per seed;
    # the per-seed weights form a valid distribution summing to 1.
    per_seed_weight = mod_stats.drop_duplicates("seed").set_index("seed")["weight"]
    assert np.isclose(per_seed_weight.dropna().sum(), 1.0, atol=1e-6)


@needs_lgb
def test_do_all_unc_rmse_weighting(synthetic_aq):
    from normet.pipeline.do_all import do_all_unc

    out, _ = do_all_unc(
        df=synthetic_aq.copy(),
        target="PM2.5",
        backend="lightgbm",
        covariates=FEATURES,
        split_method="ts",
        train_fraction=0.8,
        model_config=FAST_LGB,
        n_samples=5,
        n_models=2,
        weighted_method="rmse",
        n_cores=1,
    )
    assert "weighted" in out.columns
    assert np.isfinite(out["weighted"]).all()


@needs_lgb
def test_do_all_unc_rejects_bad_weighted_method(synthetic_aq):
    from normet.pipeline.do_all import do_all_unc

    with pytest.raises(ValueError, match="weighted_method"):
        do_all_unc(
            df=synthetic_aq.copy(),
            target="PM2.5",
            backend="lightgbm",
            covariates=FEATURES,
            model_config=FAST_LGB,
            n_models=2,
            weighted_method="bogus",
        )
