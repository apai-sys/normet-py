"""Tests for the ensemble uncertainty pipeline ``do_all_unc``.

Uses the LightGBM backend with a deliberately tiny model config so the
multi-model loop stays fast while still exercising the full assembly path
(per-seed normalise, weighting, quantile bands).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# log_transform: fit on log1p(value), report in the original units
# ---------------------------------------------------------------------------


def _multiplicative_aq(rng: np.random.Generator) -> pd.DataFrame:
    """A target with *multiplicative* noise spanning roughly 1 to 15.

    The level of the noise grows with the level of the signal, which is the
    regime a log-scale fit exists for: in log1p space the noise is roughly
    homoscedastic, so a log fit should beat a linear one on logRMSE.
    """
    n = 720
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    t2m = 10 + 8 * np.sin(2 * np.pi * np.arange(n) / (24 * 30)) + rng.normal(0, 1, n)
    blh = 800 + 400 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 50, n)
    log_pm = 1.5 + 0.004 * (t2m - 10) + 0.0004 * (blh - 800) + rng.normal(0, 0.3, n)
    pm = np.expm1(log_pm) * np.exp(rng.normal(0, 0.5, n))
    return pd.DataFrame({"date": dates, "PM2.5": pm, "t2m": t2m, "blh": blh})


LOG_COMMON = dict(
    backend="lightgbm",
    covariates=["t2m", "blh"],
    split_method="ts",
    train_fraction=0.8,
    model_config=FAST_LGB,
    n_samples=5,
    n_cores=1,
)


@needs_lgb
def test_do_all_log_transform_original_scale_and_log_space_prep():
    from normet.pipeline.do_all import do_all

    df = _multiplicative_aq(np.random.default_rng(7))
    out_lin, _, dfp_lin = do_all(df, target="PM2.5", **LOG_COMMON)
    out_log, _, dfp_log = do_all(df, target="PM2.5", log_transform=True, **LOG_COMMON)

    # The prepared frame is the frame the model trained on: log1p of the
    # linear one, same rows, same split.
    assert np.allclose(dfp_log["value"], np.log1p(dfp_lin["value"]))
    assert (dfp_log["set"] == dfp_lin["set"]).all()

    # The result is in the original units: observed is exactly the input
    # target, and the back-transform keeps normalised non-negative.
    assert np.allclose(out_log["observed"], df.set_index("date")["PM2.5"].reindex(out_log.index))
    assert (out_log["normalised"] >= 0).all()


@needs_lgb
def test_do_all_log_transform_default_path_bit_identical():
    """An explicit log_transform=False must equal the legacy call exactly."""
    from normet.pipeline.do_all import do_all

    df = _multiplicative_aq(np.random.default_rng(7))
    out_default, _, dfp_default = do_all(df, target="PM2.5", **LOG_COMMON)
    out_explicit, _, dfp_explicit = do_all(df, target="PM2.5", log_transform=False, **LOG_COMMON)
    assert out_default.equals(out_explicit)
    assert dfp_default.equals(dfp_explicit)


@needs_lgb
def test_do_all_log_transform_log_fit_wins_on_logrmse():
    from normet.pipeline.do_all import do_all
    from normet.utils.metrics import Stats

    df = _multiplicative_aq(np.random.default_rng(7))
    out_lin, _, _ = do_all(df, target="PM2.5", **LOG_COMMON)
    out_log, _, _ = do_all(df, target="PM2.5", log_transform=True, **LOG_COMMON)
    s_lin = Stats(out_lin, mod="normalised", obs="observed", statistic=["logRMSE"])
    s_log = Stats(out_log, mod="normalised", obs="observed", statistic=["logRMSE"])
    assert s_log.loc[0, "logRMSE"] < s_lin.loc[0, "logRMSE"]


@needs_lgb
def test_do_all_unc_log_transform_original_scale():
    from normet.pipeline.do_all import do_all_unc

    df = _multiplicative_aq(np.random.default_rng(7))
    out, mod_stats = do_all_unc(
        df=df,
        target="PM2.5",
        n_models=3,
        log_transform=True,
        **LOG_COMMON,
    )
    assert np.allclose(out["observed"], df.set_index("date")["PM2.5"].reindex(out.index))
    for col in ("mean", "median", "lower_bound", "upper_bound", "weighted"):
        assert (out[col] >= 0).all()
    assert not mod_stats.empty
