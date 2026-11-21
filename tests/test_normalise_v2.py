"""Comprehensive v2 test suite for Meteorological Normalisation Layers 1–3."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd
import pytest

import normet as nm
from normet.analysis import (
    CrossFitNormaliseResult,
    NormalisationDiagnostics,
    UncDecomposition,
    ale,
    crossfit_normalise,
    diagnose_normalisation,
    normalise,
    normalise_auto,
    pdp,
    rolling,
    support_audit,
)
from normet.backends import LgbModel, backend_registry
from normet.backends.lgb_backend import train_lgb
from normet.pipeline import do_all, do_all_multisite, do_all_unc, multisite_apply
from normet.utils import (
    blocked_time_cv,
    compute_duan_smearing,
    impute_values,
    prepare_data,
)

needs_lgb = pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None, reason="lightgbm not installed"
)

FAST_LGB: dict[str, Any] = {
    "n_trials": 2,
    "cv_folds": 2,
    "nrounds": 25,
    "early_stopping_rounds": 8,
}


# ---------------------------------------------------------------------------
# Lightweight deterministic stub models & helpers
# ---------------------------------------------------------------------------


class _StubModel:
    """Fast linear/callable regression model matching registered 'flaml' backend."""

    backend: str = "flaml"

    def __init__(self, features: list[str], fn: Any) -> None:
        self.feature_names_in_ = np.array(features, dtype=object)
        self.feature_importances_ = np.ones(len(features), dtype=float)
        self._fn = fn

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._fn(X), dtype=float).reshape(-1)


class _FakeICLPipeline:
    """Torch-free stand-in implementing ICLPipeline for do_all(..., chronos_mode='icn')."""

    def forecast(
        self,
        target_history: np.ndarray,
        past: pd.DataFrame,
        future_met: pd.DataFrame,
        cols: list[str],
        target: str,
    ) -> np.ndarray:
        num_cols = [
            c
            for c in cols
            if c in past.columns and pd.api.types.is_numeric_dtype(past[c]) and c == "w"
        ]
        if not num_cols:
            num_cols = [
                c for c in cols if c in past.columns and pd.api.types.is_numeric_dtype(past[c])
            ]
        Xp = past[num_cols].to_numpy(dtype=float)
        yp = past[target].to_numpy(dtype=float)
        good = np.isfinite(yp) & np.all(np.isfinite(Xp), axis=1)
        Xp, yp = Xp[good], yp[good]
        if Xp.shape[0] == 0 or Xp.shape[1] == 0:
            return np.full(len(future_met), float(np.nanmean(yp)) if yp.size else np.nan)
        w = np.linalg.solve(Xp.T @ Xp + 1e-6 * np.eye(Xp.shape[1]), Xp.T @ yp)
        intercept = yp.mean() - w @ Xp.mean(0)
        Xf = future_met[num_cols].to_numpy(dtype=float)
        return (intercept + Xf @ w).astype(np.float64)


# ===========================================================================
# Layer 1: Log Back-Transform, Auto-Convergence, Batch Reduce, Imputation, EMS
# ===========================================================================


def test_log_transform_arithmetic_smearing_and_geometric_jensen_bias():
    """Arithmetic eliminates meteorological Jensen bias; Duan smearing fixes residual bias."""
    rng = np.random.default_rng(123)
    n = 400
    w = rng.normal(0.0, 1.0, n)
    mu = 2.0
    beta = 0.8
    sigma_eps = 0.6
    z_clean = mu + beta * w
    z_obs = z_clean + rng.normal(0.0, sigma_eps, n)
    y_orig = np.expm1(z_obs)

    df_log = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "value": z_obs,
            "w": w,
        }
    )
    model = _StubModel(["w"], lambda X: mu + beta * X["w"].to_numpy(float))

    phi = compute_duan_smearing(df_log["value"], model.predict(df_log))
    expected_phi = float(np.mean(np.exp(z_obs - z_clean)))
    assert phi == pytest.approx(expected_phi, rel=1e-6)
    assert phi > 1.05

    res_geom = normalise(
        df_log,
        model,
        covariates=["w"],
        variables_resample=["w"],
        n_samples=200,
        seed=7,
        n_cores=1,
        log_transform=True,
        back_transform="geometric",
    )
    res_arith = normalise(
        df_log,
        model,
        covariates=["w"],
        variables_resample=["w"],
        n_samples=200,
        seed=7,
        n_cores=1,
        log_transform=True,
        back_transform="arithmetic",
    )
    res_smear = normalise(
        df_log,
        model,
        covariates=["w"],
        variables_resample=["w"],
        n_samples=200,
        seed=7,
        n_cores=1,
        log_transform=True,
        back_transform="smearing",
        smearing_factor=phi,
    )

    mean_geom = float(res_geom["normalised"].mean())
    mean_arith = float(res_arith["normalised"].mean())
    mean_smear = float(res_smear["normalised"].mean())
    mean_clean = float(np.mean(np.expm1(z_clean)))
    mean_obs = float(np.mean(y_orig))

    # Strict Jensen ordering: geometric < arithmetic < smearing
    assert mean_geom < mean_arith < mean_smear
    # Arithmetic matches E[expm1(E[z|w])]; smearing matches E[y]
    assert abs(mean_arith - mean_clean) / mean_clean < 0.08
    assert abs(mean_smear - mean_obs) / mean_obs < 0.08


def test_compute_duan_smearing_handles_nans_and_validation():
    assert compute_duan_smearing([1.0, np.nan, 2.0], [1.0, 0.5, 2.0]) == pytest.approx(1.0)
    assert compute_duan_smearing([np.nan], [1.0]) == 1.0


def test_normalise_auto_datetime_index_converged_mc_se_and_log_modes():
    """normalise_auto handles DatetimeIndex, emits mc_se opt-in, and matches modes across batches."""
    rng = np.random.default_rng(11)
    n = 120
    dates = pd.date_range("2024-03-01", periods=n, freq="h")
    w = rng.normal(0.0, 1.0, n)
    z = 1.5 + 0.5 * w + 0.3 * rng.normal(size=n)
    df_idx = pd.DataFrame({"value": z, "w": w}, index=dates)
    model = _StubModel(["w"], lambda X: 1.5 + 0.5 * X["w"].to_numpy(float))

    out_default = normalise_auto(
        df_idx,
        model,
        covariates=["w"],
        variables_resample=["w"],
        convergence_tol="50%",
        batch_size=10,
        max_samples=50,
        seed=5,
        n_cores=1,
    )
    assert isinstance(out_default["converged"], bool) and out_default["converged"] is True
    assert set(out_default["res"].columns) == {"date", "observed", "normalised"}

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for bt in ("arithmetic", "smearing", "geometric"):
            extra: dict[str, Any] = {"smearing_factor": 1.15} if bt == "smearing" else {}
            r10 = normalise_auto(
                df_idx,
                model,
                covariates=["w"],
                variables_resample=["w"],
                convergence_tol="0.0001%",
                batch_size=10,
                max_samples=40,
                sampling="balanced",
                seed=9,
                n_cores=1,
                log_transform=True,
                back_transform=bt,
                return_mc_se=True,
                **extra,
            )
            r20 = normalise_auto(
                df_idx,
                model,
                covariates=["w"],
                variables_resample=["w"],
                convergence_tol="0.0001%",
                batch_size=20,
                max_samples=40,
                sampling="balanced",
                seed=9,
                n_cores=1,
                log_transform=True,
                back_transform=bt,
                return_mc_se=True,
                **extra,
            )
            assert r10["converged"] is False
            assert {"date", "observed", "normalised", "mc_se"} == set(r10["res"].columns)
            assert np.isfinite(r10["res"]["mc_se"]).all()
            assert (r10["res"]["mc_se"] > 0).all()
            np.testing.assert_allclose(
                r10["res"]["normalised"].mean(),
                r20["res"]["normalised"].mean(),
                rtol=0.05,
            )


def test_batch_reduce_parity_quantiles_mc_se_and_duplicate_dates():
    """batch_reduce with quantiles + mc_se matches vectorised path, even with duplicate dates."""
    rng = np.random.default_rng(21)
    n = 15
    dates = pd.date_range("2024-02-01", periods=n, freq="h")
    base = pd.DataFrame(
        {
            "date": dates,
            "value": rng.uniform(5.0, 20.0, n),
            "ws": rng.uniform(1.0, 10.0, n),
            "t2m": rng.normal(15.0, 4.0, n),
        }
    )
    dup = pd.concat([base, base.iloc[:5]], ignore_index=True)
    dup.loc[len(base) :, "ws"] += 0.5  # distinct met values on duplicate timestamps

    model = _StubModel(
        ["ws", "t2m"],
        lambda X: 1.2 + 0.1 * X["ws"].to_numpy(float) + 0.05 * X["t2m"].to_numpy(float),
    )

    common = dict(
        df=dup,
        model=model,
        covariates=["ws", "t2m"],
        variables_resample=["ws", "t2m"],
        n_samples=12,
        return_quantiles=[0.1, 0.5, 0.9],
        return_mc_se=True,
        log_transform=True,
        back_transform="arithmetic",
        seed=42,
        n_cores=1,
    )
    vec = normalise(**common, batch_size=0)
    batched = normalise(**common, batch_size=4)
    mem = normalise(**common, memory_save=True)

    assert len(vec) == len(batched) == len(mem) == n
    for col in ("observed", "normalised", "q100", "q500", "q900", "mc_se"):
        np.testing.assert_allclose(vec[col], batched[col], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(vec[col], mem[col], rtol=1e-10, atol=1e-10)


def test_default_variables_resample_excludes_cyclical_time_keeps_wd_sin_cos():
    """hour_sin/dow_sin/doy_cos are held fixed while ws/wd_sin/wd_cos are resampled."""
    n = 48
    dates = pd.date_range("2024-04-01", periods=n, freq="h")
    hours = np.arange(n)
    hour_sin = np.sin(2 * np.pi * (hours % 24) / 24.0)
    hour_cos = np.cos(2 * np.pi * (hours % 24) / 24.0)
    dow_sin = np.sin(2 * np.pi * dates.dayofweek.to_numpy() / 7.0)
    doy_cos = np.cos(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)
    wd_rad = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    wd_sin = np.sin(wd_rad)
    wd_cos = np.cos(wd_rad)
    ws = np.linspace(1.0, 12.0, n)

    df = pd.DataFrame(
        {
            "date": dates,
            "value": 20.0 + 10.0 * hour_sin + 5.0 * ws + 4.0 * wd_sin,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "doy_cos": doy_cos,
            "ws": ws,
            "wd_sin": wd_sin,
            "wd_cos": wd_cos,
        }
    )
    covs = ["hour_sin", "hour_cos", "dow_sin", "doy_cos", "ws", "wd_sin", "wd_cos"]
    model = _StubModel(
        covs,
        lambda X: (
            20.0
            + 10.0 * X["hour_sin"].to_numpy(float)
            + 3.0 * X["hour_cos"].to_numpy(float)
            + 2.0 * X["dow_sin"].to_numpy(float)
            + 1.0 * X["doy_cos"].to_numpy(float)
            + 5.0 * X["ws"].to_numpy(float)
            + 4.0 * X["wd_sin"].to_numpy(float)
            + 2.0 * X["wd_cos"].to_numpy(float)
        ),
    )

    out = normalise(
        df, model, covariates=covs, n_samples=n, sampling="balanced", seed=10, n_cores=1
    )
    # Cyclical time covariates stay fixed per row while ws, wd_sin, and wd_cos marginalise to pool means.
    expected = (
        20.0
        + 10.0 * hour_sin
        + 3.0 * hour_cos
        + 2.0 * dow_sin
        + 1.0 * doy_cos
        + 5.0 * float(ws.mean())
        + 4.0 * float(wd_sin.mean())
        + 2.0 * float(wd_cos.mean())
    )
    np.testing.assert_allclose(out["normalised"], expected, atol=1e-10)


def test_prepare_data_and_impute_leak_free_circular_and_embargo():
    """Verify train_median leak freedom, circular wd interp, max_gap, none, and blocked+embargo."""
    n = 40
    dates = pd.date_range("2024-05-01", periods=n, freq="h")
    ws = np.full(n, 5.0)
    ws[2] = np.nan  # in training
    ws[35] = 1_000_000.0  # huge outlier strictly in testing block

    wd = np.full(n, 90.0)
    wd[3] = 350.0
    wd[4] = np.nan
    wd[5] = 10.0  # midpoint on circle is 0/360, linear would wrongly be 180

    df = pd.DataFrame({"date": dates, "pm": np.arange(n, dtype=float) + 10.0, "ws": ws, "wd": wd})

    # 1. Blocked split + embargo marks boundary rows; ts split isolates test outlier deterministically
    prep_blocked = prepare_data(
        df,
        target="pm",
        covariates=["ws", "wd"],
        split_method="blocked",
        train_fraction=0.75,
        impute="train_median",
        embargo=2,
    )
    assert "embargo" in set(prep_blocked["set"].unique())

    prep_ts = prepare_data(
        df,
        target="pm",
        covariates=["ws", "wd"],
        split_method="ts",
        train_fraction=0.75,
        impute="train_median",
        embargo=2,
    )
    assert "embargo" in set(prep_ts["set"].unique())
    assert prep_ts.loc[2, "ws"] == pytest.approx(5.0)
    assert (prep_ts.loc[prep_ts["set"] == "training", "ws"] < 100.0).all()
    assert prep_ts.loc[35, "set"] == "testing" and prep_ts.loc[35, "ws"] > 100.0

    # 2. time_interp circular_cols + max_gap
    df_gap = df.copy()
    df_gap.loc[10:15, "ws"] = np.nan  # 6-row gap > max_gap=2 -> falls back to training median (5.0)
    df_gap["set"] = ["training"] * 30 + ["testing"] * 10
    interp = impute_values(
        df_gap,
        dropna=False,
        strategy="time_interp",
        max_gap=2,
        circular_cols=("wd",),
    )
    mid_wd = float(interp.loc[4, "wd"]) % 360.0
    assert min(mid_wd, 360.0 - mid_wd) < 1.0  # ~0/360, not 180!
    assert np.allclose(interp.loc[10:15, "ws"], 5.0)

    # 3. strategy="none" drops NA targets but preserves predictor NaN
    df_na_t = df.copy()
    df_na_t.loc[0, "pm"] = np.nan
    raw_none = prepare_data(
        df_na_t, target="pm", covariates=["ws", "wd"], split_method="ts", impute="none"
    )
    assert raw_none["value"].notna().all()
    assert raw_none["ws"].isna().any()


@needs_lgb
def test_lgb_preserves_nan_and_replaces_inf_plus_blocked_cv():
    """LgbModel._prepare_array preserves np.nan and maps +-inf -> np.nan; blocked cv trains."""
    rng = np.random.default_rng(7)
    n = 120
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "value": 10.0 + 2.0 * rng.normal(size=n),
            "ws": rng.uniform(1, 8, n),
            "t2m": rng.normal(15, 5, n),
        }
    )
    df.loc[5, "ws"] = np.nan
    model = train_lgb(
        df,
        target="value",
        covariates=["ws", "t2m"],
        model_config={**FAST_LGB, "cv_strategy": "blocked", "cv_embargo": 4},
        seed=42,
        n_cores=1,
    )
    assert isinstance(model, LgbModel)
    probe = pd.DataFrame({"ws": [np.nan, np.inf, -np.inf, 3.0], "t2m": [10.0, 12.0, 14.0, 15.0]})
    prepared = model._prepare_array(probe)
    assert np.isnan(prepared[0, 0])
    assert np.isnan(prepared[1, 0])
    assert np.isnan(prepared[2, 0])
    assert prepared[3, 0] == pytest.approx(3.0)
    preds = model.predict(probe)
    assert preds.shape == (4,) and np.isfinite(preds).all()


def test_unc_decomposition_estimator_scale_ems_and_mc_scaling():
    """Verify unbiased EMS properties, 1/J scaling, and scale='estimator' frame/summary/plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from normet.analysis.uncertainty import _decompose

    rng = np.random.default_rng(101)
    n_models = 6
    a = rng.normal(0.0, 2.0, size=n_models)
    sigma_w = 4.0

    # Purely additive: Z[i, j, t] = a_i + w_{j, t} => exact EMS components with 0 interaction.
    results = {}
    for n_j in (8, 32):
        w = rng.normal(0.0, sigma_w, size=(n_j, 20))
        z = a[:, None, None] + w[None, :, :]
        vw, vm, vi, vt = _decompose(z)
        idx = pd.date_range("2024-01-01", periods=20, freq="D")
        results[n_j] = UncDecomposition(
            var_weather=vw,
            var_model=vm,
            var_interaction=vi,
            var_total=vt,
            index=idx,
            target="PM2.5",
            n_models=n_models,
            n_weather_draws=n_j,
            seed=1,
        )
        # Model EMS component must equal unbiased sample variance var(a, ddof=1) identically
        assert np.allclose(results[n_j].var_model_est, np.var(a, ddof=1), rtol=1e-9, atol=1e-9)
        assert np.allclose(results[n_j].var_interaction_mc, 0.0, atol=1e-10)
        assert np.allclose(
            results[n_j].pct_weather_mc
            + results[n_j].pct_model_est
            + results[n_j].pct_interaction_mc,
            1.0,
        )

    # Weather estimator variance scales as 1/J when J grows 4x
    ratio_w = float(np.mean(results[8].var_weather_mc) / np.mean(results[32].var_weather_mc))
    assert 2.5 < ratio_w < 6.5

    est = results[8]
    f_est = est.frame(scale="estimator")
    assert {
        "var_weather_mc",
        "var_model_est",
        "var_interaction_mc",
        "var_estimator_total",
        "pct_weather_mc",
        "pct_model_est",
        "pct_interaction_mc",
    } <= set(f_est.columns)
    s_est = est.summary(scale="estimator")
    assert list(s_est["source"]) == ["weather_mc", "model", "interaction_mc", "total"]
    ax = est.plot(scale="estimator")
    assert ax is not None
    plt.close("all")


# ===========================================================================
# Layer 2: Paired Estimator, Blocked/Cross-Fit Normalisation, Diagnostics, ICN
# ===========================================================================


def test_paired_estimator_preserves_event_and_handles_near_zero_floor():
    """paired='additive'|'ratio'|'log_ratio' preserve short events cleanly without near-zero blowup."""
    rng = np.random.default_rng(33)
    n = 100
    w = rng.normal(0.0, 1.0, n)
    event = np.zeros(n)
    event[40:50] = 12.0  # abrupt emission jump
    obs = 20.0 + 4.0 * w + event
    obs[5] = 1.0  # paired with near-zero factual check below

    df = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=n, freq="h"), "value": obs, "w": w}
    )
    # Inject a near-zero factual point on row 5 via special covariate value
    df.loc[5, "w"] = 100.0
    model_zero = _StubModel(
        ["w"],
        lambda X: np.where(X["w"].to_numpy(float) > 90, 0.0, 20.0 + 4.0 * X["w"].to_numpy(float)),
    )
    pool = df.loc[df["w"] < 90].copy()

    res_add = normalise(
        df,
        model_zero,
        covariates=["w"],
        variables_resample=["w"],
        resample_df=pool,
        n_samples=60,
        seed=1,
        n_cores=1,
        paired="additive",
    )
    assert {"observed", "normalised", "factual", "weather_effect", "paired"} <= set(res_add.columns)
    np.testing.assert_allclose(
        res_add["weather_effect"], res_add["factual"] - res_add["normalised"]
    )
    np.testing.assert_allclose(res_add["paired"], res_add["observed"] - res_add["weather_effect"])
    # Outside row 5 where model matches DGP, event jump at 40:50 is ~ +12
    assert res_add["paired"].iloc[40:50].mean() - res_add["paired"].iloc[
        10:30
    ].mean() == pytest.approx(12.0, abs=0.5)

    res_ratio = normalise(
        df,
        model_zero,
        covariates=["w"],
        variables_resample=["w"],
        resample_df=pool,
        n_samples=60,
        seed=1,
        n_cores=1,
        paired="ratio",
        paired_floor=1e-3,
    )
    # Near-zero factual at row 5 yields NaN under ratio rather than huge explosion
    assert np.isnan(res_ratio["paired"].iloc[5])
    assert np.isfinite(res_ratio["paired"].drop(res_ratio.index[5])).all()

    res_lr = normalise(
        df,
        model_zero,
        covariates=["w"],
        variables_resample=["w"],
        resample_df=pool,
        n_samples=60,
        seed=1,
        n_cores=1,
        paired="log_ratio",
    )
    # log_ratio stays finite and non-negative even at zero factual
    assert np.isfinite(res_lr["paired"]).all()
    assert (res_lr["paired"] >= 0).all()


def test_blocked_time_cv_complete_oof_and_embargo():
    """blocked_time_cv partitions all rows once into contiguous blocks with embargo exclusion."""
    n = 53
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n, freq="h"), "x": np.arange(n)})
    emb = 4
    folds = list(blocked_time_cv(df, n_splits=5, embargo=emb))
    assert len(folds) == 5

    all_te: list[int] = []
    for tr, te in folds:
        all_te.extend(te.tolist())
        # Contiguous test block
        assert np.array_equal(te, np.arange(te[0], te[-1] + 1))
        # No train index within `emb` steps of any test index
        dists = np.min(np.abs(tr[:, None] - te[None, :]), axis=1)
        assert (dists > emb).all()

    assert sorted(all_te) == list(range(n))


@needs_lgb
def test_crossfit_normalise_and_do_all_cross_fit_recovery():
    """Out-of-fold crossfit_normalise mitigates event absorption vs in-sample tree fit."""
    rng = np.random.default_rng(77)
    n = 360
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    w = rng.normal(0.0, 1.0, n)
    df = pd.DataFrame(
        {
            "date": dates,
            "pm": 10.0 + 4.0 * w + rng.normal(0.0, 0.4, n),
            "w": w,
            "t2m": 15.0 + 3.0 * rng.normal(size=n),
        }
    )
    # Inject a short localized step event (10 hours) inside fold 2
    df.loc[150:160, "pm"] += 25.0
    # Unused helper feature correlated via overfitting shouldn't prevent recovery out-of-fold;
    # include time vars so in-sample trees can overfit the spike window while OOF cannot!
    cfg_overfit = {
        "n_trials": 2,
        "cv_folds": 2,
        "nrounds": 80,
        "early_stopping_rounds": 15,
        "num_leaves_min": 31,
        "num_leaves_max": 31,
    }
    cf = crossfit_normalise(
        df,
        target="pm",
        covariates=["w", "t2m", "date_unix", "hour"],
        variables_resample=["w", "t2m"],
        backend="lightgbm",
        model_config=cfg_overfit,
        n_splits=5,
        embargo=6,
        paired="additive",
        n_samples=25,
        seed=12,
        n_cores=1,
    )
    assert isinstance(cf, CrossFitNormaliseResult)
    assert len(cf.normalised) == n
    assert set(cf.oof_metrics["fold"].astype(str)) >= {"0", "1", "2", "3", "4", "all"}
    assert len(cf.fold_models) == 5

    # Compare do_all(..., cross_fit=True) vs plain in-sample do_all on event jump
    df_norm_cf, cf_obj, _ = do_all(
        df,
        target="pm",
        backend="lightgbm",
        covariates=["w", "t2m", "date_unix", "hour"],
        variables_resample=["w", "t2m"],
        cross_fit=True,
        cv_folds=5,
        embargo=6,
        model_config=cfg_overfit,
        n_samples=25,
        seed=12,
        n_cores=1,
    )
    assert isinstance(cf_obj, CrossFitNormaliseResult)
    bg = float(df_norm_cf["paired"].iloc[80:130].mean())
    ev = float(df_norm_cf["paired"].iloc[150:161].mean())
    assert (ev - bg) > 18.0  # most of the +25 jump recovered cleanly OOF

    # Also test crossfit log_transform=True with smearing
    cf_log = crossfit_normalise(
        df.assign(pm=np.maximum(df["pm"], 1.0)),
        target="pm",
        covariates=["w", "t2m"],
        backend="lightgbm",
        model_config=FAST_LGB,
        n_splits=3,
        embargo=4,
        log_transform=True,
        back_transform="smearing",
        n_samples=10,
        seed=3,
        n_cores=1,
    )
    assert len(cf_log.smearing_factors) == 3
    assert all(f > 0 for f in cf_log.smearing_factors)


def test_diagnostics_wrf_overcorrection_support_and_residual_audit():
    """diagnose_normalisation distinguishes good deweathering, leaky, sign-flipped, and extrapolated."""
    rng = np.random.default_rng(55)
    n = 120
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    ws = rng.normal(5.0, 1.5, n)
    t2m = rng.normal(15.0, 4.0, n)
    obs = 30.0 - 4.0 * ws + 2.0 * t2m + rng.normal(0.0, 0.2, n)
    met = pd.DataFrame(
        {"date": dates, "ws": ws, "t2m": t2m, "set": ["training"] * 90 + ["testing"] * 30}
    )

    # Case 1: Clean deweathering removes ws & t2m dependence
    df_good = pd.DataFrame(
        {
            "date": dates,
            "observed": obs,
            "normalised": 30.0 + rng.normal(0.0, 0.2, n),
            "mc_se": np.full(n, 0.1),
            "factual": obs + rng.normal(0.0, 0.1, n),
        }
    )
    diag_good = diagnose_normalisation(df_good, met, met_cols=["ws", "t2m"], nonlinear_wrf=True)
    assert isinstance(diag_good, NormalisationDiagnostics)
    assert diag_good.wrf_linear > 0.85
    assert diag_good.wrf_nonlinear is not None and diag_good.wrf_nonlinear > 0.75
    assert not diag_good.overcorrection_flags
    assert diag_good.mc_se_summary is not None and diag_good.mc_se_summary[
        "median_se"
    ] == pytest.approx(0.1)
    assert isinstance(diag_good.summary(), pd.DataFrame) and "NormalisationDiagnostics" in str(
        diag_good
    )

    # Case 2: Leaky normalisation + sign flip + overfit factual + shifted pool
    df_bad = pd.DataFrame(
        {
            "date": dates,
            "observed": obs,
            # Invert relationship on ws (over-correction) and leak t2m strongly
            "normalised": 30.0 + 3.0 * ws + 1.8 * t2m,
            "factual": np.where(np.arange(n) < 90, obs, obs + 10.0),  # test RMSE >> train RMSE
        }
    )
    pool_shifted = met.assign(ws=met["ws"] - 10.0)
    diag_bad = diagnose_normalisation(
        df_bad, met, met_cols=["ws", "t2m"], resample_df=pool_shifted, nonlinear_wrf=False
    )
    assert "ws" in diag_bad.overcorrection_flags
    assert diag_bad.support_fraction is not None and diag_bad.support_fraction < 0.9
    assert diag_bad.extrapolation_rate is not None and diag_bad.extrapolation_rate > 0.1
    assert (
        diag_bad.residual_diagnostics is not None
        and diag_bad.residual_diagnostics["overfit_ratio"] > 1.5
    )
    assert len(diag_bad.warnings) >= 3

    sup = support_audit(met, pool_shifted, ["ws", "t2m"])
    assert sup["support_fraction"] < 1.0 and sup["extrapolation_rate"] > 0.5


def test_do_all_forwards_layer1_2_3_and_chronos_icn():
    """do_all forwards stratify_by/conditional_k/paired/quantiles/sampling and chronos_mode='icn'."""
    rng = np.random.default_rng(19)
    n = 96
    dates = pd.date_range("2024-06-01", periods=n, freq="h")
    w = rng.normal(0.0, 1.0, n)
    t2m = rng.normal(20.0, 3.0, n)
    df = pd.DataFrame(
        {
            "date": dates,
            "pm25": 25.0 + 3.0 * w + 0.5 * t2m + rng.normal(0, 0.3, n),
            "w": w,
            "t2m": t2m,
        }
    )

    class _MockBackend:
        name = "mock_v2"

        def train(self, frame, **kw):
            return _StubModel(
                ["w", "t2m"],
                lambda X: 25.0 + 3.0 * X["w"].to_numpy(float) + 0.5 * X["t2m"].to_numpy(float),
            )

        def save(self, model, path=".", filename="x"):
            return str(path)

        def load(self, path=".", filename=None):
            return None

    backend_registry._backends["mock_v2"] = _MockBackend()
    try:
        out, model, df_prep = do_all(
            df,
            target="pm25",
            backend="mock_v2",
            covariates=["w", "t2m"],
            variables_resample=["w", "t2m"],
            stratify_by="hour",
            conditional_k=10,
            conditional_on={"t2m": lambda v: np.isfinite(v)},
            paired="additive",
            return_quantiles=[0.1, 0.9],
            return_mc_se=True,
            back_transform="arithmetic",
            sampling="balanced",
            block_hours=4,
            impute="train_median",
            n_samples=10,
            n_cores=1,
        )
        assert {"observed", "normalised", "paired", "q100", "q900", "mc_se"} <= set(out.columns)
    finally:
        backend_registry._backends.pop("mock_v2", None)

    # Chronos-2 ICN dispatch without torch
    n_c = 160
    df_c = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n_c, freq="h"),
            "pm": 15.0 + 4.0 * rng.normal(size=n_c),
            "w": rng.normal(size=n_c),
        }
    )
    out_c, est_c, _ = do_all(
        df_c,
        target="pm",
        backend="chronos-2",
        chronos_mode="icn",
        covariates=["w"],
        variables_resample=["w"],
        n_samples=3,
        seed=2,
        model_config={
            "pipeline": _FakeICLPipeline(),
            "context_length": 48,
            "prediction_length": 24,
            "block_h": 24,
            "stride": 24,
            "min_context": 24,
        },
    )
    assert {
        "observed",
        "normalised",
        "paired",
        "weather_effect",
        "anchor_spread",
        "mc_spread",
        "hour_contrib",
    } <= set(out_c.columns)
    assert out_c["normalised"].notna().any()
    assert est_c.context_length == 48


# ===========================================================================
# Layer 3: Balanced Sampling, Block Trajectory, Uncertainty, Rolling, Multisite, ALE
# ===========================================================================


def test_balanced_sampling_reduces_variance_and_zeroes_on_exact_cycle():
    """sampling='balanced' matches population mean, lowers MC SD, and has 0 SD on exact cycle."""
    n = 16
    w = np.linspace(0.0, 15.0, n)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "value": 10.0 + 2.0 * w,
            "w": w,
        }
    )
    model = _StubModel(["w"], lambda X: 10.0 + 2.0 * X["w"].to_numpy(float))
    pop_mean = float((10.0 + 2.0 * w).mean())

    # Exact full cycle: n_samples == len(pool) -> zero variance across seeds
    bal_full = [
        normalise(
            df,
            model,
            covariates=["w"],
            variables_resample=["w"],
            n_samples=n,
            sampling="balanced",
            seed=s,
            n_cores=1,
        )["normalised"].to_numpy()
        for s in (1, 2, 3, 4)
    ]
    for arr in bal_full:
        np.testing.assert_allclose(arr, pop_mean, atol=1e-11)

    # Half-cycle: strictly lower per-row MC variance across seeds than iid (finite-population factor)
    iid_mat = np.vstack(
        [
            normalise(
                df,
                model,
                covariates=["w"],
                variables_resample=["w"],
                n_samples=8,
                sampling="iid",
                seed=s,
                n_cores=1,
            )["normalised"].to_numpy()
            for s in range(10, 40)
        ]
    )
    bal_mat = np.vstack(
        [
            normalise(
                df,
                model,
                covariates=["w"],
                variables_resample=["w"],
                n_samples=8,
                sampling="balanced",
                seed=s,
                n_cores=1,
            )["normalised"].to_numpy()
            for s in range(10, 40)
        ]
    )
    assert np.mean(np.var(bal_mat, axis=0)) < np.mean(np.var(iid_mat, axis=0))

    # kNN conditional + balanced exact cycle when n_samples == conditional_k
    k = 6
    df_knn = df.assign(t2m=np.linspace(-1.0, 1.0, n))
    m_knn = _StubModel(["w", "t2m"], lambda X: 10.0 + 2.0 * X["w"].to_numpy(float))
    knn_a = normalise(
        df_knn,
        m_knn,
        covariates=["w", "t2m"],
        variables_resample=["w"],
        conditional_k=k,
        n_samples=k,
        sampling="balanced",
        seed=1,
        n_cores=1,
    )
    knn_b = normalise(
        df_knn,
        m_knn,
        covariates=["w", "t2m"],
        variables_resample=["w"],
        conditional_k=k,
        n_samples=k,
        sampling="balanced",
        seed=99,
        n_cores=1,
    )
    np.testing.assert_allclose(knn_a["normalised"], knn_b["normalised"], atol=1e-11)


def test_block_hours_contiguous_trajectories_and_path_parity():
    """block_hours keeps consecutive pool hours adjacent and agrees across execution paths."""
    from normet.analysis.normalise import generate_resampled

    n = 32
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    pool = pd.DataFrame(
        {
            "date": dates,
            "value": np.arange(n, dtype=float),
            "ws": np.arange(n, dtype=float),
            "wd": np.arange(n, dtype=float) * 2.0,
        }
    )
    draw = generate_resampled(
        pool, ["ws", "wd"], replace=True, seed=5, resample_df=pool, block_hours=4
    )
    # Within each 4-hour block, consecutive resampled ws steps increment by +1 mod n
    for b in range(0, n, 4):
        ws_blk = draw["ws"].iloc[b : b + 4].to_numpy(int)
        expected = (ws_blk[0] + np.arange(4)) % n
        np.testing.assert_array_equal(ws_blk, expected)

    model = _StubModel(
        ["ws", "wd"], lambda X: 5.0 + X["ws"].to_numpy(float) + 0.5 * X["wd"].to_numpy(float)
    )
    kw = dict(
        df=pool,
        model=model,
        covariates=["ws", "wd"],
        variables_resample=["ws", "wd"],
        block_hours=4,
        n_samples=8,
        seed=3,
        n_cores=1,
    )
    vec = normalise(**kw, batch_size=0)
    bat = normalise(**kw, batch_size=3)
    mem = normalise(**kw, memory_save=True)
    np.testing.assert_allclose(vec["normalised"], bat["normalised"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(vec["normalised"], mem["normalised"], rtol=1e-10, atol=1e-10)


@needs_lgb
def test_do_all_unc_v2_methods(synthetic_aq):
    """do_all_unc supports block_bootstrap, crossfit_ensemble, student_t CI, inv_mse & uniform."""
    df = synthetic_aq.iloc[:150].copy()
    feats = ["t2m", "blh", "u10", "v10"]

    out_bb, stats_bb = do_all_unc(
        df,
        target="PM2.5",
        backend="lightgbm",
        covariates=feats,
        model_config=FAST_LGB,
        unc_method="block_bootstrap",
        ci_method="student_t",
        weighted_method="inv_mse",
        block_length=24,
        n_models=3,
        n_samples=6,
        confidence_level=0.9,
        seed=7,
        n_cores=1,
    )
    assert {"mean", "std", "lower_bound", "upper_bound", "weighted"} <= set(out_bb.columns)
    assert (out_bb["lower_bound"] <= out_bb["upper_bound"]).all()
    assert not stats_bb.empty and "weight" in stats_bb.columns

    out_cf, stats_cf = do_all_unc(
        df,
        target="PM2.5",
        backend="lightgbm",
        covariates=feats,
        model_config=FAST_LGB,
        unc_method="crossfit_ensemble",
        weighted_method="uniform",
        cv_folds=3,
        embargo=4,
        n_models=2,
        n_samples=6,
        seed=8,
        n_cores=1,
    )
    assert (out_cf["lower_bound"] <= out_cf["upper_bound"]).all()
    weights = stats_cf.drop_duplicates("seed")["weight"].to_numpy()
    np.testing.assert_allclose(weights, 0.5, atol=1e-6)


@needs_lgb
def test_rolling_v2_summary_log_stratified_paired_and_refit(synthetic_aq):
    """rolling supports include_summary=True, log_transform, stratify_by, paired, and refit_per_window."""
    df = synthetic_aq.iloc[: 24 * 12].copy()
    df["PM2.5"] = np.maximum(df["PM2.5"], 1.0)
    feats = ["t2m", "blh", "u10", "v10"]
    out = rolling(
        df,
        target="PM2.5",
        backend="lightgbm",
        covariates=feats,
        model_config=FAST_LGB,
        window_days=5,
        rolling_every=3,
        n_samples=6,
        log_transform=True,
        stratify_by="hour",
        paired="additive",
        include_summary=True,
        refit_per_window=True,
        seed=4,
        n_cores=1,
    )
    assert {"observed", "rolling_mean", "rolling_std", "n_windows"} <= set(out.columns)
    assert (out["n_windows"] >= 1).any()
    assert out["rolling_mean"].notna().any()


def test_multisite_pooled_pool_and_error_handling():
    """do_all_multisite supports reference_pool='pooled', attrs['failed_sites'], and on_error='raise'."""
    rng = np.random.default_rng(14)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    s1 = pd.DataFrame(
        {
            "date": dates,
            "pm": 20.0 + 2.0 * rng.uniform(0, 5, n),
            "w": np.full(n, 0.0) + rng.uniform(0, 2, n),
            "site": "cold",
        }
    )
    s2 = pd.DataFrame(
        {
            "date": dates,
            "pm": 40.0 + 2.0 * rng.uniform(8, 12, n),
            "w": rng.uniform(10, 12, n),
            "site": "warm",
        }
    )
    df = pd.concat([s1, s2], ignore_index=True)

    class _SiteSlope:
        name = "mock_ms"

        def train(self, frame, **kw):
            return _StubModel(["w"], lambda X: 10.0 + 2.0 * X["w"].to_numpy(float))

        def save(self, model, path=".", filename="x"):
            return str(path)

        def load(self, path=".", filename=None):
            return None

    backend_registry._backends["mock_ms"] = _SiteSlope()
    try:
        res_local = do_all_multisite(
            df,
            "site",
            "pm",
            backend="mock_ms",
            covariates=["w"],
            variables_resample=["w"],
            n_samples=40,
            seed=1,
            n_cores=1,
        )
        res_pooled = do_all_multisite(
            df,
            "site",
            "pm",
            backend="mock_ms",
            covariates=["w"],
            variables_resample=["w"],
            n_samples=40,
            seed=1,
            n_cores=1,
            reference_pool="pooled",
        )
        # Under pooled reference meteorology both sites share the same weather range (~0..12 -> mean w ~6),
        # so normalised predictions match (~22), whereas local keeps cold low (~12) and warm high (~32)!
        c_local = res_local.loc[res_local["site"] == "cold", "normalised"].mean()
        w_local = res_local.loc[res_local["site"] == "warm", "normalised"].mean()
        c_pool = res_pooled.loc[res_pooled["site"] == "cold", "normalised"].mean()
        w_pool = res_pooled.loc[res_pooled["site"] == "warm", "normalised"].mean()
        assert abs(w_local - c_local) > 15.0
        assert abs(w_pool - c_pool) < 5.0
        assert res_pooled.attrs["failed_sites"] == {}
    finally:
        backend_registry._backends.pop("mock_ms", None)

    # Error modes on multisite_apply
    def _sometimes_fail(df: pd.DataFrame) -> pd.DataFrame:
        if (df["w"] > 5).any():
            raise ValueError("site explosion")
        return pd.DataFrame({"mean_pm": [df["pm"].mean()]})

    res_warn = multisite_apply(df, "site", _sometimes_fail, n_cores=1, on_error="warn")
    assert set(res_warn["site"]) == {"cold"}
    assert "warm" in res_warn.attrs["failed_sites"]

    with pytest.raises(ValueError, match="site explosion"):
        multisite_apply(df, "site", _sometimes_fail, n_cores=1, on_error="raise")


def test_ale_correlated_additive_recovery_and_pdp_log_transform():
    """ale recovers exact linear slopes under 0.99 correlation with zero weighted mean; pdp handles log."""
    rng = np.random.default_rng(66)
    n = 600
    x1 = rng.normal(10.0, 3.0, n)
    # 0.99+ correlated predictor x2
    x2 = x1 + rng.normal(0.0, 0.25, n)
    assert np.corrcoef(x1, x2)[0, 1] > 0.99

    df = pd.DataFrame({"x1": x1, "x2": x2})
    model = _StubModel(
        ["x1", "x2"],
        lambda X: 5.0 + 3.0 * X["x1"].to_numpy(float) - 1.5 * X["x2"].to_numpy(float),
    )

    ale_df = ale(
        df, model, var_list=["x1", "x2"], n_bins=12, quantile_range=(0.02, 0.98), n_cores=1
    )
    assert list(ale_df.columns) == ["variable", "value", "ale", "bin_low", "bin_high", "n_bin"]

    for var, expected_slope in (("x1", 3.0), ("x2", -1.5)):
        sub = ale_df[ale_df["variable"] == var]
        # Bin-count weighted mean == 0
        w_mean = float(np.sum(sub["n_bin"] * sub["ale"]) / np.sum(sub["n_bin"]))
        assert w_mean == pytest.approx(0.0, abs=1e-10)
        # Slope matches exact partial derivative even on 0.99-correlated predictors
        slope = float(np.polyfit(sub["value"], sub["ale"], 1)[0])
        assert slope == pytest.approx(expected_slope, rel=1e-4, abs=1e-4)

    # pdp with log_transform modes
    log_df = pd.DataFrame({"x1": np.linspace(0.0, 2.0, 20), "x2": rng.normal(0.0, 0.5, 20)})
    log_model = _StubModel(
        ["x1", "x2"],
        lambda X: 1.0 + 0.5 * X["x1"].to_numpy(float) + 0.4 * X["x2"].to_numpy(float),
    )
    pdp_arith = pdp(
        log_df,
        log_model,
        var_list=["x1"],
        grid_points=5,
        log_transform=True,
        back_transform="arithmetic",
        n_cores=1,
    )
    pdp_smear = pdp(
        log_df,
        log_model,
        var_list=["x1"],
        grid_points=5,
        log_transform=True,
        back_transform="smearing",
        smearing_factor=1.2,
        n_cores=1,
    )
    pdp_geom = pdp(
        log_df,
        log_model,
        var_list=["x1"],
        grid_points=5,
        log_transform=True,
        back_transform="geometric",
        n_cores=1,
    )
    assert (pdp_geom["pdp_mean"] < pdp_arith["pdp_mean"]).all()
    assert (pdp_arith["pdp_mean"] < pdp_smear["pdp_mean"]).all()
    # Geometric point equals expm1(mean_z) and delta-method SD is positive
    assert (pdp_geom["pdp_std"] > 0).all()


def test_v2_top_level_exports_available():
    for name in (
        "blocked_time_cv",
        "compute_duan_smearing",
        "CrossFitNormaliseResult",
        "crossfit_normalise",
        "NormalisationDiagnostics",
        "diagnose_normalisation",
        "support_audit",
        "ale",
    ):
        assert hasattr(nm, name), f"Missing export: {name}"
