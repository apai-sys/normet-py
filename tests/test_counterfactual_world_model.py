"""Tests for counterfactual world-model upgrades grounded in arXiv:2611.24778v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import normet as nm
from normet.foundation import (
    Chronos2Estimator,
    ICNDesign,
    ICNDrawBank,
    build_icn_draw_bank,
    debias_icn_residuals,
    normalise_in_context,
)


class _RidgeICLPipeline:
    """Torch-free ridge ICL stand-in."""

    def forecast(
        self,
        target_history: np.ndarray,
        past: pd.DataFrame,
        future_met: pd.DataFrame,
        cols: list[str],
        target: str,
    ) -> np.ndarray:
        met = [c for c in cols if c in past.columns and pd.api.types.is_numeric_dtype(past[c])]
        Xp = past[met].to_numpy(dtype=float)
        yp = np.asarray(target_history, dtype=float)
        good = np.isfinite(yp) & np.all(np.isfinite(Xp), axis=1)
        Xp, yp = Xp[good], yp[good]
        if Xp.shape[0] == 0 or Xp.shape[1] == 0:
            return np.full(len(future_met), float(np.nanmean(yp)) if yp.size else np.nan)
        XtX = Xp.T @ Xp + 1e-4 * np.eye(Xp.shape[1])
        w = np.linalg.solve(XtX, Xp.T @ yp)
        intercept = float(yp.mean() - w @ Xp.mean(axis=0))
        Xf = future_met[met].to_numpy(dtype=float)
        return (intercept + Xf @ w).astype(np.float64)


def test_counterfactual_separability_audit_and_diagnose_normalisation() -> None:
    rng = np.random.default_rng(101)
    n = 360
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    w = rng.normal(0.0, 1.0, n)
    X = w[:, None]

    # 1. Pure additive homoscedastic DGP: low rho_lb and low RC-score
    y_add = 40.0 + 6.0 * w + rng.normal(0.0, 1.0, n)
    y_fact_add = 40.0 + 6.0 * w
    audit_add = nm.counterfactual_separability_audit(y_add, y_fact_add, X, nonlinear=True, seed=1)
    assert audit_add["interaction_rho_lb_raw"] < 0.05
    assert audit_add["rc_score_linear"] < 0.05
    assert audit_add["floor_saturation_rate"] == 0.0

    # 2. Multiplicative heteroscedastic DGP: raw rho_lb high, log1p rho_lb low
    eps = np.exp(rng.normal(0.0, 0.35, n))
    y_mult = np.exp(3.2 + 0.9 * w) * eps
    # Factual conditional mean under lognormal noise
    y_fact_mult = np.exp(3.2 + 0.9 * w + 0.5 * 0.35**2)
    audit_mult = nm.counterfactual_separability_audit(
        y_mult, y_fact_mult, X, nonlinear=True, seed=2
    )
    assert audit_mult["interaction_rho_lb_raw"] > 0.15
    assert audit_mult["interaction_rho_lb_log"] < 0.05
    assert audit_mult["recommended_coordinate"] == "log1p"

    met_df = pd.DataFrame({"date": dates, "w": w})
    norm_df = pd.DataFrame(
        {
            "date": dates,
            "observed": y_mult,
            "normalised": float(np.mean(y_mult)) + rng.normal(0.0, 0.2, n),
            "factual": y_fact_mult,
        }
    )
    diag_mult = nm.diagnose_normalisation(norm_df, met_df, met_cols=["w"], nonlinear_wrf=True)
    assert diag_mult.counterfactual_diagnostics is not None
    assert "cf_recommended_coordinate" in set(diag_mult.summary()["metric"])
    assert any("log_transform=True" in msg for msg in diag_mult.warnings)

    # 3. Misspecified factual predictor (wrong weather slope): RC-score > 0.25
    y_fact_bad = 40.0 + 1.5 * w
    audit_bad = nm.counterfactual_separability_audit(y_add, y_fact_bad, X, nonlinear=False, seed=3)
    assert audit_bad["rc_score_linear"] > 0.25

    # 4. Floor-clamped DGP (>20% zeros): floor_saturation_rate > 0.15 and warning emitted
    y_floor = np.maximum(0.0, 2.0 + 4.0 * w + rng.normal(0.0, 0.5, n))
    y_fact_floor = np.maximum(0.0, 2.0 + 4.0 * w)
    audit_floor = nm.counterfactual_separability_audit(
        y_floor, y_fact_floor, X, nonlinear=False, seed=4
    )
    assert audit_floor["floor_saturation_rate"] > 0.15
    norm_floor = pd.DataFrame(
        {
            "date": dates,
            "observed": y_floor,
            "normalised": np.full(n, 2.0) + rng.normal(0.0, 0.1, n),
            "factual": y_fact_floor,
        }
    )
    diag_floor = nm.diagnose_normalisation(norm_floor, met_df, met_cols=["w"], nonlinear_wrf=False)
    assert any("saturation" in msg.lower() for msg in diag_floor.warnings)


def test_icn_quantile_mean_reduces_skewness_bias() -> None:
    """When skewness varies with meteorology, quantile_mean beats median on conditional-mean bias."""
    from scipy.stats import norm as normal_dist

    rng = np.random.default_rng(17)
    n = 900
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    w = rng.normal(0.0, 1.0, n)
    # Right-skewed lognormal-like response whose scale grows strongly with w:
    # median = exp(mu(w)), while mean = exp(mu(w) + 0.5 * sigma(w)^2).
    mu_w = 2.0 + 0.3 * w
    sig_w = np.clip(0.2 + 0.5 * np.maximum(w, 0.0), 0.2, 1.2)
    z = rng.normal(0.0, 1.0, n)
    y = np.exp(mu_w + sig_w * z)
    df = pd.DataFrame({"y": y, "w": w}, index=idx)

    class _ExactSkewQuantilePipe:
        def forecast(self, target_history, past, future_met, cols, target):
            wf = future_met["w"].to_numpy(dtype=float)
            return np.exp(2.0 + 0.3 * wf)

        def forecast_quantiles_batch(self, items, cols, target, quantiles):
            out = []
            for _h, _p, fut in items:
                wf = fut["w"].to_numpy(dtype=float)
                mu_f = 2.0 + 0.3 * wf
                sig_f = np.clip(0.2 + 0.5 * np.maximum(wf, 0.0), 0.2, 1.2)
                q_mat = np.stack(
                    [np.exp(mu_f + sig_f * float(normal_dist.ppf(q))) for q in quantiles], axis=0
                )
                out.append(q_mat)
            return np.stack(out, axis=0)

    est = Chronos2Estimator(met_covariates=["w"])
    base_kw: dict[str, Any] = dict(
        context_length=200,
        block_h=100,
        stride=100,
        direction="forward",
        n_resamples=8,
        seed=5,
    )
    res_med = normalise_in_context(
        est,
        df,
        "y",
        design=ICNDesign(**base_kw, point_estimate="median"),
        pipeline=_ExactSkewQuantilePipe(),
    )
    res_mean = normalise_in_context(
        est,
        df,
        "y",
        design=ICNDesign(**base_kw, point_estimate="quantile_mean"),
        pipeline=_ExactSkewQuantilePipe(),
    )
    # True E[Y | w] integrated over the empirical pool
    true_pop_mean = float(np.mean(np.exp(mu_w + 0.5 * sig_w**2)))
    fin = np.isfinite(res_med.level.to_numpy())
    bias_med = abs(float(np.mean(res_med.level.to_numpy()[fin])) - true_pop_mean)
    bias_mean = abs(float(np.mean(res_mean.level.to_numpy()[fin])) - true_pop_mean)
    assert bias_mean < bias_med


def test_icn_rank_transport_preserves_extreme_shocks_and_heteroscedasticity() -> None:
    """paired_mode='rank' handles multiplicative + extreme shocks without clipping or zero blow-up."""
    from scipy.stats import norm as normal_dist

    rng = np.random.default_rng(29)
    n = 800
    idx = pd.date_range("2023-03-01", periods=n, freq="h")
    w = rng.normal(0.0, 1.0, n)
    # Emission trajectory with a sudden +20 extreme shock at [500:540]
    e_true = np.full(n, 5.0)
    e_true[500:540] = 25.0
    # Weather-dependent dispersion scale (multiplicative + near-zero floor when w < -1)
    scale_w = np.exp(0.7 * w)
    y = np.maximum(0.0, e_true * scale_w + 0.2 * scale_w * rng.normal(0.0, 0.2, n))
    df = pd.DataFrame({"y": y, "w": w}, index=idx)

    # True counterfactual emission truth under reference scale_w=1 (or mean(scale_w))
    class _MultQuantilePipe:
        def forecast(self, target_history, past, future_met, cols, target):
            wf = future_met["w"].to_numpy(dtype=float)
            return 5.0 * np.exp(0.7 * wf)

        def forecast_quantiles_batch(self, items, cols, target, quantiles):
            out = []
            for _h, _p, fut in items:
                wf = fut["w"].to_numpy(dtype=float)
                loc = 5.0 * np.exp(0.7 * wf)
                sd = 0.2 * np.exp(0.7 * wf)
                out.append(
                    np.stack(
                        [np.maximum(0.0, loc + sd * float(normal_dist.ppf(q))) for q in quantiles],
                        axis=0,
                    )
                )
            return np.stack(out, axis=0)

    est = Chronos2Estimator(met_covariates=["w"])
    common: dict[str, Any] = dict(
        context_length=200,
        block_h=80,
        stride=80,
        direction="forward",
        n_resamples=12,
        seed=7,
    )
    pipe = _MultQuantilePipe()
    res_add = normalise_in_context(
        est, df, "y", design=ICNDesign(**common, paired_mode="additive"), pipeline=pipe
    )
    res_rank = normalise_in_context(
        est,
        df,
        "y",
        design=ICNDesign(**common, paired_mode="rank"),
        pipeline=pipe,
        quantiles=(0.1, 0.5, 0.9),
    )

    assert (res_rank.paired.dropna() >= 0.0).all()
    assert res_rank.paired_quantiles is not None
    # Extreme shock [500:540] (well above baseline q95) is NOT clipped at q95
    shock_rank = float(res_rank.paired.iloc[505:535].mean())
    base_rank = float(res_rank.paired.iloc[250:450].mean())
    assert shock_rank > base_rank + 10.0

    # Under multiplicative scaling, rank transport beats naive additive residual reuse
    ref_mean = float(np.mean(np.exp(0.7 * w)))
    truth_cf = e_true * ref_mean
    valid = np.isfinite(res_rank.paired.to_numpy()) & np.isfinite(res_add.paired.to_numpy())
    rmse_rank = float(np.sqrt(np.mean((res_rank.paired.to_numpy()[valid] - truth_cf[valid]) ** 2)))
    rmse_add = float(np.sqrt(np.mean((res_add.paired.to_numpy()[valid] - truth_cf[valid]) ** 2)))
    assert rmse_rank < rmse_add

    # Also verify additive and log_ratio modes under both log_transform settings
    for lt in (False, True):
        for pm in ("additive", "log_ratio"):
            res_m = normalise_in_context(
                est,
                df,
                "y",
                design=ICNDesign(**common, log_transform=lt, paired_mode=pm),
                pipeline=pipe,
            )
            assert np.isfinite(res_m.paired.dropna()).all()

    # Pipeline without forecast_quantiles_batch raises TypeError for rank or quantile_mean
    with pytest.raises(TypeError, match="forecast_quantiles_batch"):
        normalise_in_context(
            est,
            df,
            "y",
            design=ICNDesign(**common, paired_mode="rank"),
            pipeline=_RidgeICLPipeline(),
        )


def test_icn_past_only_covariates_forwarding_and_isolation() -> None:
    rng = np.random.default_rng(42)
    n = 600
    idx = pd.date_range("2023-05-01", periods=n, freq="h")
    w = rng.normal(0.0, 1.0, n)
    regime = rng.choice([-1.0, 1.0], size=n)
    y = 20.0 + 3.0 * w + 2.0 * regime + 0.1 * rng.normal(0.0, 1.0, n)
    df = pd.DataFrame({"y": y, "w": w, "precursor": regime}, index=idx)

    seen_calls: list[tuple[list[str], list[str], list[str]]] = []

    class _InspectPipe(_RidgeICLPipeline):
        def forecast(self, target_history, past, future_met, cols, target):
            seen_calls.append((list(past.columns), list(future_met.columns), list(cols)))
            return super().forecast(target_history, past, future_met, cols, target)

    est = Chronos2Estimator()
    design = ICNDesign(
        context_length=200,
        block_h=80,
        stride=80,
        direction="forward",
        n_resamples=2,
        seed=1,
    )
    res = normalise_in_context(
        est,
        df,
        "y",
        design=design,
        pipeline=_InspectPipe(),
        past_only_covariates=["precursor"],
    )
    assert seen_calls
    for past_cols, fut_cols, passed_cols in seen_calls:
        assert "precursor" in past_cols
        assert "precursor" not in fut_cols
        assert "precursor" not in passed_cols
        assert passed_cols == ["w"]
    assert res.diagnostics["past_only_covariates"] == ["precursor"]

    # Also verify Chronos2ICLPipeline forwards past_only_cols to estimator._forecast_blocks
    from normet.foundation.in_context import Chronos2ICLPipeline

    recorded: dict[str, Any] = {}

    class _StubEst:
        def _resolve_covariates(self, frame, tgt, exclude=()):
            return [c for c in frame.columns if c != tgt and c not in set(exclude)]

        def _q_index(self, q):
            return 0

        def _forecast_blocks(self, items, cols, horizon, past_only_cols=(), cross_learning=False):
            recorded["cols"] = list(cols)
            recorded["past_only_cols"] = tuple(past_only_cols)
            return np.zeros((len(items), 1, horizon), dtype=float)

    pipe_c = Chronos2ICLPipeline(_StubEst(), past_only_cols=["precursor"])
    pipe_c.forecast(
        np.ones(20),
        df.iloc[:20],
        df.iloc[20:30],
        ["w"],
        "y",
    )
    assert recorded["cols"] == ["w"]
    assert recorded["past_only_cols"] == ("precursor",)


def test_icn_factual_debias_and_draw_bank_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(71)
    n = 700
    idx = pd.date_range("2023-06-01", periods=n, freq="h")
    w = rng.normal(0.0, 1.0, n)
    true_level = 30.0
    y = true_level + 6.0 * w + rng.normal(0.0, 0.3, n)
    df = pd.DataFrame({"y": y, "w": w}, index=idx)

    # Biased predictor that captures only half the true weather slope (3*w instead of 6*w)
    class _HalfSlopePipe:
        def forecast(self, target_history, past, future_met, cols, target):
            return 30.0 + 3.0 * future_met["w"].to_numpy(dtype=float)

        def forecast_quantiles_batch(self, items, cols, target, quantiles):
            out = []
            for _h, _p, fut in items:
                med = 30.0 + 3.0 * fut["w"].to_numpy(dtype=float)
                out.append(np.stack([med + 1.0 * (float(q) - 0.5) for q in quantiles], axis=0))
            return np.stack(out, axis=0)

    est = Chronos2Estimator(met_covariates=["w"])
    design = ICNDesign(
        context_length=200,
        block_h=80,
        stride=80,
        direction="forward",
        n_resamples=64,
        point_estimate="quantile_mean",
        paired_mode="additive",
        seed=9,
    )
    bank = build_icn_draw_bank(est, df, "y", design=design, pipeline=_HalfSlopePipe())
    res_biased = bank.aggregate()
    assert res_biased.factual is not None
    assert res_biased.factual.notna().any()

    debiased = debias_icn_residuals(res_biased, df[["w"]], met_cols=["w"], method="ridge", seed=9)
    info = debiased.diagnostics["residual_debias"]
    assert info["applied"] is True
    assert info["rc_score_before"] > 0.5
    assert info["rc_score_after"] < 0.05

    valid = np.isfinite(res_biased.paired.to_numpy())
    rmse_before = float(np.sqrt(np.mean((res_biased.paired.to_numpy()[valid] - true_level) ** 2)))
    rmse_after = float(np.sqrt(np.mean((debiased.paired.to_numpy()[valid] - true_level) ** 2)))
    assert rmse_after < 0.3 * rmse_before

    # When already unbiased, min_rc_score gates out debiasing
    res_unbiased = normalise_in_context(
        est,
        df,
        "y",
        design=ICNDesign(
            context_length=200, block_h=80, stride=80, direction="forward", n_resamples=8, seed=9
        ),
        pipeline=_RidgeICLPipeline(),
    )
    gated = debias_icn_residuals(res_unbiased, df[["w"]], met_cols=["w"], min_rc_score=0.10)
    assert gated.diagnostics["residual_debias"]["applied"] is False

    # DrawBank save/load round-trips factual, paired_mode, and point_estimate bit-identically
    save_path = tmp_path / "bank_v2.json.gz"
    bank.save(save_path)
    loaded = ICNDrawBank.load(save_path)
    assert loaded.design.paired_mode == "additive"
    assert loaded.design.point_estimate == "quantile_mean"
    res_loaded = loaded.aggregate()
    pd.testing.assert_series_equal(res_biased.level, res_loaded.level)
    pd.testing.assert_series_equal(res_biased.paired, res_loaded.paired)
    assert res_loaded.factual is not None
    pd.testing.assert_series_equal(res_biased.factual, res_loaded.factual)


def test_normalise_ox_invariant_method_icn_f3_conservation() -> None:
    pytest.importorskip("lightgbm")
    bm = nm.make_synthetic_benchmark(
        "F3_titration",
        n_days=75,
        event_start_day=35,
        event_duration_days=12,
        event_reduction_pct=-45.0,
        trend_pct_per_year=-6.0,
        seed=19,
    )
    design = ICNDesign(
        context_length=240,
        min_context=120,
        block_h=96,
        stride=96,
        direction="bidirectional",
        n_resamples=3,
        seed=19,
    )
    ox_res = nm.normalise_ox_invariant(
        bm.df,
        o3_col="O3",
        no2_col="NO2",
        no_col="NO",
        covariates=bm.covariates,
        variables_resample=bm.met_covariates,
        unit="ug_m3",
        method="icn",
        backend="lightgbm",
        model_config={
            "n_trials": 2,
            "cv_folds": 2,
            "nrounds": 80,
            "early_stopping_rounds": 15,
            "deterministic": True,
        },
        n_samples=30,
        seed=19,
        n_cores=1,
        icn_design=design,
        icn_pipeline=_RidgeICLPipeline(),
    )
    assert isinstance(ox_res, nm.ConservedOxResult)
    assert ox_res.conservation_max_err < 1e-10
    sp = ox_res.species
    assert len(sp) > 500
    assert (sp["O3_normalised"] >= 0.0).all()
    assert (sp["NO2_normalised"] >= 0.0).all()
    assert (sp["NO_normalised"] >= 0.0).all()

    ev_m = bm.truth["event_mask"].reindex(sp.index).to_numpy(dtype=bool)
    assert float(ox_res.titration_reallocation.loc[ev_m].mean()) > 0.0
