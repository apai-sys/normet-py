"""Decomposition estimators.

Meteorology and emission attribution, LMDI closure, Shapley/TreeSHAP
interactions and cross-fit stitching.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

import normet.analysis.decomposition
from normet.analysis import (
    compare_attributions,
    cross_fit_decompose,
    decom_emi,
    decom_emi_shapley,
    decom_met,
    decom_met_treeshap,
    decompose,
    normalise,
    shapley_interactions,
    suggest_met_groups,
)
from normet.core.exceptions import ConfigError

dmod = sys.modules["normet.analysis.decomposition"]
needs_lgb = pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None, reason="lightgbm not installed"
)


class _CountingLinearModel:
    backend = "flaml"

    def __init__(self, features: list[str], weights: dict[str, float], bias: float = 10.0) -> None:
        self.feature_names_in_ = np.array(features)
        self.feature_importances_ = np.asarray(
            [abs(weights.get(f, 0.1)) for f in features], dtype=float
        )
        self.weights = dict(weights)
        self.bias = float(bias)
        self.predict_calls = 0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.predict_calls += 1
        out = np.full(len(X), self.bias, dtype=float)
        for col, w in self.weights.items():
            out = out + w * X[col].to_numpy(dtype=float)
        return out


class _InteractionTimeModel:
    backend = "flaml"
    feature_names_in_ = np.array(["date_unix", "day_julian", "hour", "ws"])
    feature_importances_ = np.array([4.0, 3.0, 2.0, 1.0])

    def __init__(self, t_mid: float) -> None:
        self.t_mid = float(t_mid)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        tau = (X["date_unix"].to_numpy(dtype=float) - self.t_mid) / (365.25 * 86400.0)
        doy = X["day_julian"].to_numpy(dtype=float)
        hr = X["hour"].to_numpy(dtype=float)
        ws = X["ws"].to_numpy(dtype=float)
        seas = 4.0 * np.cos(2.0 * np.pi * doy / 365.0)
        diurnal = 2.5 * np.sin(2.0 * np.pi * hr / 24.0 + 0.3)
        event_interaction = -3.0 * np.where(tau > 0.0, 1.0, 0.0) * np.where(doy <= 90.0, 1.0, 0.0)
        return 25.0 + 3.0 * tau + seas + diurnal + event_interaction + 2.0 * ws


class _TreeContribMock:
    backend = "flaml"

    def __init__(self, features: list[str], weights: dict[str, float], bias: float = 12.0) -> None:
        self.feature_names_in_ = np.array(features)
        self.feature_importances_ = np.asarray([abs(weights[f]) for f in features], dtype=float)
        self.weights = weights
        self.bias = bias

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.full(len(X), self.bias, dtype=float)
        for f, w in self.weights.items():
            out = out + w * X[f].to_numpy(dtype=float)
        return out

    def predict_contrib(self, X: pd.DataFrame, num_threads: int | None = None) -> np.ndarray:
        cols = [
            self.weights[f] * (X[f].to_numpy(dtype=float) - X[f].mean())
            for f in self.feature_names_in_
        ]
        bias_col = np.full(
            len(X), self.bias + sum(self.weights[f] * X[f].mean() for f in self.feature_names_in_)
        )
        return np.column_stack(cols + [bias_col])


def test_normalise_empty_variables_resample_fast_path():
    dates = pd.date_range("2024-01-01", periods=24, freq="h")
    df = pd.DataFrame(
        {
            "date": dates,
            "value": np.linspace(10.0, 20.0, 24),
            "ws": np.linspace(1.0, 5.0, 24),
            "t2m": np.linspace(5.0, 15.0, 24),
        }
    )
    model = _CountingLinearModel(["ws", "t2m"], {"ws": 1.5, "t2m": 0.4}, bias=5.0)
    expected = model.predict(df)
    model.predict_calls = 0

    out_agg = normalise(df, model, covariates=["ws", "t2m"], variables_resample=[], n_samples=25)
    assert model.predict_calls == 1
    np.testing.assert_allclose(out_agg["normalised"], expected)

    model.predict_calls = 0
    out_q = normalise(
        df,
        model,
        covariates=["ws", "t2m"],
        variables_resample=[],
        n_samples=15,
        return_quantiles=(0.1, 0.9),
    )
    assert model.predict_calls == 1
    np.testing.assert_allclose(out_q["q100"], expected)
    np.testing.assert_allclose(out_q["q900"], expected)

    model.predict_calls = 0
    out_draws = normalise(
        df,
        model,
        covariates=["ws", "t2m"],
        variables_resample=[],
        n_samples=8,
        aggregate=False,
    )
    assert model.predict_calls == 1
    seed_cols = [c for c in out_draws.columns if c != "observed"]
    assert len(seed_cols) == 8
    assert seed_cols == sorted(seed_cols)
    for sc in seed_cols:
        np.testing.assert_allclose(out_draws[sc], expected)

    model.predict_calls = 0
    df_log = df.copy()
    df_log["value"] = np.log1p(df["value"])
    out_log = normalise(
        df_log,
        model,
        covariates=["ws", "t2m"],
        variables_resample=[],
        n_samples=6,
        log_transform=True,
    )
    assert model.predict_calls == 1
    np.testing.assert_allclose(out_log["normalised"], np.expm1(expected))


def test_decom_met_include_diagnostics_separates_residual_and_biased_pool():
    rng = np.random.default_rng(7)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    ws = rng.uniform(1.0, 6.0, n)
    tr = rng.uniform(5.0, 15.0, n)
    noise = rng.normal(0.0, 1.5, n)
    model = _CountingLinearModel(["ws", "tr"], {"ws": -1.2, "tr": 2.0}, bias=15.0)
    raw_df = pd.DataFrame({"date": dates, "ws": ws, "tr": tr, "value": 0.0})
    raw_df["value"] = model.predict(raw_df) + noise

    # Clean air pool forces large positive mean transport shift (met_base != 0)
    clean_pool = pd.DataFrame({"date": dates[:10], "tr": np.full(10, 0.0)})
    res = decom_met(
        raw_df,
        model,
        n_samples=15,
        resample_pools={"clean": clean_pool},
        include_diagnostics=True,
    )
    assert {"predicted", "met_explained", "residual"} <= set(res.columns)
    np.testing.assert_allclose(res["predicted"], model.predict(raw_df), atol=1e-10)
    np.testing.assert_allclose(res["met_explained"], res["ws"] + res["tr"], atol=1e-10)
    np.testing.assert_allclose(res["observed"], res["predicted"] + res["residual"], atol=1e-10)
    # True residual equals noise, whereas legacy met_noise absorbs -met_base under biased pools!
    np.testing.assert_allclose(res["residual"], noise, atol=1e-10)
    assert abs(float(res["met_base"].iloc[0])) > 10.0
    assert abs(float(res["met_noise"].mean())) > 10.0
    np.testing.assert_allclose(res["met_noise"], res["residual"] - res["met_base"], atol=1e-10)


def test_decom_emi_exact_base_and_calendar_stratify_by(monkeypatch):
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "ws": np.sin(np.arange(60) / 3.0) + 4.0,
            "regime": ["A" if i % 2 == 0 else "B" for i in range(60)],
        }
    )
    df["value"] = 20.0 + 0.05 * np.arange(60) + 2.0 * df["ws"]
    model = _CountingLinearModel(
        ["ws", "date_unix", "day_julian", "hour"],
        {"ws": 2.0, "date_unix": 1e-7, "day_julian": 0.01, "hour": 0.0},
        bias=15.0,
    )

    res = decom_emi(df, model, n_samples=8, exact_base=True)
    np.testing.assert_allclose(res["emi_noise"], 0.0, atol=1e-12)
    expected_base = float(
        model.predict(
            df.assign(
                date_unix=pd.DatetimeIndex(dates).astype("int64") // 10**9,
                day_julian=pd.DatetimeIndex(dates).dayofyear,
                hour=0,
            )
        ).mean()
    )
    np.testing.assert_allclose(res["emi_base"], expected_base, rtol=1e-10)

    seen_strata: list[object] = []

    def spy_norm(df, model, *, covariates, variables_resample, stratify_by=None, **kw):
        seen_strata.append((set(variables_resample), stratify_by))
        d = df.set_index("date")
        return pd.DataFrame(
            {"observed": d["value"], "normalised": np.full(len(d), 10.0)}, index=d.index
        )

    monkeypatch.setattr(dmod, "normalise", spy_norm)
    decom_emi(df, model, n_samples=2, stratify_by=("season", "hour", "regime"))
    # base + date_unix: day_julian & hour still resampled -> only non-calendar "regime" active
    assert seen_strata[0][1] == ["regime"]
    assert seen_strata[1][1] == ["regime"]
    # day_julian fixed: "season" activates while "hour" remains off until hour is fixed
    assert seen_strata[2][1] == ["season", "regime"]
    assert seen_strata[3][1] == ["season", "hour", "regime"]


def test_decom_met_n_cores_and_chronos_guards(monkeypatch):
    seen_cores: list[int | None] = []
    norm_cores: list[int | None] = []

    def fake_bm(df, **kw):
        seen_cores.append(kw.get("n_cores"))
        return df, _CountingLinearModel(["ws", "wd"], {"ws": 1.0, "wd": 0.5})

    def fake_norm(df, model, **kw):
        norm_cores.append(kw.get("n_cores"))
        d = df.set_index("date")
        return pd.DataFrame({"observed": d["value"], "normalised": np.zeros(len(d))}, index=d.index)

    monkeypatch.setattr(dmod, "build_model", fake_bm)
    monkeypatch.setattr(dmod, "normalise", fake_norm)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=16, freq="h"),
            "value": np.ones(16),
            "ws": np.arange(16, dtype=float),
            "wd": np.arange(16, dtype=float),
        }
    )
    decom_met(df, None, covariates=["ws", "wd"], backend="flaml", n_cores=3, n_samples=2)
    assert seen_cores == [3]
    # The per-coalition normalise calls (the decomposition's dominant cost)
    # must receive the thread count too: one base plus one per feature.
    assert norm_cores == [3, 3, 3]

    df_ch = df.rename(columns={"value": "PM2.5"})
    for bad_kw in (
        {"stratify_by": "month"},
        {"conditional_k": 5},
        {"return_uncertainty": True},
        {"scale": "multiplicative"},
    ):
        with pytest.raises(ConfigError, match="not available on the chronos-2 backend"):
            decompose(
                df_ch,
                target="PM2.5",
                method="meteorology",
                backend="chronos-2",
                covariates=["ws", "wd"],
                **bad_kw,
            )


def test_rank_by_sensitivity_averages_multiple_anchors():
    anchors_seen: list[pd.Timestamp] = []

    class _FakeEst:
        context_length = 4
        prediction_length = 2
        min_context_coverage = 0.5

        def _context_coverage(self, arr):
            return 1.0

        def covariate_sensitivity(
            self, indexed, target, *, anchor, horizon, met_features, random_state
        ):
            anchors_seen.append(anchor)
            val = 1.0 if met_features == ["a"] else 3.0
            return {"mean_abs_shift": val}

    idx = pd.date_range("2024-01-01", periods=20, freq="h")
    indexed = pd.DataFrame({"value": np.ones(20)}, index=idx)
    from normet.analysis.decomposition import DecomposeConfig, _rank_by_sensitivity

    ordered_single = _rank_by_sensitivity(_FakeEst(), indexed, ["a", "b"], DecomposeConfig())
    assert ordered_single == ["b", "a"]
    assert anchors_seen == [idx[-2], idx[-2]]

    anchors_seen.clear()
    ordered_multi = _rank_by_sensitivity(
        _FakeEst(), indexed, ["a", "b"], DecomposeConfig(n_sensitivity_anchors=4)
    )
    assert ordered_multi == ["b", "a"]
    assert len(set(anchors_seen)) == 4


def test_chronos_deweather_single_pass_when_met_empty():
    from normet.foundation.estimator import Chronos2Estimator

    est = Chronos2Estimator(context_length=4, prediction_length=2, batch_size=8)
    est._pipeline = object()
    est._quantiles = np.array([0.1, 0.5, 0.9])
    calls: list[int] = []

    def fake_blocks(items, cols, h):
        calls.append(len(items))
        return np.ones((len(items), 3, h), dtype=float)

    est._forecast_blocks = fake_blocks  # type: ignore[method-assign]
    idx = pd.date_range("2024-01-01", periods=8, freq="h")
    df = pd.DataFrame({"value": np.arange(8, dtype=float), "ws": np.ones(8)}, index=idx)
    out = est.deweather(df, "value", met_features=[], n_samples=12, schema="normet")
    assert len(calls) == 2  # 2 horizons after 4-h context
    assert all(c == 1 for c in calls)
    assert out["normalised"].notna().all()


def test_suggest_met_groups_physical_and_qc_exclusion():
    feats = [
        "ws",
        "blh",
        "t2m",
        "ssrd",
        "sp",
        "tp",
        "traj_footprint_sw",
        "traj_inflow_sin",
        "traj_n_endpoints",
        "traj_age_max_h_500m",
        "hour",
        "weekday",
        "traffic_flow",
    ]
    g = suggest_met_groups(feats)
    assert g == {
        "dynamics": ["ws", "blh"],
        "thermodynamics": ["t2m", "ssrd"],
        "synoptic_removal": ["sp", "tp"],
        "transport": ["traj_footprint_sw", "traj_inflow_sin"],
        "other": ["traffic_flow"],
    }


def test_shapley_interactions_recovers_exact_main_and_pairwise_effects(monkeypatch):
    n = 24
    t = np.arange(n, dtype=float)
    main = {"a": np.sin(t / 4.0), "b": 0.5 * np.cos(t / 3.0), "c": 0.1 * t}
    ab = 0.4 + 0.3 * np.sin(t / 5.0)

    class _GameModel:
        backend = "flaml"
        feature_names_in_ = np.array(["a", "b", "c", "hour"])
        feature_importances_ = np.array([3.0, 2.0, 1.0, 0.5])

    def fake_norm(df, model, *, covariates, variables_resample, **kw):
        fixed = {f for f in main if f in covariates and f not in variables_resample}
        d = df.set_index("date")
        rows = ((d.index - d.index[0]) // pd.Timedelta("1h")).to_numpy()
        val = 10.0 + sum((main[f][rows] for f in fixed), np.zeros(len(rows)))
        if {"a", "b"} <= fixed:
            val = val + ab[rows]
        return pd.DataFrame({"observed": d["value"], "normalised": val}, index=d.index)

    monkeypatch.setattr(dmod, "normalise", fake_norm)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "value": np.ones(n) * 15.0,
            "a": 1.0,
            "b": 2.0,
            "c": 3.0,
            "hour": t % 24,
        }
    )
    si = shapley_interactions(df, _GameModel(), n_samples=2)
    np.testing.assert_allclose(si.main_effects["a"], main["a"], atol=1e-12)
    np.testing.assert_allclose(si.main_effects["b"], main["b"], atol=1e-12)
    np.testing.assert_allclose(si.main_effects["c"], main["c"], atol=1e-12)
    np.testing.assert_allclose(si.pairwise["a:b"], ab, atol=1e-12)
    np.testing.assert_allclose(si.pairwise["a:c"], 0.0, atol=1e-12)
    np.testing.assert_allclose(si.pairwise["b:c"], 0.0, atol=1e-12)
    closure = si.main_effects.sum(axis=1) + si.pairwise.sum(axis=1)
    np.testing.assert_allclose(closure, si.decomposition["met_explained"], atol=1e-12)
    assert set(si.summary["term"]) == {"a", "b", "c", "a:b", "a:c", "b:c"}


def test_decom_emi_shapley_separates_event_interaction_from_seasonality():
    # 108h = 4.5 days: 160 steps spans ~2 years across day_julian and alternates hour 0 / 12
    dates = pd.date_range("2023-01-01", periods=160, freq="108h")
    t_unix = (dates.astype("int64") // 10**9).to_numpy(dtype=float)
    df = pd.DataFrame(
        {
            "date": dates,
            "ws": np.sin(np.arange(len(dates)) / 4.0) + 3.0,
            "date_unix": t_unix,
            "day_julian": dates.dayofyear.to_numpy(dtype=float),
            "hour": dates.hour.to_numpy(dtype=float),
        }
    )
    model = _InteractionTimeModel(t_mid=float(np.median(t_unix)))
    df["value"] = model.predict(df)

    res = decom_emi_shapley(df, model, n_samples=8, include_interactions=True)
    np.testing.assert_allclose(
        res["emi_base"] + res["date_unix"] + res["day_julian"] + res["hour"] + res["emi_noise"],
        res["emi_total"],
        atol=1e-11,
    )
    np.testing.assert_allclose(
        res["emi_base"]
        + res["date_unix_main"]
        + res["day_julian_main"]
        + res["hour_main"]
        + res["date_unix:day_julian"]
        + res["date_unix:hour"]
        + res["day_julian:hour"]
        + res["emi_noise"],
        res["emi_total"],
        atol=1e-11,
    )
    # Non-repeating early-year trend*seasonal lockdown step shows up specifically in date_unix:day_julian,
    # while CRN preserves exact zero interaction between decoupled terms (date_unix:hour == 0).
    assert res["date_unix:day_julian"].std() > 0.2
    np.testing.assert_allclose(res["date_unix:hour"], 0.0, atol=1e-10)

    # Opt-in exact_base eliminates base MC noise completely
    res_eb = decom_emi_shapley(df, model, n_samples=8, include_interactions=False, exact_base=True)
    np.testing.assert_allclose(res_eb["emi_noise"], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        res_eb["emi_base"]
        + res_eb["date_unix"]
        + res_eb["day_julian"]
        + res_eb["hour"]
        + res_eb["emi_noise"],
        res_eb["emi_total"],
        atol=1e-11,
    )


def test_decom_met_uncertainty_and_scaling_with_samples():
    rng = np.random.default_rng(11)
    n = 48
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "ws": rng.uniform(1.0, 8.0, n),
            "t2m": rng.normal(12.0, 5.0, n),
            "blh": rng.uniform(200.0, 1200.0, n),
            "value": 0.0,
        }
    )

    class _NonlinearMet:
        backend = "flaml"
        feature_names_in_ = np.array(["ws", "t2m", "blh"])
        feature_importances_ = np.array([3.0, 2.0, 1.0])

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return (
                20.0
                + 2.0 * X["ws"].to_numpy(float)
                + 0.4 * X["ws"].to_numpy(float) * X["t2m"].to_numpy(float)
                - 0.01 * X["blh"].to_numpy(float)
            )

    model = _NonlinearMet()
    df["value"] = model.predict(df)

    low_mc = decom_met(
        df,
        model,
        attribution="shapley",
        n_samples=6,
        return_uncertainty=True,
        include_diagnostics=True,
    )
    high_mc = decom_met(
        df,
        model,
        attribution="shapley",
        n_samples=60,
        return_uncertainty=True,
        include_diagnostics=True,
    )
    assert {"emi_total_se", "ws_se", "t2m_se", "blh_se"} <= set(low_mc.columns)
    assert (low_mc["ws_se"] >= 0.0).all() and (low_mc["emi_total_se"] > 0.0).all()
    assert float(high_mc["emi_total_se"].median()) < float(low_mc["emi_total_se"].median())
    assert float(high_mc["ws_se"].median()) < float(low_mc["ws_se"].median())

    sampled = decom_met(
        df,
        model,
        attribution="shapley",
        n_permutations=4,
        n_samples=10,
        return_uncertainty=True,
    )
    assert (sampled["ws_se"] > 0.0).all()
    np.testing.assert_allclose(
        sampled["ws"] + sampled["t2m"] + sampled["blh"],
        sampled["observed"] - sampled["emi_total"] - (sampled["met_noise"] + sampled["met_base"]),
        atol=1e-10,
    )


def test_multiplicative_lmdi_closure_for_met_and_emi():
    rng = np.random.default_rng(21)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "ws": rng.uniform(1.5, 6.5, n),
            "blh": rng.uniform(300.0, 1200.0, n),
            "date_unix": (dates.astype("int64") // 10**9).to_numpy(dtype=float),
        }
    )
    # Positive multiplicative concentrations predicted in linear units
    model_lin = _CountingLinearModel(["ws", "blh"], {"ws": -1.5, "blh": -0.01}, bias=35.0)
    df["value"] = np.maximum(1.0, model_lin.predict(df))

    res_met = decom_met(
        df,
        model_lin,
        scale="multiplicative",
        include_diagnostics=True,
        n_samples=12,
    )
    np.testing.assert_allclose(res_met["ws"] + res_met["blh"], res_met["met_explained"], atol=1e-10)
    np.testing.assert_allclose(
        res_met["emi_total"] + res_met["ws"] + res_met["blh"], res_met["predicted"], atol=1e-10
    )
    np.testing.assert_allclose(
        res_met["ws_factor"] * res_met["blh_factor"],
        (1.0 + res_met["predicted"]) / (1.0 + res_met["emi_total"]),
        atol=1e-10,
    )

    # Also test emission multiplicative mode on log1p-space model
    df_log = df.copy()
    df_log["value"] = np.log1p(df["value"])
    res_emi = decom_emi(
        df_log,
        model_lin,
        scale="multiplicative",
        log_transform=True,
        exact_base=True,
        n_samples=6,
    )
    np.testing.assert_allclose(res_emi["emi_noise"], 0.0, atol=1e-12)
    np.testing.assert_allclose(res_emi["observed"], np.expm1(df_log["value"]), atol=1e-10)
    np.testing.assert_allclose(res_emi["emi_base"], res_emi["emi_total"], atol=1e-10)

    # A model carrying a calendar feature exercises the multiplicative
    # component and ``_factor`` construction. The model_lin above has only
    # ws/blh, so present_time_vars == ["base"], its components are empty and
    # the closure assertions are true by construction.
    model_trend = _CountingLinearModel(
        ["ws", "blh", "date_unix"],
        {"ws": -0.1, "blh": -0.0005, "date_unix": 1e-9},
        bias=3.0,
    )
    res_trend = decom_emi(
        df_log,
        model_trend,
        scale="multiplicative",
        log_transform=True,
        exact_base=True,
        n_samples=6,
    )
    assert "date_unix_factor" in res_trend.columns
    assert float(res_trend["date_unix"].std()) > 0.0
    np.testing.assert_allclose(
        res_trend["emi_base"] + res_trend["date_unix"] + res_trend["emi_noise"],
        res_trend["emi_total"],
        atol=1e-8,
    )
    # The LMDI factor is the exact multiplicative ratio implied by the
    # component; a wrong weight or a missing _factor column would break this.
    np.testing.assert_allclose(
        res_trend["date_unix_factor"],
        (1.0 + res_trend["emi_total"]) / (1.0 + res_trend["emi_base"]),
        rtol=1e-10,
    )


def test_cross_fit_decompose_embargo_folds_and_stitching(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "value": 20.0 + np.linspace(0, 5, 40),
            "ws": np.sin(np.arange(40) / 3.0) + 4.0,
        }
    )
    seen_folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def spy_train(*, df, covariates, **kw):
        seen_folds.append((pd.Timestamp(df["date"].min()), pd.Timestamp(df["date"].max())))
        assert len(df) < 40
        weights = {c: (1.5 if c == "ws" else 1e-6) for c in covariates}
        return _CountingLinearModel(list(covariates), weights, bias=12.0)

    monkeypatch.setattr(dmod, "train_model", spy_train)
    out_met = cross_fit_decompose(
        df,
        covariates=["ws"],
        backend="flaml",
        method="meteorology",
        n_folds=4,
        embargo_hours=48.0,
        n_samples=6,
        include_diagnostics=True,
    )
    assert len(seen_folds) == 4
    assert len(out_met) == 40
    np.testing.assert_allclose(
        out_met["met_base"] + out_met["ws"] + out_met["met_noise"], out_met["met_total"], atol=1e-10
    )
    np.testing.assert_allclose(
        out_met["predicted"], out_met["emi_total"] + out_met["met_explained"], atol=1e-10
    )

    # Fold 0 test covers days 0..9 (2024-01-01..2024-01-10); 48h embargo keeps train >= 2024-01-13
    assert seen_folds[0][0] >= pd.Timestamp("2024-01-12")

    out_emi = cross_fit_decompose(
        df,
        covariates=["ws", "date_unix"],
        backend="flaml",
        method="emission",
        block_freq="20D",
        n_samples=6,
    )
    assert len(out_emi) == 40
    np.testing.assert_allclose(
        out_emi["emi_base"] + out_emi["date_unix"] + out_emi["emi_noise"],
        out_emi["emi_total"],
        atol=1e-10,
    )


@needs_lgb
def test_cross_fit_and_treeshap_with_real_lightgbm():
    rng = np.random.default_rng(123)
    n = 160
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    ws = rng.uniform(1.0, 7.0, n)
    t2m = rng.normal(12.0, 3.0, n)
    pm = 35.0 - 2.5 * ws + 0.8 * t2m + rng.normal(0.0, 0.8, n)
    # 10-h emission spike
    pm[70:80] += 15.0
    df = pd.DataFrame({"date": dates, "ws": ws, "t2m": t2m, "pm25": pm})

    cf = cross_fit_decompose(
        df,
        target="pm25",
        covariates=["ws", "t2m", "hour"],
        backend="lightgbm",
        method="meteorology",
        n_folds=4,
        embargo_hours=4.0,
        model_config={"n_trials": 1, "cv_folds": 2, "nrounds": 25},
        n_samples=6,
        n_cores=1,
        include_diagnostics=True,
    )
    assert len(cf) == n
    np.testing.assert_allclose(cf["observed"], cf["predicted"] + cf["residual"], atol=1e-10)

    from normet import train_model
    from normet.utils.prepare import prepare_data

    prep = prepare_data(df, "pm25", ["ws", "t2m", "hour"], seed=0)
    lgb_model = train_model(
        prep,
        backend="lightgbm",
        covariates=["ws", "t2m", "hour"],
        model_config={"n_trials": 1, "cv_folds": 2, "nrounds": 25},
        n_cores=1,
    )
    shap_df = decom_met_treeshap(df, lgb_model, target="pm25", n_cores=1)
    np.testing.assert_allclose(
        shap_df["predicted"], lgb_model.predict(prep[["ws", "t2m", "hour"]]), atol=1e-8
    )
    np.testing.assert_allclose(
        shap_df["emi_total"] + shap_df["ws"] + shap_df["t2m"], shap_df["predicted"], atol=1e-8
    )

    shap_grp = decom_met_treeshap(df, lgb_model, target="pm25", groups={"all": ["ws", "t2m"]})
    np.testing.assert_allclose(shap_grp["all"], shap_df["ws"] + shap_df["t2m"], atol=1e-8)

    comp = compare_attributions(cf, shap_df)
    assert list(comp.index) == ["ws", "t2m"]
    assert comp["pearson_r"].notna().all()


def test_conditional_k_on_manifold_parity_and_cache(tmp_path):
    rng = np.random.default_rng(9)
    n = 80
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    ws = np.linspace(1.0, 10.0, n)
    # Strong collinear coupling: traj_x is tightly tied to ws!
    traj_x = 50.0 * ws + rng.normal(0.0, 0.2, n)
    df = pd.DataFrame({"date": dates, "value": 10.0, "ws": ws, "traj_x": traj_x})
    model = _CountingLinearModel(["ws", "traj_x"], {"ws": 0.0, "traj_x": 1.0}, bias=0.0)

    n_uncond = normalise(
        df,
        model,
        covariates=["ws", "traj_x"],
        variables_resample=["traj_x"],
        n_samples=15,
        conditional_k=None,
        batch_size=0,
        cache=str(tmp_path / "c"),
    )
    n_cond_vec = normalise(
        df,
        model,
        covariates=["ws", "traj_x"],
        variables_resample=["traj_x"],
        n_samples=15,
        conditional_k=3,
        batch_size=0,
        memory_save=False,
        cache=str(tmp_path / "c"),
    )
    n_cond_mem = normalise(
        df,
        model,
        covariates=["ws", "traj_x"],
        variables_resample=["traj_x"],
        n_samples=15,
        conditional_k=3,
        batch_size=0,
        memory_save=True,
    )
    pd.testing.assert_frame_equal(n_cond_vec, n_cond_mem)
    assert not n_uncond["normalised"].equals(n_cond_vec["normalised"])

    err_cond = float(np.sqrt(np.mean((n_cond_vec["normalised"] - df["traj_x"].to_numpy()) ** 2)))
    err_uncond = float(np.sqrt(np.mean((n_uncond["normalised"] - df["traj_x"].to_numpy()) ** 2)))
    assert err_cond < 0.1 * err_uncond

    # Mock TreeSHAP comparison smoke test (no lightgbm needed)
    tree_m = _TreeContribMock(["ws", "traj_x"], {"ws": 1.0, "traj_x": 1.0})
    df2 = df.copy()
    df2["value"] = tree_m.predict(df2)
    inter = decom_met(df2, tree_m, n_samples=10)
    cond = decom_met_treeshap(df2, tree_m)
    cmp = compare_attributions(inter, cond)
    assert "ood_discrepancy" in cmp.columns and len(cmp) == 2


def test_decom_emi_shapley_no_time_players_keeps_original_units(monkeypatch):
    """The zero-time-feature path must not double-log the already-log1p frame.

    With ``model=None`` and ``scale='multiplicative'`` the shared prepare step
    log1p-trains the target and returns a log-space model. Delegating that
    state to ``decom_emi`` -- whose own prepare re-derives ``is_log_space``
    from ``log_transform=False`` -- log1p'd the predictions a second time and
    reported ``observed`` in log1p units.
    """
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    ws = np.linspace(1.0, 6.0, 60)
    df = pd.DataFrame({"date": dates, "ws": ws, "value": 40.0 + 3.0 * ws})

    def fake_bm(df, **kw):
        return df, _CountingLinearModel(["ws"], {"ws": 0.2}, bias=2.0)

    monkeypatch.setattr(dmod, "build_model", fake_bm)
    common = dict(
        model=None,
        covariates=["ws"],
        backend="flaml",
        scale="multiplicative",
        n_samples=6,
    )
    shap = decom_emi_shapley(df, **common)
    direct = decom_emi(df, **common)

    np.testing.assert_allclose(shap["observed"], df["value"], atol=1e-10)
    np.testing.assert_allclose(shap["observed"].mean(), df["value"].mean(), atol=1e-10)
    assert float(shap["observed"].mean()) > 10.0  # concentrations, not log1p units
    np.testing.assert_allclose(shap["emi_total"], direct["emi_total"], atol=1e-10)
    np.testing.assert_allclose(shap["emi_base"], direct["emi_base"], atol=1e-10)
    assert set(shap.columns) == set(direct.columns)


def test_decom_met_return_uncertainty_keeps_the_default_point_estimates():
    """SE columns must not move the estimator.

    ``return_uncertainty=True`` used to request ``aggregate=False`` and
    average ``expm1(draw)``, while the default averages the log-space draws
    and expm1s once -- a Jensen gap that survives more draws.
    """
    rng = np.random.default_rng(31)
    n = 48
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "ws": rng.uniform(1.0, 6.0, n),
            "t2m": rng.normal(10.0, 4.0, n),
        }
    )
    log_model = _CountingLinearModel(["ws", "t2m"], {"ws": 0.15, "t2m": -0.04}, bias=3.0)
    df_log = df.copy()
    df_log["value"] = log_model.predict(df_log)  # already on the log1p scale

    point = decom_met(df_log, log_model, log_transform=True, n_samples=32)
    wide = decom_met(df_log, log_model, log_transform=True, n_samples=32, return_uncertainty=True)
    assert {"emi_total_se", "ws_se", "t2m_se"} <= set(wide.columns)
    for col in ("observed", "emi_total", "ws", "t2m", "met_total", "met_base", "met_noise"):
        np.testing.assert_allclose(wide[col], point[col], rtol=1e-12, atol=1e-12)
    assert (wide[["emi_total_se", "ws_se", "t2m_se"]] >= 0.0).all().all()

    # The same guarantee on the multiplicative LMDI path (linear-space model).
    lin_model = _CountingLinearModel(["ws", "t2m"], {"ws": -0.4, "t2m": 0.6}, bias=22.0)
    df_lin = df.copy()
    df_lin["value"] = np.maximum(1.0, lin_model.predict(df_lin))
    point_m = decom_met(df_lin, lin_model, scale="multiplicative", n_samples=32)
    wide_m = decom_met(
        df_lin, lin_model, scale="multiplicative", n_samples=32, return_uncertainty=True
    )
    assert {"emi_total_se", "ws_se", "t2m_se"} <= set(wide_m.columns)
    for col in ("emi_total", "ws", "t2m", "met_total", "met_base", "met_noise"):
        np.testing.assert_allclose(wide_m[col], point_m[col], rtol=1e-12, atol=1e-12)


def test_shapley_interactions_time_only_model_returns_empty_summary():
    """No meteorological player is valid input, not a KeyError on 'std'."""
    n = 48
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "date": dates,
            "date_unix": (dates.astype("int64") // 10**9).to_numpy(dtype=float),
            "hour": dates.hour.to_numpy(dtype=float),
        }
    )
    model = _CountingLinearModel(["date_unix", "hour"], {"date_unix": 2e-7, "hour": 0.05}, bias=3.0)
    df["value"] = model.predict(df)

    si = shapley_interactions(df, model, n_samples=2)
    assert list(si.summary.columns) == [
        "term",
        "kind",
        "mean",
        "std",
        "mean_abs",
        "max_abs",
    ]
    assert si.summary.empty
    assert si.main_effects.shape == (n, 0)
    assert si.pairwise.shape == (n, 0)
    assert {"observed", "emi_total", "met_total", "met_base", "met_noise"} <= set(
        si.decomposition.columns
    )

    # decom_met accepts the same input; shapley_interactions must too.
    met = decom_met(df, model, n_samples=2)
    assert {"observed", "emi_total", "met_total", "met_base", "met_noise"} <= set(met.columns)


def test_decom_emi_default_column_order_matches_the_base_branch():
    """Default additive output must keep components first and totals last.

    ``decom_emi`` used to write ``emi_total``/``emi_base``/``emi_noise``
    before the calendar components, silently breaking positional consumers
    (``out.columns[1:-3]`` no longer selected the components).
    """
    n = 40
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "date": dates,
            "ws": np.sin(np.arange(n) / 3.0) + 4.0,
            "date_unix": (dates.astype("int64") // 10**9).to_numpy(dtype=float),
            "hour": dates.hour.to_numpy(dtype=float),
        }
    )
    model = _CountingLinearModel(
        ["ws", "date_unix", "hour"],
        {"ws": 1.0, "date_unix": 1e-7, "hour": 0.1},
        bias=10.0,
    )
    df["value"] = model.predict(df)

    res = decom_emi(df, model, n_samples=4)
    assert list(res.columns) == [
        "observed",
        "date_unix",
        "hour",
        "emi_total",
        "emi_noise",
        "emi_base",
    ]
    np.testing.assert_allclose(
        res["emi_base"] + res["date_unix"] + res["hour"] + res["emi_noise"],
        res["emi_total"],
        atol=1e-10,
    )


def test_zero_resample_quantiles_match_the_per_seed_long_table():
    """The fast path must reproduce the resampling path's quantiles.

    Quantiles are computed over the long per-seed table, where each input
    row appears ``n_samples`` times. With nothing resampled every seed
    predicts the same value, so two equal-date rows predicting 6 and 7 have
    pooled quantiles 6 / 7 at 10 % / 90 %, not 6.1 / 6.9 from the two raw
    rows.
    """
    ts = pd.Timestamp("2024-01-01 00:00")
    df = pd.DataFrame(
        {
            "date": [ts, ts],
            "value": [5.0, 5.0],
            "ws": [0.0, 1.0],
            "dummy": [0.0, 1.0],
        }
    )
    model = _CountingLinearModel(["ws"], {"ws": 1.0}, bias=6.0)
    fast = normalise(
        df,
        model,
        covariates=["ws"],
        variables_resample=[],
        n_samples=4,
        return_quantiles=(0.1, 0.9),
    )
    slow = normalise(
        df,
        model,
        covariates=["ws", "dummy"],
        variables_resample=["dummy"],
        n_samples=4,
        return_quantiles=(0.1, 0.9),
        batch_size=0,
    )
    assert len(fast) == len(slow) == 1
    np.testing.assert_allclose(fast["q100"], [6.0])
    np.testing.assert_allclose(fast["q900"], [7.0])
    pd.testing.assert_frame_equal(
        fast[["observed", "normalised", "q100", "q900"]],
        slow[["observed", "normalised", "q100", "q900"]],
    )


def test_multiplicative_exact_base_converges_to_the_monte_carlo_limit():
    """``exact_base`` must be the noise-free limit of the MC path, not a
    different estimand: for a linear-space model the baseline is
    ``log1p(mean(preds))``, not ``mean(log1p(preds))`` (a Jensen gap that
    does not shrink with n_samples)."""
    n = 60
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "date_unix": np.linspace(-30.0, 30.0, n),
            "value": 30.0,
        }
    )
    model = _CountingLinearModel(["date_unix"], {"date_unix": 0.05}, bias=2.0)

    exact = decom_emi(df, model, scale="multiplicative", n_samples=1000, exact_base=True)
    mc = decom_emi(df, model, scale="multiplicative", n_samples=1000, exact_base=False)

    expected_base = float(np.mean(model.predict(df)))
    np.testing.assert_allclose(exact["emi_base"].iloc[0], expected_base, rtol=1e-12)
    np.testing.assert_allclose(exact["emi_base"].iloc[0], mc["emi_base"].iloc[0], rtol=1e-2)
    np.testing.assert_allclose(mc["emi_base"].iloc[0], expected_base, rtol=1e-2)
    # The same base shift is subtracted from every time component:
    # a different estimand moves the whole date_unix series, not just noise.
    np.testing.assert_allclose(
        exact["date_unix"].mean(), mc["date_unix"].mean(), rtol=0.0, atol=0.05
    )
    np.testing.assert_allclose(exact["date_unix"], mc["date_unix"], atol=0.2)


def test_cross_fit_decompose_imputes_from_training_rows_only(monkeypatch):
    """A fold's training medians must not see the held-out block.

    The old whole-record ``impute_values`` filled training NaNs with a
    median that included the test block (and filled the test block's NaNs
    with a statistic that included their own values).
    """
    n = 40
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    ws = np.r_[np.full(19, 100.0), np.nan, np.linspace(1.0, 2.0, 19), np.nan]
    df = pd.DataFrame({"date": dates, "ws": ws, "value": 20.0 + np.linspace(0.0, 5.0, n)})

    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    def spy_train(*, df, covariates, **kw):
        train_frames.append(df.copy())
        return _CountingLinearModel(list(covariates), {"ws": 1.0}, bias=10.0)

    real_decom_met = dmod.decom_met

    def spy_decom_met(df, model=None, **kw):
        test_frames.append(df.copy())
        return real_decom_met(df, model=model, **kw)

    monkeypatch.setattr(dmod, "train_model", spy_train)
    monkeypatch.setattr(dmod, "decom_met", spy_decom_met)

    out = cross_fit_decompose(
        df,
        covariates=["ws"],
        backend="flaml",
        method="meteorology",
        n_folds=2,
        n_samples=4,
    )
    assert len(out) == n
    # Fold 0 trains on rows 20..39 and holds out rows 0..19, whose values
    # are ~100. The training-only median is ~1.5; the full-record median is
    # ~50, so the leak is easy to distinguish.
    expected_fill = float(np.nanmedian(ws[20:40]))
    full_fill = float(np.nanmedian(ws))
    assert abs(full_fill - expected_fill) > 10.0

    fold0_train = train_frames[0]
    assert float(fold0_train.loc[fold0_train["date"] == dates[39], "ws"].iloc[0]) == (
        pytest.approx(expected_fill)
    )
    fold0_test = test_frames[0]
    assert float(fold0_test.loc[fold0_test["date"] == dates[19], "ws"].iloc[0]) == (
        pytest.approx(expected_fill)
    )
