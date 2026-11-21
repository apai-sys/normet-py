"""Comprehensive tests for Data Preparation, Model Training, and Evaluation v2."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import normet
from normet.analysis.verification import (
    calibration_curve,
    crps_skill_score,
    fair_crps_from_samples,
    interval_coverage,
    interval_score,
    pit_diagnostics,
)
from normet.backends import backend_registry
from normet.backends.flaml_backend import train_flaml
from normet.backends.lgb_backend import LgbModel, train_lgb
from normet.exceptions import DataError
from normet.model.io import load_model, load_model_metadata, save_model
from normet.model.predict import ml_predict
from normet.model.train import build_model, train_model
from normet.utils.cv import cv_score
from normet.utils.featureeng import (
    DEFAULT_MET_BOUNDS,
    LagDiagnostics,
    add_lag_features,
    add_met_physics_features,
    add_rolling_features,
    analyze_lag,
    screen_aq_data,
)
from normet.utils.features import extract_features, model_feature_names, permutation_importance
from normet.utils.metrics import Stats, compare_models, exceedance_stats, modStats
from normet.utils.prepare import (
    add_date_variables,
    check_data,
    impute_values,
    prepare_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hourly_df(n: int = 120, seed: int = 42, *, tz: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="h", tz=tz)
    ws = np.clip(3.0 + 1.5 * np.sin(np.arange(n) / 12.0) + rng.normal(0, 0.3, n), 0.2, 20.0)
    wd = (np.arange(n) * 15.0) % 360.0
    temp = 15.0 + 5.0 * np.sin(np.arange(n) / 24.0) + rng.normal(0, 0.5, n)
    noise = rng.normal(0, 1.0, n)
    value = 40.0 - 2.0 * ws + 0.8 * temp + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "date": dates,
            "value": value,
            "ws": ws,
            "wd": wd,
            "temp": temp,
            "noise": noise,
        }
    )


class _SimpleLinearModel:
    """Minimal deterministic sklearn-like model recording feature order."""

    def __init__(
        self,
        features: list[str],
        coefs: np.ndarray,
        intercept: float = 0.0,
        *,
        backend: str = "flaml",
    ) -> None:
        self.backend = backend
        self.feature_names_in_ = np.array(features, dtype=object)
        self.coefs = np.asarray(coefs, dtype=float)
        self.intercept = float(intercept)
        self.last_columns: list[str] | None = None
        # Reverse-order importances to ensure extract_features != model_feature_names
        self.feature_importances_ = np.arange(1, len(features) + 1, dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.last_columns = list(X.columns)
        arr = X.to_numpy(dtype=float)
        return arr @ self.coefs + self.intercept


# ---------------------------------------------------------------------------
# Layer 1: Data preparation & feature engineering
# ---------------------------------------------------------------------------


def test_prepare_data_deduplicate_modes(caplog: pytest.LogCaptureFixture) -> None:
    df = _make_hourly_df(12)
    dup = df.iloc[[2]].copy()
    dup["value"] = df.loc[2, "value"] + 10.0
    df_dup = pd.concat([df, dup], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        prepare_data(df_dup, "value", ["ws", "temp"], deduplicate="raise")

    with caplog.at_level(logging.WARNING):
        df_default = prepare_data(df_dup, "value", ["ws", "temp"], deduplicate=None)
    assert len(df_default) == len(df_dup)
    assert "duplicate" in caplog.text.lower()

    df_mean = prepare_data(df_dup, "value", ["ws", "temp"], deduplicate="mean")
    df_first = prepare_data(df_dup, "value", ["ws", "temp"], deduplicate="first")
    df_last = prepare_data(df_dup, "value", ["ws", "temp"], deduplicate="last")
    assert len(df_mean) == len(df)
    assert len(df_first) == len(df)
    assert len(df_last) == len(df)

    orig_val = float(df.loc[2, "value"])
    ts = df.loc[2, "date"]
    assert float(df_first.loc[df_first["date"] == ts, "value"].iloc[0]) == pytest.approx(orig_val)
    assert float(df_last.loc[df_last["date"] == ts, "value"].iloc[0]) == pytest.approx(
        orig_val + 10.0
    )
    assert float(df_mean.loc[df_mean["date"] == ts, "value"].iloc[0]) == pytest.approx(
        orig_val + 5.0
    )


def test_check_data_strict_covariates() -> None:
    df = _make_hourly_df(10)
    df_missing = df.drop(columns=["temp"])

    df_relaxed = check_data(df_missing.copy(), ["ws", "temp"], "value", strict_covariates=False)
    assert "ws" in df_relaxed.columns
    assert "temp" not in df_relaxed.columns

    with pytest.raises(ValueError, match="temp"):
        check_data(df_missing.copy(), ["ws", "temp"], "value", strict_covariates=True)


def test_impute_time_interp_no_cross_split_or_boundary_or_oversize_leakage() -> None:
    dates = pd.date_range("2023-01-01", periods=20, freq="h")
    x = np.arange(20, dtype=float) * 10.0
    sets = np.array(["training"] * 9 + ["testing"] * 11, dtype=object)
    # 0: leading NaN -> falls back to training median
    x[0] = np.nan
    # 3: interior training gap len 1 <= max_gap(2) -> interpolated (20..40 -> 30)
    x[3] = np.nan
    # 8, 9: cross-split gap straddling training/testing boundary -> falls back to training median
    x[8] = np.nan
    x[9] = np.nan
    # 12, 13, 14: interior testing gap len 3 > max_gap(2) -> falls back to training median (no partial fill)
    x[12:15] = np.nan
    # 16: interior testing gap len 1 <= max_gap(2) -> interpolated (150..170 -> 160)
    x[16] = np.nan
    # 19: trailing NaN -> falls back to training median
    x[19] = np.nan

    df = pd.DataFrame({"date": dates, "x": x, "set": sets})
    out = impute_values(df, dropna=False, strategy="time_interp", max_gap=2)

    # Interior interpolations succeed strictly within their split
    assert out.loc[3, "x"] == pytest.approx(30.0)
    assert out.loc[16, "x"] == pytest.approx(160.0)

    # After interior training interpolation fills index 3 -> 30.0, training median = median([10,20,30,40,50,60,70]) = 40.0
    train_med = float(np.median([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]))
    for idx in (0, 8, 9, 12, 13, 14, 19):
        assert out.loc[idx, "x"] == pytest.approx(train_med)

    # Modifying testing boundary neighbor (row 10) must NOT alter training/cross-split rows
    df_pert = df.copy()
    df_pert.loc[10, "x"] = 9999.0
    out_pert = impute_values(df_pert, dropna=False, strategy="time_interp", max_gap=2)
    assert out_pert.loc[8, "x"] == pytest.approx(train_med)
    assert out_pert.loc[9, "x"] == pytest.approx(train_med)


def test_add_date_variables_timezone_local_vs_utc_epoch() -> None:
    ts_local = pd.DatetimeIndex([pd.Timestamp("2023-06-15 01:00:00", tz="Asia/Shanghai")])
    df = pd.DataFrame({"date": ts_local})
    out = add_date_variables(df)

    expected_epoch = int(ts_local.tz_convert("UTC")[0].timestamp())
    assert int(out.loc[0, "date_unix"]) == expected_epoch
    assert int(out.loc[0, "hour"]) == 1
    assert int(out.loc[0, "day_julian"]) == pd.Timestamp("2023-06-15").dayofyear
    assert int(out.loc[0, "weekday"]) == 4


def test_time_aware_lag_rolling_and_analyze_lag_with_gaps() -> None:
    df = _make_hourly_df(30)
    df_gap = df.drop(index=[5]).reset_index(drop=True)

    lag_df = add_lag_features(df_gap, cols=["value"], lags=[1], freq="h")
    t_06 = df.loc[6, "date"]
    assert pd.isna(lag_df.loc[lag_df["date"] == t_06, "value_lag1"].iloc[0])
    t_07 = df.loc[7, "date"]
    assert lag_df.loc[lag_df["date"] == t_07, "value_lag1"].iloc[0] == pytest.approx(
        df.loc[6, "value"]
    )

    # Default min_periods=window=3: at 07:00, [05:00(NaN), 06:00, 07:00] has only 2 points -> NaN
    roll_default = add_rolling_features(
        df_gap, cols=["value"], windows=[3], aggs=("mean",), causal=True, freq="h"
    )
    assert len(roll_default) == len(df_gap)
    assert pd.isna(roll_default.loc[roll_default["date"] == t_07, "value_roll3_mean"].iloc[0])

    # With min_periods=2, 07:00 averages only [06:00, 07:00] (skipping missing 05:00)
    roll_mp2 = add_rolling_features(
        df_gap, cols=["value"], windows=[3], aggs=("mean",), min_periods=2, causal=True, freq="h"
    )
    expected_roll = np.mean([df.loc[6, "value"], df.loc[7, "value"]])
    assert roll_mp2.loc[roll_mp2["date"] == t_07, "value_roll3_mean"].iloc[0] == pytest.approx(
        expected_roll
    )

    diag = analyze_lag(
        _make_hourly_df(120).drop(index=[20, 21]),
        target="value",
        driver="ws",
        max_lag=12,
        freq="h",
    )
    assert isinstance(diag, LagDiagnostics)
    assert len(diag.acf) == 13
    assert np.isfinite(diag.acf.dropna().to_numpy(dtype=float)).all()


def test_screen_aq_data_and_met_physics_features() -> None:
    df = _make_hourly_df(30)
    df["t2m"] = 20.0
    df["d2m"] = 10.0
    df["rh"] = 60.0
    df["blh"] = 500.0
    df.loc[0, "value"] = -5.0
    df.loc[5:12, "value"] = 42.123
    df.loc[2, "rh"] = 150.0
    df.loc[20, "value"] = 5000.0

    assert "rh" in DEFAULT_MET_BOUNDS
    screened, flags = screen_aq_data(
        df,
        "value",
        negative_action="nan",
        flatline_window=6,
        bounds={"rh": DEFAULT_MET_BOUNDS["rh"]},
        iqr_multiplier=3.0,
        return_flags=True,
    )
    assert pd.isna(screened.loc[0, "value"])
    assert screened.loc[5:12, "value"].isna().all()
    assert pd.isna(screened.loc[2, "rh"])
    assert pd.isna(screened.loc[20, "value"])
    assert isinstance(flags, pd.DataFrame)

    # Case 1: with explicit RH present
    phys_with_rh = add_met_physics_features(
        df[["ws", "wd", "blh", "t2m", "d2m", "rh"]].copy(), temp_unit="auto"
    )
    assert {"u_wind", "v_wind", "ventilation_coef", "vpd", "dewpoint_depression"}.issubset(
        phys_with_rh.columns
    )

    # Case 2: without RH -> derives rh_derived, matches between Celsius & Kelvin
    phys_c = add_met_physics_features(
        df[["ws", "wd", "blh", "t2m", "d2m"]].copy(), temp_unit="auto"
    )
    df_k = df[["ws", "wd", "blh", "t2m", "d2m"]].copy()
    df_k["t2m"] = df_k["t2m"] + 273.15
    df_k["d2m"] = df_k["d2m"] + 273.15
    phys_k = add_met_physics_features(df_k, temp_unit="auto")

    expected_cols = {
        "u_wind",
        "v_wind",
        "ventilation_coef",
        "vpd",
        "dewpoint_depression",
        "rh_derived",
    }
    assert expected_cols.issubset(phys_c.columns)
    assert np.allclose(phys_c["dewpoint_depression"], 10.0)
    assert np.allclose(phys_k["dewpoint_depression"], 10.0)
    assert np.allclose(phys_c["vpd"], phys_k["vpd"], rtol=1e-5)
    assert np.allclose(phys_c["rh_derived"], phys_k["rh_derived"], rtol=1e-4)
    assert (phys_c["rh_derived"] > 0).all() and (phys_c["rh_derived"] <= 100.0).all()


# ---------------------------------------------------------------------------
# Layer 2: Model feature order, backends, build_model, cache, IO
# ---------------------------------------------------------------------------


def test_model_feature_names_ml_predict_strict_and_permutation_importance() -> None:
    df = _make_hourly_df(120)
    df["set"] = ["training"] * 80 + ["testing"] * 40
    df["value"] = 3.0 * df["ws"] + 0.5 * df["temp"]

    features = ["ws", "temp", "noise"]
    model = _SimpleLinearModel(features, np.array([3.0, 0.5, 0.0]))

    assert extract_features(model) == ["noise", "temp", "ws"]
    assert model_feature_names(model) == ["ws", "temp", "noise"]

    preds = ml_predict(model, df, strict_features=True)
    assert model.last_columns == ["ws", "temp", "noise"]
    assert np.allclose(preds, df["value"])

    with pytest.raises(DataError):
        ml_predict(model, df.drop(columns=["temp"]), strict_features=True)

    imp_rmse = permutation_importance(
        df,
        model,
        target="value",
        subset="testing",
        metric="RMSE",
        n_repeats=5,
        block_size=4,
        seed=7,
    )
    assert list(imp_rmse.columns) == [
        "variable",
        "importance_mean",
        "importance_std",
        "baseline_score",
        "metric",
    ]
    ws_imp = float(imp_rmse.loc[imp_rmse["variable"] == "ws", "importance_mean"].iloc[0])
    noise_imp = float(imp_rmse.loc[imp_rmse["variable"] == "noise", "importance_mean"].iloc[0])
    assert ws_imp > noise_imp
    assert noise_imp == pytest.approx(0.0, abs=1e-9)

    df_log = df.copy()
    df_log["value"] = np.log1p(np.exp(0.02 * df["ws"]))
    model_log = _SimpleLinearModel(["ws", "temp", "noise"], np.array([0.02, 0.0, 0.0]))
    imp_r2 = permutation_importance(
        df_log,
        model_log,
        target="value",
        subset="testing",
        metric="R2",
        n_repeats=3,
        log_transform=True,
        seed=7,
    )
    assert (imp_r2["metric"] == "R2").all()


def test_lgb_backend_categoricals_objectives_deterministic_warning_and_weights(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pytest.importorskip("lightgbm")
    df = _make_hourly_df(90, seed=11)
    rng = np.random.default_rng(11)
    df["station"] = pd.Series(rng.choice(["urban", "suburban", "rural"], size=len(df))).astype(
        "category"
    )
    df["value"] = df["value"] + df["station"].map(
        {"urban": 15.0, "suburban": 5.0, "rural": -5.0}
    ).astype(float)
    covs = ["ws", "temp", "station"]

    with caplog.at_level(logging.WARNING):
        m1 = train_lgb(
            df,
            target="value",
            covariates=covs,
            model_config={
                "objective": "huber",
                "alpha": 1.5,
                "n_trials": 2,
                "cv_folds": 2,
                "deterministic": True,
                "nonexistent_magic_param": 99,
            },
            seed=123,
        )
    assert isinstance(m1, LgbModel)
    assert m1.categorical_features == ["station"]
    assert set(m1.category_levels["station"]) == {"rural", "suburban", "urban"}
    assert "nonexistent_magic_param" in caplog.text

    m2 = train_lgb(
        df,
        target="value",
        covariates=covs,
        model_config={
            "objective": "huber",
            "alpha": 1.5,
            "n_trials": 2,
            "cv_folds": 2,
            "deterministic": True,
        },
        seed=123,
    )
    p1 = ml_predict(m1, df)
    p2 = ml_predict(m2, df)
    assert np.allclose(p1, p2)

    df_unseen = df.iloc[:5].copy()
    df_unseen["station"] = "brand_new_station"
    p_unseen = ml_predict(m1, df_unseen)
    assert p_unseen.shape == (5,)
    assert np.isfinite(p_unseen).all()

    # Legacy __setstate__ compatibility using slotted state dict
    state = {
        "_booster": m1.booster,
        "backend": "lightgbm",
        "feature_names": list(m1.feature_names),
        "use_gpu": False,
    }
    legacy = LgbModel.__new__(LgbModel)
    legacy.__setstate__(state)
    assert legacy.categorical_features == []
    assert legacy.category_levels == {}
    assert legacy.predict(df[["ws", "temp", "station"]].assign(station=0.0).iloc[:3]).shape == (3,)

    w = np.linspace(0.5, 5.0, len(df))
    m_q = train_lgb(
        df,
        target="value",
        covariates=["ws", "temp"],
        model_config={"objective": "quantile", "alpha": 0.9, "n_trials": 1, "cv_folds": 2},
        sample_weight=w,
        seed=5,
    )
    assert np.isfinite(ml_predict(m_q, df)).all()


def test_flaml_backend_cv_strategy_and_sample_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_calls: list[dict[str, Any]] = []

    class _FakeAutoML:
        best_estimator = "fake"
        best_config: dict[str, Any] = {}
        best_loss = 0.0

        def fit(self, **kwargs: Any) -> None:
            recorded_calls.append(kwargs)
            self._X_len = len(kwargs["X_train"])

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X), dtype=float)

    monkeypatch.setattr("normet.backends.flaml_backend._import_flaml_automl", lambda: _FakeAutoML)

    df = _make_hourly_df(60)
    w = np.linspace(1.0, 2.0, len(df))
    train_flaml(
        df,
        target="value",
        covariates=["ws", "temp"],
        model_config={"cv_strategy": "blocked", "cv_folds": 3, "time_budget": 1},
        sample_weight=w,
    )
    assert len(recorded_calls) == 1
    call = recorded_calls[0]
    assert call["split_type"] == "group"
    assert len(call["groups"]) == len(df)
    assert len(np.unique(call["groups"])) == 3
    assert np.all(np.diff(call["groups"]) >= 0)
    assert np.allclose(call["sample_weight"], w)

    recorded_calls.clear()
    train_flaml(
        df,
        target="value",
        covariates=["ws", "temp"],
        model_config={"cv_strategy": "time", "cv_folds": 3, "time_budget": 1},
    )
    assert recorded_calls[0]["split_type"] == "time"
    assert "groups" not in recorded_calls[0]
    assert "sample_weight" not in recorded_calls[0]


def test_build_model_forwarding_weight_alignment_and_train_cache_keys(tmp_path: Path) -> None:
    df = _make_hourly_df(40)
    df.loc[5, "ws"] = np.nan
    weights = np.arange(1, 41, dtype=float)
    date_to_weight = dict(zip(df["date"], weights, strict=True))

    seen_calls: list[dict[str, Any]] = []

    class _RecordingBackend:
        name = "rec"

        def train(
            self,
            df_train: pd.DataFrame,
            target: str = "value",
            covariates: list[str] | None = None,
            variables: list[str] | None = None,
            model_config: dict[str, Any] | None = None,
            seed: int = 1,
            verbose: bool = False,
            n_cores: int | None = None,
            sample_weight: Any = None,
        ) -> _SimpleLinearModel:
            covs = list(covariates or [])
            seen_calls.append(
                {
                    "n_rows": len(df_train),
                    "dates": df_train["date"].to_numpy().copy(),
                    "covariates": covs,
                    "sample_weight": (
                        None if sample_weight is None else np.asarray(sample_weight).copy()
                    ),
                }
            )
            return _SimpleLinearModel(covs, np.ones(len(covs)), backend="rec")

        def save(self, model: object, path: str = ".", filename: str = "automl.joblib") -> str:
            return str(Path(path) / filename)

        def load(self, path: str = ".", filename: str | None = None) -> object:
            return {}

    backend_registry.register(_RecordingBackend())
    try:
        df_out, _model = build_model(
            df,
            "value",
            backend="rec",
            covariates=["ws", "temp"],
            split_method="ts",
            train_fraction=0.75,
            impute="none",
            embargo=2,
            deduplicate="mean",
            strict_covariates=True,
            sample_weight=weights,
        )
        assert (df_out["set"] == "embargo").sum() == 2
        # The backend receives weights aligned row-for-row with the frame it
        # gets (which still carries the ``set`` column and filters itself).
        expected_w = np.array(
            [date_to_weight[pd.Timestamp(d)] for d in seen_calls[0]["dates"]], dtype=float
        )
        assert seen_calls[0]["sample_weight"] is not None
        assert len(seen_calls[0]["sample_weight"]) == seen_calls[0]["n_rows"]
        assert np.allclose(seen_calls[0]["sample_weight"], expected_w)

        seen_calls.clear()
        df_clean = _make_hourly_df(30)
        w1 = np.ones(len(df_clean))
        w2 = np.full(len(df_clean), 2.0)

        train_model(
            df_clean,
            target="value",
            covariates=["ws", "temp"],
            backend="rec",
            cache=tmp_path,
            sample_weight=w1,
        )
        train_model(
            df_clean,
            target="value",
            covariates=["ws", "temp"],
            backend="rec",
            cache=tmp_path,
            sample_weight=w1,
        )
        assert len(seen_calls) == 1

        # Reordering covariates misses cache
        train_model(
            df_clean,
            target="value",
            covariates=["temp", "ws"],
            backend="rec",
            cache=tmp_path,
            sample_weight=w1,
        )
        assert len(seen_calls) == 2

        # Changing sample_weight misses cache
        train_model(
            df_clean,
            target="value",
            covariates=["ws", "temp"],
            backend="rec",
            cache=tmp_path,
            sample_weight=w2,
        )
        assert len(seen_calls) == 3

        # Changing date column misses cache
        df_shifted = df_clean.copy()
        df_shifted["date"] = df_shifted["date"] + pd.Timedelta(days=1)
        train_model(
            df_shifted,
            target="value",
            covariates=["ws", "temp"],
            backend="rec",
            cache=tmp_path,
            sample_weight=w1,
        )
        assert len(seen_calls) == 4
    finally:
        backend_registry._backends.pop("rec", None)


def test_save_load_model_with_metadata_sidecar(tmp_path: Path) -> None:
    model = _SimpleLinearModel(["ws", "temp"], np.array([1.5, -0.2]))
    out_path = Path(
        save_model(
            model,
            path=tmp_path,
            filename="mymodel.joblib",
            metadata={"site": "MY1", "split": "blocked"},
        )
    )
    assert out_path.exists()
    meta_file = Path(f"{out_path}.meta.json")
    assert meta_file.exists()

    raw_meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert raw_meta["features"] == ["ws", "temp"]
    assert "normet_version" in raw_meta
    assert "saved_at" in raw_meta
    assert raw_meta["metadata"] == {"site": "MY1", "split": "blocked"}

    loaded_meta = load_model_metadata(tmp_path, filename="mymodel.joblib")
    assert loaded_meta == raw_meta

    loaded_model, meta = load_model(tmp_path, filename="mymodel.joblib", return_metadata=True)
    assert model_feature_names(loaded_model) == ["ws", "temp"]
    assert meta["metadata"]["site"] == "MY1"


# ---------------------------------------------------------------------------
# Layer 3: Metrics, Exceedance, Compare Models, CV, Probabilistic Verification
# ---------------------------------------------------------------------------


def test_stats_and_modstats_extended_metrics_effective_n_weekday_and_log() -> None:
    rng = np.random.default_rng(7)
    n = 168
    dates = pd.date_range("2023-01-02", periods=n, freq="h")
    obs = 20.0 + 5.0 * np.sin(np.arange(n) / 6.0) + rng.normal(0, 0.5, n)
    mod = obs * 0.9 + 1.5

    df = pd.DataFrame(
        {
            "date": dates,
            "value": obs,
            "value_predict": mod,
            "ws": np.ones(n),
            "set": ["testing"] * n,
        }
    )

    stats = Stats(
        df,
        mod="value_predict",
        obs="value",
        statistic=[
            "RMSE",
            "NSE",
            "R2_oos",
            "KGE",
            "NRMSE",
            "FB",
            "FE",
            "MG",
            "VG",
            "Spearman",
            "slope",
            "intercept",
            "r",
        ],
        effective_n=False,
    )
    for col in [
        "NSE",
        "R2_oos",
        "KGE",
        "NRMSE",
        "FB",
        "FE",
        "MG",
        "VG",
        "Spearman",
        "slope",
        "intercept",
    ]:
        assert col in stats.columns
        assert np.isfinite(stats.loc[0, col])
    assert stats.loc[0, "slope"] == pytest.approx(0.9, rel=1e-6)
    assert stats.loc[0, "intercept"] == pytest.approx(1.5, rel=1e-6)

    # Autocorrelated AR(1) pair: effective_n=True yields less significant p_level than naive
    rng_ac = np.random.default_rng(123)
    n_ac = 120
    x_ac = np.zeros(n_ac)
    y_ac = np.zeros(n_ac)
    for t in range(1, n_ac):
        e1 = rng_ac.normal()
        e2 = rng_ac.normal()
        x_ac[t] = 0.92 * x_ac[t - 1] + e1
        y_ac[t] = 0.92 * y_ac[t - 1] + 0.35 * e1 + 0.94 * e2
    df_ac = pd.DataFrame({"mod": x_ac, "obs": y_ac})
    s_naive = Stats(df_ac, mod="mod", obs="obs", statistic=["r"], effective_n=False)
    s_eff = Stats(df_ac, mod="mod", obs="obs", statistic=["r"], effective_n=True)
    sig_rank = {"": 0, "+": 1, "*": 2, "**": 3, "***": 4}
    assert sig_rank[str(s_eff.loc[0, "p_level"])] < sig_rank[str(s_naive.loc[0, "p_level"])]

    model = _SimpleLinearModel(["ws"], np.array([np.log1p(25.0)]))
    df_log = df.copy()
    df_log["value"] = np.log1p(obs)
    ms_wd = modStats(
        df_log,
        model,
        subset="testing",
        by="weekday",
        log_transform=True,
        back_transform="smearing",
        smearing_factor=1.05,
        effective_n=True,
    )
    assert {"1", "2", "3", "4", "5", "6", "7"}.issubset(set(ms_wd["weekday"].astype(str)))


def test_exceedance_stats_and_compare_models() -> None:
    rng = np.random.default_rng(12)
    n = 120
    dates = pd.date_range("2023-03-01", periods=n, freq="h")
    ws = rng.uniform(1.0, 10.0, n)
    true_y = 30.0 + 3.0 * ws
    df = pd.DataFrame(
        {
            "date": dates,
            "value": true_y,
            "ws": ws,
            "set": ["testing"] * n,
        }
    )
    model_a = _SimpleLinearModel(["ws"], np.array([3.0]), intercept=30.0 + 0.1)
    model_b = _SimpleLinearModel(["ws"], np.array([1.0]), intercept=40.0)

    ex = exceedance_stats(
        y_pred=ml_predict(model_a, df),
        y_true=true_y,
        threshold=45.0,
    )
    assert isinstance(ex, pd.DataFrame)
    for col in ["POD", "FAR", "CSI", "FBI", "HSS", "tail_RMSE"]:
        assert col in ex.columns

    cmp_res = compare_models(
        df,
        model_a,
        model_b,
        target="value",
        subset="testing",
        metric="RMSE",
        block_size=12,
        n_bootstrap=200,
        seed=42,
    )
    expected_keys = {
        "metric",
        "n",
        "score_a",
        "score_b",
        "diff",
        "ci_low",
        "ci_high",
        "dm_stat",
        "dm_pvalue",
        "summary",
    }
    assert expected_keys.issubset(cmp_res.keys())
    assert cmp_res["score_a"] < cmp_res["score_b"]
    # diff is score_b - score_a: positive when model A is better
    assert cmp_res["diff"] > 0
    assert cmp_res["dm_pvalue"] < 0.05


def test_cv_score_blocked_embargo_impute_log_pooled_and_oof() -> None:
    df = _make_hourly_df(90, seed=9)
    # Keep original positive scale; cv_score(log_transform=True) handles log1p internally
    df["value"] = np.maximum(df["value"], 1.0)
    df.loc[45, "ws"] = np.nan

    class _CvBackend:
        name = "cvrec"

        def train(
            self,
            df_train: pd.DataFrame,
            target: str = "value",
            covariates: list[str] | None = None,
            variables: list[str] | None = None,
            model_config: dict[str, Any] | None = None,
            seed: int = 1,
            verbose: bool = False,
            n_cores: int | None = None,
            **kwargs: Any,
        ) -> _SimpleLinearModel:
            covs = list(covariates or [])
            assert not df_train[covs].isna().any().any()
            return _SimpleLinearModel(
                covs,
                np.zeros(len(covs)),
                intercept=float(df_train[target].mean()),
                backend="cvrec",
            )

        def save(self, model: object, path: str = ".", filename: str = "automl.joblib") -> str:
            return str(Path(path) / filename)

        def load(self, path: str = ".", filename: str | None = None) -> object:
            return {}

    backend_registry.register(_CvBackend())
    try:
        res = cv_score(
            df,
            target="value",
            covariates=["ws", "temp"],
            backend="cvrec",
            n_splits=3,
            strategy="blocked",
            embargo=2,
            impute="train_median",
            log_transform=True,
            back_transform="smearing",
            include_pooled=True,
            return_oof=True,
        )
        assert isinstance(res, tuple)
        scores_df, oof_df = res
        assert isinstance(scores_df, pd.DataFrame)
        assert isinstance(oof_df, pd.DataFrame)
        assert (scores_df["fold"].astype(str) == "pooled").any()
        assert {"date", "observed", "predicted", "fold"}.issubset(oof_df.columns)
        assert len(oof_df) > 0
        assert np.isfinite(oof_df["predicted"]).all()
        assert (oof_df["predicted"] > 0).all()
    finally:
        backend_registry._backends.pop("cvrec", None)


def test_probabilistic_verification_suite() -> None:
    rng = np.random.default_rng(101)
    n, k = 300, 40
    mu = rng.normal(10.0, 1.5, size=n)
    sigma = np.full(n, 1.3)
    obs_cal = rng.normal(mu, sigma)
    samples_cal = rng.normal(mu[:, None], sigma[:, None], size=(n, k))
    samples_bad = rng.normal(mu[:, None] + 5.0, sigma[:, None] * 0.2, size=(n, k))

    crps_cal = fair_crps_from_samples(samples_cal, obs_cal)
    crps_bad = fair_crps_from_samples(samples_bad, obs_cal)
    assert np.isfinite(crps_cal) and crps_cal > 0.0
    assert crps_cal < crps_bad
    assert fair_crps_from_samples(np.zeros((5, 3)), np.zeros(5)) == pytest.approx(0.0)

    ss = crps_skill_score(crps_cal, crps_bad)
    assert 0.0 < ss < 1.0
    assert crps_skill_score(0.0, crps_bad) == pytest.approx(1.0)

    lower = mu - 1.645 * sigma
    upper = mu + 1.645 * sigma
    is_sharp = interval_score(lower, upper, obs_cal, alpha=0.1)
    is_miss = interval_score(lower + 10.0, upper + 10.0, obs_cal, alpha=0.1)
    assert is_sharp < is_miss

    cov = interval_coverage(lower=lower, upper=upper, obs=obs_cal, alpha=0.1)
    assert "empirical_coverage" in cov.columns
    assert 0.80 <= float(cov.loc[0, "empirical_coverage"]) <= 0.99

    q_df = pd.DataFrame({"q050": lower, "q950": upper})
    cov_pairs = interval_coverage(preds=q_df, obs=obs_cal, pairs=[("q050", "q950", 0.10)])
    assert float(cov_pairs.loc[0, "nominal_coverage"]) == pytest.approx(0.90)
    assert 0.80 <= float(cov_pairs.loc[0, "empirical_coverage"]) <= 0.99

    cov_auto = interval_coverage(preds=q_df, obs=obs_cal)
    assert float(cov_auto.loc[0, "nominal_coverage"]) == pytest.approx(0.90)
    assert float(cov_auto.loc[0, "empirical_coverage"]) == pytest.approx(
        float(cov_pairs.loc[0, "empirical_coverage"])
    )

    pit_good = pit_diagnostics(samples_cal, obs_cal, n_bins=10, seed=42)
    pit_under = pit_diagnostics(samples_bad, obs_cal, n_bins=10, seed=42)
    assert pit_good["ks_pvalue"] > 0.01
    assert pit_under["ks_pvalue"] < 0.01

    y_pred = mu
    y_true = mu + rng.normal(0, 0.5, size=n)
    cal = calibration_curve(y_pred, y_true, n_bins=8, strategy="quantile")
    assert isinstance(cal, pd.DataFrame)
    assert len(cal) == 8
    assert np.allclose(cal["pred_mean"], cal["obs_mean"], atol=0.6)


def test_top_level_v2_exports() -> None:
    expected = [
        "screen_aq_data",
        "add_met_physics_features",
        "model_feature_names",
        "permutation_importance",
        "load_model_metadata",
        "exceedance_stats",
        "compare_models",
        "fair_crps_from_samples",
        "crps_skill_score",
        "interval_score",
        "interval_coverage",
        "pit_diagnostics",
        "calibration_curve",
    ]
    for name in expected:
        assert hasattr(normet, name), f"Missing top-level export: {name}"
    assert hasattr(normet.utils, "DEFAULT_MET_BOUNDS")
