"""Torch-free tests for Chronos-2 core primitives and first-principles operators.

Covers:
  - ``past_only_covariates`` across inference, rolling counterfactuals, and training inputs
  - Native probabilistic rollout (``counterfactual(..., rollout="quantile")``)
  - ``Chronos2Estimator.save()`` / ``load()`` persistence and ``finetune_multisite()``
  - ``chronos_scm`` / ``Chronos2Estimator.synthetic_control()`` + ``to_scm_frame()``
  - ``chronos_anomaly_scores`` and ``detect_chronos_events``
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import normet as nm
from normet.foundation import (
    AdaptiveConformalCalibrator,
    Chronos2Estimator,
    ChronosEmbedder,
    InsufficientContextError,
    calibrate_quantiles,
    chronos_anomaly_scores,
    chronos_attention_attribution,
    chronos_impute,
    chronos_pdp,
    chronos_scm,
    detect_chronos_events,
    select_donors_by_embedding,
)

LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def _stub_estimator(
    *,
    context_length: int = 96,
    prediction_length: int = 48,
    spread: float = 2.0,
    slope_fn: Any = None,
    past_only_covariates: list[str] | None = None,
) -> tuple[Chronos2Estimator, list[dict[str, Any]]]:
    """Build a torch-free estimator backed by a deterministic covariate-aware stub."""
    calls: list[dict[str, Any]] = []

    class _FakePipe:
        quantiles = list(LEVELS)

        def predict(
            self,
            inputs: list[dict[str, Any]],
            prediction_length: int,
            cross_learning: bool = False,
            **extra: Any,
        ) -> list[np.ndarray]:
            calls.append(
                {
                    "inputs": inputs,
                    "prediction_length": prediction_length,
                    "cross_learning": cross_learning,
                    "extra": extra,
                }
            )
            out: list[np.ndarray] = []
            for item in inputs:
                hist = np.asarray(item["target"], dtype=float)
                base = float(np.nanmedian(hist))
                fc = item.get("future_covariates") or {}
                if slope_fn is not None:
                    med = np.asarray(
                        slope_fn(hist, item.get("past_covariates") or {}, fc), dtype=float
                    )
                else:
                    # Combine any known-future numeric covariates linearly around their past mean
                    delta = np.zeros(prediction_length, dtype=float)
                    pc = item.get("past_covariates") or {}
                    for k, fvals in fc.items():
                        arr = np.asarray(fvals)
                        if np.issubdtype(arr.dtype, np.number):
                            pmean = float(np.nanmean(np.asarray(pc[k], dtype=float)))
                            delta += arr.astype(float) - pmean
                    med = base + delta
                block = np.vstack([med + 2.0 * spread * (q - 0.5) for q in LEVELS])
                out.append(block[None, ...])
            return out

        def save_pretrained(self, save_directory: str | Path) -> None:
            p = Path(save_directory)
            p.mkdir(parents=True, exist_ok=True)
            (p / "config.json").write_text('{"architectures": ["Chronos2Model"]}', encoding="utf-8")

    est = Chronos2Estimator(
        context_length=context_length,
        prediction_length=prediction_length,
        device="cpu",
        past_only_covariates=past_only_covariates,
    )
    est._pipeline = _FakePipe()
    est._quantiles = np.asarray(LEVELS, dtype=float)
    return est, calls


def _hourly_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    ws = 2.0 + np.abs(rng.normal(2.0, 0.8, n))
    co = 0.5 + 0.1 * rng.standard_normal(n)
    site_type = np.where(np.arange(n) % 2 == 0, "urban", "rural")
    value = 40.0 - 3.0 * ws + rng.normal(0, 0.5, n)
    return pd.DataFrame({"value": value, "ws": ws, "co": co, "site_type": site_type}, index=idx)


# ------------------------------------------------------- past_only_covariates


def test_past_only_covariates_channel_and_validation() -> None:
    df = _hourly_frame(300)
    est, calls = _stub_estimator(
        context_length=48, prediction_length=24, past_only_covariates=["co"]
    )

    cols = est._resolve_covariates(df, "value")
    assert "ws" in cols and "site_type" in cols
    assert "co" not in cols

    est.predict_quantiles(df, "value", anchor=df.index[200], horizon=24)
    sent = calls[-1]["inputs"][0]
    assert "co" in sent["past_covariates"]
    assert "co" not in sent["future_covariates"]
    assert "ws" in sent["future_covariates"]

    with pytest.raises(ValueError, match="both known-future and past-only"):
        est.predict_quantiles(
            df, "value", anchor=df.index[200], covariates=["ws", "co"], past_only_covariates=["co"]
        )
    with pytest.raises(KeyError, match="absent from frame"):
        est.predict_quantiles(df, "value", anchor=df.index[200], past_only_covariates=["missing"])


def test_past_only_covariates_do_not_leak_in_roll_forward() -> None:
    df = _hourly_frame(420)
    # Inject a huge post-intervention jump into the past-only column `co`
    interv = df.index[320]
    df.loc[df.index >= interv, "co"] = 9999.0

    est, calls = _stub_estimator(
        context_length=48, prediction_length=24, past_only_covariates=["co"]
    )
    est.counterfactual(
        df,
        "value",
        interv,
        features=["ws"],
        validation_hours=48,
        past_only_covariates=["co"],
    )
    # Inspect the last counterfactual block (which rolled past the intervention)
    last_inp = calls[-1]["inputs"][0]
    assert "co" not in last_inp["future_covariates"]
    assert np.max(last_inp["past_covariates"]["co"]) < 100.0


def test_training_inputs_exclude_past_only_from_known_future(monkeypatch) -> None:
    recorded: list[tuple[list[dict[str, Any]], int, list[str]]] = []

    fake_preprocess = types.ModuleType("chronos.chronos2.preprocess")

    def _from_list_of_dicts(
        data: list[dict[str, Any]], *, prediction_length: int, known_covariates_names: list[str]
    ) -> list[dict[str, Any]]:
        recorded.append((data, prediction_length, known_covariates_names))
        return [{"n_items": len(data)}]

    fake_preprocess.from_list_of_dicts = _from_list_of_dicts  # type: ignore[attr-defined]
    fake_c2 = types.ModuleType("chronos.chronos2")
    fake_c2.preprocess = fake_preprocess  # type: ignore[attr-defined]
    fake_chronos = types.ModuleType("chronos")
    fake_chronos.chronos2 = fake_c2  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "chronos", fake_chronos)
    monkeypatch.setitem(sys.modules, "chronos.chronos2", fake_c2)
    monkeypatch.setitem(sys.modules, "chronos.chronos2.preprocess", fake_preprocess)

    df = _hourly_frame(200)
    est = Chronos2Estimator(use_calendar=False)
    est._training_inputs(df, "value", ["ws"], horizon=12, past_only_cols=["co"])

    assert len(recorded) == 1
    items, horizon, known = recorded[0]
    assert horizon == 12
    assert known == ["ws"]
    assert set(items[0]["past_covariates"]) == {"ws", "co"}


# ------------------------------------------------------- rollout & persistence


def test_counterfactual_quantile_rollout_and_calibration() -> None:
    df = _hourly_frame(500)
    interv = df.index[400]
    est, calls = _stub_estimator(context_length=96, prediction_length=24, spread=0.2)

    res = est.counterfactual(
        df,
        "value",
        interv,
        features=["ws"],
        validation_hours=72,
        calibrated=True,
        rollout="quantile",
    )
    assert res.summary["rollout"] == "quantile"
    assert res.calibration is not None
    # Two native rollout calls (placebo validation + post-intervention) passed limit_prediction_length=False
    native_calls = [c for c in calls if c["extra"].get("limit_prediction_length") is False]
    assert [c["prediction_length"] for c in native_calls] == [72, 100]

    with pytest.raises(ValueError, match="rollout"):
        est.counterfactual(df, "value", interv, features=["ws"], rollout="bogus")


def test_save_and_load_roundtrip_and_bare_dir(monkeypatch, tmp_path) -> None:
    unloaded = Chronos2Estimator()
    with pytest.raises(RuntimeError, match="No Chronos-2 pipeline"):
        unloaded.save(tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        Chronos2Estimator.load(tmp_path / "does_not_exist")

    est, _ = _stub_estimator(
        context_length=128,
        prediction_length=24,
        past_only_covariates=["co"],
    )
    est.met_covariates = ["ws"]
    est.target_col = "NO2"
    est.feature_cols = ["ws"]

    out_dir = est.save(tmp_path / "ckpt")
    assert (out_dir / "normet_estimator.json").exists()
    meta = json.loads((out_dir / "normet_estimator.json").read_text(encoding="utf-8"))
    assert meta["past_only_covariates"] == ["co"]
    assert meta["met_covariates"] == ["ws"]

    import normet.foundation.estimator as est_mod

    fake_pipe = est._pipeline
    monkeypatch.setattr(est_mod, "resolve_device", lambda req=None: req or "cpu")
    monkeypatch.setattr(est_mod, "_load_pipeline_on", lambda name, dev, auto: (fake_pipe, dev))

    loaded = Chronos2Estimator.load(out_dir, device="cpu")
    assert loaded._pipeline is fake_pipe
    assert loaded.context_length == 128
    assert loaded.prediction_length == 24
    assert loaded.met_covariates == ["ws"]
    assert loaded.past_only_covariates == ["co"]
    assert loaded.target_col == "NO2"
    assert loaded.feature_cols == ["ws"]

    # Bare HuggingFace / LoRA directory without normet_estimator.json
    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()
    bare = Chronos2Estimator.load(bare_dir)
    assert bare.context_length == 2048
    assert bare.device == "cpu"


def test_finetune_multisite_contracts(monkeypatch, tmp_path) -> None:
    import importlib.util

    est, _ = _stub_estimator(context_length=48, prediction_length=12)
    frames = {"S1": _hourly_frame(200, seed=1), "S2": _hourly_frame(220, seed=2)}

    with pytest.raises(ValueError, match="frames is empty"):
        est.finetune_multisite({}, "value", mode="full")
    with pytest.raises(ValueError, match="mode must be"):
        est.finetune_multisite(frames, "value", mode="bogus")

    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None if name == "peft" else object()
    )
    with pytest.raises(ImportError, match="requires peft"):
        est.finetune_multisite(frames, "value", mode="lora")

    # Torch-free stub of preprocess + pipe.fit
    fake_preprocess = types.ModuleType("chronos.chronos2.preprocess")
    fake_preprocess.from_list_of_dicts = (  # type: ignore[attr-defined]
        lambda data, **kw: [{"n_series": len(data), **kw}]
    )
    fake_c2 = types.ModuleType("chronos.chronos2")
    fake_c2.preprocess = fake_preprocess  # type: ignore[attr-defined]
    fake_chronos = types.ModuleType("chronos")
    fake_chronos.chronos2 = fake_c2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chronos", fake_chronos)
    monkeypatch.setitem(sys.modules, "chronos.chronos2", fake_c2)
    monkeypatch.setitem(sys.modules, "chronos.chronos2.preprocess", fake_preprocess)

    fit_calls: list[dict[str, Any]] = []
    tuned_pipe = types.SimpleNamespace(quantiles=list(LEVELS))

    def _fit(inputs: Any, **kwargs: Any) -> Any:
        fit_calls.append({"inputs": inputs, **kwargs})
        return tuned_pipe

    est._pipeline.fit = _fit
    orig_pipe = est._pipeline
    tuned = est.finetune_multisite(
        frames,
        "value",
        mode="full",
        covariates=["ws"],
        past_only_covariates=["co"],
        validation_frames=[frames["S1"]],
        num_steps=3,
        output_dir=tmp_path,
    )
    assert est._pipeline is orig_pipe
    assert tuned._pipeline is tuned_pipe
    assert tuned.met_covariates == ["ws"]
    assert tuned.past_only_covariates == ["co"]
    assert fit_calls[0]["inputs"][0]["n_series"] == 2
    assert fit_calls[0]["inputs"][0]["known_covariates_names"] == ["ws"]
    assert fit_calls[0]["validation_inputs"][0]["n_series"] == 1


# ------------------------------------------------------- chronos_scm


def _panel_data(n: int = 320, freq: str = "h", seed: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq=freq)
    donor_a = 30.0 + 4.0 * np.sin(np.arange(n) / 6.0) + rng.normal(0, 0.3, n)
    donor_b = 20.0 + 2.0 * np.cos(np.arange(n) / 8.0) + rng.normal(0, 0.3, n)
    ws = 3.0 + rng.normal(0, 0.5, n)
    co = 0.4 + rng.normal(0, 0.05, n)
    treated = 10.0 + donor_a + 0.5 * donor_b - 2.0 * ws
    cut_idx = int(n * 0.75)
    treated[cut_idx:] -= 8.0  # policy reduction

    long_panel = pd.concat(
        [
            pd.DataFrame({"date": dates, "ID": "MY1", "value": treated}),
            pd.DataFrame({"date": dates, "ID": "D1", "value": donor_a}),
            pd.DataFrame({"date": dates, "ID": "D2", "value": donor_b}),
        ],
        ignore_index=True,
    )
    met_df = pd.DataFrame({"ws": ws, "co": co}, index=dates)
    return long_panel, met_df


def test_chronos_scm_hourly_with_met_and_to_scm_frame() -> None:
    panel, met_df = _panel_data(n=360, freq="h")
    est, calls = _stub_estimator(context_length=96, prediction_length=24, spread=0.5)
    cutoff = pd.date_range("2020-01-01", periods=360, freq="h")[270]

    res = est.synthetic_control(
        panel,
        treated_unit="MY1",
        cutoff_date=cutoff,
        met_df=met_df,
        met_covariates=["ws"],
        past_only_covariates=["co"],
        calibrated=True,
        rollout="quantile",
    )
    assert res.summary["method"] == "chronos_scm"
    assert res.summary["treated_unit"] == "MY1"
    assert res.summary["n_donors"] == 2
    assert res.summary["met_covariates"] == ["ws"]
    assert res.calibration is not None

    # Last native call receives donors + met in future_covariates, NOT past-only `co`
    native_call = [c for c in calls if c["extra"].get("limit_prediction_length") is False][-1]
    inp = native_call["inputs"][0]
    assert set(inp["future_covariates"]) == {"D1", "D2", "ws"}
    assert "co" in inp["past_covariates"]

    scm_df = res.to_scm_frame()
    assert list(scm_df.columns) == ["observed", "synthetic", "effect", "lower", "upper"]
    assert scm_df.index.name == "date"
    assert scm_df.loc[scm_df.index < cutoff, "synthetic"].isna().all()
    assert scm_df.loc[scm_df.index >= cutoff, "effect"].mean() < -4.0


def test_chronos_scm_monthly_panel_and_guards() -> None:
    panel, met_df = _panel_data(n=80, freq="MS")
    est, _ = _stub_estimator(context_length=24, prediction_length=12)
    cutoff = pd.date_range("2020-01-01", periods=80, freq="MS")[60]

    res = chronos_scm(est, panel, treated_unit="MY1", cutoff_date=cutoff)
    assert len(res.counterfactual_p50) == 20
    assert res.summary["validation_hours"] == 12

    with pytest.raises(ValueError, match="Treated unit"):
        chronos_scm(est, panel, treated_unit="MISSING", cutoff_date=cutoff)
    with pytest.raises(ValueError, match="Donor unit"):
        chronos_scm(est, panel, treated_unit="MY1", donors=["NOPE"], cutoff_date=cutoff)
    with pytest.raises(ValueError, match="collide"):
        bad_met = met_df.rename(columns={"ws": "D1"})
        chronos_scm(est, panel, treated_unit="MY1", cutoff_date=cutoff, met_df=bad_met)


# ------------------------------------------------------- anomaly detection


def test_chronos_anomaly_scores_and_detect_events() -> None:
    # Construct meteorology-driven signal y = 20 + 10 * stagnation (weather) + anomalies
    rng = np.random.default_rng(21)
    n = 300
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    stagnation = np.sin(2.0 * np.pi * np.arange(n) / 24.0)
    # Normal stagnant high concentration around hour 150 (pure weather)
    y = 20.0 + 10.0 * stagnation + 0.1 * rng.standard_normal(n)
    # Inject a 4-hour emission spike (+18) at [180:184] and a 3-hour drop (-18) at [240:243]
    y[180:184] += 18.0
    y[240:243] -= 18.0
    df = pd.DataFrame({"value": y, "stag": stagnation}, index=idx)

    # Exact linear weather model in stub: expected = 20 + 10 * stag, spread = 2.0
    est, _ = _stub_estimator(
        context_length=48,
        prediction_length=24,
        spread=2.0,
        slope_fn=lambda hist, pc, fc: 20.0 + 10.0 * np.asarray(fc["stag"], dtype=float),
    )

    scores = est.anomaly_scores(df, "value", covariates=["stag"])
    assert list(scores.columns) == [
        "observed",
        "expected",
        "lower",
        "upper",
        "pit",
        "tail_p",
        "surprise",
        "z_score",
    ]
    assert scores.iloc[:48]["pit"].isna().all()
    # Weather-driven peak at hour 150 is expected -> low surprise, near-zero z-score
    assert scores.iloc[150]["tail_p"] > 0.5
    assert abs(scores.iloc[150]["z_score"]) < 0.3

    # Injected spike / drop hit extreme tails
    assert (scores.iloc[180:184]["tail_p"] <= 0.01).all()
    assert (scores.iloc[180:184]["z_score"] > 5.0).all()
    assert (scores.iloc[240:243]["z_score"] < -5.0).all()

    events = est.detect_events(df, "value", covariates=["stag"], alpha=0.05, min_length=3)
    assert len(events) == 2
    assert set(events["direction"]) == {"high", "low"}
    high_row = events[events["direction"] == "high"].iloc[0]
    low_row = events[events["direction"] == "low"].iloc[0]
    assert high_row["start"] == idx[180] and high_row["end"] == idx[183] and high_row["n"] == 4
    assert low_row["start"] == idx[240] and low_row["end"] == idx[242] and low_row["n"] == 3

    only_high = detect_chronos_events(
        est, df, "value", covariates=["stag"], alpha=0.05, min_length=3, direction="high"
    )
    assert len(only_high) == 1 and only_high.iloc[0]["direction"] == "high"

    # Calibration widening scales down z-score and raises tail_p
    wide_est = calibrate_quantiles(est, df.iloc[:144], "value", df.index[144], covariates=["stag"])
    # Inflate threshold manually to verify calibration integration cleanly suppresses borderline noise
    wide_est.calibration.threshold = [12.0] * len(wide_est.calibration.threshold)
    cal_scores = chronos_anomaly_scores(wide_est, df, "value", covariates=["stag"])
    assert np.nanmax(cal_scores["z_score"].abs()) < np.nanmax(scores["z_score"].abs())
    assert cal_scores.iloc[180]["tail_p"] > scores.iloc[180]["tail_p"]
    assert cal_scores.iloc[240]["tail_p"] > scores.iloc[240]["tail_p"]

    # Top-level re-exports match foundation exports
    assert nm.chronos_scm is chronos_scm
    assert nm.chronos_anomaly_scores is chronos_anomaly_scores
    assert nm.detect_chronos_events is detect_chronos_events

    with pytest.raises(InsufficientContextError):
        chronos_anomaly_scores(est, df.iloc[:30], "value", covariates=["stag"])


# ------------------------------------------------------- embeddings & ACI


def test_select_donors_by_embedding_zero_leakage_and_ranking() -> None:
    n = 160
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    t_pre = 20.0 + 5.0 * np.sin(np.arange(n) / 8.0)
    d_close = t_pre * 1.01
    d_far = np.linspace(50.0, 10.0, n)
    d_low_cov = np.full(n, np.nan)
    d_low_cov[:12] = 15.0  # < 10% coverage over 120 pre rows when >120? wait: 120 pre rows -> 10%

    panel = pd.concat(
        [
            pd.DataFrame({"date": dates, "ID": "T", "value": t_pre}),
            pd.DataFrame({"date": dates, "ID": "D_CLOSE", "value": d_close}),
            pd.DataFrame({"date": dates, "ID": "D_FAR", "value": d_far}),
            pd.DataFrame({"date": dates, "ID": "D_SPARSE", "value": d_low_cov}),
        ],
        ignore_index=True,
    )
    # Heavily corrupt post-cutoff period of D_CLOSE to confirm strictly pre-cutoff slicing
    cutoff = dates[120]
    panel.loc[(panel["ID"] == "D_CLOSE") & (panel["date"] >= cutoff), "value"] = -99999.0
    panel.loc[(panel["ID"] == "T") & (panel["date"] >= cutoff), "value"] = 99999.0

    class _FakeSigPipe:
        def embed(self, windows, batch_size=32):
            embs = []
            for w in windows:
                arr = np.asarray(w, dtype=float)
                ok = arr[np.isfinite(arr)]
                # Project mean, std, early-half vs late-half delta onto 4D latent patch tensor
                mid = max(1, len(ok) // 2)
                vec = np.array(
                    [
                        float(np.mean(ok)),
                        float(np.std(ok)),
                        float(np.mean(ok[mid:]) - np.mean(ok[:mid])),
                        float(ok[-1] - ok[0]),
                    ],
                    dtype=np.float32,
                )
                embs.append(np.tile(vec, (1, 4, 1)))
            return embs, None

    emb = ChronosEmbedder(min_coverage=0.15)
    emb._pipeline = _FakeSigPipe()

    ranked = select_donors_by_embedding(
        panel,
        treated_unit="T",
        cutoff_date=cutoff,
        embedder=emb,
        context_length=120,
        top_k=1,
        min_similarity=0.9,
    )
    assert list(ranked.columns) == ["unit", "similarity", "rank", "selected"]
    assert "D_SPARSE" not in set(ranked["unit"])
    assert ranked.iloc[0]["unit"] == "D_CLOSE"
    assert ranked.iloc[0]["rank"] == 1
    assert bool(ranked.iloc[0]["selected"]) is True
    assert bool(ranked.iloc[1]["selected"]) is False
    assert float(ranked.iloc[0]["similarity"]) > 0.999


def test_adaptive_conformal_calibrator_and_anomaly_integration() -> None:
    rng = np.random.default_rng(13)
    n = 240
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    expected = pd.Series(np.zeros(n), index=idx)
    lower = pd.Series(np.full(n, -1.0), index=idx)
    upper = pd.Series(np.full(n, 1.0), index=idx)
    obs_vals = np.concatenate([rng.normal(0.0, 0.5, 100), rng.normal(0.0, 3.0, n - 100)])
    observed = pd.Series(obs_vals, index=idx)

    aci = AdaptiveConformalCalibrator(alpha=0.2, gamma=0.1, window=48)
    cal = aci.calibrate_series(observed, expected, lower, upper)
    assert list(cal.columns) == [
        "lower_cal",
        "upper_cal",
        "scale",
        "alpha_t",
        "covered",
        "running_coverage",
    ]
    assert float(cal["scale"].iloc[0]) == pytest.approx(1.0)
    assert float(cal["scale"].iloc[150:].mean()) > float(cal["scale"].iloc[20:90].mean())
    assert 0.55 <= float(cal["running_coverage"].iloc[-1]) <= 0.95

    df = _hourly_frame(260)
    est, _ = _stub_estimator(context_length=48, prediction_length=24, spread=1.0)
    plain = chronos_anomaly_scores(est, df, "value", covariates=["ws"], adaptive=False)
    assert "aci_scale" not in plain.columns
    adap = est.anomaly_scores(
        df, "value", covariates=["ws"], adaptive=True, aci_gamma=0.1, aci_window=48
    )
    assert "aci_scale" in adap.columns and "aci_alpha" in adap.columns
    assert np.isfinite(adap["aci_scale"].dropna()).all()


# ------------------------------------------------------- full-stack integration


def test_run_scm_chronos_backend_and_placebo_in_space() -> None:
    dates = pd.date_range("2022-01-01", periods=160, freq="D")
    rng = np.random.default_rng(17)
    d1 = 20.0 + 3.0 * np.sin(np.arange(160) / 5.0) + rng.normal(0, 0.2, 160)
    d2 = 15.0 + 2.0 * np.cos(np.arange(160) / 7.0) + rng.normal(0, 0.2, 160)
    d3 = 18.0 + 1.5 * np.sin(np.arange(160) / 9.0) + rng.normal(0, 0.2, 160)
    treated = 1.0 * d1 + 0.5 * d2 + 0.3 * d3 + rng.normal(0, 0.2, 160)
    cutoff = dates[120]
    treated[dates >= cutoff] -= 10.0

    panel = pd.concat(
        [
            pd.DataFrame({"date": dates, "ID": "T", "value": treated}),
            pd.DataFrame({"date": dates, "ID": "D1", "value": d1}),
            pd.DataFrame({"date": dates, "ID": "D2", "value": d2}),
            pd.DataFrame({"date": dates, "ID": "D3", "value": d3}),
        ],
        ignore_index=True,
    )
    est, _ = _stub_estimator(context_length=40, prediction_length=20, spread=0.5)
    out = nm.run_scm(
        df=panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date=cutoff.strftime("%Y-%m-%d"),
        donors=None,
        scm_backend="chronos",
        estimator=est,
        validation_steps=10,
    )
    assert {"observed", "synthetic", "effect", "lower", "upper"} <= set(out.columns)
    assert float(out.loc[out.index >= cutoff, "effect"].mean()) < -5.0

    plc = nm.placebo_in_space(
        df=panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date=cutoff.strftime("%Y-%m-%d"),
        scm_backend="chronos",
        estimator=est,
        validation_steps=10,
        n_cores=1,
    )
    assert set(plc["placebos"]) == {"D1", "D2", "D3"}
    assert 0.0 < float(plc["p_value"]) <= 1.0


def test_do_all_and_decompose_chronos_icn_integration(monkeypatch) -> None:
    rng = np.random.default_rng(29)
    n = 240
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    ws = 2.5 + np.abs(rng.normal(1.5, 0.5, n))
    t2m = 10.0 + 3.0 * np.sin(2.0 * np.pi * np.arange(n) / 24.0)
    pm25 = 40.0 - 3.0 * ws + 0.5 * t2m + rng.normal(0, 0.3, n)
    pm25[140:152] += 15.0
    raw = pd.DataFrame({"date": idx, "PM2.5": pm25, "ws": ws, "t2m": t2m})
    pool_df = raw[["date", "ws", "t2m"]].copy()

    est, _ = _stub_estimator(context_length=48, prediction_length=24, spread=1.0)
    fake_pipe = est._pipeline
    monkeypatch.setattr(Chronos2Estimator, "_load_pipeline", lambda self: fake_pipe)

    def _init_with_quantiles(self, *args, **kwargs):
        self._quantiles = np.asarray(LEVELS, dtype=float)

    orig_init = Chronos2Estimator.__init__

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._quantiles = np.asarray(LEVELS, dtype=float)

    monkeypatch.setattr(Chronos2Estimator, "__init__", _patched_init)

    out, model, _ = nm.do_all(
        raw,
        target="PM2.5",
        covariates=["ws", "t2m"],
        backend="chronos-2",
        model_config={
            "method": "icn",
            "estimate": "paired",
            "context_length": 48,
            "prediction_length": 24,
        },
        n_samples=2,
        log_transform=True,
    )
    assert isinstance(model, Chronos2Estimator)
    assert list(out.columns) == ["observed", "normalised", "q100", "q500", "q900"]
    assert (out["normalised"].dropna() >= 0.0).all()

    dec = nm.decompose(
        raw,
        target="PM2.5",
        method="meteorology",
        backend="chronos-2",
        covariates=["ws", "t2m"],
        model=est,
        model_config={
            "method": "icn",
            "estimate": "paired",
            "context_length": 48,
            "prediction_length": 24,
        },
        resample_df=pool_df,
        n_samples=2,
    )
    assert {"observed", "emi_total", "met_total", "ws", "t2m"} <= set(dec.columns)
    assert np.isfinite(dec["met_total"].dropna()).all()
    np.testing.assert_allclose(
        (dec["ws"] + dec["t2m"]).dropna().to_numpy(),
        dec["met_total"].dropna().to_numpy(),
        atol=1e-6,
    )


def test_events_chronos_method() -> None:
    rng = np.random.default_rng(33)
    n = 240
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    stag = np.sin(2.0 * np.pi * np.arange(n) / 24.0)
    y = 20.0 + 10.0 * stag + 0.1 * rng.standard_normal(n)
    y[160:164] += 18.0
    df = pd.DataFrame({"date": idx, "value": y, "stag": stag})
    est, _ = _stub_estimator(
        context_length=48,
        prediction_length=24,
        spread=2.0,
        slope_fn=lambda hist, pc, fc: 20.0 + 10.0 * np.asarray(fc["stag"], dtype=float),
    )

    scores = nm.anomaly_scores(
        df, value_col="value", method="chronos", estimator=est, covariates=["stag"]
    )
    assert scores.name == "score"
    assert int(np.nanargmax(scores.to_numpy())) in range(160, 164)

    events = nm.detect_events(
        df,
        value_col="value",
        method="chronos",
        estimator=est,
        covariates=["stag"],
        alpha=0.05,
        min_length=3,
    )
    assert len(events) == 1 and int(events.iloc[0]["n"]) == 4


# ------------------------------------------------------- attention, PDP, impute, finetune polish


def test_chronos_attention_attribution_torch_free() -> None:
    df = _hourly_frame(160, seed=7)
    est, _ = _stub_estimator(context_length=64, prediction_length=16, past_only_covariates=["co"])

    def _inspect_attentions(item: dict[str, Any], horizon: int):
        # preprocess.py row order: [target, *sorted(past_only), *sorted(known_future)]
        # i.e. [value, co, site_type, ws] -> n_series = 4
        n_ctx, n_out, n_heads, n_series = 4, 1, 2, 4
        seq_len = n_ctx + 1 + n_out
        g0 = np.zeros((seq_len, n_heads, n_series, n_series), dtype=np.float32)
        g1 = np.zeros_like(g0)
        # Forecast query row (idx -1), target (idx 0) attends heavily to ws (idx 3) in layer 0
        # and evenly to target (0), co (1), site_type (2), ws (3) in layer 1
        g0[-n_out:, :, 0, :] = [0.2, 0.1, 0.2, 0.5]
        g1[-n_out:, :, 0, :] = [0.4, 0.2, 0.1, 0.3]
        g0[:n_ctx, :, 0, :] = [0.5, 0.2, 0.1, 0.2]
        g1[:n_ctx, :, 0, :] = [0.5, 0.2, 0.1, 0.2]

        t0 = np.zeros((n_series, n_heads, seq_len, seq_len), dtype=np.float32)
        t1 = np.zeros_like(t0)
        t0[0, :, -n_out:, :n_ctx] = [0.1, 0.2, 0.3, 0.4]
        t1[0, :, -n_out:, :n_ctx] = [0.1, 0.2, 0.3, 0.4]
        return [g0, g1], [t0, t1], n_ctx, n_out

    est._pipeline.inspect_attentions = _inspect_attentions

    res = est.attention_attribution(df, "value", anchor=df.index[100], horizon=16)
    gw = res["group_weights"]
    tw = res["time_weights"]
    assert list(gw.columns) == ["variable", "role", "attention", "share", "context_attention"]
    assert gw["share"].sum() == pytest.approx(1.0)
    by_var = gw.set_index("variable")
    assert by_var.loc["value", "role"] == "target"
    assert by_var.loc["co", "role"] == "past_only"
    assert by_var.loc["ws", "role"] == "known_future"
    assert by_var.loc["ws", "share"] == pytest.approx(0.4)
    assert res["target_self_share"] == pytest.approx(0.3)
    assert res["covariate_share"] == pytest.approx(0.7)
    assert res["n_layers"] == 2 and res["layers"] == [0, 1]
    assert list(tw.columns) == ["patch_idx", "date", "attention", "share"]
    assert len(tw) == 4 and tw["share"].iloc[-1] > tw["share"].iloc[0]

    # Single / last layer selection
    res_last = chronos_attention_attribution(est, df, "value", anchor=df.index[100], layers="last")
    assert res_last["layers"] == [1]
    assert res_last["target_self_share"] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="out of range"):
        chronos_attention_attribution(est, df, "value", anchor=df.index[100], layers=5)


def test_chronos_pdp_and_top_level_export() -> None:
    df = _hourly_frame(200, seed=5)
    est, _ = _stub_estimator(
        context_length=48,
        prediction_length=12,
        spread=1.5,
        slope_fn=lambda hist, pc, fc: 30.0 - 2.5 * np.asarray(fc["ws"], dtype=float),
        past_only_covariates=["co"],
    )
    res = est.pdp(df, "value", var_list=["ws"], grid_points=6, n_anchors=3, horizon=12)
    assert list(res.columns) == [
        "variable",
        "value",
        "pdp_mean",
        "pdp_std",
        "pdp_lower",
        "pdp_upper",
    ]
    assert len(res) == 6
    assert (res["pdp_lower"] <= res["pdp_mean"]).all()
    assert (res["pdp_mean"] <= res["pdp_upper"]).all()
    # Linear slope -2.5 on ws
    diffs = np.diff(res["pdp_mean"].to_numpy()) / np.diff(res["value"].to_numpy())
    np.testing.assert_allclose(diffs, -2.5, rtol=1e-5)
    assert nm.chronos_pdp is chronos_pdp
    assert nm.chronos_attention_attribution is chronos_attention_attribution
    assert nm.chronos_impute is chronos_impute


def test_chronos_impute_bidirectional_and_boundaries() -> None:
    n = 160
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    ws = np.linspace(1.0, 5.0, n)
    true_y = 20.0 + 3.0 * ws
    df = pd.DataFrame({"value": true_y.copy(), "ws": ws}, index=idx)
    # Create: leading gap [0:5), interior long multi-chunk gap [60:80), trailing gap [155:160)
    df.iloc[0:5, df.columns.get_loc("value")] = np.nan
    df.iloc[60:80, df.columns.get_loc("value")] = np.nan
    df.iloc[155:160, df.columns.get_loc("value")] = np.nan

    est, _ = _stub_estimator(
        context_length=32,
        prediction_length=8,
        spread=1.0,
        slope_fn=lambda hist, pc, fc: 20.0 + 3.0 * np.asarray(fc["ws"], dtype=float),
    )

    out = est.impute(df, "value", covariates=["ws"], direction="bidirectional")
    assert list(out.columns) == ["observed", "imputed", "is_imputed", "lower", "upper"]
    assert out["is_imputed"].iloc[0:5].all()
    assert out["is_imputed"].iloc[60:80].all()
    assert out["is_imputed"].iloc[155:160].all()
    assert (~out["is_imputed"].iloc[5:60]).all()
    np.testing.assert_allclose(
        out.loc[out["is_imputed"], "imputed"], true_y[out["is_imputed"]], atol=1e-5
    )
    assert (out.loc[out["is_imputed"], "lower"] < out.loc[out["is_imputed"], "imputed"]).all()
    assert (out.loc[out["is_imputed"], "upper"] > out.loc[out["is_imputed"], "imputed"]).all()

    # max_gap=6 skips [60:80) while keeping 5-hour boundary gaps
    capped = chronos_impute(est, df, "value", covariates=["ws"], max_gap=6)
    assert (~capped["is_imputed"].iloc[60:80]).all()
    assert capped["is_imputed"].iloc[0:5].all()

    # No missing values -> zero forecast calls
    clean = pd.DataFrame({"value": true_y, "ws": ws}, index=idx)
    noop = est.impute(clean, "value", covariates=["ws"])
    assert not noop["is_imputed"].any()


def test_finetune_forwards_min_past_and_callbacks_and_validates(monkeypatch, tmp_path) -> None:
    est, _ = _stub_estimator(context_length=48, prediction_length=12)
    df = _hourly_frame(200, seed=1)

    with pytest.raises(ValueError, match="min_past"):
        est.finetune(df, "value", mode="full", min_past=0)

    fake_preprocess = types.ModuleType("chronos.chronos2.preprocess")
    fake_preprocess.from_list_of_dicts = lambda data, **kw: [{"n_series": len(data), **kw}]  # type: ignore[attr-defined]
    fake_c2 = types.ModuleType("chronos.chronos2")
    fake_c2.preprocess = fake_preprocess  # type: ignore[attr-defined]
    fake_chronos = types.ModuleType("chronos")
    fake_chronos.chronos2 = fake_c2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chronos", fake_chronos)
    monkeypatch.setitem(sys.modules, "chronos.chronos2", fake_c2)
    monkeypatch.setitem(sys.modules, "chronos.chronos2.preprocess", fake_preprocess)

    recorded: list[dict[str, Any]] = []
    est._pipeline.fit = lambda inputs, **kw: (
        recorded.append(kw),
        types.SimpleNamespace(quantiles=list(LEVELS)),
    )[1]
    cb = object()
    est.finetune(
        df,
        "value",
        mode="full",
        covariates=["ws"],
        min_past=24,
        callbacks=[cb],
        num_steps=2,
        output_dir=tmp_path,
    )
    assert recorded[0]["min_past"] == 24
    assert recorded[0]["callbacks"] == [cb]
