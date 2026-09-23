"""Tests for the Chronos-2 foundation estimator.

Split into two groups. The structural tests exercise covariate resolution,
calendar encoding and the guard rails, and need no model weights. The inference
tests are gated on ``chronos-forecasting`` plus ``torch`` being installed and hit
the real 119M-parameter checkpoint, so they use short contexts and horizons.

The covariate test is the one that matters most: a Chronos call without
covariates is a univariate forecast, and calling that "de-weathering" is exactly
the failure this module exists to prevent.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from normet.foundation import (
    Chronos2Estimator,
    CounterfactualResult,
    InsufficientContextError,
    IrregularIndexError,
    to_normet_frame,
    to_regular_index,
)
from normet.foundation.estimator import add_calendar_covariates


def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


needs_chronos = pytest.mark.skipif(
    not (_has("chronos") and _has("torch")),
    reason="chronos-forecasting/torch not installed",
)


@pytest.fixture(scope="session")
def loaded_pipeline(chronos2_pipeline):
    """The session-shared checkpoint, plus its native quantile levels."""
    return chronos2_pipeline, np.asarray(chronos2_pipeline.quantiles, dtype=float)


@pytest.fixture
def make_estimator(loaded_pipeline, chronos_device):
    """Build estimators that reuse the session-loaded pipeline."""
    pipeline, quantiles = loaded_pipeline

    def _factory(**kwargs) -> Chronos2Estimator:
        kwargs.setdefault("device", chronos_device)
        est = Chronos2Estimator(**kwargs)
        est._pipeline = pipeline
        est._quantiles = quantiles
        return est

    return _factory


@pytest.fixture
def met_frame() -> pd.DataFrame:
    """Hourly frame whose target is driven by wind speed plus a diurnal cycle."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=3000, freq="h")
    ws = 2.0 + np.abs(rng.normal(2.0, 1.0, len(idx)))
    blh = 300.0 + 500.0 * np.abs(np.sin(2 * np.pi * np.arange(len(idx)) / 24.0))
    diurnal = 20.0 * (1 + np.sin(2 * np.pi * idx.hour / 24.0))
    value = 60.0 / ws + diurnal + rng.normal(0, 2.0, len(idx))
    df = pd.DataFrame({"value": value, "ws": ws, "blh": blh}, index=idx)
    return add_calendar_covariates(df)


# --------------------------------------------------------------- structural


def test_rejects_chronos_one_checkpoints():
    """Chronos-1/Bolt silently drop covariates, so they must not be selectable."""
    with pytest.raises(ValueError, match="does not accept covariates"):
        Chronos2Estimator(model_name="amazon/chronos-t5-base")
    with pytest.raises(ValueError, match="does not accept covariates"):
        Chronos2Estimator(model_name="amazon/chronos-bolt-small")


def test_default_model_is_chronos_two():
    assert Chronos2Estimator().model_name == "amazon/chronos-2"


def test_batch_size_cannot_be_zero():
    """A zero or negative batch would make the chunking loop skip every draw."""
    assert Chronos2Estimator(batch_size=0).batch_size == 1
    assert Chronos2Estimator(batch_size=-4).batch_size == 1
    assert Chronos2Estimator().batch_size == 32


def test_calendar_covariates_are_cyclical(met_frame):
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"):
        assert col in met_frame
        assert np.isfinite(met_frame[col]).all()
        assert met_frame[col].abs().max() <= 1.0 + 1e-6
    # midnight and the following midnight encode identically
    assert met_frame["hour_sin"].iloc[0] == pytest.approx(met_frame["hour_sin"].iloc[24])


def test_calendar_covariates_require_datetime_index():
    with pytest.raises(TypeError, match="DatetimeIndex"):
        add_calendar_covariates(pd.DataFrame({"value": [1.0, 2.0]}))


def test_covariate_resolution_picks_up_met_and_calendar(met_frame):
    est = Chronos2Estimator()
    cols = est._resolve_covariates(met_frame, "value")
    assert "ws" in cols and "blh" in cols
    assert "hour_sin" in cols
    assert "value" not in cols


def test_explicit_covariates_are_validated(met_frame):
    est = Chronos2Estimator(met_covariates=["ws", "not_a_column"])
    with pytest.raises(KeyError, match="not_a_column"):
        est._resolve_covariates(met_frame, "value")


def test_fit_records_metadata_without_training(met_frame):
    est = Chronos2Estimator()
    out = est.fit(met_frame[["ws", "blh"]], met_frame["value"])
    assert out is est
    assert est.target_col == "value"
    assert est.feature_cols == ["ws", "blh"]


def test_counterfactual_rejects_insufficient_history(met_frame):
    est = Chronos2Estimator(context_length=2048)
    with pytest.raises(ValueError, match="pre-intervention hours"):
        est.counterfactual(met_frame, "value", "2020-01-05", validation_hours=720)


def test_counterfactual_rejects_empty_post_window(met_frame):
    est = Chronos2Estimator(context_length=64)
    with pytest.raises(ValueError, match="no post-intervention data"):
        est.counterfactual(met_frame, "value", met_frame.index[-1] + pd.Timedelta("1h"))


def test_counterfactual_result_to_dataframe_roundtrip():
    idx = pd.date_range("2020-01-01", periods=5, freq="h")
    s = pd.Series(np.arange(5.0), index=idx)
    res = CounterfactualResult(
        observed=s,
        counterfactual_p50=s,
        counterfactual_p10=s,
        counterfactual_p90=s,
        absolute_impact_p50=s * 0,
        relative_impact_pct_p50=s * 0,
        pre_intervention_bias_pct=1.5,
    )
    out = res.to_dataframe()
    assert list(out.columns) == [
        "observed",
        "counterfactual_p50",
        "counterfactual_p10",
        "counterfactual_p90",
        "impact_p50",
        "impact_pct_p50",
    ]
    assert len(out) == 5


# ---------------------------------------------------------------- inference


@needs_chronos
def test_native_quantile_levels_are_exposed(make_estimator):
    est = make_estimator()
    q = est.quantile_levels
    assert len(q) >= 9
    assert q.min() < 0.1 < 0.5 < 0.9 < q.max()


@needs_chronos
def test_predict_quantiles_are_ordered_and_aligned(met_frame, make_estimator):
    est = make_estimator(context_length=512, prediction_length=48)
    out = est.predict_quantiles(met_frame, "value", anchor=met_frame.index[1000], horizon=48)
    assert len(out) == 48
    assert (out.index == met_frame.index[1000:1048]).all()
    assert (out["q0.1"] <= out["q0.5"] + 1e-6).all()
    assert (out["q0.5"] <= out["q0.9"] + 1e-6).all()


@needs_chronos
def test_forecast_responds_to_meteorology(met_frame, make_estimator):
    """Permuting the future weather must move the forecast.

    If it does not, the model is autoregressing and no de-weathering it produces
    carries meteorological information.
    """
    est = make_estimator(context_length=512, prediction_length=48)
    sens = est.covariate_sensitivity(
        met_frame, "value", anchor=met_frame.index[1000], horizon=48, random_state=0
    )
    assert sens["mean_abs_shift"] > 0.1
    assert sens["pct_of_prediction"] > 1.0


@needs_chronos
def test_counterfactual_reports_its_own_pre_intervention_bias(met_frame, make_estimator):
    est = make_estimator(context_length=512, prediction_length=48)
    res = est.counterfactual(
        met_frame,
        "value",
        intervention_date=met_frame.index[2000],
        validation_hours=96,
        max_horizon=96,
    )
    assert np.isfinite(res.pre_intervention_bias_pct)
    # an untreated series should not produce a large apparent effect
    assert abs(res.summary["net_impact_pct"]) < 50.0
    assert res.summary["validation_hours"] == 96
    assert len(res.counterfactual_p50) == 96
    assert (res.counterfactual_p10 <= res.counterfactual_p90 + 1e-6).all()


@needs_chronos
def test_deweather_returns_expectation_over_resampled_weather(met_frame, make_estimator):
    est = make_estimator(context_length=512, prediction_length=96)
    small = met_frame.iloc[:800]
    out = est.deweather(small, "value", met_features=["ws", "blh"], n_samples=2, random_state=0)
    assert list(out.columns) == ["observed", "dew_p10", "dew_p50", "dew_p90"]
    assert len(out) == len(small)
    assert np.isfinite(out["dew_p50"]).all()
    # the seeded context region is passed through unchanged
    assert out["dew_p50"].iloc[:512].to_numpy() == pytest.approx(small["value"].to_numpy()[:512])


@needs_chronos
def test_batching_draws_does_not_change_the_answer(met_frame, make_estimator):
    """Draws sent together must give what draws sent one at a time give.

    Batching buys ~3x on a GPU and nothing on a single-threaded CPU, which is
    compute-bound rather than dispatch-bound; it is therefore only worth having
    if it leaves the numbers alone. It does: ``predict`` is deterministic, and
    the draws for a given ``random_state`` are taken in the same order whichever
    way they are sent, so the two runs differ only by float32 accumulation order.

    ``batch_size=3`` against ``n_samples=4`` also exercises a ragged final chunk.
    """
    small = met_frame.iloc[:800]
    kwargs = {"met_features": ["ws", "blh"], "n_samples": 4, "random_state": 0}
    one_at_a_time = make_estimator(context_length=512, prediction_length=96, batch_size=1)
    batched = make_estimator(context_length=512, prediction_length=96, batch_size=3)
    a = one_at_a_time.deweather(small, "value", **kwargs)["dew_p50"].to_numpy()
    b = batched.deweather(small, "value", **kwargs)["dew_p50"].to_numpy()
    assert a == pytest.approx(b, rel=1e-4)


@needs_chronos
def test_predict_batches_blocks_without_changing_them(met_frame, make_estimator):
    """``predict`` rolls over independent blocks, so batching them is safe.

    Each block conditions on observed history rather than on the previous
    block's output. The frame length here leaves a short final block, which has
    to leave the batch because a batch shares one horizon.
    """
    frame = met_frame.iloc[:700]  # 512 context + 48 + a 44 h tail
    one_at_a_time = make_estimator(context_length=512, prediction_length=48, batch_size=1)
    batched = make_estimator(context_length=512, prediction_length=48, batch_size=8)
    one_at_a_time.target_col = batched.target_col = "value"
    a = one_at_a_time.predict(frame)
    b = batched.predict(frame)
    assert len(a) == len(frame)
    assert np.isfinite(a).all()
    assert a[:512] == pytest.approx(frame["value"].to_numpy()[:512])  # seeded head
    assert a == pytest.approx(b, rel=1e-4)


def test_empty_context_is_refused_not_silently_projected(met_frame):
    """A context with no observed target must raise, not return a forecast.

    Chronos-2 masks NaNs rather than failing, so an all-missing conditioning
    window yields a forecast scaled to nothing -- observed at Belfast Centre,
    where a record gap across the window produced a business-as-usual level of
    0.25 ug/m3 against 26 ug/m3 observed, i.e. an apparent +10,000% effect.
    """
    est = Chronos2Estimator(context_length=512, prediction_length=48)
    blank = met_frame.copy()
    blank.loc[blank.index[:1500], "value"] = np.nan
    with pytest.raises(InsufficientContextError, match="finite"):
        est.predict_quantiles(blank, "value", anchor=blank.index[1000], horizon=48)


def test_partial_context_is_allowed(met_frame):
    """Ordinary gappy records must still project; only near-empty ones are refused."""
    est = Chronos2Estimator(context_length=512, prediction_length=48, min_context_coverage=0.25)
    gappy = met_frame.copy()
    gappy.loc[gappy.index[600:900], "value"] = np.nan  # ~60% of a 512 h context intact
    est._check_context(gappy["value"].to_numpy()[488:1000])


# ------------------------------------------------ record gaps on rolling paths
#
# A stub stands in for the forward pass: it forecasts the mean of the observed
# context, and it refuses -- exactly as the real one does -- any block whose
# context is too sparse. So these tests pin which blocks are left NaN without
# loading weights, and would fail if a sparse block ever reached the model.

GAP_L, GAP_STEP = 48, 24


def _stub_estimator() -> Chronos2Estimator:
    est = Chronos2Estimator(context_length=GAP_L, prediction_length=GAP_STEP, device="cpu")
    est._pipeline = object()
    est._quantiles = np.array([0.1, 0.5, 0.9])

    def forecast(items, cols, horizon):
        for hist, _, _ in items:
            est._check_context(hist)
        return np.stack([np.full((3, horizon), np.nanmean(hist)) for hist, _, _ in items])

    est._forecast_blocks = forecast  # type: ignore[method-assign]
    return est


def _gappy_frame(gap: tuple[int, int] = (100, 250), n: int = 400) -> pd.DataFrame:
    t = np.arange(n)
    value = 10.0 + np.sin(t / 5.0)
    value[gap[0] : gap[1]] = np.nan
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"value": value, "ws": 1.0 + t / n}, index=idx)


def _sparse_blocks(value: np.ndarray, cover: float = 0.25) -> list[tuple[int, int]]:
    """Blocks whose context has less than ``cover`` of the target observed."""
    spans = [(s, min(s + GAP_STEP, len(value))) for s in range(GAP_L, len(value), GAP_STEP)]
    return [(s, e) for s, e in spans if np.isfinite(value[max(0, s - GAP_L) : s]).mean() < cover]


def _nan_rows(value: np.ndarray, blocks: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(len(value), dtype=bool)
    for s, e in blocks:
        mask[s:e] = True
    return mask


def test_deweather_leaves_blocks_after_a_long_gap_nan(caplog):
    """A gap longer than the context used to abort the whole run."""
    df = _gappy_frame()
    sparse = _sparse_blocks(df["value"].to_numpy())
    assert len(sparse) == 5  # the blocks the 150 h gap starves of context

    with caplog.at_level("WARNING"):
        out = _stub_estimator().deweather(df, "value", n_samples=2, random_state=0)

    expected = _nan_rows(df["value"].to_numpy(), sparse)
    np.testing.assert_array_equal(out["dew_p50"].isna().to_numpy(), expected)
    assert any("5 of 15 de-weathering blocks" in r.getMessage() for r in caplog.records)


def test_predict_leaves_blocks_after_a_long_gap_nan():
    df = _gappy_frame()
    pred = _stub_estimator().predict(df)

    expected = _nan_rows(df["value"].to_numpy(), _sparse_blocks(df["value"].to_numpy()))
    np.testing.assert_array_equal(np.isnan(pred), expected)


def test_rolling_paths_refuse_when_no_block_has_context():
    df = _gappy_frame(gap=(10, 400))
    with pytest.raises(InsufficientContextError, match="nothing can be projected"):
        _stub_estimator().deweather(df, "value", n_samples=2, random_state=0)
    with pytest.raises(InsufficientContextError, match="nothing can be projected"):
        _stub_estimator().predict(df)


def test_a_record_without_gaps_skips_nothing(caplog):
    df = _gappy_frame(gap=(0, 0))
    with caplog.at_level("WARNING"):
        out = _stub_estimator().deweather(df, "value", n_samples=2, random_state=0)
    assert out["dew_p50"].notna().all()
    assert not [r for r in caplog.records if "left NaN" in r.getMessage()]


def test_zero_shot_decomposition_survives_a_long_gap():
    """Every coalition skips the same blocks, so the gap stays NaN and the rest closes."""
    from normet import decompose

    df = _gappy_frame().reset_index().rename(columns={"index": "date"})
    df["blh"] = 500.0 + np.arange(len(df))
    res = decompose(
        df,
        _stub_estimator(),
        target="value",
        method="meteorology",
        backend="chronos-2",
        covariates=["ws", "blh"],
        groups={"wind": ["ws"], "mixing": ["blh"]},
        n_samples=2,
    )

    expected = _nan_rows(df["value"].to_numpy(), _sparse_blocks(df["value"].to_numpy()))
    np.testing.assert_array_equal(res["emi_total"].isna().to_numpy(), expected)
    ok = ~expected & res["observed"].notna().to_numpy()
    np.testing.assert_allclose(
        (res["met_base"] + res["met_noise"] + res["wind"] + res["mixing"])[ok],
        res["met_total"][ok],
    )


@needs_chronos
def test_deweather_with_a_long_gap_on_the_real_model(met_frame, make_estimator):
    """Same contract on the checkpoint: no exception, NaN exactly where starved."""
    est = make_estimator(context_length=256, prediction_length=48)
    frame = met_frame.iloc[:900].copy()
    frame.loc[frame.index[400:700], "value"] = np.nan

    out = est.deweather(frame, "value", met_features=["ws", "blh"], n_samples=2, random_state=0)

    value = frame["value"].to_numpy()
    spans = [(s, min(s + 48, len(value))) for s in range(256, len(value), 48)]
    sparse = [(s, e) for s, e in spans if np.isfinite(value[max(0, s - 256) : s]).mean() < 0.25]
    assert sparse  # the gap does starve some blocks
    mask = np.zeros(len(value), dtype=bool)
    for s, e in sparse:
        mask[s:e] = True
    np.testing.assert_array_equal(out["dew_p50"].isna().to_numpy(), mask)


def test_irregular_index_is_refused(met_frame):
    """Dropped rows must raise: Chronos-2 reads position as time, not the timestamp.

    A frame that drops its missing hours is read as if those hours never existed,
    shifting every later point against its own calendar covariates. On the UK
    network this raised projected business-as-usual levels by roughly 40%.
    """
    est = Chronos2Estimator(context_length=512, prediction_length=48)
    gappy = met_frame.drop(met_frame.index[700:740])
    with pytest.raises(IrregularIndexError, match="not uniformly spaced"):
        est.predict_quantiles(gappy, "value", anchor=gappy.index[1000], horizon=48)


def test_to_regular_index_restores_the_grid(met_frame):
    gappy = met_frame.drop(met_frame.index[700:740])
    fixed = to_regular_index(gappy)
    assert len(fixed) == len(met_frame)
    assert (fixed.index == met_frame.index).all()
    assert fixed["value"].isna().sum() == 40  # the dropped hours come back as missing
    Chronos2Estimator()._check_index(fixed.index)  # no longer raises


def test_regular_index_passes_unchanged(met_frame):
    Chronos2Estimator()._check_index(met_frame.index)


# ----------------------------------------------------------- normet schema


@pytest.fixture
def dew_frame() -> pd.DataFrame:
    """A frame shaped like ``Chronos2Estimator.deweather``'s native output."""
    idx = pd.date_range("2020-01-01", periods=48, freq="h")
    base = np.linspace(10.0, 30.0, len(idx))
    return pd.DataFrame(
        {
            "observed": base + 2.0,
            "dew_p10": base - 3.0,
            "dew_p50": base,
            "dew_p90": base + 3.0,
        },
        index=idx,
    )


def test_quantile_column_names_match_normalise():
    """The two formatters must not drift; report._auto_plot parses these names."""
    from normet.analysis.normalise import _format_quantile_name
    from normet.foundation.estimator import _qname

    for q in (0.0, 0.025, 0.1, 0.5, 0.9, 0.975, 1.0):
        assert _qname(q) == _format_quantile_name(q)


def test_to_normet_frame_matches_normalise_schema(dew_frame):
    out = to_normet_frame(
        dew_frame,
        point_col="dew_p50",
        quantile_cols={0.1: "dew_p10", 0.5: "dew_p50", 0.9: "dew_p90"},
    )
    assert out.index.name == "date"
    assert {"observed", "normalised"} <= set(out.columns)
    assert [c for c in out.columns if c.startswith("q")] == ["q100", "q500", "q900"]
    assert out["normalised"].to_numpy() == pytest.approx(dew_frame["dew_p50"].to_numpy())
    assert (out.index == dew_frame.index).all()


def test_to_normet_frame_reports_missing_columns(dew_frame):
    with pytest.raises(KeyError, match="dew_p50"):
        to_normet_frame(dew_frame.drop(columns=["dew_p50"]), point_col="dew_p50")
    with pytest.raises(KeyError, match="not_a_column"):
        to_normet_frame(dew_frame, point_col="dew_p50", quantile_cols={0.1: "not_a_column"})


def test_converted_frame_drives_normalise_plot(dew_frame):
    """The rename is only worth anything if the plotting entry point accepts it."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from normet import normalise_plot

    out = to_normet_frame(
        dew_frame, point_col="dew_p50", quantile_cols={0.1: "dew_p10", 0.9: "dew_p90"}
    )
    ax = normalise_plot(out, ci_low="q100", ci_high="q900")
    assert ax is not None


def test_converted_frame_drives_the_html_report(dew_frame, tmp_path):
    """report._auto_plot dispatches on the column names, so this is the real check."""
    pytest.importorskip("matplotlib")
    from normet import generate_html_report, make_run
    from normet.report import _auto_plot

    out = to_normet_frame(
        dew_frame, point_col="dew_p50", quantile_cols={0.1: "dew_p10", 0.9: "dew_p90"}
    )
    run = make_run(out, kind="chronos_deweather")
    assert _auto_plot(run) is not None, "report did not recognise the converted frame"
    path = generate_html_report(run, tmp_path / "report.html")
    assert path.exists() and path.stat().st_size > 0


def test_counterfactual_result_converts_to_normet_schema():
    idx = pd.date_range("2020-01-01", periods=6, freq="h")
    obs = pd.Series(np.arange(6.0), index=idx)
    post = obs.iloc[3:]
    res = CounterfactualResult(
        observed=obs,
        counterfactual_p50=post,
        counterfactual_p10=post - 1,
        counterfactual_p90=post + 1,
        absolute_impact_p50=post * 0,
        relative_impact_pct_p50=post * 0,
        pre_intervention_bias_pct=1.5,
    )
    out = res.to_normet_frame()
    assert out.index.name == "date"
    assert list(out.columns) == ["observed", "normalised", "q100", "q900"]
    # the projection only exists after the intervention; history stays observed-only
    assert out["normalised"].iloc[:3].isna().all()
    assert out["normalised"].iloc[3:].to_numpy() == pytest.approx(post.to_numpy())
    assert out["observed"].to_numpy() == pytest.approx(obs.to_numpy())


def test_deweather_rejects_unknown_schema_before_loading_weights(met_frame):
    """The guard must fire up front -- deweather is minutes of forward passes."""
    est = Chronos2Estimator(context_length=64, prediction_length=12)
    with pytest.raises(ValueError, match="schema must be"):
        est.deweather(met_frame, "value", schema="normett")
    assert est._pipeline is None


needs_torch = pytest.mark.skipif(not _has("torch"), reason="torch not installed")


@needs_torch
def test_resolve_device_returns_an_explicit_request_unchanged():
    """A named device is the caller's decision, including one torch cannot serve.

    Second-guessing it would hide torch's own error behind a silent downgrade.
    """
    from normet.foundation import resolve_device

    for name in ("cpu", "cuda", "mps", "cuda:1", "nonsense"):
        assert resolve_device(name) == name


@needs_torch
@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_resolve_device_prefers_an_accelerator_over_the_cpu(monkeypatch, cuda, mps, expected):
    """Apple Silicon must not be skipped.

    The check this replaced was ``"cuda" if is_available() else "cpu"``, which put
    every Mac on the CPU path however capable its GPU — and de-weathering spends
    one full forward pass per Monte-Carlo sample, so that is the difference
    between minutes and hours.
    """
    import torch

    from normet.foundation import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    if getattr(torch.backends, "mps", None) is not None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    elif mps:
        pytest.skip("this torch build has no mps backend to stub")
    assert resolve_device(None) == expected


@needs_torch
def test_auto_selected_device_falls_back_to_the_cpu_but_an_explicit_one_does_not():
    """MPS can refuse a dtype that CUDA and the CPU accept, and it fails at load.

    Falling back is right when we guessed; when the caller named the device, the
    error is the answer they asked for.
    """
    import normet.foundation.estimator as est_mod

    calls: list[str] = []

    class _Boom:
        @staticmethod
        def from_pretrained(name, device_map):
            calls.append(device_map)
            if device_map != "cpu":
                raise RuntimeError("no kernel for this dtype")
            return object()

    monkey = pytest.MonkeyPatch()
    monkey.setattr(est_mod, "_import_foundation", lambda: (None, _Boom))
    try:
        pipe, landed = est_mod._load_pipeline_on("amazon/chronos-2", "mps", auto=True)
        assert landed == "cpu" and pipe is not None
        assert calls == ["mps", "cpu"]

        calls.clear()
        with pytest.raises(RuntimeError, match="no kernel"):
            est_mod._load_pipeline_on("amazon/chronos-2", "mps", auto=False)
        assert calls == ["mps"]
    finally:
        monkey.undo()


@needs_chronos
def test_do_all_runs_the_zero_shot_path_and_returns_the_estimator(chronos2_pipeline):
    """`do_all` keeps its three-tuple shape but skips the middle step entirely.

    The AutoML path is prepare -> train -> normalise. Nothing is fitted here, so
    the model slot holds the loaded estimator instead of a trained model, and
    model_config carries the constructor settings the search parameters would
    have carried.
    """
    from normet import do_all
    from normet.foundation import Chronos2Estimator

    n = 900
    rng = np.random.default_rng(3)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.arange(n)
    blh = 800 + 400 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 40, n)
    t2m = 10 + 8 * np.sin(2 * np.pi * t / (24 * 30)) + rng.normal(0, 1, n)
    pm = np.clip(20 - 0.01 * blh + 0.2 * (t2m - t2m.mean()) ** 2 + rng.normal(0, 2, n), 0, None)
    df = pd.DataFrame({"date": dates, "PM2.5": pm, "t2m": t2m, "blh": blh})

    out, model, df_prep = do_all(
        df,
        target="PM2.5",
        backend="chronos-2",
        covariates=["t2m", "blh"],
        variables_resample=["t2m", "blh"],
        n_samples=2,
        model_config={"context_length": 512, "prediction_length": 48, "device": "cpu"},
    )

    assert isinstance(model, Chronos2Estimator)
    assert model.context_length == 512 and model.device == "cpu"
    # Same schema as normalise, so the existing plot and report paths apply.
    assert {"observed", "normalised"} <= set(out.columns)
    assert out.index.name == "date"
    assert len(out) == n
    assert out["normalised"].notna().all()
    # The leading context is seeded with observations, and the tail is not.
    assert np.allclose(out["normalised"][:512], out["observed"][:512])
    assert not np.allclose(out["normalised"][512:], out["observed"][512:])
    assert "set" in df_prep.columns


@needs_chronos
def test_do_all_zero_shot_defaults_to_a_sane_sample_count(chronos2_pipeline):
    """300 Monte-Carlo samples is a tree-ensemble budget, not a transformer one.

    Each sample here is a full forward pass, so carrying the AutoML default
    across would run for days. An explicit n_samples must still win.
    """
    from normet.foundation import (
        CHRONOS_BACKEND,
        CHRONOS_DEFAULT_SAMPLES,
        resolve_n_samples,
    )
    from normet.pipeline import CHRONOS_BACKEND as REEXPORTED

    # These live in normet.foundation next to the model they describe; the
    # pipeline re-exports the backend name because that is where callers meet it.
    assert REEXPORTED == CHRONOS_BACKEND

    assert resolve_n_samples(None, "flaml") == 300
    assert resolve_n_samples(None, CHRONOS_BACKEND) == CHRONOS_DEFAULT_SAMPLES
    assert resolve_n_samples(300, CHRONOS_BACKEND) == 300
    assert resolve_n_samples(1, "flaml") == 1


@needs_chronos
def test_do_all_zero_shot_needs_meteorological_covariates():
    """Without covariates the run is a plain forecast wearing the wrong name.

    This must fail before the ~500 MB checkpoint download, not after it.
    """
    from normet import do_all

    df = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=48, freq="h"), "PM2.5": range(48)}
    )
    with pytest.raises(ValueError, match="meteorological covariates"):
        do_all(df, target="PM2.5", backend="chronos-2", covariates=[])


@needs_chronos
def test_embed_multisite_returns_one_vector_per_site(chronos2_pipeline):
    """The long-to-wide pivot is what connects multi-site frames to the embedder.

    Site keys must come back as the caller's own values, not stringified, so the
    result joins against their frame directly.
    """
    from normet import embed_multisite

    n = 400
    rng = np.random.default_rng(11)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.arange(n)
    frames = []
    for site, period in ((101, 24), (202, 12), (303, 168)):
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "site": site,
                    "PM2.5": 20 + 6 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.5, n),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)

    vectors = embed_multisite(df, "site", "PM2.5", context_length=256, device="cpu")
    assert set(vectors) == {101, 202, 303}
    assert all(v.shape == (768,) for v in vectors.values())
    # Sites with different dynamics must not collapse onto the same vector.
    assert not np.allclose(vectors[101], vectors[303])


@needs_chronos
def test_zero_shot_meteorology_decomposition_adds_up(chronos2_pipeline):
    """Nested de-weathering, one feature fixed at a time -- same shape as decom_met.

    The identity that matters is that the pieces reconstruct the whole: observed
    minus emi_total is met_total, and met_total minus met_base minus the summed
    per-feature contributions is met_noise. If the successive differences were
    misaligned this would not close.
    """
    from normet import decompose

    n = 800
    rng = np.random.default_rng(5)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.arange(n)
    blh = 800 + 400 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 40, n)
    t2m = 10 + 8 * np.sin(2 * np.pi * t / (24 * 30)) + rng.normal(0, 1, n)
    pm = np.clip(40 - 0.01 * blh + 0.3 * t2m + rng.normal(0, 1.5, n), 0, None)
    df = pd.DataFrame({"date": dates, "PM2.5": pm, "t2m": t2m, "blh": blh})

    out = decompose(
        df,
        target="PM2.5",
        method="meteorology",
        backend="chronos-2",
        covariates=["t2m", "blh"],
        n_samples=2,
        model_config={"context_length": 256, "prediction_length": 48, "device": "cpu"},
    )

    for col in ("observed", "emi_total", "t2m", "blh", "met_total", "met_base", "met_noise"):
        assert col in out.columns, col
    assert len(out) == n

    np.testing.assert_allclose(
        out["met_total"], out["observed"] - out["emi_total"], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        out["met_noise"],
        out["met_total"] - (out["met_base"] + out[["t2m", "blh"]].sum(axis=1)),
        rtol=1e-6,
        atol=1e-6,
    )


@needs_chronos
def test_zero_shot_decomposition_honours_an_explicit_variable_order(chronos2_pipeline):
    """Without fitted importances the order comes from covariate sensitivity.

    That is a measurement, so it can move between runs; variable_order pins it
    for results that stay comparable. A wrong set must be rejected rather than
    silently partially applied.
    """
    from normet import decompose
    from normet.exceptions import ConfigError

    n = 700
    rng = np.random.default_rng(6)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.arange(n)
    blh = 800 + 400 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 40, n)
    t2m = 10 + 8 * np.sin(2 * np.pi * t / (24 * 30)) + rng.normal(0, 1, n)
    pm = np.clip(40 - 0.01 * blh + 0.3 * t2m + rng.normal(0, 1.5, n), 0, None)
    df = pd.DataFrame({"date": dates, "PM2.5": pm, "t2m": t2m, "blh": blh})

    common = dict(
        target="PM2.5",
        method="meteorology",
        backend="chronos-2",
        covariates=["t2m", "blh"],
        n_samples=2,
        model_config={"context_length": 256, "prediction_length": 48, "device": "cpu"},
    )
    out = decompose(df, variable_order=["blh", "t2m"], **common)
    assert list(out.columns).index("blh") < list(out.columns).index("t2m")

    with pytest.raises(ConfigError, match="variable_order"):
        decompose(df, variable_order=["blh"], **common)


@needs_chronos
def test_zero_shot_decomposition_takes_groups_and_shapley(chronos2_pipeline):
    """The zero-shot path shares decom_met's attribution machinery.

    Shapley over two features and Shapley over two one-feature groups evaluate
    the same four coalitions with the same seed, so they must agree exactly; and
    the Shapley split must still add up to the prediction minus emi_total.
    """
    from normet import decompose

    n = 700
    rng = np.random.default_rng(8)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.arange(n)
    blh = 800 + 400 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 40, n)
    t2m = 10 + 8 * np.sin(2 * np.pi * t / (24 * 30)) + rng.normal(0, 1, n)
    pm = np.clip(40 - 0.01 * blh + 0.3 * t2m + rng.normal(0, 1.5, n), 0, None)
    df = pd.DataFrame({"date": dates, "PM2.5": pm, "t2m": t2m, "blh": blh})

    common = dict(
        target="PM2.5",
        method="meteorology",
        backend="chronos-2",
        covariates=["t2m", "blh"],
        n_samples=2,
        model_config={"context_length": 256, "prediction_length": 48, "device": "cpu"},
    )
    per_feature = decompose(df, attribution="shapley", **common)
    grouped = decompose(df, groups={"temperature": ["t2m"], "mixing": ["blh"]}, **common)

    np.testing.assert_allclose(grouped["temperature"], per_feature["t2m"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(grouped["mixing"], per_feature["blh"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(grouped["emi_total"], per_feature["emi_total"], rtol=1e-6)
    np.testing.assert_allclose(
        grouped["met_noise"],
        grouped["met_total"] - grouped["met_base"] - grouped[["temperature", "mixing"]].sum(axis=1),
        rtol=1e-6,
        atol=1e-6,
    )


# --------------------------------------------------- categorical covariates


@pytest.fixture
def categorical_frame() -> pd.DataFrame:
    """Hourly frame whose level is set by a categorical regime, not a number.

    ``regime`` is the only thing that separates the two levels: it is a string,
    so a numeric-only covariate path drops it and the forecast falls back to the
    pooled mean. The numeric column ``ws`` is deliberately uninformative here so
    the test cannot pass by reading it instead.
    """
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=1200, freq="h")
    regime = np.where((np.arange(len(idx)) // 24) % 2 == 0, "calm", "windy")
    level = np.where(regime == "calm", 70.0, 20.0)
    ws = 2.0 + np.abs(rng.normal(2.0, 1.0, len(idx)))
    value = level + rng.normal(0, 1.5, len(idx))
    return pd.DataFrame({"value": value, "ws": ws, "regime": regime}, index=idx)


def test_is_categorical_splits_on_pandas_numeric(categorical_frame):
    from normet.foundation.estimator import _is_categorical

    assert _is_categorical(categorical_frame["regime"])
    assert _is_categorical(categorical_frame["regime"].astype("category"))
    assert not _is_categorical(categorical_frame["ws"])
    assert not _is_categorical(categorical_frame["value"])
    # booleans read as numeric, which is what we want: the ordering is already there
    assert not _is_categorical(pd.Series([True, False, True]))


def test_categorical_covariate_survives_resolution(categorical_frame):
    """A string column used to be dropped by the numeric-dtype filter."""
    est = Chronos2Estimator(use_calendar=False)
    cols = est._resolve_covariates(categorical_frame, "value")
    assert "regime" in cols
    assert "ws" in cols


def test_categorical_reaches_the_input_dict_unconverted(categorical_frame):
    """Numerics are float32; categoricals stay object so upstream reads them as categories."""
    est = Chronos2Estimator(use_calendar=False)
    past = categorical_frame.iloc[:100]
    future = categorical_frame.iloc[100:124]
    item = est._make_input(past["value"].to_numpy(), past, future, ["ws", "regime"])

    assert item["past_covariates"]["ws"].dtype == np.float32
    assert item["past_covariates"]["regime"].dtype == object
    assert set(item["past_covariates"]["regime"]) <= {"calm", "windy"}
    assert item["future_covariates"]["regime"].shape == (24,)


def test_categorical_missing_values_are_not_imputed(categorical_frame):
    """NaN gets its own category upstream, so filling it here would invent a level."""
    frame = categorical_frame.copy()
    frame.loc[frame.index[:10], "regime"] = np.nan
    est = Chronos2Estimator(use_calendar=False)
    past = frame.iloc[:100]
    item = est._make_input(past["value"].to_numpy(), past, frame.iloc[100:124], ["regime"])

    values = item["past_covariates"]["regime"]
    assert pd.isna(pd.Series(values)).sum() == 10


@needs_chronos
def test_categorical_covariate_moves_the_forecast(make_estimator, categorical_frame):
    """The regime label must change the forecast, or it never reached the model.

    Two calls over the same history, differing only in the future value of
    ``regime``. If the label is being dropped the two forecasts are identical;
    if it is encoded, the one told "calm" must sit above the one told "windy",
    because that is the only thing separating a level of 70 from a level of 20.
    """
    est = make_estimator(context_length=512, prediction_length=24, use_calendar=False)
    anchor = categorical_frame.index[1000]

    calm = categorical_frame.copy()
    calm.loc[calm.index >= anchor, "regime"] = "calm"
    windy = categorical_frame.copy()
    windy.loc[windy.index >= anchor, "regime"] = "windy"

    common = dict(target="value", anchor=anchor, horizon=24, quantiles=(0.5,))
    q_calm = est.predict_quantiles(calm, covariates=["regime"], **common)["q0.5"]
    q_windy = est.predict_quantiles(windy, covariates=["regime"], **common)["q0.5"]

    assert not np.allclose(q_calm.to_numpy(), q_windy.to_numpy())
    assert q_calm.mean() > q_windy.mean()


# ------------------------------------------- multivariate / multisite joint


@pytest.fixture
def two_species_frame() -> pd.DataFrame:
    """One site, two species at very different levels, sharing a diurnal driver.

    The levels are far apart (~70 and ~20) on purpose: it is what makes a
    transposed variate visible. A joint call that returns the variates in the
    wrong order would otherwise pass every structural assertion.
    """
    rng = np.random.default_rng(11)
    idx = pd.date_range("2020-01-01", periods=1200, freq="h")
    driver = np.sin(2 * np.pi * np.arange(len(idx)) / 24.0)
    ws = 2.0 + np.abs(rng.normal(2.0, 1.0, len(idx)))
    no2 = 70.0 + 8.0 * driver + rng.normal(0, 1.5, len(idx))
    nox = 20.0 + 3.0 * driver + rng.normal(0, 1.0, len(idx))
    return pd.DataFrame({"no2": no2, "nox": nox, "ws": ws}, index=idx)


@pytest.fixture
def site_frames(two_species_frame) -> dict[str, pd.DataFrame]:
    """Three sites holding the same target at levels 70 / 45 / 20."""
    rng = np.random.default_rng(3)
    idx = two_species_frame.index
    frames = {}
    for name, level in (("kerbside", 70.0), ("urban", 45.0), ("rural", 20.0)):
        driver = np.sin(2 * np.pi * np.arange(len(idx)) / 24.0)
        frames[name] = pd.DataFrame(
            {
                "no2": level + 6.0 * driver + rng.normal(0, 1.5, len(idx)),
                "ws": 2.0 + np.abs(rng.normal(2.0, 1.0, len(idx))),
            },
            index=idx,
        )
    return frames


def test_multivariate_refuses_a_single_target(two_species_frame):
    """One variate is not a joint task; the univariate method is the honest route."""
    est = Chronos2Estimator(use_calendar=False)
    with pytest.raises(ValueError, match="at least two targets"):
        est.predict_quantiles_multivariate(
            two_species_frame, ["no2"], anchor=two_species_frame.index[1000]
        )


def test_multivariate_refuses_an_absent_target(two_species_frame):
    est = Chronos2Estimator(use_calendar=False)
    with pytest.raises(KeyError, match="target columns absent"):
        est.predict_quantiles_multivariate(
            two_species_frame, ["no2", "pm25"], anchor=two_species_frame.index[1000]
        )


def test_multivariate_excludes_every_target_from_the_covariates(two_species_frame):
    """A target must not also be fed in as a covariate of itself."""
    est = Chronos2Estimator(use_calendar=False)
    cols = est._resolve_covariates(two_species_frame, ["no2", "nox"])
    assert cols == ["ws"]


def test_multisite_refuses_empty_input():
    est = Chronos2Estimator(use_calendar=False)
    with pytest.raises(ValueError, match="nothing to forecast"):
        est.predict_quantiles_multisite({}, "no2", anchor="2020-01-01")


def test_multisite_refuses_a_site_without_the_target(site_frames):
    est = Chronos2Estimator(use_calendar=False)
    frames = dict(site_frames)
    frames["broken"] = frames["urban"].drop(columns=["no2"])
    with pytest.raises(KeyError, match="has no column"):
        est.predict_quantiles_multisite(
            frames, "no2", anchor=site_frames["urban"].index[1000], horizon=6
        )


def test_multisite_refuses_a_site_missing_a_covariate(site_frames):
    """The batch shares one covariate schema, so a ragged site is an error, not a gap."""
    est = Chronos2Estimator(use_calendar=False)
    frames = dict(site_frames)
    frames["broken"] = frames["urban"].drop(columns=["ws"])
    with pytest.raises(KeyError, match="missing covariates"):
        est.predict_quantiles_multisite(
            frames, "no2", anchor=site_frames["urban"].index[1000], horizon=6
        )


@needs_chronos
def test_multivariate_returns_one_frame_per_variate_in_order(make_estimator, two_species_frame):
    """Each variate must come back at its own level, not its neighbour's."""
    est = make_estimator(context_length=512, prediction_length=24, use_calendar=False)
    out = est.predict_quantiles_multivariate(
        two_species_frame,
        ["no2", "nox"],
        anchor=two_species_frame.index[1000],
        horizon=24,
        quantiles=(0.1, 0.5, 0.9),
    )

    assert set(out) == {"no2", "nox"}
    for name, frame in out.items():
        assert list(frame.columns) == ["q0.1", "q0.5", "q0.9"]
        assert len(frame) == 24
        assert np.isfinite(frame.to_numpy()).all()
    # levels are ~70 and ~20; a transposition would swap these
    assert out["no2"]["q0.5"].mean() > 45.0
    assert out["nox"]["q0.5"].mean() < 45.0


@needs_chronos
def test_multivariate_quantiles_are_ordered(make_estimator, two_species_frame):
    est = make_estimator(context_length=512, prediction_length=24, use_calendar=False)
    out = est.predict_quantiles_multivariate(
        two_species_frame, ["no2", "nox"], anchor=two_species_frame.index[1000], horizon=12
    )
    for frame in out.values():
        assert (frame["q0.1"] <= frame["q0.5"] + 1e-6).all()
        assert (frame["q0.5"] <= frame["q0.9"] + 1e-6).all()


@needs_chronos
@pytest.mark.parametrize("cross_learning", [True, False])
def test_multisite_keeps_each_site_at_its_own_level(make_estimator, site_frames, cross_learning):
    """Sharing across the batch must not pull the sites onto a common level.

    This is the failure cross-learning could plausibly cause, so it is asserted
    in both modes: kerbside sits near 70 and rural near 20, and if the joint
    call blurred them together the gap would collapse.
    """
    est = make_estimator(context_length=512, prediction_length=24, use_calendar=False)
    out = est.predict_quantiles_multisite(
        site_frames,
        "no2",
        anchor=site_frames["urban"].index[1000],
        horizon=24,
        cross_learning=cross_learning,
    )

    assert list(out) == list(site_frames)
    for frame in out.values():
        assert len(frame) == 24
        assert np.isfinite(frame.to_numpy()).all()
    assert out["kerbside"]["q0.5"].mean() > out["urban"]["q0.5"].mean()
    assert out["urban"]["q0.5"].mean() > out["rural"]["q0.5"].mean()


@needs_chronos
def test_multisite_without_cross_learning_matches_the_per_site_loop(make_estimator, site_frames):
    """``cross_learning=False`` is batching, nothing more, so it must agree with a loop."""
    est = make_estimator(context_length=512, prediction_length=24, use_calendar=False)
    anchor = site_frames["urban"].index[1000]
    joint = est.predict_quantiles_multisite(
        site_frames, "no2", anchor=anchor, horizon=12, cross_learning=False
    )
    for name, frame in site_frames.items():
        alone = est.predict_quantiles(frame, "no2", anchor=anchor, horizon=12)
        np.testing.assert_allclose(joint[name].to_numpy(), alone.to_numpy(), rtol=1e-4, atol=1e-4)


@needs_chronos
def test_cross_learning_actually_changes_the_answer(make_estimator, site_frames):
    """If the flag were not reaching the pipeline, both modes would be identical."""
    est = make_estimator(context_length=512, prediction_length=24, use_calendar=False)
    anchor = site_frames["urban"].index[1000]
    common = dict(target="no2", anchor=anchor, horizon=12)
    shared = est.predict_quantiles_multisite(site_frames, cross_learning=True, **common)
    apart = est.predict_quantiles_multisite(site_frames, cross_learning=False, **common)
    assert any(
        not np.allclose(shared[name].to_numpy(), apart[name].to_numpy()) for name in site_frames
    )


# --------------------------------------------------------------- fine-tuning


def test_fit_still_trains_nothing(met_frame):
    """The sklearn-shaped path must stay free; fine-tuning is asked for by name.

    ``do_all`` calls ``fit`` as a matter of course. If that ever started
    optimising, a routine call would silently become a GPU job.
    """
    est = Chronos2Estimator()
    same = est.fit(met_frame[["ws", "blh"]], met_frame["value"])
    assert same is est
    assert est._pipeline is None
    assert est.target_col == "value"
    assert est.feature_cols == ["ws", "blh"]


def test_finetune_rejects_an_unknown_mode(met_frame):
    est = Chronos2Estimator()
    with pytest.raises(ValueError, match="mode must be"):
        est.finetune(met_frame, "value", mode="adapter")


def test_finetune_rejects_an_absent_target(met_frame):
    est = Chronos2Estimator()
    with pytest.raises(KeyError, match="target column absent"):
        est.finetune(met_frame, "pm25", mode="full")


@pytest.mark.skipif(_has("peft"), reason="peft installed, so lora is available")
def test_lora_refuses_rather_than_falling_back_to_full(met_frame):
    """Upstream warns and silently does full fine-tuning instead. That is not the ask.

    Full fine-tuning updates every one of 119M parameters where LoRA would have
    trained adapters; treating them as interchangeable hides both the cost and
    the overfitting risk.
    """
    est = Chronos2Estimator()
    with pytest.raises(ImportError, match="requires peft"):
        est.finetune(met_frame, "value", mode="lora")


@needs_chronos
def test_training_inputs_keep_covariates_known_into_the_future(met_frame):
    """Meteorology is known across the forecast window; training must say so.

    A run that prepared the covariates as past-only would adapt the model to a
    problem normet never poses -- de-weathering exists precisely because the
    weather over the counterfactual window is available.
    """
    est = Chronos2Estimator(context_length=256, use_calendar=False)
    prepared = est._training_inputs(met_frame, "value", ["ws", "blh"], horizon=24)

    assert len(prepared) == 1
    item = prepared[0]
    assert item["n_targets"] == 1
    assert item["n_covariates"] == 2
    assert item["n_future_covariates"] == 2
    assert item["future_covariates"].shape[-1] == 24


@needs_chronos
def test_training_inputs_accept_a_categorical_covariate(categorical_frame):
    est = Chronos2Estimator(use_calendar=False)
    prepared = est._training_inputs(categorical_frame, "value", ["ws", "regime"], horizon=12)
    assert prepared[0]["n_covariates"] == 2
    assert np.isfinite(prepared[0]["context"].numpy()).any()


@needs_chronos
@pytest.mark.slow
def test_full_finetune_returns_a_new_estimator_on_tuned_weights(
    make_estimator, met_frame, tmp_path
):
    """Two optimiser steps: enough to prove the path, not to claim an improvement.

    What is asserted is the contract -- a *new* estimator comes back, the
    original keeps its pretrained pipeline, and the tuned one still forecasts.
    Whether fine-tuning helps is a question about data, not about this wiring.
    """
    est = make_estimator(context_length=256, prediction_length=24, use_calendar=False)
    before = est._pipeline

    tuned = est.finetune(
        met_frame.iloc[:600],
        "value",
        mode="full",
        prediction_length=24,
        covariates=["ws", "blh"],
        num_steps=2,
        batch_size=1,
        learning_rate=1e-6,
        # Upstream defaults this to chronos-2-finetuned/<timestamp> in the
        # working directory, which for a test run is the repository.
        output_dir=tmp_path,
        report_to=[],
    )

    assert tuned is not est
    assert est._pipeline is before, "the original must keep its pretrained weights"
    assert tuned._pipeline is not before
    assert tuned.target_col == "value"
    assert tuned.feature_cols == ["ws", "blh"]

    out = tuned.predict_quantiles(
        met_frame.iloc[:700], "value", anchor=met_frame.index[600], horizon=12
    )
    assert len(out) == 12
    assert np.isfinite(out.to_numpy()).all()
