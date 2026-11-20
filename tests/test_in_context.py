"""Tests for the in-context normalisation (ICN) module.

These run without torch/checkpoints: the heavy forward pass is behind
:class:`normet.foundation.in_context.ICLPipeline`, and the test injects
:class:`FakeICLPipeline`, a torch-free stand-in that fits a ridge regression of
the past target on the past meteorology and applies it to the future.

The headline assertion (research plan WP3.1) is the one that separates ICN from
the baseline ``deweather`` configuration: *future-only intervention recovers the
truth, past+future intervention does not.* ``FakeICLPipeline`` makes that
concrete -- with the observed past it learns the true weather response and the
counterfactual forecast tracks the (weather-driven) truth; with the past
meteorology swapped for random pool draws it cannot, and the forecast collapses
to a constant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from normet.foundation import (
    Chronos2Estimator,
    ICNDesign,
    MetPool,
    build_icn_draw_bank,
    normalise_in_context,
    normalise_in_context_multisite,
    normalise_in_context_multivariate,
)


class FakeICLPipeline:
    """Torch-free stand-in satisfying :class:`ICLPipeline`.

    Fits a ridge regression of the past target on the past meteorology and
    extrapolates to the future meteorology. Future-only intervention keeps the
    observed past, so the fit recovers the true weather response; past+future
    intervention swaps the past meteorology for random pool draws, destroying it.
    """

    def forecast(
        self,
        target_history: np.ndarray,
        past: pd.DataFrame,
        future_met: pd.DataFrame,
        cols: list[str],
        target: str,
    ) -> np.ndarray:
        met = [c for c in cols if c in past.columns]
        Xp = past[met].to_numpy(dtype=float)
        yp = past[target].to_numpy(dtype=float)
        good = np.isfinite(yp) & np.all(np.isfinite(Xp), axis=1)
        Xp, yp = Xp[good], yp[good]
        if Xp.shape[0] == 0 or Xp.shape[1] == 0:
            return np.full(len(future_met), float(np.nanmean(yp)) if yp.size else np.nan)
        XtX = Xp.T @ Xp + 1e-6 * np.eye(Xp.shape[1])
        w = np.linalg.solve(XtX, Xp.T @ yp)
        intercept = yp.mean() - w @ Xp.mean(0)
        Xf = future_met[met].to_numpy(dtype=float)
        return (intercept + Xf @ w).astype(np.float64)


def _synthetic(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """y = 10 + 5*w + noise, with a +5 step event in the future of the first anchor.

    ``w`` is the (only) weather driver, so the counterfactual truth is exactly a
    function of ``w``. The ICN level estimate averages the forecast over the
    counterfactual weather (like ``deweather``), so it is a *weather-removed*
    baseline -- its variance comes only from how strongly the fitted weather
    response survives. A future-only intervention keeps the observed past and
    recovers that response (level carries weather amplitude); a past+future
    intervention swaps the past meteorology for random pool draws and kills the
    response (level collapses to a constant).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="h")
    # White-noise weather: independent of position, so a random pool draw carries no
    # relationship to the observed target, while the observed past (same series)
    # lets the pipeline learn the true 5*w response. This is what makes the
    # past+future intervention collapse: swapped past weather is uninformative.
    w = rng.standard_normal(n)
    y = 10.0 + 5.0 * w + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame({"y": y, "w": w}, index=idx)
    # event: +10 for 24 h starting at hour 1100 (inside the first anchor's future)
    df.loc[df.index[1100:1124], "y"] += 10.0
    return df


def _estimator() -> Chronos2Estimator:
    est = Chronos2Estimator()
    est.fit(_synthetic()[["w"]], _synthetic()["y"])
    return est


def test_icn_design_validation() -> None:
    with pytest.raises(ValueError):
        ICNDesign(intervene="bogus")
    with pytest.raises(ValueError):
        ICNDesign(pool="bogus")
    with pytest.raises(ValueError):
        ICNDesign(direction="bogus")
    with pytest.raises(ValueError):
        ICNDesign(aggregate="bogus")


def test_metpool_draws() -> None:
    df = _synthetic()
    pool = MetPool(df[["w"]], ["w"])
    rng = np.random.default_rng(1)
    pt = pool.sample_pointwise(50, rng)
    tr = pool.sample_trajectory(50, rng)
    cl = pool.sample_climatology(df.index[2000], 50, rng)
    assert len(pt) == 50 and len(tr) == 50 and len(cl) == 50
    frac, per = pool.support_check(df.iloc[1000:1050])
    assert 0.0 <= frac <= 1.0
    assert set(per) == {"w"}


class _NumericFakeICLPipeline(FakeICLPipeline):
    """The fake over numeric covariates only, holding a categorical column at
    its factual values, as the Chronos adapter does."""

    def forecast(
        self,
        target_history: np.ndarray,
        past: pd.DataFrame,
        future_met: pd.DataFrame,
        cols: list[str],
        target: str,
    ) -> np.ndarray:
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(past[c])]
        return super().forecast(target_history, past, future_met, numeric, target)


def test_a_categorical_covariate_is_not_resampled_as_weather() -> None:
    """A site or season label Chronos accepts is not weather with a range.

    Inferred meteorological columns used to be every non-calendar covariate,
    categoricals included; the A2 support check then read the label with
    ``dtype=float`` and raised. The label must stay a factual covariate and
    the counterfactual pool must hold the numeric columns alone.
    """
    df = _synthetic(n=1500)
    df["site"] = np.where(np.arange(len(df)) % 2, "A", "B")
    est = Chronos2Estimator()
    est.fit(df[["w", "site"]], df["y"])
    design = ICNDesign(
        intervene="future_only",
        pool="record_trajectory",
        block_h=96,
        stride=96,
        context_length=400,
        n_resamples=2,
        seed=1,
    )
    res = normalise_in_context(est, df, "y", design=design, pipeline=_NumericFakeICLPipeline())
    assert np.isfinite(res.level.to_numpy()).any()


def test_future_only_recovers_truth() -> None:
    df = _synthetic()
    est = _estimator()
    design = ICNDesign(
        intervene="future_only",
        pool="record_trajectory",
        block_h=168,
        stride=168,
        context_length=1000,
        n_resamples=5,
        seed=3,
    )
    res = normalise_in_context(est, df, "y", design=design, pipeline=FakeICLPipeline())
    level = res.level.to_numpy()
    covered = np.isfinite(level)
    # The counterfactual truth is 10 + 2*w (std ~ 1.5). A future-only intervention
    # keeps the observed past, so the fitted weather response is recovered and the
    # level carries the full weather amplitude instead of collapsing to a constant.
    assert np.std(level[covered]) > 0.6, "future-only level must keep the weather amplitude"
    # the event is preserved: observed - level shows the +10 bump
    weather_effect = res.weather_effect.to_numpy()
    assert weather_effect[1110] > 5.0, "event must survive future-only intervention"


def test_past_future_does_not_recover() -> None:
    df = _synthetic()
    est = _estimator()
    design = ICNDesign(
        intervene="past_future",
        pool="record_trajectory",
        block_h=168,
        stride=168,
        context_length=1000,
        n_resamples=5,
        seed=3,
    )
    res = normalise_in_context(est, df, "y", design=design, pipeline=FakeICLPipeline())
    level = res.level.to_numpy()
    covered = np.isfinite(level)
    # swapping the past meteorology for random pool draws destroys the fit, so the
    # level collapses to a constant and loses the weather amplitude.
    assert np.std(level[covered]) < 0.3, "past+future level should collapse to a constant"


def test_result_shapes_and_uncovered() -> None:
    df = _synthetic()
    est = _estimator()
    design = ICNDesign(context_length=2500, block_h=168, stride=168, n_resamples=3, seed=2)
    res = normalise_in_context(est, df, "y", design=design, pipeline=FakeICLPipeline())
    assert len(res.level) == len(df)
    assert len(res.paired) == len(df)
    assert res.spread_by_source.shape[0] == len(df)
    assert res.hour_contrib.shape[0] == len(df)
    # a single forward pass leaves the first context_length hours uncovered
    assert int((res.hour_contrib.to_numpy() == 0).sum()) >= design.context_length


def test_schema_normet_renames_level() -> None:
    df = _synthetic()
    est = _estimator()
    design = ICNDesign(context_length=1000, n_resamples=3, seed=4)
    res = normalise_in_context(
        est, df, "y", design=design, pipeline=FakeICLPipeline(), schema="normet"
    )
    assert res.level.name == "normalised"


def test_deweather_untouched() -> None:
    # the new method is additive; the existing de-weathering entry point remains
    assert hasattr(Chronos2Estimator, "deweather")
    assert hasattr(Chronos2Estimator, "normalise_in_context")


def test_d4_multi_anchor_overlap() -> None:
    df = _synthetic()
    est = _estimator()
    # stride < block_h -> anchors overlap, so interior hours are covered by >1 anchor
    design = ICNDesign(context_length=1000, block_h=168, stride=84, n_resamples=3, seed=5)
    res = normalise_in_context(est, df, "y", design=design, pipeline=FakeICLPipeline())
    assert int(res.hour_contrib.max()) > 1, "overlapping anchors must cover hours >1 times"
    assert res.diagnostics["n_anchors"] > (3000 - 168) // 168


def test_d5_bidirectional_fills_hole() -> None:
    df = _synthetic()
    est = _estimator()
    # a single forward pass leaves the first context_length hours uncovered; the
    # reverse orientation covers them, so bidirectional should leave almost none.
    fwd = normalise_in_context(
        est,
        df,
        "y",
        design=ICNDesign(context_length=1000, direction="forward", n_resamples=3, seed=6),
        pipeline=FakeICLPipeline(),
    )
    bid = normalise_in_context(
        est,
        df,
        "y",
        design=ICNDesign(context_length=1000, direction="bidirectional", n_resamples=3, seed=6),
        pipeline=FakeICLPipeline(),
    )
    assert fwd.diagnostics["uncovered_hours"] >= 1000
    # bidirectional fills the middle of the record: the uncovered count drops to a
    # fraction of the forward-only hole (the first ~context hours stay uncovered on
    # both ends, which is expected -- both orientations still need a conditioning
    # window). It must not leave the forward-size hole behind.
    assert bid.diagnostics["uncovered_hours"] < fwd.diagnostics["uncovered_hours"] * 0.6


def test_lead_time_weighting_runs() -> None:
    df = _synthetic()
    est = _estimator()
    design = ICNDesign(
        context_length=1000, block_h=168, stride=84, lead_tau=24.0, n_resamples=3, seed=7
    )
    res = normalise_in_context(est, df, "y", design=design, pipeline=FakeICLPipeline())
    assert np.isfinite(res.level.to_numpy()).any()
    # weighting must not change the shape contract
    assert len(res.level) == len(df)
    assert int(res.hour_contrib.max()) > 1


# --------------------------------------------------------------------------- #
# 2026-11-11: short records, honest aggregation, separated spread
# --------------------------------------------------------------------------- #
def test_min_context_gives_a_short_record_an_estimate() -> None:
    """A record shorter than the context produced no output at all."""
    df = _synthetic(n=1900)  # ~79 days, C = 2048
    est = _estimator()
    base = dict(context_length=2048, block_h=168, stride=168, n_resamples=3, seed=8)
    none = normalise_in_context(est, df, "y", design=ICNDesign(**base), pipeline=FakeICLPipeline())
    assert np.isfinite(none.level.to_numpy()).sum() == 0
    short = normalise_in_context(
        est, df, "y", design=ICNDesign(**base, min_context=400), pipeline=FakeICLPipeline()
    )
    # anchors now start at 400 h, so the rest of the record is estimated
    assert np.isfinite(short.level.to_numpy()).sum() > 1000
    assert short.diagnostics["min_context"] == 400


def test_without_min_context_a_three_month_record_keeps_one_block() -> None:
    """3 months at C=2048: one partial anchor over the last block, 6 % of hours."""
    df = _synthetic(n=91 * 24)
    est = _estimator()
    design = ICNDesign(context_length=2048, block_h=168, stride=168, n_resamples=2, seed=11)
    res = normalise_in_context(est, df, "y", design=design, pipeline=FakeICLPipeline())
    assert np.isfinite(res.level.to_numpy()).sum() == 91 * 24 - 2048


def test_aggregate_names_are_three_estimators() -> None:
    """mean, median and trimmed must differ on a skewed draw set."""
    from normet.foundation.in_context import _aggregate

    items = [(1.0, 0), (2.0, 1), (2.0, 2), (3.0, 3), (100.0, 30)]  # one heavy tail
    mean = _aggregate(items, "mean", 0.0)
    median = _aggregate(items, "median", 0.0)
    trimmed = _aggregate(items, "trimmed", 0.0)
    assert mean == pytest.approx(21.6)
    assert median == pytest.approx(2.0)
    assert trimmed < mean
    # the deprecated alias is the trimmed mean
    assert _aggregate(items, "quantile_mean", 0.0) == pytest.approx(trimmed)
    # lead weighting moves the mean towards the early draw
    assert _aggregate(items, "mean", 1.0) < mean


def test_anchor_spread_is_not_the_resample_spread() -> None:
    """With one draw per anchor, mc spread is 0 while the anchors still disagree."""
    df = _synthetic()
    est = _estimator()
    design = ICNDesign(context_length=1000, block_h=168, stride=84, n_resamples=1, seed=9)
    res = normalise_in_context(est, df, "y", design=design, pipeline=FakeICLPipeline())
    both = (res.hour_contrib.to_numpy() >= 2) & np.isfinite(res.level.to_numpy())
    assert both.any()
    assert np.nanmax(res.spread_by_source["mc"].to_numpy()[both]) < 1e-12
    assert np.nanmax(res.spread_by_source["anchor"].to_numpy()[both]) > 0.0


def test_support_gate_is_recorded() -> None:
    """The A2 gate reports what it dropped; a record pool is inside its own range."""
    df = _synthetic()
    est = _estimator()
    design = ICNDesign(context_length=1000, n_resamples=2, seed=10, support_min_frac=0.5)
    res = normalise_in_context(est, df, "y", design=design, pipeline=FakeICLPipeline())
    assert res.diagnostics["low_support"] == []
    assert res.diagnostics["design"]["support_min_frac"] == 0.5


class BatchFakeICLPipeline(FakeICLPipeline):
    """Torch-free batched pipeline emitting deterministic quantiles around the ridge median."""

    def __init__(self, spread: float = 2.0) -> None:
        self.spread = float(spread)
        self.batch_calls = 0
        self.quantile_calls = 0

    def forecast_batch(
        self,
        items: list[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]],
        cols: list[str],
        target: str,
    ) -> np.ndarray:
        self.batch_calls += 1
        return np.stack([self.forecast(h, p, f, cols, target) for h, p, f in items])

    def forecast_quantiles_batch(
        self,
        items: list[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]],
        cols: list[str],
        target: str,
        quantiles: list[float],
    ) -> np.ndarray:
        self.quantile_calls += 1
        med = np.stack([self.forecast(h, p, f, cols, target) for h, p, f in items])
        return np.stack([med + self.spread * (float(q) - 0.5) for q in quantiles], axis=1)


def test_batched_pipeline_matches_scalar_bit_for_bit() -> None:
    df = _synthetic()
    est = _estimator()
    design = ICNDesign(
        context_length=1000,
        block_h=168,
        stride=84,
        direction="bidirectional",
        n_resamples=4,
        seed=12,
    )
    scalar = normalise_in_context(est, df, "y", design=design, pipeline=FakeICLPipeline())
    bpipe = BatchFakeICLPipeline()
    batched = normalise_in_context(est, df, "y", design=design, pipeline=bpipe)

    np.testing.assert_allclose(batched.level.to_numpy(), scalar.level.to_numpy(), equal_nan=True)
    np.testing.assert_allclose(batched.paired.to_numpy(), scalar.paired.to_numpy(), equal_nan=True)
    np.testing.assert_allclose(
        batched.spread_by_source.to_numpy(),
        scalar.spread_by_source.to_numpy(),
        equal_nan=True,
    )
    assert bpipe.batch_calls < batched.diagnostics["n_anchors"]


def test_icn_design_v1_preset_and_overrides() -> None:
    v1 = ICNDesign.v1()
    assert v1.intervene == "future_only"
    assert v1.pool == "record_trajectory"
    assert v1.gap_h == 0
    assert v1.block_h == 336
    assert v1.stride == 168
    assert v1.direction == "bidirectional"
    assert v1.context_length == 2048
    assert v1.aggregate == "trimmed"
    assert v1.n_resamples == 5
    short = ICNDesign.v1(min_context=336, n_resamples=3)
    assert short.min_context == 336
    assert short.n_resamples == 3


def test_external_met_pool_reversed_and_support_gate() -> None:
    df = _synthetic(n=1800)
    est = _estimator()
    # Narrow external pool triggers the A2 support gate on extreme factual windows
    pool_df = df[["w"]].copy()
    pool_df["w"] = np.clip(pool_df["w"], -0.5, 0.5)
    pool_with_date = pool_df.reset_index().rename(columns={"index": "date"})
    pool = MetPool.from_frame(pool_with_date, ["w"])

    rev_pool = pool.reversed()
    assert list(rev_pool.df.index) == list(pool.df.index[::-1])
    np.testing.assert_allclose(rev_pool.df["w"].to_numpy(), pool.df["w"].to_numpy()[::-1])

    design = ICNDesign(
        context_length=800,
        block_h=96,
        stride=96,
        direction="bidirectional",
        n_resamples=3,
        support_min_frac=0.95,
        seed=5,
    )
    res = normalise_in_context(
        est, df, "y", design=design, pipeline=BatchFakeICLPipeline(), pool=pool_with_date
    )
    assert res.diagnostics["external_pool"] is True
    assert len(res.diagnostics["low_support"]) > 0


def test_icn_quantiles_and_to_normet_frame() -> None:
    df = _synthetic()
    est = _estimator()
    design = ICNDesign(context_length=1000, block_h=168, stride=168, n_resamples=3, seed=14)
    bpipe = BatchFakeICLPipeline(spread=4.0)
    res = normalise_in_context(
        est,
        df,
        "y",
        design=design,
        pipeline=bpipe,
        quantiles=(0.1, 0.5, 0.9),
    )
    assert bpipe.quantile_calls > 0
    assert res.quantiles is not None
    assert list(res.quantiles.columns) == ["q100", "q500", "q900"]
    valid = res.quantiles.dropna()
    assert (valid["q100"] < valid["q500"]).all()
    assert (valid["q500"] < valid["q900"]).all()
    np.testing.assert_allclose(valid["q500"].to_numpy(), res.level.loc[valid.index].to_numpy())

    lvl_frame = res.to_normet_frame("level")
    assert list(lvl_frame.columns) == ["observed", "normalised", "q100", "q500", "q900"]
    assert lvl_frame.index.name == "date"

    assert res.paired_quantiles is not None
    assert list(res.paired_quantiles.columns) == ["q100", "q500", "q900"]
    valid_pq = res.paired_quantiles.dropna()
    assert (valid_pq["q100"] < valid_pq["q500"]).all()
    assert (valid_pq["q500"] < valid_pq["q900"]).all()

    paired_frame = res.to_normet_frame("paired")
    assert list(paired_frame.columns) == ["observed", "normalised", "q100", "q500", "q900"]
    np.testing.assert_allclose(
        paired_frame["normalised"].to_numpy(), res.paired.to_numpy(), equal_nan=True
    )

    # Without quantiles, paired frame keeps concise 2-column schema
    res_no_q = normalise_in_context(est, df, "y", design=design, pipeline=bpipe)
    assert res_no_q.paired_quantiles is None
    assert list(res_no_q.to_normet_frame("paired").columns) == ["observed", "normalised"]

    with pytest.raises(TypeError, match="forecast_quantiles_batch"):
        normalise_in_context(
            est, df, "y", design=design, pipeline=FakeICLPipeline(), quantiles=(0.1, 0.5, 0.9)
        )
    with pytest.raises(ValueError, match="estimate"):
        res.to_normet_frame("bogus")


def test_log_transform_icn_additive_pairing_and_guards() -> None:
    df = _synthetic(n=1800, seed=19)
    # Shift values near zero so linear paired would explode or cross zero, keeping >= 0
    df["y"] = np.maximum(df["y"] - 8.5, 0.0)
    est = Chronos2Estimator()
    est.fit(df[["w"]], df["y"])
    design = ICNDesign(
        context_length=600,
        block_h=96,
        stride=96,
        direction="bidirectional",
        n_resamples=4,
        log_transform=True,
        seed=21,
    )
    bpipe = BatchFakeICLPipeline(spread=0.4)
    res = normalise_in_context(
        est, df, "y", design=design, pipeline=bpipe, quantiles=(0.1, 0.5, 0.9)
    )
    assert (res.level.dropna() >= 0.0).all()
    assert (res.paired.dropna() >= 0.0).all()
    assert res.quantiles is not None and (res.quantiles.dropna().to_numpy() >= 0.0).all()
    assert (
        res.paired_quantiles is not None and (res.paired_quantiles.dropna().to_numpy() >= 0.0).all()
    )

    bad_df = df.copy()
    bad_df.iloc[10, bad_df.columns.get_loc("y")] = -2.0
    with pytest.raises(ValueError, match="non-negative"):
        normalise_in_context(est, bad_df, "y", design=design, pipeline=bpipe)


def test_icn_draw_bank_matches_direct_across_aggregations() -> None:
    df = _synthetic(n=1800, seed=23)
    est = Chronos2Estimator()
    est.fit(df[["w"]], df["y"])
    design = ICNDesign(
        context_length=600,
        block_h=96,
        stride=48,
        direction="bidirectional",
        aggregate="trimmed",
        lead_tau=0.0,
        n_resamples=3,
        seed=9,
    )
    bpipe = BatchFakeICLPipeline(spread=2.0)
    bank = build_icn_draw_bank(
        est, df, "y", design=design, pipeline=bpipe, quantiles=(0.1, 0.5, 0.9)
    )
    calls_after_build = bpipe.quantile_calls
    assert calls_after_build > 0

    # Zero extra model calls across all re-aggregation modes and orientations
    for direction, agg, tau in (
        ("forward", "mean", 0.0),
        ("reverse", "median", 0.0),
        ("bidirectional", "trimmed", 24.0),
    ):
        from_bank = bank.aggregate(direction=direction, aggregate=agg, lead_tau=tau)
        assert bpipe.quantile_calls == calls_after_build
        direct = normalise_in_context(
            est,
            df,
            "y",
            design=ICNDesign(
                context_length=600,
                block_h=96,
                stride=48,
                direction=direction,
                aggregate=agg,
                lead_tau=tau,
                n_resamples=3,
                seed=9,
            ),
            pipeline=BatchFakeICLPipeline(spread=2.0),
            quantiles=(0.1, 0.5, 0.9),
        )
        np.testing.assert_allclose(
            from_bank.level.to_numpy(), direct.level.to_numpy(), equal_nan=True
        )
        np.testing.assert_allclose(
            from_bank.paired.to_numpy(), direct.paired.to_numpy(), equal_nan=True
        )
        assert from_bank.paired_quantiles is not None and direct.paired_quantiles is not None
        np.testing.assert_allclose(
            from_bank.paired_quantiles.to_numpy(),
            direct.paired_quantiles.to_numpy(),
            equal_nan=True,
        )

    fwd_only = build_icn_draw_bank(
        est,
        df,
        "y",
        design=ICNDesign(
            context_length=600, block_h=96, stride=96, direction="forward", n_resamples=2
        ),
        pipeline=bpipe,
    )
    with pytest.raises(ValueError, match="no cached 'reverse'"):
        fwd_only.aggregate(direction="reverse")


def test_empty_met_cols_fast_path() -> None:
    df = _synthetic(n=1500, seed=31)
    est = _estimator()
    bpipe = BatchFakeICLPipeline()
    design = ICNDesign(
        context_length=500, block_h=100, stride=100, direction="forward", n_resamples=5, seed=1
    )
    res = normalise_in_context(est, df, "y", design=design, pipeline=bpipe, met_features=[])
    covered = np.isfinite(res.paired.to_numpy())
    assert covered.any()
    np.testing.assert_allclose(
        res.paired.to_numpy()[covered], df["y"].to_numpy()[covered], atol=1e-12
    )
    # 10 anchors * 1 factual item (not 1 + 5) fits cleanly into a single batch
    assert bpipe.batch_calls == 1


def test_normalise_in_context_multivariate() -> None:
    rng = np.random.default_rng(7)
    n = 1556
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    w = rng.standard_normal(n)
    y1 = 30.0 + 4.0 * w + 0.1 * rng.standard_normal(n)
    y2 = 50.0 - 3.0 * w + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame({"y1": y1, "y2": y2, "w": w}, index=idx)

    est = Chronos2Estimator()
    seen_cols: list[list[str]] = []

    class _FakeMultiVarPipe:
        def forecast_multivariate_batch(
            self,
            items: list[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]],
            cols: list[str],
            targets: list[str],
            quantiles: list[float] | None = None,
        ) -> np.ndarray:
            seen_cols.append(list(cols))
            batches: list[np.ndarray] = []
            for hist, past, fut in items:
                xp = past[cols].to_numpy(dtype=float)
                xf = fut[cols].to_numpy(dtype=float)
                per_var: list[np.ndarray] = []
                for vi in range(len(targets)):
                    yp = hist[vi].astype(float)
                    good = np.isfinite(yp) & np.all(np.isfinite(xp), axis=1)
                    beta = np.linalg.lstsq(
                        np.column_stack([np.ones(good.sum()), xp[good]]), yp[good], rcond=None
                    )[0]
                    med = beta[0] + xf @ beta[1:]
                    if quantiles is None:
                        per_var.append(med)
                    else:
                        per_var.append(np.stack([med + 2.0 * (float(q) - 0.5) for q in quantiles]))
                batches.append(np.stack(per_var, axis=0))
            return np.stack(batches, axis=0)

    design = ICNDesign(
        context_length=500, block_h=96, stride=96, direction="bidirectional", n_resamples=3, seed=4
    )
    out = normalise_in_context_multivariate(
        est,
        df,
        ["y1", "y2"],
        design=design,
        pipeline=_FakeMultiVarPipe(),
        quantiles=(0.1, 0.5, 0.9),
    )
    assert set(out) == {"y1", "y2"}
    assert all("y1" not in c and "y2" not in c and c == ["w"] for c in seen_cols)
    for key in ("y1", "y2"):
        res = out[key]
        assert np.isfinite(res.level.to_numpy()).all()
        assert res.quantiles is not None and res.paired_quantiles is not None
        assert list(res.quantiles.columns) == ["q100", "q500", "q900"]


def test_normalise_in_context_multisite_cross_learning_slot_batching() -> None:
    df_long = _synthetic(n=1800, seed=41)
    # 1800 - 1224 = 576 = 6 * 96 so SHORT anchors align on exact LONG anchor timestamps
    df_short = _synthetic(n=1224, seed=42).copy()
    df_short.index = df_long.index[-1224:]
    frames = {"LONG": df_long, "SHORT": df_short}
    est = Chronos2Estimator(met_covariates=["w"])

    recorded_batches: list[list[tuple[np.ndarray, pd.DataFrame, pd.DataFrame]]] = []

    class _RecordingSlotBatchPipe(BatchFakeICLPipeline):
        def forecast_batch(self, items, cols, target):
            recorded_batches.append(items)
            # In slot batching, items within a single batch come from distinct
            # sites in the same slot (each site's factual future is distinct or matched
            # separately from counterfactual slots).
            return np.stack([self.forecast(h, p, f, cols, target) for h, p, f in items])

    design = ICNDesign(
        context_length=800,
        min_context=400,
        block_h=96,
        stride=96,
        direction="forward",
        n_resamples=2,
        seed=8,
    )
    bpipe = _RecordingSlotBatchPipe()
    res_cl = normalise_in_context_multisite(
        est, frames, "y", design=design, pipeline=bpipe, cross_learning=True
    )
    assert set(res_cl) == {"LONG", "SHORT"}
    assert np.isfinite(res_cl["LONG"].level.to_numpy()).any()
    assert np.isfinite(res_cl["SHORT"].level.to_numpy()).any()

    # Co-occurring anchor timestamps have batch size == 2 (one call per slot: 1 factual + 2 cf)
    assert any(len(b) == 2 for b in recorded_batches)
    # Within a batch of 2, both entries share the exact same future horizon and belong to
    # distinct site histories rather than mixing factual + counterfactual for a single site.
    for batch in recorded_batches:
        if len(batch) == 2:
            assert batch[0][1].index[0] != batch[1][1].index[0] or not np.allclose(
                batch[0][0], batch[1][0], equal_nan=True
            )

    res_apart = normalise_in_context_multisite(
        est, frames, "y", design=design, pipeline=FakeICLPipeline(), cross_learning=False
    )
    assert set(res_apart) == {"LONG", "SHORT"}
    assert np.isfinite(res_apart["SHORT"].level.to_numpy()).any()
