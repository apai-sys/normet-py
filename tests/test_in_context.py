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
    normalise_in_context,
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
