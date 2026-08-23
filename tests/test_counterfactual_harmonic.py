"""Tests for the harmonic / calendar counterfactual.

``test_predict_is_invariant_to_horizon_length`` is the one that matters.
``_build_feature_matrix`` used to let ``pd.get_dummies(..., drop_first=True)``
pick the reference level from whatever data it was handed, so a short prediction
window dropped a different day-of-week x hour column than ``fit`` had. The
``reindex`` in ``predict`` then filled that column with zeros, quietly putting
every row of that cell on the reference level: predicting the same timestamp
gave a different answer depending on how long the surrounding window was
(0.95 ug/m3 apart on a 34 ug/m3 mean, measured before the fix).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from normet.counterfactual import HarmonicCounterfactual, HarmonicCounterfactualResult


@pytest.fixture(scope="module")
def baseline() -> pd.Series:
    """Hourly series with a diurnal cycle, a weekday step and an annual harmonic."""
    idx = pd.date_range("2019-01-01", periods=24 * 400, freq="h")
    rng = np.random.default_rng(0)
    y = (
        30.0
        + 10.0 * np.sin(2 * np.pi * idx.hour / 24.0)
        + 5.0 * (idx.dayofweek < 5)
        + 8.0 * np.sin(2 * np.pi * idx.dayofyear / 365.25)
        + rng.normal(0, 1.0, len(idx))
    )
    return pd.Series(y, index=idx)


@pytest.fixture(scope="module")
def fitted(baseline) -> HarmonicCounterfactual:
    return HarmonicCounterfactual().fit(baseline)


def test_predict_is_invariant_to_horizon_length(fitted, baseline):
    """The same timestamp must get the same prediction in any window."""
    t0 = baseline.index[0]
    long_idx = pd.date_range("2020-02-05", periods=24 * 30, freq="h")
    short_idx = pd.date_range("2020-02-05", periods=24 * 3, freq="h")

    from_long = fitted.predict(long_idx, t0_timestamp=t0).loc[short_idx]
    from_short = fitted.predict(short_idx, t0_timestamp=t0)

    pd.testing.assert_series_equal(from_long, from_short)


def test_predict_is_invariant_for_a_single_hour(fitted, baseline):
    """The degenerate horizon: one row covers one dow x hour cell."""
    t0 = baseline.index[0]
    long_idx = pd.date_range("2020-02-05", periods=24 * 30, freq="h")
    one = pd.date_range("2020-02-05 00:00", periods=1, freq="h")

    np.testing.assert_allclose(
        fitted.predict(long_idx, t0_timestamp=t0).loc[one].to_numpy(),
        fitted.predict(one, t0_timestamp=t0).to_numpy(),
    )


def test_design_matrix_columns_do_not_depend_on_the_window(fitted, baseline):
    """168 dow x hour cells minus the one dropped reference, in every window."""
    t0 = baseline.index[0]
    dh_fit = [c for c in fitted.feature_names if c.startswith("dh_")]
    assert len(dh_fit) == 167

    short = fitted._build_feature_matrix(
        pd.date_range("2020-02-05", periods=24 * 3, freq="h"), t0_timestamp=t0
    )
    assert [c for c in short.columns if c.startswith("dh_")] == dh_fit


def test_fit_requires_a_datetime_index():
    with pytest.raises(ValueError, match="pd.DatetimeIndex"):
        HarmonicCounterfactual().fit(pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2]))


def test_evaluate_intervention_requires_a_datetime_index():
    with pytest.raises(ValueError, match="pd.DatetimeIndex"):
        HarmonicCounterfactual().evaluate_intervention(
            pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2]),
            train_end_date="2019-12-31",
            intervention_start_date="2020-03-23",
        )


def test_predict_before_fit_is_refused():
    with pytest.raises(RuntimeError, match="not fitted"):
        HarmonicCounterfactual().predict(
            pd.date_range("2020-01-01", periods=24, freq="h"),
            t0_timestamp=pd.Timestamp("2020-01-01"),
        )


def test_evaluate_intervention_recovers_an_injected_step():
    """A 40% cut must come back as roughly -40%, with near-zero validation bias."""
    idx = pd.date_range("2019-01-01", periods=24 * 500, freq="h")
    rng = np.random.default_rng(1)
    base = (
        30.0
        + 10.0 * np.sin(2 * np.pi * idx.hour / 24.0)
        + 5.0 * (idx.dayofweek < 5)
        + 8.0 * np.sin(2 * np.pi * idx.dayofyear / 365.25)
        + rng.normal(0, 2.0, len(idx))
    )
    observed = base * np.where(idx >= pd.Timestamp("2020-03-23"), 0.6, 1.0)

    res = HarmonicCounterfactual().evaluate_intervention(
        pd.Series(observed, index=idx),
        train_end_date="2019-12-31",
        intervention_start_date="2020-03-23",
        val_start_date="2020-01-01",
    )

    assert isinstance(res, HarmonicCounterfactualResult)
    assert res.summary_metrics["net_impact_pct"] == pytest.approx(-40.0, abs=3.0)
    # The validation window is pre-intervention, so the counterfactual should
    # track the observations there; a large bias means the fit is extrapolating
    # badly and the headline impact cannot be trusted either.
    assert abs(res.summary_metrics["validation_bias_pct"]) < 5.0


def test_confidence_band_brackets_the_counterfactual():
    idx = pd.date_range("2019-01-01", periods=24 * 200, freq="h")
    rng = np.random.default_rng(2)
    s = pd.Series(30.0 + rng.normal(0, 3.0, len(idx)), index=idx)

    res = HarmonicCounterfactual().evaluate_intervention(
        s, train_end_date="2019-04-30", intervention_start_date="2019-06-01"
    )

    assert (res.counterfactual_p10 <= res.counterfactual_bau).all()
    assert (res.counterfactual_bau <= res.counterfactual_p90).all()
    assert (res.counterfactual_p10 >= 0.0).all()

    df = res.to_dataframe()
    assert list(df.columns) == [
        "observed",
        "counterfactual_bau",
        "counterfactual_p10",
        "counterfactual_p90",
        "impact",
        "impact_pct",
    ]
    assert np.isfinite(df.to_numpy()).all()
