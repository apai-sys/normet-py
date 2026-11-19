"""Conformal effect interval + RMSPE-ratio placebo test."""

import numpy as np
import pandas as pd
import pytest

from normet.causal import (
    conformal_effect_interval,
    placebo_in_space,
    rmspe_ratio_test,
    run_scm,
    scm,
)


def test_conformal_effect_interval_recovers_shock(scm_panel):
    out = run_scm(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=["D1", "D2", "D3", "D4", "D5", "D6"],
        scm_backend="scm",
    )
    ci = conformal_effect_interval(out, cutoff_date="2023-05-01", n_perm=400, ci_level=0.95)
    # Observed shock is +10; the CI should bracket something near it.
    assert ci["att"] > 5.0
    assert ci["low"] < ci["att"] < ci["high"]
    # p-value should be small for such a large effect.
    assert ci["p_value"] < 0.05


def test_conformal_interval_builds_the_null_from_the_raw_effect():
    """A noiseless effect of exactly +10 must give the degenerate CI [10, 10].

    The sub-sampling null has to be imposed (post-period residuals shifted by
    the observed ATT); drawing block means from the *raw* effect series
    instead lets every mixed pre/post window carry part of the effect under
    test, so the null spreads over 0..10 and the interval comes back wide
    around the very estimate being tested.
    """
    idx = pd.date_range("2023-01-01", periods=200, freq="D")
    effect = np.r_[np.zeros(100), np.full(100, 10.0)]
    scm_out = pd.DataFrame({"effect": effect}, index=idx)

    ci = conformal_effect_interval(scm_out, cutoff_date="2023-04-11", n_perm=200)

    assert ci["att"] == pytest.approx(10.0)
    assert ci["low"] == pytest.approx(10.0)
    assert ci["high"] == pytest.approx(10.0)
    assert ci["p_value"] < 0.01


def test_rmspe_ratio_with_placebos(scm_panel):
    pl = placebo_in_space(
        scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=["D1", "D2", "D3", "D4", "D5", "D6"],
        scm_backend="scm",
    )
    out = rmspe_ratio_test(pl, cutoff_date="2023-05-01")
    assert "treated_ratio" in out and "placebo_ratios" in out
    # Treated unit has a real shock → its ratio should be largest (rank 1).
    assert out["rank"] == 1
    assert out["treated_ratio"] > out["placebo_ratios"].max()


def test_an_undefined_treated_rmspe_ratio_gives_nan_not_significance():
    """A zero pre-RMSPE leaves the treated ratio undefined.

    ``NaN >= r`` is False for every placebo, so the old p-value came back
    ``1/(n+1)`` -- the smallest possible p, read as the strongest effect --
    from a comparison that never happened.
    """
    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    treated = pd.DataFrame({"effect": np.zeros(10)}, index=idx)  # pre-RMSPE = 0
    placebo = pd.DataFrame(
        {"effect": np.r_[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0]}, index=idx
    )
    out = rmspe_ratio_test(
        {"treated": treated, "placebos": {"D1": placebo, "D2": placebo * 2.0}},
        cutoff_date="2023-01-06",
    )
    assert np.isfinite(out["placebo_ratios"]).all()
    assert np.isnan(out["treated_ratio"])
    assert np.isnan(out["p_value"])
    assert np.isnan(out["rank"])
