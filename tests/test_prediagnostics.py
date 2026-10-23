"""Pre-trend diagnostics for synthetic-control fits.

The ``scm_panel`` fixture is the clean case: the treated unit equals
0.5*D1 + 0.5*D2 plus tiny noise before the cutoff, so the pre-period is
parallel and the test must stay quiet. Drift is then injected into the
treated unit's pre-period and the test must catch it.
"""

import numpy as np
import pandas as pd
import pytest

from normet.causal import run_scm, scm_diagnostics
from normet.causal.prediagnostics import PreTrend, pre_trend_test
from normet.causal.scm import scm

CUTOFF = "2023-05-01"
DONORS = ["D1", "D2", "D3", "D4", "D5", "D6"]


def _fit(panel: pd.DataFrame) -> pd.DataFrame:
    return scm(
        df=panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date=CUTOFF,
        donors=DONORS,
    )["synthetic"]


def test_pre_trend_is_quiet_on_a_parallel_pre_period(scm_panel):
    out = _fit(scm_panel)
    pt = pre_trend_test(out, CUTOFF)

    assert isinstance(pt, PreTrend)
    assert pt.n_periods == 120
    assert pt.verdict.startswith("ok")
    assert pt.p_slope >= 0.05
    assert 0.7 <= pt.rmpe_drift <= 1.3
    assert str(pt) == pt.verdict


def _drifted_panel(scm_panel: pd.DataFrame, ramp: float = 2.0) -> pd.DataFrame:
    """The toy panel with a linear pre-period ramp on the treated unit.

    The ramp starts at zero at the first day and reaches ``ramp`` at the day
    before the cutoff; the donors carry no trend, so the synthetic control
    cannot reproduce it and it survives in the pre-period residuals.
    """
    panel = scm_panel.copy()
    dates = pd.to_datetime(panel["date"])
    cutoff_ts = pd.Timestamp(CUTOFF)
    pre_days = dates < cutoff_ts
    first = dates.min()
    n_pre = int(pre_days.sum() // len(DONORS))  # one row per date per unit
    t = (dates - first).dt.days
    ramp_values = np.where(pre_days, ramp * t / max(n_pre - 1, 1), 0.0)
    is_t = panel["ID"].to_numpy() == "T"
    panel.loc[is_t, "value"] += ramp_values[is_t]
    return panel


def test_pre_trend_catches_an_injected_drift(scm_panel):
    """A slow linear ramp: the slope is decisive even though the drift ratio
    stays mild (the fit absorbs much of a monotone ramp's level)."""
    out = _fit(_drifted_panel(scm_panel))
    pt = pre_trend_test(out, CUTOFF)

    assert pt.verdict.startswith("fail")
    assert pt.p_slope < 0.01
    assert pt.rmpe_drift > 1.0
    assert pt.slope > 0


def test_pre_trend_flags_a_late_fit_breakdown_without_a_trend(scm_panel):
    """Noise that doubles in the final third of the pre-period: no trend (the
    slope stays quiet), but the drift ratio says the fit stops tracking."""
    rng = np.random.default_rng(7)
    panel = scm_panel.copy()
    dates = pd.to_datetime(panel["date"])
    cutoff_ts = pd.Timestamp(CUTOFF)
    pre_max = dates[dates < cutoff_ts].max()
    is_t = panel["ID"].to_numpy() == "T"
    is_late = ((dates > pre_max - pd.Timedelta(days=40)) & (dates < cutoff_ts)).to_numpy()
    extra = np.where(is_t & is_late, rng.normal(0, 1.25, len(panel)), 0.0)
    panel["value"] = panel["value"] + extra
    out = _fit(panel)
    pt = pre_trend_test(out, CUTOFF)

    assert pt.verdict.startswith("caution")
    assert "deteriorates" in pt.verdict
    assert pt.p_slope >= 0.01
    assert pt.rmpe_drift > 1.25


def test_pre_trend_reads_a_bare_dataframe_and_a_dict_the_same(scm_panel):
    out = _fit(scm_panel)
    pt_df = pre_trend_test(out, CUTOFF)
    pt_dict = pre_trend_test({"synthetic": out, "weights": None}, CUTOFF)

    assert pt_df.p_slope == pt_dict.p_slope
    assert pt_df.verdict == pt_dict.verdict


def test_pre_trend_is_insufficient_on_a_short_pre_period(scm_panel):
    # Four pre-period days: the counterfactual machinery still runs, but the
    # trend test refuses a p-value from two degrees of freedom.
    panel = scm_panel[pd.to_datetime(scm_panel["date"]) >= pd.Timestamp("2023-04-27")]
    out = _fit(panel)
    pt = pre_trend_test(out, CUTOFF)

    assert pt.n_periods == 4
    assert pt.verdict.startswith("insufficient")
    assert not np.isfinite(pt.slope)
    assert not np.isfinite(pt.p_slope)
    assert not np.isfinite(pt.rmpe_drift)


def test_pre_trend_rejects_a_frame_without_the_columns():
    with pytest.raises(ValueError, match="observed"):
        pre_trend_test(pd.DataFrame({"a": [1.0, 2.0]}), CUTOFF)
    with pytest.raises(ValueError, match="'synthetic' key"):
        pre_trend_test({"effect": pd.Series([1.0])}, CUTOFF)


def test_scm_diagnostics_carries_the_pre_trend(scm_panel):
    out = _fit(scm_panel)
    diag = scm_diagnostics(out, CUTOFF)
    pt = pre_trend_test(out, CUTOFF)

    expected_old = {
        "pre_n",
        "pre_rmse",
        "pre_mae",
        "pre_mape",
        "pre_r2",
        "post_n",
        "att",
        "att_cum",
        "post_rmse",
        "hhi",
        "effective_n_donors",
        "top_donors",
        "top_donor_share",
        "n_donors",
    }
    assert expected_old <= set(diag.keys())
    assert set(diag["pre_trend"]) == {"slope", "p_slope", "rmpe_drift", "n_periods", "verdict"}
    assert diag["pre_trend"]["p_slope"] == pt.p_slope
    assert diag["pre_trend"]["slope"] == pt.slope
    assert diag["pre_trend"]["verdict"] == pt.verdict


def test_run_scm_result_feeds_the_test(scm_panel):
    res = run_scm(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date=CUTOFF,
        donors=DONORS,
    )
    pt = pre_trend_test(res, CUTOFF)
    assert pt.n_periods == 120
    assert pt.verdict.startswith("ok")
