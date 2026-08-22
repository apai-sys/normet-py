"""Tests for :func:`normet.causal.panel.prepare_panel`.

``prepare_panel`` is the screen standing between a ragged real-world panel and
``scm()``'s ridge fit, which drops any date where *any* unit is missing. A
handful of sparse donors can therefore collapse the usable sample to zero rows
without raising anything, which is the failure mode this function exists to
prevent -- so what it refuses and what it drops matter at least as much as what
it passes through.

Two of the cases here are regressions against silent-empty-result bugs the
implementation comments call out by name: tz-aware input compared against
tz-naive bounds, and sub-daily input reindexed onto a daily grid before being
aggregated onto it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from normet.causal.panel import prepare_panel
from normet.exceptions import DataError

CUTOFF = "2023-01-15"


def _long(frames: dict[str, pd.Series]) -> pd.DataFrame:
    """Stack ``{unit: series-indexed-by-date}`` into a ragged long panel.

    NaNs are dropped rather than carried, so a unit's absence from a date is
    expressed the way a real feed expresses it: a missing row.
    """
    rows = []
    for unit, series in frames.items():
        for date, value in series.dropna().items():
            rows.append({"date": date, "site": unit, "no2": float(value)})
    return pd.DataFrame(rows)


def _dense(n: int = 20, seed: int = 0) -> dict[str, pd.Series]:
    """One treated unit and three complete donors on a daily grid."""
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    return {
        unit: pd.Series(rng.normal(30.0, 2.0, n), index=dates) for unit in ("T", "D1", "D2", "D3")
    }


# ------------------------------------------------------------------ happy path


def test_dense_panel_survives_intact():
    out = prepare_panel(
        _long(_dense()),
        date_col="date",
        unit_col="site",
        outcome_col="no2",
        cutoff_date=CUTOFF,
        treated_unit="T",
    )

    assert list(out.columns) == ["date", "site", "no2"]
    assert set(out["site"]) == {"T", "D1", "D2", "D3"}
    assert not out["no2"].isna().any()
    # 20 dates x 4 units, nothing dropped.
    assert len(out) == 80


def test_diagnostics_ride_along_in_attrs():
    """The donor count and pre-period length are what callers screen on."""
    out = prepare_panel(
        _long(_dense()),
        date_col="date",
        unit_col="site",
        outcome_col="no2",
        cutoff_date=CUTOFF,
        treated_unit="T",
    )

    assert out.attrs["n_donors"] == 3
    assert out.attrs["n_pre_period_rows"] == 14  # 01-01 .. 01-14
    assert out.attrs["donor_ratio"] == pytest.approx(3 / 14)
    coverage = out.attrs["coverage"]
    assert set(coverage.columns) == {"pre_coverage", "post_coverage"}
    assert coverage.loc["D1", "pre_coverage"] == pytest.approx(1.0)


def test_small_gaps_are_interpolated_not_dropped():
    frames = _dense()
    frames["D1"].iloc[5:7] = np.nan  # a two-day outage, well inside coverage
    out = prepare_panel(
        _long(frames),
        date_col="date",
        unit_col="site",
        outcome_col="no2",
        cutoff_date=CUTOFF,
        treated_unit="T",
    )

    assert "D1" in set(out["site"])
    assert not out["no2"].isna().any()


# --------------------------------------------------------------------- screens


def test_donor_below_min_coverage_is_dropped():
    frames = _dense()
    frames["D3"].iloc[3:] = np.nan  # reports for 3 of 20 days
    out = prepare_panel(
        _long(frames),
        date_col="date",
        unit_col="site",
        outcome_col="no2",
        cutoff_date=CUTOFF,
        treated_unit="T",
        min_coverage=0.65,
    )

    assert "D3" not in set(out["site"])
    assert set(out["site"]) == {"T", "D1", "D2"}
    assert out.attrs["n_donors"] == 2


def test_treated_unit_is_kept_below_min_coverage():
    """Dropping the treated unit for sparseness would leave nothing to treat."""
    frames = _dense()
    frames["T"].iloc[2:16] = np.nan  # 6 of 20 days, far below the bar
    out = prepare_panel(
        _long(frames),
        date_col="date",
        unit_col="site",
        outcome_col="no2",
        cutoff_date=CUTOFF,
        treated_unit="T",
        min_coverage=0.65,
    )

    assert "T" in set(out["site"])
    assert not out["no2"].isna().any()


def test_unit_still_empty_after_interpolation_is_dropped(caplog):
    """Interpolation cannot invent a unit that never reported in the window."""
    frames = _dense()
    # D3 reports only *before* the requested window, so its column reindexes to
    # all-NaN and survives the coverage screen only because the bar is off.
    frames["D3"] = pd.Series([10.0, 11.0], index=pd.to_datetime(["2022-12-01", "2022-12-02"]))

    with caplog.at_level(logging.WARNING):
        out = prepare_panel(
            _long(frames),
            date_col="date",
            unit_col="site",
            outcome_col="no2",
            cutoff_date=CUTOFF,
            date_from="2023-01-01",
            date_to="2023-01-20",
            min_coverage=0.0,
        )

    assert "D3" not in set(out["site"])
    assert "still incomplete after interpolation" in caplog.text


def test_large_donor_pool_warns_about_the_ridge_fit(caplog):
    """Donors approaching the pre-period length destabilise scm()'s ridge."""
    dates = pd.date_range("2023-01-01", periods=8, freq="D")
    rng = np.random.default_rng(1)
    frames = {
        unit: pd.Series(rng.normal(30.0, 2.0, len(dates)), index=dates)
        for unit in ["T", *[f"D{i}" for i in range(10)]]
    }

    with caplog.at_level(logging.WARNING):
        out = prepare_panel(
            _long(frames),
            date_col="date",
            unit_col="site",
            outcome_col="no2",
            cutoff_date="2023-01-05",
            treated_unit="T",
        )

    assert out.attrs["n_donors"] == 10
    assert out.attrs["donor_ratio"] == pytest.approx(10 / 4)
    assert "max_donor_ratio" in caplog.text


# ------------------------------------------------------------------- refusals


def test_unparseable_dates_are_refused():
    df = _long(_dense())
    df["date"] = df["date"].astype(object)
    df.loc[0, "date"] = "not-a-date"
    with pytest.raises(DataError, match="invalid `date` values"):
        prepare_panel(
            df,
            date_col="date",
            unit_col="site",
            outcome_col="no2",
            cutoff_date=CUTOFF,
        )


def test_cutoff_outside_the_window_is_refused():
    """An empty pre- or post-period is a specification error, not a result."""
    with pytest.raises(DataError, match="empty pre- or post-period"):
        prepare_panel(
            _long(_dense()),
            date_col="date",
            unit_col="site",
            outcome_col="no2",
            cutoff_date="2022-06-01",
        )


def test_unknown_treated_unit_is_refused():
    with pytest.raises(DataError, match="not found"):
        prepare_panel(
            _long(_dense()),
            date_col="date",
            unit_col="site",
            outcome_col="no2",
            cutoff_date=CUTOFF,
            treated_unit="nowhere",
        )


def test_treated_unit_with_no_usable_data_fails_loudly():
    """Silently dropping the treated unit would return a donors-only panel."""
    frames = _dense()
    frames["T"] = pd.Series([5.0], index=pd.to_datetime(["2022-11-01"]))

    with pytest.raises(DataError, match="no usable data"):
        prepare_panel(
            _long(frames),
            date_col="date",
            unit_col="site",
            outcome_col="no2",
            cutoff_date=CUTOFF,
            date_from="2023-01-01",
            date_to="2023-01-20",
            treated_unit="T",
            min_coverage=0.0,
        )


# ---------------------------------------------------------------- regressions


def test_tz_aware_input_is_normalised_not_silently_emptied():
    """A tz-aware column against tz-naive bounds reindexes to all-NaN.

    pandas raises nothing for that comparison -- it just matches no timestamps
    -- so the symptom is an empty panel rather than an error.
    """
    frames = _dense()
    aware = {
        unit: series.set_axis(series.index.tz_localize("UTC")) for unit, series in frames.items()
    }

    out = prepare_panel(
        _long(aware),
        date_col="date",
        unit_col="site",
        outcome_col="no2",
        cutoff_date=CUTOFF,
        treated_unit="T",
    )

    assert len(out) == 80
    assert out["date"].dt.tz is None
    assert not out["no2"].isna().any()


def test_subdaily_input_is_aggregated_before_it_meets_a_daily_grid():
    """Hourly readings only touch a daily grid at midnight.

    Reindexing before resampling would classify every non-midnight reading as
    missing, so the panel would arrive dense-looking but built entirely from
    interpolation.
    """
    # Four readings a day and never at midnight, so the grid and the readings
    # share no timestamp at all.
    stamps = pd.date_range("2023-01-01 01:00", periods=20 * 4, freq="6h")
    rng = np.random.default_rng(2)
    frames = {
        unit: pd.Series(rng.normal(30.0, 2.0, len(stamps)), index=stamps)
        for unit in ("T", "D1", "D2")
    }

    out = prepare_panel(
        _long(frames),
        date_col="date",
        unit_col="site",
        outcome_col="no2",
        cutoff_date=CUTOFF,
        treated_unit="T",
        freq="D",
    )

    per_unit = out.groupby("site").size()
    assert set(per_unit) == {20}
    assert not out["no2"].isna().any()
    # The daily value is the mean of that day's readings, not a midnight sample.
    first_day = frames["D1"].loc["2023-01-01"].mean()
    got = out[(out["site"] == "D1") & (out["date"] == pd.Timestamp("2023-01-01"))]
    assert got["no2"].iloc[0] == pytest.approx(first_day)
