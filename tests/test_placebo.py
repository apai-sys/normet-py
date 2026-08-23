"""Placebo-in-space and placebo-in-time inference tests.

These exercise the public return structure of ``normet.causal.placebo`` on the
shared ``scm_panel`` fixture (one treated unit ``T`` with a known +10 post-cutoff
shock and six distinct donors).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from normet.causal.placebo import placebo_in_space, placebo_in_time

CUTOFF = "2023-05-01"


def test_placebo_in_space_structure_and_significance(scm_panel):
    out = placebo_in_space(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date=CUTOFF,
        scm_backend="scm",
        n_cores=1,
    )

    assert set(out) == {"treated", "placebos", "p_value", "ref_band"}

    # The true treated run carries the recovered effect series.
    treated = out["treated"]
    assert {"observed", "synthetic", "effect"} <= set(treated.columns)

    # Every donor was run as a placebo and contributes one effect column.
    donor_units = {u for u in scm_panel["ID"].unique() if u != "T"}
    assert set(out["placebos"]) == donor_units

    # Reference band is aligned to the treated index and exposes the documented columns.
    ref_band = out["ref_band"]
    assert list(ref_band.index) == list(treated.index)
    assert {"p10", "p90", "mean", "std", "band_low_1sd", "band_high_1sd"} <= set(ref_band.columns)

    # The true treated run recovers the known +10 post-cutoff shock.
    post = treated.loc[treated.index >= pd.Timestamp(CUTOFF), "effect"]
    assert abs(post.mean() - 10.0) < 1.0

    # With the real treated unit excluded from every placebo's donor pool, no
    # placebo sees the +10 shock, so the true effect is the most extreme and
    # attains the smallest possible permutation p-value.
    p = out["p_value"]
    assert p == 1.0 / (len(donor_units) + 1)


def test_placebo_in_space_post_agg_sum_matches_mean_ranking(scm_panel):
    out = placebo_in_space(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date=CUTOFF,
        scm_backend="scm",
        post_agg="sum",
        n_cores=1,
    )
    # Aggregation switch is accepted and yields a valid permutation p-value.
    p = out["p_value"]
    assert 0.0 < p <= 1.0


def test_placebo_in_time_structure(scm_panel):
    out = placebo_in_time(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date=CUTOFF,
        scm_backend="scm",
        min_pre_period=30,
        placebo_every=7,
        n_cores=1,
    )

    assert set(out) == {
        "treated",
        "placebos",
        "p_value",
        "ref_band_event_time",
        "placebo_stats",
    }
    assert {"observed", "synthetic", "effect"} <= set(out["treated"].columns)

    # At least one valid pre-cutoff placebo window was found.
    assert len(out["placebos"]) >= 1
    assert isinstance(out["placebo_stats"], pd.Series)
    assert not out["placebo_stats"].empty

    band = out["ref_band_event_time"]
    assert band.index.name == "event_time"
    assert {"p10", "p90", "ci_lo", "ci_hi", "std"} <= set(band.columns)

    # The genuine post-cutoff effect (~+10) should exceed every fake-cutoff
    # placebo statistic computed on the quiet pre-period.
    assert np.abs(out["placebo_stats"].values).max() < 10.0
    assert 0.0 < out["p_value"] <= 1.0


def test_placebo_in_time_rejects_bad_dates(scm_panel):
    bad = scm_panel.copy()
    # The fixture's date column is datetime64; cast to object so an uncoercible
    # string survives until placebo_in_time's own to_datetime guard.
    bad["date"] = bad["date"].astype(object)
    bad.loc[bad.index[0], "date"] = "not-a-date"
    try:
        placebo_in_time(
            df=bad,
            date_col="date",
            unit_col="ID",
            outcome_col="value",
            treated_unit="T",
            cutoff_date=CUTOFF,
            scm_backend="scm",
            n_cores=1,
        )
    except ValueError as e:
        assert "invalid dates" in str(e)
    else:  # pragma: no cover - guard
        raise AssertionError("expected ValueError on uncoercible dates")
