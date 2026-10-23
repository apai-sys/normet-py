"""Staggered synthetic controls: several treated units, each with its own cutoff.

The panel is the clean case: both treated units share a common pre-period
signal that the donors span (0.5*D1 + 0.5*D2), so every pre-period fits
perfectly and an injected post-cutoff step is read almost exactly. With the
step the effects are far outside the donor-only placebo reference (the p-value
sits at its minimum, 1/(n_donors + 1)); without it the effects sit at zero and
the p-value at its maximum (1.0). The summary rows are checked against the
units' own effects, and the input guards against the staggered-specific
mistakes: a treated unit inside the donor pool, a misaligned cutoff list, and
a treated unit whose cutoff falls after its last observation. The explicit
pool test also pins the backend's own constraint: a placebo's reduced pool
must be large enough for the default backend to fit (three donors), so the
pool is tested at four of five donors.
"""

import numpy as np
import pandas as pd
import pytest

from normet import run_scm_staggered

N_DAYS = 160
T1_CUTOFF = "2023-03-02"  # day 60
T2_CUTOFF = "2023-04-01"  # day 90
DONORS = ["D1", "D2", "D3", "D4", "D5"]


def _staggered_panel(effect: float = 5.0) -> pd.DataFrame:
    """Two treated units (T1 at day 60, T2 at day 90) plus five donors; the
    treated pre-period is 0.5*D1 + 0.5*D2 plus the step after each cutoff."""
    t = np.arange(N_DAYS)
    dates = pd.date_range("2023-01-01", periods=N_DAYS, freq="D")
    base = {}
    for j, name in enumerate(DONORS):
        base[name] = 5 + j + 3 * np.sin(2 * np.pi * t / (30 - 4 * j) + j)
    common = 0.5 * base["D1"] + 0.5 * base["D2"]
    series = {
        "T1": common + effect * (t >= 60),
        "T2": common + effect * (t >= 90),
        **base,
    }
    rows = []
    for name, vals in series.items():
        for i, d in enumerate(dates):
            rows.append({"date": d, "ID": name, "value": float(vals[i])})
    return pd.DataFrame(rows)


def _run(df, **kw):
    kw = {
        "treated_units": ["T1", "T2"],
        "cutoff_dates": {"T1": T1_CUTOFF, "T2": T2_CUTOFF},
        "date_col": "date",
        "unit_col": "ID",
        "outcome_col": "value",
        "n_cores": 2,
        **kw,
    }
    return run_scm_staggered(df, **kw)


# ------------------------------------------------------------------- the fit
def test_staggered_reads_each_units_step_and_ranks_it():
    out = _run(_staggered_panel())

    by_unit = out.set_index("unit")
    assert list(by_unit["cutoff_date"])[:2] == [T1_CUTOFF, T2_CUTOFF]
    assert by_unit.loc["T1", "n_pre"] == 60
    assert by_unit.loc["T1", "n_post"] == 100
    assert by_unit.loc["T2", "n_pre"] == 90
    assert by_unit.loc["T2", "n_post"] == 70
    # The step is read almost exactly, and the pre-period is a clean fit.
    assert np.isclose(by_unit.loc["T1", "effect"], 5.0, atol=0.05)
    assert np.isclose(by_unit.loc["T2", "effect"], 5.0, atol=0.05)
    assert by_unit.loc[["T1", "T2"], "pre_r2"].min() > 0.99
    # Five donors: with no placebo reaching the step the p-value sits at its
    # minimum, (0 + 1) / (5 + 1).
    assert by_unit.loc[["T1", "T2"], "p_value"].tolist() == [
        pytest.approx(1 / 6),
        pytest.approx(1 / 6),
    ]


def test_staggered_null_gives_the_maximum_p():
    out = _run(_staggered_panel(effect=0.0))

    by_unit = out.set_index("unit")
    assert by_unit.loc[["T1", "T2"], "effect"].abs().max() < 1e-3
    assert by_unit.loc[["T1", "T2"], "p_value"].tolist() == [1.0, 1.0]
    assert out.loc[out["unit"] == "mean", "p_value"].iloc[0] == 1.0


def test_staggered_summary_rows_aggregate_the_units():
    out = _run(_staggered_panel())

    units = out[out["unit"].isin(["T1", "T2"])]
    assert list(out["unit"]) == ["T1", "T2", "mean", "median"]
    mean_row = out[out["unit"] == "mean"].iloc[0]
    median_row = out[out["unit"] == "median"].iloc[0]
    assert np.isclose(mean_row["effect"], units["effect"].mean(), rtol=1e-12)
    assert np.isclose(median_row["effect"], units["effect"].median(), rtol=1e-12)
    # The cross-unit test is stated on the mean; the median row is descriptive.
    assert np.isclose(mean_row["p_value"], 1 / 6)
    assert np.isnan(median_row["p_value"])
    assert mean_row["cutoff_date"] is None or pd.isna(mean_row["cutoff_date"])


def test_staggered_list_cutoffs_match_the_dict_form():
    df = _staggered_panel()
    out_dict = _run(df)
    out_list = _run(df, cutoff_dates=[T1_CUTOFF, T2_CUTOFF])
    pd.testing.assert_frame_equal(out_dict, out_list)


def test_staggered_explicit_donor_pool_sets_the_reference_size():
    # Four of the five donors: the reference holds four placebos, so the
    # minimum p-value is (0 + 1) / (4 + 1) — the pool is honoured, and the
    # reduced pool (three donors) is still large enough for the default
    # backend to fit each placebo.
    out = _run(_staggered_panel(), donors=["D1", "D2", "D3", "D4"])
    by_unit = out.set_index("unit")
    assert by_unit.loc["T1", "p_value"] == pytest.approx(0.2)
    assert np.isclose(by_unit.loc["T1", "effect"], 5.0, atol=0.05)


def test_staggered_post_agg_sum_reads_the_cumulative_effect():
    out = _run(_staggered_panel(effect=2.0), post_agg="sum")
    by_unit = out.set_index("unit")
    # 2 per day over 100 (T1) and 70 (T2) post days.
    assert np.isclose(by_unit.loc["T1", "effect"], 200.0, atol=2.0)
    assert np.isclose(by_unit.loc["T2", "effect"], 140.0, atol=2.0)


# ------------------------------------------------------------------- the guards
def test_staggered_rejects_an_empty_or_duplicate_treated_list():
    df = _staggered_panel()
    with pytest.raises(ValueError, match="must not be empty"):
        _run(df, treated_units=[])
    with pytest.raises(ValueError, match="must be unique"):
        _run(df, treated_units=["T1", "T1"], cutoff_dates={"T1": T1_CUTOFF})


def test_staggered_rejects_units_absent_from_the_panel():
    with pytest.raises(ValueError, match="not in the panel"):
        _run(_staggered_panel(), treated_units=["T1", "Nope"])


def test_staggered_rejects_a_misaligned_cutoff_list():
    df = _staggered_panel()
    with pytest.raises(ValueError, match="align"):
        _run(df, cutoff_dates=[T1_CUTOFF])
    with pytest.raises(ValueError, match="dict"):
        _run(df, cutoff_dates="2023-03-02")


def test_staggered_rejects_a_treated_unit_inside_the_donors():
    with pytest.raises(ValueError, match="must not include treated"):
        _run(_staggered_panel(), donors=["D1", "T2"])


def test_staggered_rejects_a_design_with_no_donors_left():
    df = _staggered_panel()
    with pytest.raises(ValueError, match="No donor units"):
        _run(
            df,
            treated_units=["T1", "T2", "D1", "D2", "D3", "D4", "D5"],
            cutoff_dates={
                "T1": T1_CUTOFF,
                "T2": T2_CUTOFF,
                "D1": T1_CUTOFF,
                "D2": T2_CUTOFF,
                "D3": T1_CUTOFF,
                "D4": T2_CUTOFF,
                "D5": T1_CUTOFF,
            },
        )


def test_staggered_rejects_a_cutoff_after_the_units_last_observation():
    with pytest.raises(ValueError, match="No post-cutoff observations"):
        _run(
            _staggered_panel(),
            cutoff_dates={"T1": T1_CUTOFF, "T2": "2024-01-01"},
        )
