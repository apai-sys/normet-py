"""Diagnostics for SCM fits."""

import numpy as np
import pandas as pd

from normet.causal.diagnostics import _pre_period_fit, loo_weight_stability, scm_diagnostics
from normet.causal.scm import scm


def test_pre_mape_ignores_exact_zero_observations():
    """One zero observation must not turn the whole percentage metric to NaN."""
    idx = pd.date_range("2023-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {"observed": [0.0, 2.0, 4.0], "synthetic": [1.0, 1.0, 2.0], "effect": [0.0, 0.0, 0.0]},
        index=idx,
    )
    diag = _pre_period_fit(df, pd.Timestamp("2023-02-01"))

    # |0-1| is skipped (no denominator); |2-1|/2 and |4-2|/4 average to 0.5.
    assert diag["pre_mape"] == 0.5
    assert np.isfinite(diag["pre_rmse"])


def test_scm_diagnostics_keys_and_signs(scm_panel):
    out = scm(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=["D1", "D2", "D3", "D4", "D5", "D6"],
    )
    diag = scm_diagnostics(out, cutoff_date="2023-05-01")

    expected = {
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
    assert expected <= set(diag.keys())

    # Sanity on numbers
    assert diag["pre_n"] > 0 and diag["post_n"] > 0
    assert diag["n_donors"] == 6
    # HHI is in (1/n_donors, 1]
    assert 1.0 / 6 - 1e-9 <= diag["hhi"] <= 1.0
    # ATT should be near +10 (the toy shock)
    assert 8.0 < diag["att"] < 12.0
    # Pre-period fit on toy data should be excellent
    assert diag["pre_r2"] > 0.9


def test_loo_weight_stability_shape(scm_panel):
    donors = ["D1", "D2", "D3", "D4", "D5", "D6"]
    out = loo_weight_stability(
        scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=donors,
    )
    assert {"dropped_donor", "mean_abs_drift", "max_abs_drift", "effect_shift"} <= set(out.columns)
    assert set(out["dropped_donor"]) <= set(donors)


def _noise_panel(n_donors: int, n_pre: int = 12, n_post: int = 6) -> pd.DataFrame:
    """Donors on independent random walks, and a treated unit of pure noise."""
    import numpy as np

    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=n_pre + n_post, freq="MS")
    rows = [
        (d, f"D{j}", v)
        for j in range(n_donors)
        for d, v in zip(dates, 50 + np.cumsum(rng.normal(0, 3, len(dates))), strict=True)
    ]
    rows += [(d, "T", v) for d, v in zip(dates, rng.normal(50, 5, len(dates)), strict=True)]
    return pd.DataFrame(rows, columns=["date", "ID", "value"])


def _pre_r2(fit, panel: pd.DataFrame) -> float:
    donors = sorted(set(panel["ID"]) - {"T"})
    out = fit(
        df=panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2021-01-01",
        donors=donors,
    )
    return scm_diagnostics(out, cutoff_date="2021-01-01")["pre_r2"]


def test_augmented_scm_fits_noise_once_donors_outnumber_the_pre_period():
    # Why the SCM Studio checks a fit against a shuffled series before reading its
    # pre-period R²: the augmented estimator's ridge features are the units'
    # pre-period values, the treated unit's own among them. With more donors than
    # pre-period dates it reproduces a treated unit that is pure noise; with few
    # it cannot.
    many = _noise_panel(n_donors=40)  # 40 donors, 12 pre-period months
    few = _noise_panel(n_donors=4)
    assert _pre_r2(scm, many) > 0.9
    assert _pre_r2(scm, few) < 0.5
