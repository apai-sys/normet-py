"""Diagnostics for SCM fits."""

import pandas as pd

from normet.causal.diagnostics import loo_weight_stability, scm_diagnostics
from normet.causal.scm import scm


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
