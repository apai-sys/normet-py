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
