"""Toy-panel sanity checks for the SCM backend."""

import numpy as np
import pandas as pd

from normet.causal.scm import scm


def test_scm_recovers_known_effect(scm_panel):
    cutoff = "2023-05-01"
    out = scm(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date=cutoff,
        donors=["D1", "D2", "D3", "D4", "D5", "D6"],
    )
    syn = out["synthetic"]
    weights = out["weights"]

    # Donors D1 and D2 should dominate the weights (true generating process)
    top_two = weights.sort_values(ascending=False).head(2).index.tolist()
    assert set(top_two) == {"D1", "D2"}

    # Post-cutoff mean effect should be near the +10 shock
    post = syn.loc[syn.index >= pd.Timestamp(cutoff), "effect"]
    assert 8.5 < float(post.mean()) < 11.5

    # Pre-cutoff mean effect should be near zero
    pre = syn.loc[syn.index < pd.Timestamp(cutoff), "effect"]
    assert abs(float(pre.mean())) < 1.5


def test_scm_fallback_with_missing_donor(scm_panel):
    # Nulling one donor's outcome at a single post-cutoff date makes the donor
    # matrix non-finite, forcing the exact per-timestamp fallback path instead
    # of the batched-SVD fast path. The effect should still be recovered.
    df = scm_panel.copy()
    miss = (df["ID"] == "D3") & (df["date"] == pd.Timestamp("2023-06-01"))
    df.loc[miss, "value"] = np.nan

    out = scm(
        df=df,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=["D1", "D2", "D3", "D4", "D5", "D6"],
    )
    post = out["synthetic"].loc[out["synthetic"].index >= pd.Timestamp("2023-05-01"), "effect"]
    assert 8.0 < float(post.mean()) < 12.0


def test_scm_with_pre_covariates(scm_panel):
    # Exercises the pre-period covariate-augmentation branch.
    df = scm_panel.copy()
    df["pop"] = df["ID"].map(
        {"T": 1.0, "D1": 2.0, "D2": 3.0, "D3": 4.0, "D4": 5.0, "D5": 6.0, "D6": 7.0}
    )
    out = scm(
        df=df,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=["D1", "D2", "D3", "D4", "D5", "D6"],
        pre_covariates=["pop"],
    )
    post = out["synthetic"].loc[out["synthetic"].index >= pd.Timestamp("2023-05-01"), "effect"]
    assert 7.0 < float(post.mean()) < 13.0


def test_run_scm_dispatch(scm_panel):
    from normet.causal.run_scm import run_scm

    out = run_scm(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        donors=["D1", "D2", "D3"],
        cutoff_date="2023-05-01",
        scm_backend="scm",
    )
    assert {"observed", "synthetic", "effect"} <= set(out.columns)
