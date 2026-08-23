"""Sanity checks for the alternative SCM backends."""

import pandas as pd
import pytest

from normet.causal import run_scm
from normet.causal.variants import did_baseline, scm_abadie, scm_mcnnm, scm_robust


@pytest.mark.parametrize("backend", ["scm", "abadie", "did", "mcnnm", "robust"])
def test_run_scm_recovers_shock(scm_panel, backend):
    out = run_scm(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=["D1", "D2", "D3", "D4", "D5", "D6"],
        scm_backend=backend,
    )
    post = out.loc[out.index >= pd.Timestamp("2023-05-01"), "effect"]
    # All four backends should land near the +10 toy shock
    assert 7.0 < post.mean() < 13.0, f"{backend}: ATT={post.mean():.2f}"


def test_abadie_simplex_weights(scm_panel):
    out = scm_abadie(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=["D1", "D2", "D3", "D4", "D5", "D6"],
    )
    w = out["weights"]
    assert (w >= -1e-9).all()
    assert abs(float(w.sum()) - 1.0) < 1e-6


def test_did_synthetic_matches_pre_mean(scm_panel):
    out = did_baseline(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=["D1", "D2", "D3", "D4", "D5", "D6"],
    )
    pre_obs = (
        out["synthetic"].loc[out["synthetic"].index < pd.Timestamp("2023-05-01"), "observed"].mean()
    )
    pre_syn = (
        out["synthetic"]
        .loc[out["synthetic"].index < pd.Timestamp("2023-05-01"), "synthetic"]
        .mean()
    )
    # Pre-period averages should agree by construction
    assert abs(pre_obs - pre_syn) < 1e-6


def test_robust_recovers_weights_and_rank(scm_panel):
    # Treated == 0.5*D1 + 0.5*D2 in the pre-period. With full rank retained,
    # HSVT leaves the donor matrix intact and OLS recovers the 0.5/0.5 identity.
    out = scm_robust(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=["D1", "D2", "D3", "D4", "D5", "D6"],
        rank=6,
    )
    w = out["weights"]
    assert out["rank"] == 6
    assert abs(w["D1"] - 0.5) < 0.15
    assert abs(w["D2"] - 0.5) < 0.15
    assert w.drop(["D1", "D2"]).abs().max() < 0.15
    post = out["synthetic"].loc[out["synthetic"].index >= pd.Timestamp("2023-05-01"), "effect"]
    assert 7.0 < post.mean() < 13.0


def test_robust_ridge_alpha_path(scm_panel):
    # alpha > 0 exercises the closed-form ridge branch (vs OLS).
    out = scm_robust(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=["D1", "D2", "D3", "D4", "D5", "D6"],
        rank=6,
        alpha=1.0,
    )
    post = out["synthetic"].loc[out["synthetic"].index >= pd.Timestamp("2023-05-01"), "effect"]
    assert 5.0 < float(post.mean()) < 15.0


def test_robust_truncates_rank(scm_panel):
    # Energy-based selection should keep at most the donor count.
    out = scm_robust(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=["D1", "D2", "D3", "D4", "D5", "D6"],
        energy=0.6,
    )
    assert 1 <= out["rank"] <= 6
    assert {"observed", "synthetic", "effect"} <= set(out["synthetic"].columns)


def test_mcnnm_returns_synthetic_shape(scm_panel):
    out = scm_mcnnm(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=["D1", "D2", "D3", "D4", "D5", "D6"],
        max_iter=80,
    )
    syn = out["synthetic"]
    assert {"observed", "synthetic", "effect"} <= set(syn.columns)
    assert len(syn) == 200


def _mcnnm(scm_panel, **kw):
    return scm_mcnnm(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=["D1", "D2", "D3", "D4", "D5", "D6"],
        **kw,
    )


def test_mcnnm_cv_selects_lambda(scm_panel):
    out = _mcnnm(scm_panel, cv=3, random_state=0, max_iter=100)
    assert out["rank_lambda"] > 0
    post = out["synthetic"].loc[out["synthetic"].index >= pd.Timestamp("2023-05-01"), "effect"]
    assert 7.0 < post.mean() < 13.0


def test_mcnnm_explicit_lam_disables_cv(scm_panel):
    # An explicit lam must be used verbatim regardless of cv.
    out = _mcnnm(scm_panel, lam=5.0, cv=3, max_iter=60)
    assert out["rank_lambda"] == 5.0


def test_mcnnm_randomized_rank_runs(scm_panel):
    # Truncated randomized SVD path (requires scikit-learn, a core dependency).
    out = _mcnnm(scm_panel, max_rank=4, random_state=0, max_iter=80)
    syn = out["synthetic"]
    assert {"observed", "synthetic", "effect"} <= set(syn.columns)
    assert len(syn) == 200
