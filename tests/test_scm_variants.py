"""Sanity checks for the alternative SCM backends."""

import numpy as np
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


def _low_rank_null_panel(n_dates: int = 70, n_donors: int = 6, seed: int = 0):
    """A clean rank-2 panel (one common seasonal factor) with no treatment."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    t = np.arange(n_dates)
    common = 2.0 * np.sin(2.0 * np.pi * t / 12.0) + 0.05 * t
    rows = []
    for j in range(n_donors):
        y = (j * 1.5) + (0.7 + 0.05 * j) * common + rng.normal(0.0, 0.05, n_dates)
        rows.extend({"date": d, "code": f"D{j}", "poll": v} for d, v in zip(dates, y, strict=True))
    y = 3.0 + 0.75 * common + rng.normal(0.0, 0.05, n_dates)
    rows.extend({"date": d, "code": "T", "poll": v} for d, v in zip(dates, y, strict=True))
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame, dates[n_dates // 2]


def test_mcnnm_default_lambda_is_cross_validated_not_the_heuristic():
    """The default must not be the biased ``0.1 * sigma_max`` heuristic.

    On this clean rank-2 null panel the heuristic leaves a pre-period RMSE of
    ~0.41 and a spurious post-period ATT of +0.34; three-fold CV selects a
    lambda whose fit error is ~0.05 and ATT ~0.00. The bias adds
    symmetrically to a real effect too, so it does not cancel.
    """
    df, cutoff = _low_rank_null_panel()
    donors = [f"D{j}" for j in range(6)]
    out = scm_mcnnm(
        df,
        "date",
        "code",
        "poll",
        "T",
        cutoff.strftime("%Y-%m-%d"),
        donors,
    )
    syn = out["synthetic"]
    pre = syn[syn.index < cutoff]
    post = syn[syn.index >= cutoff]
    assert float(np.sqrt((pre["effect"] ** 2).mean())) < 0.10
    assert abs(float(post["effect"].mean())) < 0.05

    # cv=0 keeps the legacy heuristic, and its bias is what the default avoids.
    legacy = scm_mcnnm(
        df,
        "date",
        "code",
        "poll",
        "T",
        cutoff.strftime("%Y-%m-%d"),
        donors,
        cv=0,
    )
    legacy_pre = legacy["synthetic"]
    legacy_post = legacy_pre[legacy_pre.index >= cutoff]
    assert abs(float(legacy_post["effect"].mean())) > 0.20


def test_mcnnm_default_lambda_is_reproducible_without_a_random_state():
    """Unseeded CV folds would make the default counterfactual differ per call."""
    df, cutoff = _low_rank_null_panel()
    donors = [f"D{j}" for j in range(6)]
    first = scm_mcnnm(df, "date", "code", "poll", "T", cutoff.strftime("%Y-%m-%d"), donors)
    second = scm_mcnnm(df, "date", "code", "poll", "T", cutoff.strftime("%Y-%m-%d"), donors)
    assert first["rank_lambda"] == second["rank_lambda"]
