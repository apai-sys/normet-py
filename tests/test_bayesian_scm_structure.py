"""Structural tests for normet.causal.bayesian_scm.

These tests exercise the module without requiring PyMC to be installed
(except where ``@requires_pymc`` is applied).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# PyMC availability guard
# ---------------------------------------------------------------------------

pymc_available = importlib.util.find_spec("pymc") is not None
requires_pymc = pytest.mark.skipif(not pymc_available, reason="pymc not installed")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_panel(n_pre: int = 4) -> pd.DataFrame:
    """Minimal panel with one treated unit and two donors."""
    dates = pd.date_range("2023-01-01", periods=n_pre + 2, freq="D")
    rows = []
    for d in dates:
        rows.append({"date": d, "ID": "T", "value": float(np.random.default_rng(0).normal(5, 1))})
        rows.append({"date": d, "ID": "D1", "value": float(np.random.default_rng(1).normal(5, 1))})
        rows.append({"date": d, "ID": "D2", "value": float(np.random.default_rng(2).normal(5, 1))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bayesian_scm_importable():
    """bayesian_scm is callable after import."""
    from normet.causal.bayesian_scm import bayesian_scm  # noqa: PLC0415

    assert callable(bayesian_scm)


def test_bayesian_scm_in_all():
    """bayesian_scm is accessible from normet.causal namespace."""
    import normet.causal as causal_pkg  # noqa: PLC0415

    # Either in __all__ or directly importable as an attribute
    assert "bayesian_scm" in causal_pkg.__all__ or hasattr(causal_pkg, "bayesian_scm")


def test_bayesian_scm_requires_treated():
    """Calling without treated_unit should raise ValueError."""
    from normet.causal.bayesian_scm import bayesian_scm  # noqa: PLC0415

    df = _tiny_panel(n_pre=10)
    with pytest.raises(ValueError):
        # Passes treated_unit=None → ValueError before pymc is touched
        bayesian_scm(
            df,
            date_col="date",
            unit_col="ID",
            outcome_col="value",
            treated_unit=None,
            cutoff_date="2023-01-05",
        )


def test_bayesian_scm_auto_cutoff_fails_when_no_anomalies():
    """Calling with cutoff_date=None raises ValueError if no anomalies are found."""
    from normet.causal.bayesian_scm import bayesian_scm  # noqa: PLC0415

    # Create a tiny panel where values are perfectly constant, so no IQR anomalies can be found
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    rows = []
    for d in dates:
        rows.append({"date": d, "ID": "T", "value": 5.0})
        rows.append({"date": d, "ID": "D1", "value": 5.0})
    df = pd.DataFrame(rows)

    with pytest.raises(ValueError, match="no anomalies detected"):
        bayesian_scm(
            df,
            date_col="date",
            unit_col="ID",
            outcome_col="value",
            treated_unit="T",
            cutoff_date=None,
        )


@requires_pymc
def test_bayesian_scm_requires_enough_preperiod():
    """Raises ValueError when fewer than 5 complete pre-treatment rows exist."""
    from normet.causal.bayesian_scm import bayesian_scm  # noqa: PLC0415

    df = _tiny_panel(n_pre=3)  # only 3 pre-period rows → should fail
    cutoff = str(df["date"].sort_values().iloc[3])  # 4th date as cutoff

    with pytest.raises(ValueError, match="Not enough"):
        bayesian_scm(
            df,
            date_col="date",
            unit_col="ID",
            outcome_col="value",
            treated_unit="T",
            cutoff_date=cutoff,
            draws=10,
            tune=10,
            chains=1,
        )


@requires_pymc
def test_bayesian_scm_smoke(scm_panel):
    """Smoke test: bayesian_scm runs and returns the expected result structure."""
    from normet.causal.bayesian_scm import bayesian_scm  # noqa: PLC0415

    result = bayesian_scm(
        scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        draws=20,
        tune=20,
        chains=1,
        progressbar=False,
    )

    # Check top-level keys
    assert set(result.keys()) == {
        "synthetic",
        "weights",
        "weights_summary",
        "posterior_samples",
        "idata",
    }

    # Check synthetic DataFrame columns
    syn = result["synthetic"]
    assert isinstance(syn, pd.DataFrame)
    for col in ("observed", "synthetic", "effect"):
        assert col in syn.columns, f"Missing column in synthetic: {col}"
