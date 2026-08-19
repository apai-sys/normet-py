"""Smoke tests for matplotlib plotting helpers."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from normet.plotting import decomposition_stack, pdp_grid, polar_plot, scm_dashboard


@pytest.fixture
def wind_aq_df():
    rng = np.random.default_rng(0)
    n = 5000
    return pd.DataFrame(
        {
            "ws": rng.gamma(2.0, 1.5, n),
            "wd": rng.uniform(0, 360, n),
            "PM2.5": 20 + 10 * rng.normal(size=n),
        }
    )


def test_polar_plot_returns_axes(wind_aq_df):
    ax = polar_plot(wind_aq_df, target="PM2.5")
    assert ax is not None
    assert hasattr(ax, "pcolormesh")


def test_polar_plot_rejects_unknown_stat(wind_aq_df):
    with pytest.raises(ValueError):
        polar_plot(wind_aq_df, target="PM2.5", statistic="bogus")


def test_pdp_grid_minimal():
    pdp = pd.DataFrame(
        {
            "variable": ["a"] * 5 + ["b"] * 5,
            "value": list(range(5)) * 2,
            "pdp_mean": np.arange(10, dtype=float),
            "pdp_std": np.ones(10),
        }
    )
    fig = pdp_grid(pdp, cols=2)
    assert fig is not None


def test_decomposition_stack_minimal():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "observed": np.linspace(10, 20, 30),
            "feat_a": np.linspace(1, 3, 30),
            "feat_b": np.linspace(0, 2, 30),
        },
        index=dates,
    )
    ax = decomposition_stack(df)
    assert ax is not None


def test_scm_dashboard_minimal(scm_panel):
    from normet.causal.diagnostics import scm_diagnostics
    from normet.causal.scm import scm

    res = scm(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=["D1", "D2", "D3", "D4", "D5", "D6"],
    )
    diag = scm_diagnostics(res, cutoff_date="2023-05-01")
    fig = scm_dashboard(res, cutoff_date="2023-05-01", diagnostics=diag)
    assert fig is not None


def test_time_series_plot_minimal():
    from normet.plotting import time_series_plot

    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "my_val": np.sin(np.linspace(0, 10, 30)),
            "my_val_low": np.sin(np.linspace(0, 10, 30)) - 0.2,
            "my_val_high": np.sin(np.linspace(0, 10, 30)) + 0.2,
        },
        index=dates,
    )
    ax = time_series_plot(df, "my_val", ci_low="my_val_low", ci_high="my_val_high", resample="W")
    assert ax is not None

    with pytest.raises(ValueError):
        time_series_plot(df, "bogus")


def test_normalise_plot_minimal():
    from normet.plotting import normalise_plot

    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "observed": np.linspace(10, 20, 30),
            "normalised": np.linspace(11, 21, 30),
            "q025": np.linspace(9, 19, 30),
            "q975": np.linspace(12, 22, 30),
        },
        index=dates,
    )
    ax = normalise_plot(df, ci_low="q025", ci_high="q975", resample="W")
    assert ax is not None

    with pytest.raises(ValueError):
        normalise_plot(df, observed_col="bogus")


def test_plot_bayesian_scm_minimal():
    from normet.plotting import plot_bayesian_scm

    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    res = {
        "synthetic": pd.DataFrame(
            {
                "observed": np.sin(np.linspace(0, 10, 30)),
                "synthetic": np.sin(np.linspace(0, 10, 30)) + 0.1,
                "synthetic_low": np.sin(np.linspace(0, 10, 30)) - 0.2,
                "synthetic_high": np.sin(np.linspace(0, 10, 30)) + 0.3,
                "effect": -0.1 * np.ones(30),
                "effect_low": -0.3 * np.ones(30),
                "effect_high": 0.2 * np.ones(30),
            },
            index=dates,
        )
    }
    fig = plot_bayesian_scm(res, cutoff_date="2023-01-15")
    assert fig is not None

    # Test with single ax passed
    import matplotlib.pyplot as plt

    _, ax1 = plt.subplots()
    fig2 = plot_bayesian_scm(res, cutoff_date="2023-01-15", ax=ax1)
    assert fig2 is not None
