"""Smoke tests for matplotlib plotting helpers."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from normet.plotting import (
    decomposition_stack,
    pdp_grid,
    polar_plot,
    scm_dashboard,
    usable_cycles,
)


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


def test_usable_cycles_is_independent_of_datetime_resolution():
    """A daily record offers no hour cycle, whatever the column's unit.

    ``DatetimeIndex.asi8`` counts in the index's own resolution (pandas 2.x
    keeps a ``datetime64[s]`` column as seconds), and the cycle test read it
    as nanoseconds: a daily s-resolution record then measured its step as
    8.6e-5 s and was offered an "hour of day" profile it cannot show.
    """
    ns = pd.date_range("2024-01-01", periods=80, freq="D")
    coarse = pd.DatetimeIndex(pd.Series(np.array(ns, dtype="datetime64[s]")))
    assert usable_cycles(ns) == ["weekday", "month"]
    assert usable_cycles(coarse) == usable_cycles(ns)


def test_pdp_grid_prefers_quantile_envelope_with_std_fallback():
    """pdp_grid renders [pdp_lower, pdp_upper] when finite, else pdp_mean ± pdp_std."""
    from matplotlib.collections import PolyCollection

    x = np.linspace(0.0, 4.0, 5)
    pdp_q = pd.DataFrame(
        {
            "variable": ["ws"] * 5,
            "value": x,
            "pdp_mean": np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
            "pdp_std": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            "pdp_lower": np.array([6.0, 7.0, 8.0, 9.0, 10.0]),
            "pdp_upper": np.array([20.0, 21.0, 22.0, 23.0, 24.0]),
        }
    )
    fig_q = pdp_grid(pdp_q, cols=1)
    polys_q = [c for c in fig_q.axes[0].collections if isinstance(c, PolyCollection)]
    assert len(polys_q) == 1
    y_verts = polys_q[0].get_paths()[0].vertices[:, 1]
    assert np.isclose(y_verts.min(), 6.0)
    assert np.isclose(y_verts.max(), 24.0)

    # Without pdp_lower / pdp_upper, fall back to pdp_mean ± pdp_std
    pdp_s = pdp_q.drop(columns=["pdp_lower", "pdp_upper"])
    fig_s = pdp_grid(pdp_s, cols=1)
    polys_s = [c for c in fig_s.axes[0].collections if isinstance(c, PolyCollection)]
    assert len(polys_s) == 1
    y_verts_s = polys_s[0].get_paths()[0].vertices[:, 1]
    assert np.isclose(y_verts_s.min(), 9.0)
    assert np.isclose(y_verts_s.max(), 15.0)


def test_plot_attention_attribution_and_validation():
    from normet.plotting import plot_attention_attribution

    dates = pd.date_range("2024-01-01", periods=6, freq="h")
    attrib = {
        "group_weights": pd.DataFrame(
            {
                "variable": ["PM2.5", "ws", "co", "t2m"],
                "role": ["target", "known_future", "past_only", "known_future"],
                "attention": [0.4, 0.3, 0.2, 0.1],
                "share": [0.4, 0.3, 0.2, 0.1],
                "context_attention": [0.35, 0.25, 0.25, 0.15],
            }
        ),
        "time_weights": pd.DataFrame(
            {
                "patch_idx": np.arange(6),
                "date": dates,
                "attention": np.linspace(0.05, 0.3, 6),
                "share": np.linspace(0.05, 0.3, 6) / np.linspace(0.05, 0.3, 6).sum(),
            }
        ),
        "target_self_share": 0.4,
        "covariate_share": 0.6,
    }
    fig = plot_attention_attribution(attrib, top_k=3, title="Attribution test")
    assert fig is not None and len(fig.axes) == 2
    assert len(fig.axes[0].patches) == 3  # top_k respected

    # Empty time_weights should render cleanly without error
    fig_empty = plot_attention_attribution(
        {
            "group_weights": attrib["group_weights"],
            "time_weights": pd.DataFrame(columns=["patch_idx", "date", "attention", "share"]),
        }
    )
    assert fig_empty is not None

    with pytest.raises(ValueError, match="missing key"):
        plot_attention_attribution({"group_weights": attrib["group_weights"]})


def test_plot_imputation_with_gaps_and_intervals():
    from normet.plotting import plot_imputation

    dates = pd.date_range("2024-01-01", periods=24, freq="h")
    obs = np.linspace(10.0, 33.0, 24)
    obs[5:9] = np.nan  # 4-point contiguous gap
    obs[15] = np.nan  # 1-point isolated gap
    imp = np.linspace(10.0, 33.0, 24)
    is_imp = np.isnan(obs)
    lo = np.where(is_imp, imp - 2.0, np.nan)
    hi = np.where(is_imp, imp + 2.0, np.nan)

    df_idx = pd.DataFrame(
        {"observed": obs, "imputed": imp, "is_imputed": is_imp, "lower": lo, "upper": hi},
        index=dates,
    )
    fig = plot_imputation(df_idx, title="Gap fill", ylabel="PM2.5")
    assert fig is not None

    # Also accepts an explicit `date` column and no uncertainty columns
    df_col = df_idx[["observed", "imputed", "is_imputed"]].reset_index(names="date")
    fig_col = plot_imputation(df_col)
    assert fig_col is not None

    with pytest.raises(ValueError, match="missing column"):
        plot_imputation(pd.DataFrame({"observed": [1.0, 2.0]}))
