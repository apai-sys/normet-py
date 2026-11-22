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


def test_transport_plots_render_cleanly():
    import matplotlib.pyplot as plt

    from normet.analysis import (
        cluster_trajectories,
        invert_source_footprints,
        wcwt,
    )
    from normet.plotting import (
        plot_footprint_inversion,
        plot_receptor_grid,
        plot_trajectory_clusters,
    )

    rng = np.random.default_rng(0)
    times = pd.date_range("2023-01-01", periods=24, freq="6h")
    ep_rows = []
    obs_rows = []
    for i, ts in enumerate(times):
        obs_rows.append({"date": ts, "pm25": 20.0 + 30.0 * (i % 2)})
        for age in (0, -3, -6):
            ep_rows.append(
                {
                    "receptor_date": ts,
                    "age_h": float(age),
                    "lat": 51.5 + (0.4 * abs(age) if i % 2 else -0.3 * abs(age)),
                    "lon": -0.1 + float(rng.normal(0.0, 0.05)),
                    "height": 300.0,
                    "blh": 800.0,
                }
            )
    ep = pd.DataFrame(ep_rows)
    obs = pd.DataFrame(obs_rows)
    grid = wcwt(ep, obs, value_col="pm25", grid_res=0.8)
    fig_g = plot_receptor_grid(grid, regions={"N": (-2.0, 52.0, 2.0, 54.0)})
    assert fig_g is not None

    cl = cluster_trajectories(ep, n_clusters=2, series=obs, value_col="pm25")
    fig_c1 = plot_trajectory_clusters(cl, show_tsv=False)
    fig_c2 = plot_trajectory_clusters(cl, show_tsv=True)
    assert fig_c1 is not None and fig_c2 is not None

    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    fp = pd.DataFrame({"A": rng.uniform(0, 3, 60), "B": rng.uniform(0, 3, 60)}, index=dates)
    y = pd.Series(5.0 + 4.0 * fp["A"] + 2.0 * fp["B"] + rng.normal(0, 0.2, 60), index=dates)
    inv_stat = invert_source_footprints(fp, y, n_bootstrap=15)
    inv_roll = invert_source_footprints(
        fp, y, n_bootstrap=10, window_days=25, step_days=10, min_window_obs=15
    )
    fig_i1 = plot_footprint_inversion(inv_stat)
    fig_i2 = plot_footprint_inversion(inv_roll)
    assert fig_i1 is not None and fig_i2 is not None
    plt.close("all")


def test_decomposition_stack_diverging_and_shapley_heatmap():
    import matplotlib.pyplot as plt

    from normet.analysis.decomposition import ShapleyInteractionResult
    from normet.plotting import plot_shapley_interactions

    dates = pd.date_range("2024-01-01", periods=24, freq="D")
    # Emission frame must exclude emi_total automatically so totals aren't double-stacked.
    df_emi = pd.DataFrame(
        {
            "observed": np.linspace(15, 25, 24),
            "emi_total": np.linspace(14, 24, 24),
            "emi_base": np.full(24, 18.0),
            "date_unix": np.linspace(-2.0, 4.0, 24),
            "day_julian": np.sin(np.linspace(0, 2 * np.pi, 24)),
            "emi_noise": np.zeros(24),
            "date_unix_main": np.linspace(-2.0, 4.0, 24),
            "date_unix:day_julian": np.zeros(24),
        },
        index=dates,
    )
    ax_e = decomposition_stack(df_emi)
    labels_e = [txt.get_text() for txt in ax_e.get_legend().get_texts()]
    assert "emi_total" not in labels_e
    assert "date_unix:day_julian" not in labels_e

    # Meteorology frame with negative & positive anomalies uses diverging fill without overlap.
    df_met = pd.DataFrame(
        {
            "observed": 20.0 + np.cos(np.linspace(0, 2 * np.pi, 24)),
            "emi_total": np.full(24, 20.0),
            "ws": np.sin(np.linspace(0, 2 * np.pi, 24)) * 3.0,
            "t2m": np.cos(np.linspace(0, 2 * np.pi, 24)) * 2.0,
            "met_total": np.cos(np.linspace(0, 2 * np.pi, 24)),
            "met_base": np.zeros(24),
            "met_noise": np.zeros(24),
            "ws_se": np.full(24, 0.1),
        },
        index=dates,
    )
    ax_m = decomposition_stack(df_met)
    labels_m = [txt.get_text() for txt in ax_m.get_legend().get_texts()]
    assert "met_total" not in labels_m
    assert "ws_se" not in labels_m
    assert "ws" in labels_m and "t2m" in labels_m

    mat_m = pd.DataFrame([[1.5, -0.4], [-0.4, 2.0]], index=["ws", "t2m"], columns=["ws", "t2m"])
    mat_s = pd.DataFrame([[2.1, 0.8], [0.8, 1.4]], index=["ws", "t2m"], columns=["ws", "t2m"])
    dummy_res = ShapleyInteractionResult(
        decomposition=df_met,
        main_effects=df_met[["ws", "t2m"]],
        pairwise=pd.DataFrame({"ws:t2m": np.zeros(24)}, index=dates),
        mean_matrix=mat_m,
        std_matrix=mat_s,
        summary=pd.DataFrame(),
    )
    fig_s = plot_shapley_interactions(dummy_res, stat="std")
    fig_mean = plot_shapley_interactions(dummy_res, stat="mean")
    assert fig_s is not None and fig_mean is not None
    plt.close("all")
