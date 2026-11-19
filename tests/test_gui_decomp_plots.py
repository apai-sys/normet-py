"""Tests for the GUI's decomposition figures (normet.gui._decomp_plots).

Uses synthetic DataFrames shaped exactly like ``decom_emi``/``decom_met``
output (rather than running a real model) to keep this fast; the actual
decomposition math is covered by test_analysis_decomposition.py.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from normet.gui import _decomp_plots as DP


@pytest.fixture()
def emi_result() -> pd.DataFrame:
    n = 24 * 14  # two weeks hourly
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    base = 20.0
    trend = np.linspace(0, 2, n)
    seasonal = 3 * np.sin(2 * np.pi * np.arange(n) / (24 * 365))
    weekly = 1.5 * np.sin(2 * np.pi * np.arange(n) / (24 * 7))
    diurnal = 4 * np.sin(2 * np.pi * np.arange(n) / 24)
    noise = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "date": dates,
            "observed": base + trend + seasonal + weekly + diurnal + noise + rng.normal(0, 2, n),
            "date_unix": trend,
            "day_julian": seasonal,
            "weekday": weekly,
            "hour": diurnal,
            "emi_total": base + trend + seasonal + weekly + diurnal + noise,
            "emi_base": base,
            "emi_noise": noise,
        }
    )


@pytest.fixture()
def met_result(emi_result: pd.DataFrame) -> pd.DataFrame:
    n = len(emi_result)
    rng = np.random.default_rng(1)
    t2m_contrib = 2.0 * np.sin(2 * np.pi * np.arange(n) / (24 * 365))
    ws_contrib = rng.normal(0, 1.0, n)
    d = emi_result.copy()
    d["met_total"] = d["observed"] - d["emi_total"]
    d["t2m"] = t2m_contrib
    d["ws"] = ws_contrib
    d["met_base"] = float(d["met_total"].mean())
    d["met_noise"] = d["met_total"] - (d["met_base"] + t2m_contrib + ws_contrib)
    return d


def test_met_contribution_columns_excludes_metadata(met_result):
    cols = DP.met_contribution_columns(met_result)
    assert set(cols) == {"t2m", "ws"}
    assert "met_total" not in cols
    assert "met_noise" not in cols
    assert "emi_total" not in cols  # emission-decomposition column, not a met contribution


def _titles(fig) -> list[str]:
    """All axis titles, regardless of which alignment (left/center) they use."""
    return [
        ax.get_title(loc=loc)
        for ax in fig.axes
        for loc in ("left", "center", "right")
        if ax.get_title(loc=loc)
    ]


def test_emission_figure_renders_one_panel_per_component(emi_result):
    fig = DP.emission_figure(emi_result, target="PM2.5")
    # 1 overview panel + trend/seasonal/weekly/diurnal/noise
    assert len(fig.axes) == 1 + len(DP.EMI_COMPONENTS)
    titles = _titles(fig)
    assert any("normalised" in t.lower() or "PM2.5" in t for t in titles)
    assert any("trend" in t.lower() for t in titles)
    assert any("diurnal" in t.lower() for t in titles)


def test_meteorology_figure_renders_total_plus_top_contributors_and_ranking(met_result):
    fig = DP.meteorology_figure(met_result, target="PM2.5", top_k=5)
    # overview + up to 2 contributors (only t2m, ws exist) + 1 ranking axis
    assert len(fig.axes) == 1 + 2 + 1
    titles = " ".join(_titles(fig))
    assert "meteorological influence" in titles.lower()
    assert "ranking" in titles.lower()


def test_meteorology_figure_handles_no_contribution_columns(emi_result):
    """A malformed/partial result (no per-variable columns) must not crash."""
    d = emi_result.copy()
    d["met_total"] = d["observed"] - d["emi_total"]
    d["met_base"] = float(d["met_total"].mean())
    d["met_noise"] = d["met_total"] - d["met_base"]
    fig = DP.meteorology_figure(d, target="PM2.5")
    assert len(fig.axes) == 1 + 0 + 1  # overview + no contributors + ranking (met_noise only)
