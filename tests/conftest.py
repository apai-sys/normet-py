"""Shared fixtures for the normet test suite."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20240101)


@pytest.fixture(scope="session")
def synthetic_aq(rng) -> pd.DataFrame:
    """Hourly synthetic air-quality panel with a clear weather signal."""
    n = 24 * 60  # 60 days hourly
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    t2m = 10 + 8 * np.sin(2 * np.pi * np.arange(n) / (24 * 30)) + rng.normal(0, 1, n)
    blh = 800 + 400 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 50, n)
    u10 = rng.normal(0, 3, n)
    v10 = rng.normal(0, 3, n)
    # Concentration: increases when blh is low and temperature deviates from mean.
    pm = 20 - 0.01 * blh + 0.3 * (t2m - t2m.mean()) ** 2 + 0.5 * u10 + rng.normal(0, 2, n)
    pm = np.clip(pm, 0, None)
    return pd.DataFrame(
        {"date": dates, "PM2.5": pm, "t2m": t2m, "blh": blh, "u10": u10, "v10": v10}
    )


@pytest.fixture(scope="session")
def scm_panel(rng) -> pd.DataFrame:
    """
    Toy panel: one treated unit + 6 donors. Treated unit equals 0.5*D1 + 0.5*D2
    in the pre-period, then gets a shock of +10 after cutoff.

    Each donor carries a *distinct* deterministic signal (a different sinusoid
    frequency and phase), so the donors are not mutually collinear and the true
    0.5/0.5 weight identity on D1+D2 is actually recoverable. (If every donor
    shared the same signal — as in a naive construction — the synthetic-control
    weights would be unidentifiable: many donor combinations fit equally well,
    so only the *effect* would be recoverable, not the weights. Note the
    augmented SCM balances ridge *residuals*, so the donor signals must also be
    cleanly predictable — a large independent random-walk component would
    dominate the residuals and again wash out weight identity.)
    """
    n_days = 200
    t = np.arange(n_days)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    cutoff = pd.Timestamp("2023-05-01")

    # Distinct deterministic signal per donor: different period + phase so no
    # donor is a linear combination of the others.
    periods = {"D1": 30, "D2": 17, "D3": 45, "D4": 11, "D5": 60, "D6": 23}
    donors_data = {}
    for j, name in enumerate(["D1", "D2", "D3", "D4", "D5", "D6"]):
        donors_data[name] = 5 + j + 3 * np.sin(2 * np.pi * t / periods[name] + j)

    treated = 0.5 * donors_data["D1"] + 0.5 * donors_data["D2"] + rng.normal(0, 0.05, n_days)
    treated = treated + np.where(dates >= cutoff, 10.0, 0.0)

    rows = []
    for d, val in zip(dates, treated, strict=False):
        rows.append({"date": d, "ID": "T", "value": val})
    for name, arr in donors_data.items():
        for d, val in zip(dates, arr, strict=False):
            rows.append({"date": d, "ID": name, "value": val})
    return pd.DataFrame(rows)


def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


needs_flaml = pytest.mark.skipif(not _has("flaml"), reason="flaml not installed")
needs_lgb = pytest.mark.skipif(not _has("lightgbm"), reason="lightgbm not installed")
