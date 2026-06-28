"""Functional tests for synthetic-control uncertainty / effect bands."""

import numpy as np
import pandas as pd

from normet.causal.bands import effect_bands_space, uncertainty_bands

DONORS = ["D1", "D2", "D3", "D4", "D5", "D6"]


def _common(**kw):
    return dict(
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
        scm_backend="scm",
        **kw,
    )


def test_uncertainty_bands_jackknife(scm_panel):
    out = uncertainty_bands(scm_panel, **_common(method="jackknife", n_cores=1))
    assert {"treated", "low", "high", "jackknife_effects"} <= set(out)
    assert (out["low"] <= out["high"] + 1e-9).all()
    # One leave-one-donor-out effect path per donor.
    assert out["jackknife_effects"].shape[1] == len(DONORS)


def test_uncertainty_bands_bootstrap_reproducible(scm_panel):
    # Same random_state must give identical bands (per-replicate seeding).
    o1 = uncertainty_bands(
        scm_panel, **_common(method="bootstrap", B=12, random_state=0, n_cores=1)
    )
    o2 = uncertainty_bands(
        scm_panel, **_common(method="bootstrap", B=12, random_state=0, n_cores=1)
    )
    assert (o1["low"] <= o1["high"] + 1e-9).all()
    pd.testing.assert_series_equal(o1["low"], o2["low"])
    pd.testing.assert_series_equal(o1["high"], o2["high"])


def test_effect_bands_space_quantile():
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    treated = pd.DataFrame({"effect": [0.0, 1.0, 2.0, 3.0, 4.0]}, index=idx)
    placebos = {
        f"D{i}": pd.DataFrame({"effect": np.linspace(-1.0, 1.0, 5) * i}, index=idx)
        for i in range(1, 5)
    }
    out = effect_bands_space({"treated": treated, "placebos": placebos}, level=0.9)
    assert {"effect", "lower", "upper"} <= set(out.columns)
    assert (out["lower"] <= out["upper"] + 1e-9).all()


def test_effect_bands_space_no_placebos_returns_nan():
    idx = pd.date_range("2023-01-01", periods=3, freq="D")
    treated = pd.DataFrame({"effect": [1.0, 2.0, 3.0]}, index=idx)
    out = effect_bands_space({"treated": treated, "placebos": {}})
    assert out["lower"].isna().all()
    assert out["upper"].isna().all()
