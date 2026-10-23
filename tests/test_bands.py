"""Functional tests for synthetic-control uncertainty / effect bands."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from normet.causal.bands import effect_bands_space, plot_event_study, uncertainty_bands

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


def _event_study_frame() -> pd.DataFrame:
    """20 daily points before the cutoff, 10 after, with a +3 level shift."""
    idx = pd.date_range("2023-01-01", periods=30, freq="D")
    effect = np.r_[np.zeros(20), np.full(10, 3.0)]
    effect[:20:3] = 0.5  # a little pre-period scatter, well clear of the jump
    return pd.DataFrame({"effect": effect}, index=idx)


def test_plot_event_study_structure():
    df = _event_study_frame()
    ax = plot_event_study(df, cutoff_date="2023-01-21")
    plt.close(ax.figure)

    # The two series are named by period, and nothing else carries a label.
    labels = {ln.get_label() for ln in ax.lines if not ln.get_label().startswith("_")}
    assert labels == {"pre-period", "post-period"}

    pre, post = (ln for ln in ax.lines if ln.get_label() in {"pre-period", "post-period"})
    assert len(pre.get_xdata()) == 20
    assert len(post.get_xdata()) == 10
    # Event time is a period count: pre runs -20..-1, post starts at 0.
    assert pre.get_xdata()[0] == -20 and pre.get_xdata()[-1] == -1
    assert post.get_xdata()[0] == 0 and post.get_xdata()[-1] == 9
    # The shift shows up where it was injected.
    assert np.allclose(post.get_ydata(), 3.0)
    assert np.allclose(pre.get_ydata()[:3], [0.5, 0.0, 0.0])


def test_plot_event_study_band_starts_at_cutoff():
    df = _event_study_frame()
    band = pd.DataFrame(
        {
            "p10": np.full(10, 1.0),
            "p90": np.full(10, 5.0),
            "ci_lo": np.full(10, 0.5),
            "ci_hi": np.full(10, 5.5),
        }
    )
    ax = plot_event_study({"treated": df}, cutoff_date="2023-01-21", band=band)
    plt.close(ax.figure)

    (fill,) = ax.collections
    assert fill.get_label() == "placebo band"
    xs = fill.get_paths()[0].vertices[:, 0]
    # ci_lo/ci_hi win over p10/p90 when both are present, and the band covers
    # only the post-period: x runs from event time 0 to 9.
    assert xs.min() == 0 and xs.max() == 9
    assert np.allclose(fill.get_paths()[0].vertices[:, 1].min(), 0.5)


def test_plot_event_study_aligns_date_indexed_band():
    df = _event_study_frame()
    band = pd.DataFrame({"lower": 1.0, "upper": 2.0}, index=df.index)
    ax = plot_event_study(df, cutoff_date="2023-01-21", band=band)
    plt.close(ax.figure)

    # A date-indexed band spans the whole window; only its post rows are drawn.
    (fill,) = ax.collections
    xs = fill.get_paths()[0].vertices[:, 0]
    assert xs.min() == 0 and xs.max() == 9


def test_plot_event_study_rejects_bad_input():
    with pytest.raises(ValueError, match="effect"):
        plot_event_study(pd.DataFrame({"observed": [1.0]}), cutoff_date="2023-01-01")
    with pytest.raises(ValueError, match="indexed by dates"):
        plot_event_study(pd.DataFrame({"effect": [1.0, 2.0]}), cutoff_date="2023-01-01")
    with pytest.raises(ValueError, match="ci_lo"):
        plot_event_study(
            _event_study_frame(),
            cutoff_date="2023-01-21",
            band=pd.DataFrame({"low": [1.0]}),
        )
    with pytest.raises(ValueError, match="cover the post-period"):
        plot_event_study(
            _event_study_frame(),
            cutoff_date="2023-01-21",
            band=pd.DataFrame(
                {"low": [1.0], "high": [2.0]}, index=pd.date_range("2020-01-01", periods=1)
            ),
        )
