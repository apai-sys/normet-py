"""Tests for normalise_auto's adaptive convergence loop.

Uses a deterministic linear stub model registered as a mock backend, so the
Monte-Carlo variance comes purely from the meteorological resampling — no
AutoML dependency required. Ported from tests/manual/smoke_normalise_auto_series.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from normet.analysis.normalise import normalise_auto
from normet.backends import backend_registry
from normet.exceptions import ConfigError

# ---------------------------------------------------------------------------
# Mock backend + linear stub model
# ---------------------------------------------------------------------------


class _LinearStubModel:
    """pred = 5 + 2*met1 - 3*met2 — nonzero mean, variance from resampling."""

    backend = "mock_auto"
    feature_names_in_ = np.array(["met1", "met2"])

    def predict(self, X):
        return (
            5.0
            + 2.0 * np.asarray(X["met1"], dtype=float)
            - 3.0 * np.asarray(X["met2"], dtype=float)
        )


class _MockAutoBackend:
    name = "mock_auto"

    def train(self, df, **kwargs):
        return _LinearStubModel()

    def save(self, model, path=".", filename="automl.joblib"):
        return str(path)

    def load(self, path=".", filename=None):
        return _LinearStubModel()


@pytest.fixture(autouse=True)
def _mock_backend():
    backend_registry._backends["mock_auto"] = _MockAutoBackend()
    yield
    del backend_registry._backends["mock_auto"]


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2021-01-01", periods=n, freq="h")
    met1 = rng.uniform(10, 20, n)
    met2 = rng.uniform(0, 1, n)
    value = 5 + 2 * met1 - 3 * met2 + rng.normal(0, 1, n)
    return pd.DataFrame({"date": dates, "value": value, "met1": met1, "met2": met2})


@pytest.fixture
def model() -> _LinearStubModel:
    return _LinearStubModel()


KW = dict(
    covariates=["met1", "met2"],
    variables_resample=["met1", "met2"],
    batch_size=10,
    seed=1,
    verbose=False,
    n_cores=1,
)


# ---------------------------------------------------------------------------
# series criterion (default)
# ---------------------------------------------------------------------------


def test_series_loose_tol_stops_at_streak_floor(df, model):
    """Loose tolerance → every check passes → stop at batch*(streak+1)."""
    out = normalise_auto(
        df, model, convergence_tol="50%", max_samples=200, return_history=True, **KW
    )
    # checks run from batch 2; default streak 3 → batches 2,3,4 → n = 40
    assert out["best_n"] == 40
    assert set(out["res"].columns) == {"date", "observed", "normalised"}
    assert len(out["res"]) == len(df)
    assert out["res"]["normalised"].notna().all()
    assert list(out["history"].columns) == ["n", "metric", "global_mean", "stable_count"]
    assert out["history"]["stable_count"].iloc[-1] == 3


def test_series_rse_declines_like_sqrt_n(df, model):
    """RSE is CLT-based: metric(n=50)/metric(n=200) ≈ sqrt(19/4)."""
    with pytest.warns(UserWarning, match="without strict convergence"):
        out = normalise_auto(
            df, model, convergence_tol="0.0001%", max_samples=200, return_history=True, **KW
        )
    h = out["history"]
    m50 = h.loc[h["n"] == 50, "metric"].iloc[0]
    m200 = h.loc[h["n"] == 200, "metric"].iloc[0]
    assert 1.3 < m50 / m200 < 4.0, "RSE not declining like 1/sqrt(n)"


def test_strict_tol_hits_max_samples_with_warning(df, model):
    with pytest.warns(UserWarning, match="without strict convergence"):
        out = normalise_auto(df, model, convergence_tol="0.0001%", max_samples=60, **KW)
    assert out["best_n"] == 60
    assert "history" not in out  # return_history defaults to False


# ---------------------------------------------------------------------------
# global (legacy) criterion
# ---------------------------------------------------------------------------


def test_global_loose_tol_stops_at_streak_floor(df, model):
    """Legacy metric keeps streak default 5 → floor 10*(5+1) = 60."""
    out = normalise_auto(
        df, model, convergence_metric="global", convergence_tol="50%", max_samples=200, **KW
    )
    assert out["best_n"] == 60


def test_numeric_tolerance_accepted(df, model):
    """A plain fraction behaves like the equivalent percent string."""
    out = normalise_auto(
        df, model, convergence_metric="global", convergence_tol=0.5, max_samples=200, **KW
    )
    assert out["best_n"] == 60


def test_metrics_agree_at_equal_n(df, model):
    """Aggregation is metric-independent: same n ⇒ identical normalised series."""
    r_series = normalise_auto(df, model, convergence_tol="50%", max_samples=200, **KW)
    with pytest.warns(UserWarning):
        r_global = normalise_auto(
            df,
            model,
            convergence_metric="global",
            convergence_tol="0.0001%",
            max_samples=r_series["best_n"],
            **KW,
        )
    merged = r_series["res"].merge(r_global["res"], on="date", suffixes=("_s", "_g"))
    assert np.allclose(merged["normalised_s"], merged["normalised_g"])


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_bad_metric_raises_config_error(df, model):
    with pytest.raises(ConfigError, match="convergence_metric"):
        normalise_auto(df, model, convergence_metric="bogus", **KW)
