"""Tests for normet.analysis.rolling module.

We mock ``normalise`` and ``build_model`` to verify the rolling window logic
without needing heavy AutoML backends or real training.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

# Import module and get the actual module object from sys.modules
import normet.analysis.rolling
from normet.analysis.rolling import rolling

rolling_mod = sys.modules["normet.analysis.rolling"]

# ---------------------------------------------------------------------------
# Trivial Model & Mocks
# ---------------------------------------------------------------------------


class DummyModel:
    backend: str = "flaml"

    def __init__(self, features: list[str]) -> None:
        self.feature_names_in_ = np.array(features)
        self.feature_importances_ = np.ones(len(features))


def mock_normalise(
    df: pd.DataFrame,
    *,
    covariates: list[str],
    variables_resample: list[str] | None = None,
    n_samples: int = 300,
    seed: int = 0,
    **kwargs,
) -> pd.DataFrame:
    # Just return a copy of the dataframe with a 'normalised' column
    res = df.copy()
    res["normalised"] = res["value"] * 0.95
    return res


def mock_build_model(df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, DummyModel]:
    return df, DummyModel(["ws", "wd"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rolling_df() -> pd.DataFrame:
    # Create 30 days of data (e.g. 2024-01-01 to 2024-01-30)
    n = 30
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "date": dates,
            "value": rng.normal(10.0, 1.0, n),
            "ws": rng.uniform(2.0, 8.0, n),
            "wd": rng.uniform(0.0, 360.0, n),
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rolling_missing_df():
    with pytest.raises(ValueError, match="must be provided"):
        rolling(df=None)


def test_rolling_missing_value():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5)})
    # pass covariates so it gets past the model/feature verification step
    with pytest.raises(ValueError, match="does not contain the target column"):
        rolling(df=df, target="nonexistent", covariates=["ws"])


def test_rolling_window_too_large(rolling_df):
    # Span is 30 days, ask for 40 day window
    # pass model so it gets past model building
    model = DummyModel(["ws", "wd"])
    with pytest.raises(ValueError, match="Window is larger than"):
        rolling(df=rolling_df, model=model, window_days=40, rolling_every=5)


def test_rolling_runs_successfully(rolling_df, monkeypatch):
    monkeypatch.setattr(rolling_mod, "normalise", mock_normalise)
    monkeypatch.setattr(rolling_mod, "build_model", mock_build_model)

    # Use a pre-built model to test that branch
    model = DummyModel(["ws", "wd"])
    result = rolling(
        df=rolling_df,
        model=model,
        target="value",
        covariates=["ws", "wd"],
        window_days=10,
        rolling_every=5,
        n_samples=50,
        verbose=True,
    )

    # Check that observed is returned
    assert "observed" in result.columns
    # Check that rolling windows (rolling_0, rolling_1, etc.) are present
    rolling_cols = [c for c in result.columns if c.startswith("rolling_")]
    assert len(rolling_cols) > 0
    # Length should match the input
    assert len(result) == len(rolling_df)
    # Check index matches
    assert (result.index == rolling_df["date"]).all()


def test_rolling_with_training(rolling_df, monkeypatch):
    monkeypatch.setattr(rolling_mod, "normalise", mock_normalise)
    monkeypatch.setattr(rolling_mod, "build_model", mock_build_model)

    # Call without pre-built model to trigger build_model
    result = rolling(
        df=rolling_df,
        model=None,
        target="value",
        covariates=["ws", "wd"],
        window_days=12,
        rolling_every=6,
        n_samples=10,
        verbose=False,
    )

    assert "observed" in result.columns
    rolling_cols = [c for c in result.columns if c.startswith("rolling_")]
    assert len(rolling_cols) > 0


def test_rolling_no_features_raises_error(rolling_df):
    # When model is None, training needs covariates
    with pytest.raises(ValueError, match="must provide `covariates`"):
        rolling(
            df=rolling_df,
            model=None,
            covariates=None,
            window_days=10,
            rolling_every=5,
        )


def test_rolling_extracts_features_from_model(rolling_df, monkeypatch):
    """When model is given without covariates, extract from model."""
    monkeypatch.setattr(rolling_mod, "normalise", mock_normalise)
    monkeypatch.setattr(rolling_mod, "build_model", mock_build_model)

    model = DummyModel(["ws", "wd"])
    result = rolling(
        df=rolling_df,
        model=model,
        covariates=None,  # ← not provided; should be auto-extracted
        window_days=10,
        rolling_every=5,
        n_samples=10,
        verbose=False,
    )

    assert "observed" in result.columns
    rolling_cols = [c for c in result.columns if c.startswith("rolling_")]
    assert len(rolling_cols) > 0


def test_rolling_no_features_model_without_features(rolling_df):
    """Model without feature_names_in_ raises when covariates=None."""

    class FeaturelessModel:
        backend = "flaml"
        # no feature_names_in_, no feature_importances_

    with pytest.raises(ValueError, match="covariates"):
        rolling(
            df=rolling_df,
            model=FeaturelessModel(),
            covariates=None,
            window_days=10,
            rolling_every=5,
        )
