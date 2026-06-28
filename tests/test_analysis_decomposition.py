"""Tests for normet.analysis.decomposition module.

We mock ``normalise`` and ``build_model`` to verify the
decomposition workflow without requiring heavy ML backends.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

import normet.analysis.decomposition
from normet.analysis.decomposition import _effective_cores, decom_emi, decom_met, decompose

# Import module and get the actual module objects from sys.modules
from normet.exceptions import ConfigError, DataError

decomposition_mod = sys.modules["normet.analysis.decomposition"]

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
    feature_names: list[str],
    variables_resample: list[str] | None = None,
    n_samples: int = 300,
    seed: int = 0,
    **kwargs,
) -> pd.DataFrame:
    res = df.copy()
    # Simple dummy calculation
    res["normalised"] = res["value"] * 0.95
    return res


def mock_build_model(df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, DummyModel]:
    return df, DummyModel(["ws", "wd", "date_unix"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def decomp_df() -> pd.DataFrame:
    n = 20
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


def test_decompose_input_validation():
    with pytest.raises(DataError, match="`df` must be provided"):
        decompose(df=None)

    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(DataError, match="`value` must be provided"):
        decompose(df=df, value=None)

    with pytest.raises(ConfigError, match="Either `model` or `feature_names`"):
        decompose(df=df, value="value", model=None, feature_names=None)

    with pytest.raises(ConfigError, match="When training a model, `backend`"):
        decompose(df=df, value="value", model=None, feature_names=["x"], backend=None)

    with pytest.raises(ConfigError, match="Unsupported decomposition method"):
        decompose(df=df, value="value", feature_names=["x"], backend="flaml", method="invalid")


def test_effective_cores():
    assert _effective_cores(4) == 4
    assert _effective_cores(None) >= 1
    assert _effective_cores(-1) == 1


def test_decom_emi_validation():
    with pytest.raises(DataError, match="`df` must be provided"):
        decom_emi(df=None)

    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(DataError, match="target column.*must be provided"):
        decom_emi(df=df, value=None)

    with pytest.raises(ConfigError, match="Either `model` or `feature_names`"):
        decom_emi(df=df, value="value", model=None, feature_names=None)

    with pytest.raises(ConfigError, match="When training a model, `backend`"):
        decom_emi(df=df, value="value", model=None, feature_names=["x"], backend=None)


def test_decom_met_validation():
    with pytest.raises(DataError, match="`df` must be provided"):
        decom_met(df=None)

    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(DataError, match="target column.*must be provided"):
        decom_met(df=df, value=None)

    with pytest.raises(ConfigError, match="Either `model` or `feature_names`"):
        decom_met(df=df, value="value", model=None, feature_names=None)

    with pytest.raises(ConfigError, match="When training a model, `backend`"):
        decom_met(df=df, value="value", model=None, feature_names=["x"], backend=None)


def test_decompose_emission(decomp_df, monkeypatch):
    monkeypatch.setattr(decomposition_mod, "normalise", mock_normalise)
    monkeypatch.setattr(decomposition_mod, "build_model", mock_build_model)

    result = decompose(
        method="emission",
        df=decomp_df,
        value="value",
        feature_names=["ws", "wd", "date_unix"],
        backend="flaml",
        n_samples=10,
        verbose=False,
    )

    assert "observed" in result.columns
    assert "emi_total" in result.columns
    assert "emi_noise" in result.columns
    assert "emi_base" in result.columns


def test_decompose_meteorology(decomp_df, monkeypatch):
    monkeypatch.setattr(decomposition_mod, "normalise", mock_normalise)
    monkeypatch.setattr(decomposition_mod, "build_model", mock_build_model)

    result = decompose(
        method="meteorology",
        df=decomp_df,
        value="value",
        feature_names=["ws", "wd"],
        backend="flaml",
        n_samples=10,
        verbose=False,
    )

    assert "observed" in result.columns
    assert "emi_total" in result.columns
    assert "met_total" in result.columns
    assert "met_base" in result.columns
    assert "met_noise" in result.columns


def test_decompose_shap_removed(decomp_df):
    """method='shap' now raises ConfigError since SHAP support was removed."""
    with pytest.raises(ConfigError, match="Unsupported decomposition method.*shap"):
        decompose(
            method="shap",
            df=decomp_df,
            value="value",
            feature_names=["ws", "wd"],
            backend="flaml",
        )
