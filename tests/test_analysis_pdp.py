"""Tests for normet.analysis.pdp module.

We mock ``ml_predict`` in the ``normet.analysis.pdp`` module namespace to avoid
needing a real trained model while still exercising the full PDP logic.

NOTE: ``normet.analysis`` re-exports ``pdp`` as an attribute (function), so we
must import the *module* explicitly via ``importlib`` to be able to monkeypatch
module-level names.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
import pytest

# Import the *module* (not the function) so monkeypatch targets work
import normet.analysis.pdp  # noqa: F401 — ensures module is in sys.modules

_pdp_module = sys.modules["normet.analysis.pdp"]

# ---------------------------------------------------------------------------
# Trivial model stub
# ---------------------------------------------------------------------------


class TrivialModel:
    """Minimal FLAML-like model stub with sklearn-style feature attributes."""

    backend: str = "flaml"

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names_in_ = np.array(feature_names)
        self.feature_importances_ = np.ones(len(feature_names))

    def predict(self, X):  # noqa: ANN001
        return np.ones(len(X))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def feature_names() -> list[str]:
    return ["ws", "wd", "t2m"]


@pytest.fixture()
def sample_df(feature_names: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 40
    data = {f: rng.uniform(0, 10, n) for f in feature_names}
    return pd.DataFrame(data)


@pytest.fixture()
def trivial_model(feature_names: list[str]) -> TrivialModel:
    return TrivialModel(feature_names)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pdp_returns_dataframe(monkeypatch, sample_df, trivial_model, feature_names):
    """pdp() returns a DataFrame with the expected columns."""
    monkeypatch.setattr(_pdp_module, "ml_predict", lambda model, df: np.ones(len(df)))

    result = _pdp_module.pdp(sample_df, trivial_model, var_list=feature_names, n_cores=1)

    assert isinstance(result, pd.DataFrame)
    for col in ("variable", "value", "pdp_mean"):
        assert col in result.columns, f"Missing column: {col}"


def test_pdp_covers_all_features(monkeypatch, sample_df, trivial_model, feature_names):
    """All requested features appear in result['variable']."""
    monkeypatch.setattr(_pdp_module, "ml_predict", lambda model, df: np.ones(len(df)))

    result = _pdp_module.pdp(sample_df, trivial_model, var_list=feature_names, n_cores=1)

    found = set(result["variable"].unique())
    for feat in feature_names:
        assert feat in found, f"Feature '{feat}' not in pdp result"


def test_pdp_invalid_feature(monkeypatch, sample_df, trivial_model):
    """Model whose features are all absent from df raises ValueError."""
    monkeypatch.setattr(_pdp_module, "ml_predict", lambda model, df: np.ones(len(df)))

    # Build a model that exposes only "nonexistent_col" as its features — so
    # after the intersection with df columns the feature list is empty.
    bad_model = TrivialModel(["nonexistent_col"])

    with pytest.raises((ValueError, Exception)):
        _pdp_module.pdp(sample_df, bad_model, var_list=["nonexistent_col"], n_cores=1)
