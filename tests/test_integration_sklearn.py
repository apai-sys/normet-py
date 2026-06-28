"""E2E integration test using a sklearn RandomForest backend.

Tests the pipeline end-to-end without requiring flaml.
Registers a ``sklearn`` backend via ``backend_registry`` at session scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from normet import normalise
from normet.backends import backend_registry
from normet.model.train import build_model, train_model

# ---------------------------------------------------------------------------
# Sklearn backend
# ---------------------------------------------------------------------------


class _SklearnBackend:
    """Minimal backend wrapping sklearn's RandomForestRegressor."""

    name = "sklearn"

    def train(
        self,
        df: pd.DataFrame,
        value: str = "value",
        feature_names: list[str] | None = None,
        variables: list[str] | None = None,
        model_config: dict[str, Any] | None = None,
        seed: int = 7654321,
        verbose: bool = False,
        n_cores: int | None = None,
        use_gpu: bool = False,
    ) -> object:
        if variables is not None and feature_names is None:
            feature_names = variables
        if not feature_names:
            raise ValueError("feature_names required")
        # Use training subset if available
        if "set" in df.columns:
            df_train = df[df["set"] == "training"]
            if df_train.empty:
                df_train = df
        else:
            df_train = df
        model = RandomForestRegressor(n_estimators=10, random_state=seed, n_jobs=1)
        model.fit(df_train[feature_names], df_train[value])
        model.backend = "sklearn"
        model.feature_names_ = list(feature_names)
        return model

    def save(
        self,
        model: object,
        path: str = ".",
        filename: str = "sklearn.joblib",
    ) -> str:
        import joblib

        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / filename
        joblib.dump(model, str(p))
        return str(p)

    def load(
        self,
        path: str = ".",
        filename: str | None = None,
    ) -> object:
        import joblib

        p = Path(path) / filename if filename else Path(path)
        return joblib.load(str(p))


@pytest.fixture(scope="session", autouse=True)
def _register_sklearn_backend():
    """Register the sklearn backend before any test runs."""
    if not backend_registry.has("sklearn"):
        backend_registry.register(_SklearnBackend())
    yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_data() -> pd.DataFrame:
    """Small synthetic dataset that sklearn can model trivially."""
    n = 100
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 3.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(0, 0.1, n)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "date": dates,
            "value": y,
            "x1": x1,
            "x2": x2,
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSklearnBackendE2E:
    def test_backend_registered(self):
        assert backend_registry.has("sklearn")

    def test_train_predict_roundtrip(self, simple_data):
        df, model = build_model(
            df=simple_data,
            value="value",
            backend="sklearn",
            feature_names=["x1", "x2"],
            seed=42,
        )
        assert hasattr(model, "predict")
        assert getattr(model, "backend", None) == "sklearn"

        preds = model.predict(df[["x1", "x2"]])
        assert len(preds) == len(df)
        assert np.isfinite(preds).all()

    def test_train_model_directly(self, simple_data):
        model = train_model(
            df=simple_data,
            value="value",
            backend="sklearn",
            feature_names=["x1", "x2"],
            seed=42,
        )
        assert getattr(model, "backend", None) == "sklearn"

    def test_train_model_raises_no_features(self, simple_data):
        with pytest.raises(ValueError, match="feature_names"):
            train_model(df=simple_data, value="value", backend="sklearn", feature_names=None)

    def test_train_model_raises_duplicate_features(self, simple_data):
        with pytest.raises(ValueError, match="duplicate"):
            train_model(
                df=simple_data,
                value="value",
                backend="sklearn",
                feature_names=["x1", "x1"],
            )

    def test_train_model_raises_missing_column(self, simple_data):
        with pytest.raises(ValueError, match="not found"):
            train_model(
                df=simple_data,
                value="value",
                backend="sklearn",
                feature_names=["not_a_col"],
            )

    def test_normalise_with_sklearn_backend(self, simple_data):
        df, model = build_model(
            df=simple_data,
            value="value",
            backend="sklearn",
            feature_names=["x1", "x2"],
            seed=42,
        )
        result = normalise(
            df=df,
            model=model,
            feature_names=["x1", "x2"],
            n_samples=5,
            aggregate=True,
            seed=123,
            n_cores=1,
        )
        assert "normalised" in result.columns
        assert "observed" in result.columns
        assert len(result) == len(df)

    def test_normalise_with_return_quantiles(self, simple_data):
        df, model = build_model(
            df=simple_data,
            value="value",
            backend="sklearn",
            feature_names=["x1", "x2"],
            seed=42,
        )
        result = normalise(
            df=df,
            model=model,
            feature_names=["x1", "x2"],
            n_samples=10,
            aggregate=True,
            return_quantiles=[0.025, 0.5, 0.975],
            seed=123,
            n_cores=1,
        )
        # quantile columns use qNNN format
        for col in ["q025", "q500", "q975"]:
            assert col in result.columns, f"Missing quantile column: {col}"

    def test_normalise_aggregate_false(self, simple_data):
        df, model = build_model(
            df=simple_data,
            value="value",
            backend="sklearn",
            feature_names=["x1", "x2"],
            seed=42,
        )
        result = normalise(
            df=df,
            model=model,
            feature_names=["x1", "x2"],
            n_samples=3,
            aggregate=False,
            seed=123,
            n_cores=1,
        )
        assert "observed" in result.columns
        # Each seed becomes an integer-named column
        non_meta = [c for c in result.columns if c not in ("observed",)]
        assert len(non_meta) == 3
