"""Mock-based unit tests for model/train.py (build_model, train_model).

Uses a ``MockBackend`` registered in the global ``backend_registry`` so no
FLAML or other AutoML dependency is required.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from normet.backends import backend_registry
from normet.model.train import build_model, train_model

# ---------------------------------------------------------------------------
# Mock backend — protocol-compatible, records last train call
# ---------------------------------------------------------------------------


class MockBackend:
    name = "mock"
    last_call: dict[str, Any] | None = None

    def train(
        self,
        df: pd.DataFrame,
        target: str = "value",
        covariates: list[str] | None = None,
        variables: list[str] | None = None,
        model_config: dict[str, Any] | None = None,
        seed: int = 7654321,
        verbose: bool = True,
        n_cores: int | None = None,
    ) -> object:
        resolved = list(covariates or variables or [])
        MockBackend.last_call = {"variables": resolved, "value": target}

        class _StubModel:
            backend = "mock"

            def predict(self, X):
                return np.zeros(len(X))

        return _StubModel()

    def save(self, model: object, path: str = ".", filename: str = "automl.joblib") -> str:
        return str(path)

    def load(self, path: str = ".", filename: str | None = None) -> object:
        class _StubModel:
            backend = "mock"

        return _StubModel()


@pytest.fixture(autouse=True)
def _mock_backend():
    backend_registry._backends["mock"] = MockBackend()
    MockBackend.last_call = None
    yield
    del backend_registry._backends["mock"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 48
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "date": dates,
            "NO2": rng.uniform(10, 50, n),
            "t2m": rng.uniform(5, 25, n),
            "ws": rng.uniform(0, 10, n),
            "wd": rng.uniform(0, 360, n),
        }
    )


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------


class TestTrainModel:
    def test_unknown_backend(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            train_model(sample_df, backend="nonexistent", covariates=["t2m"])

    def test_empty_variables(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            train_model(sample_df, covariates=[])

    def test_duplicate_variables(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            train_model(sample_df, covariates=["t2m", "t2m"])

    def test_missing_columns(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Columns not found"):
            train_model(sample_df, covariates=["nonexistent"])

    def test_happy_path(self, sample_df: pd.DataFrame) -> None:
        model = train_model(sample_df, target="NO2", backend="mock", covariates=["t2m", "ws"])
        assert model.backend == "mock"
        preds = model.predict(sample_df[["t2m", "ws"]].head(5))
        assert len(preds) == 5

    def test_training_set_filter(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.copy()
        df["set"] = "testing"
        df.iloc[:24, df.columns.get_loc("set")] = "training"
        model = train_model(df, target="NO2", backend="mock", covariates=["t2m", "ws"])
        assert model.backend == "mock"

    def test_default_backend_is_flaml(self) -> None:
        import inspect

        from normet.model.train import train_model as _tm

        sig = inspect.signature(_tm)
        assert sig.parameters["backend"].default == "flaml"


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_empty_feature_names(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="covariates"):
            build_model(sample_df, target="NO2", covariates=[])

    def test_unknown_backend(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            build_model(sample_df, target="NO2", backend="nonexistent", covariates=["t2m"])

    def test_happy_path(self, sample_df: pd.DataFrame) -> None:
        df_prep, model = build_model(
            sample_df,
            target="NO2",
            backend="mock",
            covariates=["t2m", "ws"],
        )
        assert model.backend == "mock"
        assert "set" in df_prep.columns
        assert df_prep["set"].isin(["training", "testing"]).all()
        assert "value" in df_prep.columns
        # default split is "random"
        n_train = (df_prep["set"] == "training").sum()
        assert n_train > 0

    def test_happy_path_different_target_name(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.rename(columns={"NO2": "PM25"})
        df_prep, model = build_model(df, target="PM25", backend="mock", covariates=["t2m", "ws"])
        assert "value" in df_prep.columns
        assert model.backend == "mock"

    def test_drop_time_features(self, sample_df: pd.DataFrame) -> None:
        build_model(
            sample_df,
            target="NO2",
            backend="mock",
            covariates=["t2m", "ws", "date_unix", "day_julian"],
            drop_time_features=True,
        )
        assert MockBackend.last_call is not None
        # time features should NOT be in the variables passed to train_model
        assert "date_unix" not in MockBackend.last_call["variables"]
        assert "day_julian" not in MockBackend.last_call["variables"]
        assert "t2m" in MockBackend.last_call["variables"]
        assert "ws" in MockBackend.last_call["variables"]

    def test_ts_split(self, sample_df: pd.DataFrame) -> None:
        df_prep, model = build_model(
            sample_df,
            target="NO2",
            backend="mock",
            covariates=["t2m", "ws"],
            split_method="ts",
            train_fraction=0.8,
        )
        assert model.backend == "mock"
        # With ts split, first 80% should be training
        n_train = (df_prep["set"] == "training").sum()
        assert n_train == int(0.8 * len(df_prep))

    def test_model_config_passthrough(self, sample_df: pd.DataFrame) -> None:
        config: dict[str, Any] = {"time_budget": 10, "metric": "mae"}
        build_model(
            sample_df,
            target="NO2",
            backend="mock",
            covariates=["t2m"],
            model_config=config,
        )
