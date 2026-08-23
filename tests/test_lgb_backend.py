"""Unit tests for normet/backends/lgb_backend.py.

Uses ``unittest.mock`` to replace lightgbm so tests run without the
native library installed.  Integration tests (marked ``needs_lgb``) verify
end-to-end training.
"""

from __future__ import annotations

import importlib.util
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from normet.backends.lgb_backend import LgbModel, _LgbBackend, train_lgb


def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


needs_lgb = pytest.mark.skipif(not _has("lightgbm"), reason="lightgbm not installed")

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubBooster:
    """Minimal lightgbm Booster stand-in."""

    def __init__(self, feature_names: list[str]) -> None:
        self._feature_names = feature_names

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

    def feature_name(self) -> list[str]:
        return self._feature_names

    def feature_importance(self, importance_type: str = "gain") -> list[float]:
        return [1.0] * len(self._feature_names)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 48
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "value": rng.uniform(10, 50, n),
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
        }
    )


@pytest.fixture
def stub_booster() -> _StubBooster:
    return _StubBooster(["x1", "x2"])


# ---------------------------------------------------------------------------
# LgbModel wrapper
# ---------------------------------------------------------------------------


class TestLgbModel:
    def test_attributes(self, stub_booster: _StubBooster) -> None:
        model = LgbModel(stub_booster, ["x1", "x2"])
        assert model.backend == "lightgbm"
        assert model.feature_names == ["x1", "x2"]
        assert model.booster is stub_booster

    def test_predict(self, stub_booster: _StubBooster) -> None:
        model = LgbModel(stub_booster, ["x1", "x2"])
        result = model.predict(np.ones((5, 2)))
        assert result.shape == (5,)
        assert (result == 0.0).all()

    def test_predict_replaces_non_finite(self, stub_booster: _StubBooster) -> None:
        model = LgbModel(stub_booster, ["x1", "x2"])
        X = np.array([[1.0, np.nan], [np.inf, 3.0], [-np.inf, 5.0]])
        result = model.predict(X)
        assert result.shape == (3,)

    def test_feature_name(self, stub_booster: _StubBooster) -> None:
        model = LgbModel(stub_booster, ["x1", "x2"])
        assert model.feature_name() == ["x1", "x2"]

    def test_feature_importance(self, stub_booster: _StubBooster) -> None:
        model = LgbModel(stub_booster, ["x1", "x2"])
        assert model.feature_importance() == [1.0, 1.0]


# ---------------------------------------------------------------------------
# train_lgb — validation (no lightgbm needed)
# ---------------------------------------------------------------------------


class TestTrainLgbValidation:
    def test_empty_feature_names(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            train_lgb(sample_df, covariates=[])

    def test_duplicate_feature_names(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            train_lgb(sample_df, covariates=["x1", "x1"])

    def test_missing_columns(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Columns not found"):
            train_lgb(sample_df, covariates=["nonexistent"])

    def test_deprecated_variables(self, sample_df: pd.DataFrame) -> None:
        exc = ImportError("mock no lightgbm")
        with (
            patch("normet.backends.lgb_backend._import_lightgbm", side_effect=exc),
            pytest.warns(DeprecationWarning, match="`variables` is deprecated"),
            pytest.raises(ImportError),
        ):
            train_lgb(sample_df, variables=["x1"])


# ---------------------------------------------------------------------------
# train_lgb — integration (needs lightgbm installed)
# ---------------------------------------------------------------------------


@needs_lgb
class TestTrainLgbIntegration:
    def test_happy_path(self, sample_df: pd.DataFrame) -> None:
        model = train_lgb(
            sample_df,
            target="value",
            covariates=["x1", "x2"],
            model_config={"n_trials": 2, "cv_folds": 2, "nrounds": 10, "early_stopping_rounds": 0},
            seed=42,
        )
        assert isinstance(model, LgbModel)
        assert model.backend == "lightgbm"
        assert model.feature_names == ["x1", "x2"]
        # should produce reasonable predictions
        preds = model.predict(sample_df[["x1", "x2"]])
        assert preds.shape == (len(sample_df),)
        assert np.all(np.isfinite(preds))

    def test_training_set_filter(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.copy()
        df["set"] = "testing"
        df.iloc[:24, df.columns.get_loc("set")] = "training"
        model = train_lgb(
            df,
            target="value",
            covariates=["x1", "x2"],
            model_config={"n_trials": 2, "cv_folds": 2, "nrounds": 10, "early_stopping_rounds": 0},
            seed=42,
        )
        assert isinstance(model, LgbModel)

    def test_na_target_raises(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.copy()
        df.loc[0, "value"] = np.nan
        with pytest.raises(ValueError, match="NA"):
            train_lgb(
                df,
                target="value",
                covariates=["x1", "x2"],
                model_config={
                    "n_trials": 1,
                    "cv_folds": 2,
                    "nrounds": 5,
                    "early_stopping_rounds": 0,
                },
                seed=42,
            )

    def test_default_backend_string(self) -> None:
        assert _LgbBackend.name == "lightgbm"


# ---------------------------------------------------------------------------
# save / load (mock-based)
# ---------------------------------------------------------------------------


class TestSaveLoadLgb:
    def test_save_and_load(self, tmp_path, stub_booster: _StubBooster) -> None:
        model = LgbModel(stub_booster, ["x1", "x2"])
        from normet.backends.lgb_backend import load_lgb, save_lgb

        save_lgb(model, path=str(tmp_path), filename="test_lgb.joblib")
        loaded = load_lgb(path=str(tmp_path), filename="test_lgb.joblib")
        assert isinstance(loaded, LgbModel)
        assert loaded.backend == "lightgbm"
        assert loaded.feature_names == ["x1", "x2"]

    def test_load_auto_pick(self, tmp_path, stub_booster: _StubBooster) -> None:
        model = LgbModel(stub_booster, ["x1", "x2"])
        from normet.backends.lgb_backend import load_lgb, save_lgb

        save_lgb(model, path=str(tmp_path))
        loaded = load_lgb(path=str(tmp_path))
        assert isinstance(loaded, LgbModel)

    def test_load_file_not_found(self, tmp_path) -> None:
        from normet.backends.lgb_backend import load_lgb

        with pytest.raises(FileNotFoundError):
            load_lgb(path=str(tmp_path), filename="nonexistent.joblib")


# ---------------------------------------------------------------------------
# _LgbBackend protocol wrapper
# ---------------------------------------------------------------------------


class TestLgbBackendWrapper:
    def test_backend_registration(self) -> None:
        from normet.backends import backend_registry

        be = backend_registry.get("lightgbm")
        assert be.name == "lightgbm"
        assert hasattr(be, "train")
        assert hasattr(be, "save")
        assert hasattr(be, "load")

    @needs_lgb
    def test_train_with_variables_deprecated(self, sample_df: pd.DataFrame) -> None:
        be = _LgbBackend()
        with pytest.warns(DeprecationWarning, match="variables"):
            model = be.train(
                sample_df,
                target="value",
                variables=["x1", "x2"],
                model_config={
                    "n_trials": 1,
                    "cv_folds": 2,
                    "nrounds": 5,
                    "early_stopping_rounds": 0,
                },
                seed=42,
            )
        assert isinstance(model, LgbModel)

    def test_save_delegates(self, stub_booster: _StubBooster, tmp_path) -> None:
        be = _LgbBackend()
        model = LgbModel(stub_booster, ["x1", "x2"])
        p = be.save(model, path=str(tmp_path), filename="test.joblib")
        assert "test.joblib" in p

    def test_load_delegates(self, stub_booster: _StubBooster, tmp_path) -> None:
        be = _LgbBackend()
        model = LgbModel(stub_booster, ["x1", "x2"])
        be.save(model, path=str(tmp_path), filename="test.joblib")
        loaded = be.load(path=str(tmp_path), filename="test.joblib")
        assert isinstance(loaded, LgbModel)
