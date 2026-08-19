"""Mock-based tests for model/predict.py (ml_predict error paths).

Chunked-prediction correctness is already tested in ``test_predict_chunked.py``.
This file covers error handling and edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from normet.model.predict import ml_predict


class _StubModel:
    """FLAML-like stub whose features are a subset of ``feature_names_in_``."""

    def __init__(self, feature_names: list[str], backend: str = "flaml"):
        self.backend = backend
        self.feature_names_in_ = np.array(feature_names)
        self.feature_importances_ = np.ones(len(feature_names))

    def predict(self, X):
        return np.ones(len(X))


class _NoBackendModel:
    """Model with no ``backend`` attribute."""

    def predict(self, X):
        return np.ones(len(X))


class _NoFeatureModel:
    """Model whose features are not present in newdata."""

    backend = "flaml"

    @property
    def feature_names_in_(self):
        return ["a", "b"]

    @property
    def feature_importances_(self):
        return [1.0, 1.0]

    def predict(self, X):
        return np.ones(len(X))


class _FailingPredictModel:
    """Model whose predict raises AttributeError."""

    backend = "flaml"

    @property
    def feature_names_in_(self):
        return ["a"]

    @property
    def feature_importances_(self):
        return [1.0]

    @property
    def predict(self):
        raise AttributeError("broken")


class TestMlPredict:
    def test_unknown_backend(self) -> None:
        model = _StubModel(["a"], backend="nonexistent")
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(TypeError, match="Unsupported model backend"):
            ml_predict(model, df)

    def test_no_backend_attribute(self) -> None:
        model = _NoBackendModel()
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(TypeError, match="Unsupported model backend"):
            ml_predict(model, df)

    def test_no_features_in_newdata_falls_back(self) -> None:
        model = _NoFeatureModel()
        df = pd.DataFrame({"x": [1.0, 2.0]})  # no 'a' or 'b'
        # ml_predict catches ValueError from the feature check and falls back
        # to all newdata columns; if the model can predict on them it succeeds.
        result = ml_predict(model, df)
        assert result.shape == (2,)

    def test_empty_newdata_columns(self) -> None:
        model = _StubModel(["a"])
        df = pd.DataFrame(index=[0, 1])  # no columns
        with pytest.raises(ValueError, match="no columns"):
            ml_predict(model, df)

    def test_empty_newdata_rows(self) -> None:
        model = _StubModel(["a"])
        df = pd.DataFrame({"a": []})
        result = ml_predict(model, df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)

    def test_happy_path(self) -> None:
        model = _StubModel(["a", "b"])
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = ml_predict(model, df)
        assert result.shape == (3,)
        assert np.allclose(result, 1.0)

    def test_subset_features_available(self) -> None:
        """Some features present, some missing — predict with the intersection."""
        model = _StubModel(["a", "b", "c"])
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})  # 'c' missing
        result = ml_predict(model, df)
        assert result.shape == (1,)

    def test_chunk_size_zero(self) -> None:
        model = _StubModel(["a"])
        df = pd.DataFrame({"a": np.ones(100)})
        result = ml_predict(model, df, chunk_size=0)
        assert result.shape == (100,)

    def test_chunk_size_one(self) -> None:
        model = _StubModel(["a"])
        df = pd.DataFrame({"a": np.ones(10)})
        result = ml_predict(model, df, chunk_size=1)
        assert result.shape == (10,)
