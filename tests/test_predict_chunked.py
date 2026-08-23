"""Chunked prediction path on a stub sklearn-like model."""

import numpy as np
import pandas as pd

from normet.model.predict import ml_predict


class _IdentityModel:
    """Pretends to be a FLAML model; returns the sum of its features."""

    backend = "flaml"

    @property
    def feature_names_in_(self):  # sklearn-style attribute used by extract_features
        return ["a", "b"]

    @property
    def feature_importances_(self):
        return [1.0, 1.0]

    def predict(self, X):
        return X["a"].to_numpy() + X["b"].to_numpy()


def test_ml_predict_chunked_matches_unchunked():
    n = 1000
    df = pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.arange(n, dtype=float) * 0.5})
    full = ml_predict(_IdentityModel(), df, chunk_size=None)
    chunked = ml_predict(_IdentityModel(), df, chunk_size=137)
    np.testing.assert_allclose(full, chunked)


def test_ml_predict_chunk_size_zero_is_single_pass():
    n = 50
    df = pd.DataFrame({"a": np.ones(n), "b": np.zeros(n)})
    out = ml_predict(_IdentityModel(), df, chunk_size=0)
    np.testing.assert_allclose(out, np.ones(n))
