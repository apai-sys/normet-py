# src/normet/model/__init__.py
"""Model training, prediction, and persistence utilities.

This subpackage provides a unified interface for training, predicting,
saving, and loading models across registered AutoML backends (see
:data:`normet.backends.backend_registry`).
"""

from .io import load_model, save_model
from .predict import ml_predict, ml_predict_dask
from .train import build_model, train_model

__all__ = [
    "build_model",
    "train_model",
    "ml_predict",
    "ml_predict_dask",
    "load_model",
    "save_model",
]
