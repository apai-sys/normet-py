# src/normet/model/predict.py
"""Model prediction helpers: :func:`ml_predict` and :func:`ml_predict_dask`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backends import backend_registry
from ..utils._lazy import require
from ..utils.features import extract_features
from ..utils.logging import get_logger

log = get_logger(__name__)


def _lgb_model_type():
    """Return LgbModel class (lazy import to avoid circular dependency)."""
    from ..backends.lgb_backend import LgbModel

    return LgbModel


def ml_predict(
    model,
    newdata: pd.DataFrame,
    *,
    chunk_size: int | None = None,
    use_gpu: bool | None = None,
) -> np.ndarray:
    """
    Predict using a trained AutoML model.

    Parameters
    ----------
    model : object
        Trained model with a ``backend`` attribute matching a registered
        backend and a ``predict`` method.
    newdata : pandas.DataFrame
        Feature matrix for prediction. Must contain the columns used by the model.
    chunk_size : int, optional
        If set and the input has more than ``chunk_size`` rows, predictions
        are computed in chunks and concatenated. Useful when the model's
        predict path materialises a large intermediate (e.g., LightGBM with
        wide feature matrices). Defaults to 200_000.

    Returns
    -------
    numpy.ndarray
        1D float array of predictions aligned to ``newdata`` rows used.

    Raises
    ------
    TypeError
        If the model's backend is not registered.
    ValueError
        If none of the model's features are present in ``newdata``.
    """
    model_type = getattr(model, "backend", None)
    if not model_type or not backend_registry.has(model_type):
        available = backend_registry.available
        raise TypeError(
            f"Unsupported model backend '{model_type}'. Expected one of: {', '.join(available)}"
        )

    # Try to respect the model's feature order
    try:
        feature_cols: list[str] = extract_features(model)
        use_cols = [c for c in feature_cols if c in newdata.columns]
        if len(use_cols) != len(feature_cols):
            missing = [c for c in feature_cols if c not in newdata.columns]
            if missing:
                log.warning("Missing features in newdata (excluded from prediction): %s", missing)
        if not use_cols:
            raise ValueError("No model features found in `newdata`.")
        X = newdata.loc[:, use_cols]
    except Exception as e:
        log.debug("extract_features failed (%s); falling back to all columns in `newdata`.", e)
        X = newdata
        if X.shape[1] == 0:
            raise ValueError("`newdata` has no columns to predict on.") from e

    n_rows = len(X)
    if n_rows == 0:
        return np.array([], dtype=float)

    # Resolve effective GPU flag: explicit arg overrides model attribute.
    _use_gpu = use_gpu if use_gpu is not None else getattr(model, "use_gpu", False)

    # cuDF path: convert feature frame to GPU DataFrame for models that
    # accept cuDF natively (cuML, cuDF-native models).  LightGBM and FLAML
    # handle their own CUDA context after GPU training — no cuDF needed there.
    if _use_gpu and not isinstance(model, _lgb_model_type()):
        try:
            import cudf as _cudf

            X = _cudf.DataFrame.from_pandas(X)
        except ImportError:
            pass  # cuDF not installed; model.predict falls back to numpy
        except Exception as _e:
            log.debug("cuDF conversion failed (%s); using pandas X.", _e)

    try:
        cs = int(chunk_size) if chunk_size is not None else 200_000
        if cs > 0 and n_rows > cs:
            parts: list[np.ndarray] = []
            for start in range(0, n_rows, cs):
                stop = min(start + cs, n_rows)
                chunk = X.iloc[start:stop] if hasattr(X, "iloc") else X[start:stop]
                parts.append(np.asarray(model.predict(chunk), dtype=float).reshape(-1))
            yhat = np.concatenate(parts, axis=0)
        else:
            yhat = model.predict(X)

        return np.asarray(yhat, dtype=float).reshape(-1)

    except AttributeError:
        log.exception("Prediction failed: missing method or invalid input.")
        raise
    except Exception:
        log.exception("Unexpected error during prediction.")
        raise


def ml_predict_dask(
    model,
    ddf,
    *,
    chunk_size: int | None = None,
):
    """
    Predict over a :class:`dask.dataframe.DataFrame` lazily.

    Each partition is forwarded to :func:`ml_predict` independently.
    Useful when the prediction grid does not fit in memory.

    Parameters
    ----------
    model : object
        Trained ``normet`` model (FLAML).
    ddf : dask.dataframe.DataFrame
        Partitioned feature frame. Must contain the model's features.
    chunk_size : int, optional
        Forwarded to :func:`ml_predict` for each partition.

    Returns
    -------
    dask.array.Array
        Lazy 1-D float array of predictions; call ``.compute()`` to materialise.
    """
    # Verify dask is importable; we don't need to keep handles since we use
    # the high-level Dask DataFrame API on the user-supplied `ddf` directly.
    require("dask.dataframe", hint="pip install dask[dataframe]")
    require("dask.array", hint="pip install dask[array]")

    def _one(part: pd.DataFrame) -> np.ndarray:
        if len(part) == 0:
            return np.empty(0, dtype=float)
        return ml_predict(model, part, chunk_size=chunk_size)

    # map_partitions returns a Series; pull to a numpy via to_dask_array
    series = ddf.map_partitions(lambda p: pd.Series(_one(p), index=p.index), meta=("yhat", "f8"))
    return series.to_dask_array(lengths=True)
