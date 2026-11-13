# src/normet/model/predict.py
"""Model prediction helpers: :func:`ml_predict` and :func:`ml_predict_dask`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..backends import backend_registry
from ..utils._lazy import require
from ..utils.features import extract_features
from ..utils.logging import get_logger

log = get_logger(__name__)


def ml_predict(
    model: Any,
    newdata: pd.DataFrame,
    *,
    chunk_size: int | None = None,
    num_threads: int | None = None,
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
        are computed in chunks and concatenated, bounding the float64 copy the
        backend materialises per call. The default derives the row count from
        the frame's width for a ~256 MB payload (13 features: ~2.4M rows, so
        an ordinary frame is predicted in one call; 1,000 features: ~33k
        rows). A fixed 200k-row default split a 13-feature, 1.74M-row
        normalise into 9 calls and cost 12% against a single call at 8
        threads (4.85 s vs 4.28 s).
    num_threads : int, optional
        Thread count for backends that take one (LightGBM; ignored by the
        others). LightGBM's prediction is memory-bound and scales ~5.6x from
        1 to 8 threads on a 300-resample normalise of the bundled MY1 record
        (23.1 s -> 4.1 s, bit-identical results), but it reads the thread
        count from the OpenMP environment at first use -- which HPC test
        harnesses often pin to 1. Passing it per call lets ``normalise``'s
        ``n_cores`` mean what it says on the batch path too.

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

    # Only LightGBM's wrapper takes a thread count; passing it to a sklearn or
    # FLAML estimator would be a TypeError, so it is opted into by backend name.
    predict_kwargs: dict[str, Any] = {}
    if num_threads is not None and model_type == "lightgbm":
        predict_kwargs["num_threads"] = max(1, int(num_threads))

    try:
        # See the docstring: the default bounds each call's float64 feature
        # copy at ~256 MB instead of fixing a row count that is far too small
        # for a narrow frame and far too large for a wide one.
        default_cs = max(1_000, (256 * 1024**2) // (max(1, X.shape[1]) * 8))
        cs = int(chunk_size) if chunk_size is not None else int(default_cs)
        if cs > 0 and n_rows > cs:
            parts: list[np.ndarray] = []
            for start in range(0, n_rows, cs):
                stop = min(start + cs, n_rows)
                chunk = X.iloc[start:stop] if hasattr(X, "iloc") else X[start:stop]
                parts.append(
                    np.asarray(model.predict(chunk, **predict_kwargs), dtype=float).reshape(-1)
                )
            yhat = np.concatenate(parts, axis=0)
        else:
            yhat = model.predict(X, **predict_kwargs)

        return np.asarray(yhat, dtype=float).reshape(-1)

    except AttributeError:
        log.exception("Prediction failed: missing method or invalid input.")
        raise
    except Exception:
        log.exception("Unexpected error during prediction.")
        raise


def ml_predict_dask(
    model: Any,
    ddf: Any,
    *,
    chunk_size: int | None = None,
) -> Any:
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
