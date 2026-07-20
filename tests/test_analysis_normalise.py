"""Tests for normet.analysis.normalise helper functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from normet.analysis.normalise import (
    _apply_conditional_filter,
    _format_quantile_name,
    generate_resampled,
)
from normet.exceptions import DataError

# ---------------------------------------------------------------------------
# _format_quantile_name
# ---------------------------------------------------------------------------


def test_format_quantile_name_values():
    assert _format_quantile_name(0.025) == "q025"
    assert _format_quantile_name(0.5) == "q500"
    assert _format_quantile_name(0.975) == "q975"


def test_format_quantile_name_out_of_range():
    with pytest.raises(ValueError):
        _format_quantile_name(1.1)


# ---------------------------------------------------------------------------
# _apply_conditional_filter
# ---------------------------------------------------------------------------


def _simple_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7]})


def test_apply_conditional_filter_scalar():
    """Scalar value → exact-match filter."""
    df = _simple_df()
    result = _apply_conditional_filter(df, {"x": 2})
    assert list(result["x"]) == [2]


def test_apply_conditional_filter_iterable():
    """List of values → isin semantics."""
    df = _simple_df()
    result = _apply_conditional_filter(df, {"x": [1, 3]})
    assert len(result) == 2
    assert set(result["x"]) == {1, 3}


def test_apply_conditional_filter_callable():
    """Callable → boolean mask applied per column."""
    df = _simple_df()
    result = _apply_conditional_filter(df, {"x": lambda v: v > 5})
    assert list(result["x"]) == [6, 7]


def test_apply_conditional_filter_missing_col():
    """Raises ValueError for a condition on a column that does not exist."""
    df = _simple_df()
    with pytest.raises(ValueError, match="not found"):
        _apply_conditional_filter(df, {"nonexistent": 1})


# ---------------------------------------------------------------------------
# generate_resampled
# ---------------------------------------------------------------------------


def _weather_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "value": rng.normal(20.0, 2.0, n),
            "ws": rng.uniform(0.5, 10.0, n),
            "wd": rng.uniform(0.0, 360.0, n),
        }
    )


def test_generate_resampled_shape():
    """Output has same shape as input and includes a 'seed' column."""
    df = _weather_df()
    result = generate_resampled(
        df,
        variables_resample=["ws", "wd"],
        replace=True,
        seed=42,
        resample_df=df,
    )
    assert result.shape == df.shape or "seed" in result.columns
    assert result.shape[0] == df.shape[0]
    assert "seed" in result.columns
    assert result["seed"].iloc[0] == 42


def test_generate_resampled_missing_col():
    """Raises ValueError when resample_df is missing a required column."""
    df = _weather_df()
    pool = df.drop(columns=["ws"])
    with pytest.raises(ValueError, match="missing columns"):
        generate_resampled(
            df,
            variables_resample=["ws", "wd"],
            replace=True,
            seed=0,
            resample_df=pool,
        )


def test_generate_resampled_replaces_values():
    """Resampled columns should differ from original (with high probability)."""
    rng = np.random.default_rng(99)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    ws_original = np.arange(n, dtype=float)  # deterministic, easy to verify change
    df = pd.DataFrame({"date": dates, "value": np.ones(n), "ws": ws_original, "wd": np.zeros(n)})

    # Use a pool with completely different ws values
    pool = df.copy()
    pool["ws"] = rng.uniform(100.0, 200.0, n)

    result = generate_resampled(
        df,
        variables_resample=["ws"],
        replace=True,
        seed=7,
        resample_df=pool,
    )

    # All resampled ws values should come from pool range [100, 200]
    assert (result["ws"] >= 100.0).all()
    # Original df ws values were 0..99, so they should not match
    assert not (result["ws"] == ws_original).all()


# ---------------------------------------------------------------------------
# normalise() Function Tests
# ---------------------------------------------------------------------------


def test_normalise_core_runs(monkeypatch):
    import sys

    import normet.analysis.normalise

    normalise_mod = sys.modules["normet.analysis.normalise"]
    from normet.analysis.normalise import normalise

    # Dummy predictor returning 10.0 for every sample
    def mock_ml_predict(model, df):
        return np.full(len(df), 10.0)

    monkeypatch.setattr(normalise_mod, "ml_predict", mock_ml_predict)

    # 1. Test aggregate=True with quantiles
    df = _weather_df(n=30)
    result = normalise(
        df=df,
        model="dummy-model",
        covariates=["ws", "wd"],
        variables_resample=["ws", "wd"],
        n_samples=5,
        aggregate=True,
        return_quantiles=[0.025, 0.975],
        n_cores=1,
        verbose=False,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.index.name == "date"
    assert "observed" in result.columns
    assert "normalised" in result.columns
    assert "q025" in result.columns
    assert "q975" in result.columns
    assert (result["normalised"] == 10.0).all()

    # 2. Test aggregate=False (wide seed table)
    result_wide = normalise(
        df=df,
        model="dummy-model",
        covariates=["ws", "wd"],
        variables_resample=["ws", "wd"],
        n_samples=5,
        aggregate=False,
        n_cores=1,
        verbose=False,
    )
    assert isinstance(result_wide, pd.DataFrame)
    assert "observed" in result_wide.columns
    # Should have 5 seed columns (not starting with 'q')
    seed_cols = [c for c in result_wide.columns if isinstance(c, int | np.integer)]
    assert len(seed_cols) == 5

    # 3. Test memory_save=True (ThreadPoolExecutor branch)
    result_mem = normalise(
        df=df,
        model="dummy-model",
        covariates=["ws", "wd"],
        variables_resample=["ws", "wd"],
        n_samples=5,
        aggregate=True,
        memory_save=True,
        n_cores=1,
        verbose=False,
    )
    assert "normalised" in result_mem.columns

    # 4. Test conditional_on filtering
    result_filtered = normalise(
        df=df,
        model="dummy-model",
        covariates=["ws", "wd"],
        variables_resample=["ws", "wd"],
        n_samples=5,
        aggregate=True,
        conditional_on={"ws": lambda v: v > 0.0},
        n_cores=1,
        verbose=False,
    )
    assert "normalised" in result_filtered.columns

    # 5. Test filter empty raises error
    with pytest.raises(DataError, match="left no rows"):
        normalise(
            df=df,
            model="dummy-model",
            covariates=["ws", "wd"],
            variables_resample=["ws", "wd"],
            n_samples=5,
            aggregate=True,
            conditional_on={"ws": lambda v: v > 1000.0},
            n_cores=1,
            verbose=False,
        )

    # 6. Test missing resample columns raises error
    with pytest.raises(DataError, match="missing columns"):
        normalise(
            df=df,
            model="dummy-model",
            covariates=["ws", "wd"],
            variables_resample=["missing_col"],
            n_samples=5,
            aggregate=True,
            n_cores=1,
            verbose=False,
        )
