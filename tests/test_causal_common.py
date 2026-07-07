"""Unit tests for the shared synthetic-control primitives in causal/_common.py."""

import numpy as np
import pandas as pd
import pytest

from normet.causal._common import pivot_panel, solve_simplex_weights


def _long_panel() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    rows = []
    for unit, base in [("T", 1.0), ("A", 2.0), ("B", 3.0)]:
        for i, d in enumerate(dates):
            rows.append({"date": d, "unit": unit, "y": base + i})
    return pd.DataFrame(rows)


def test_pivot_panel_basic():
    panel, donors = pivot_panel(
        _long_panel(), date_col="date", unit_col="unit", outcome_col="y", treated_unit="T"
    )
    assert panel.shape == (5, 3)
    assert "T" in panel.columns
    assert set(donors) == {"A", "B"}
    # Sorted by date ascending.
    assert list(panel.index) == sorted(panel.index)


def test_pivot_panel_filters_unknown_donors():
    _, donors = pivot_panel(
        _long_panel(),
        date_col="date",
        unit_col="unit",
        outcome_col="y",
        treated_unit="T",
        donors=["A", "ghost"],
    )
    assert donors == ["A"]


def test_pivot_panel_missing_treated_raises():
    with pytest.raises(ValueError):
        pivot_panel(
            _long_panel(), date_col="date", unit_col="unit", outcome_col="y", treated_unit="ZZZ"
        )


def test_pivot_panel_unparseable_dates_raise():
    # date column carries an unparseable string (object dtype) → coerced to NaT.
    df = pd.DataFrame(
        {
            "date": ["2023-01-01", "not-a-date", "2023-01-03"],
            "unit": ["T", "A", "B"],
            "y": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError):
        pivot_panel(df, date_col="date", unit_col="unit", outcome_col="y", treated_unit="T")


def test_solve_simplex_weights_recovers_convex_combo():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    true_w = np.array([0.2, 0.5, 0.3])  # on the simplex
    y = X @ true_w
    w = solve_simplex_weights(X, y)
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert (w >= -1e-9).all()
    assert np.allclose(w, true_w, atol=5e-3)


def test_solve_simplex_weights_allow_negative():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3))
    true_w = np.array([1.5, -0.8, 0.3])  # sums to 1 but leaves the simplex
    y = X @ true_w
    w = solve_simplex_weights(X, y, allow_negative=True)
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert np.allclose(w, true_w, atol=1e-2)
    # The simplex-constrained solver cannot reproduce a negative weight.
    w_simplex = solve_simplex_weights(X, y, allow_negative=False)
    assert (w_simplex >= -1e-9).all()
