"""Input validation utilities for normet analysis functions.

Centralises common DataFrame boundary checks so each analysis function
doesn't reinvent the same guards.
"""

from __future__ import annotations

import pandas as pd

from ..exceptions import DataError


def require_column(df: pd.DataFrame, col: str, label: str = "column") -> None:
    """Raise ``DataError`` if ``col`` is not in ``df.columns``."""
    if col not in df.columns:
        raise DataError(
            f"{label.capitalize()} '{col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )


def require_not_empty(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """Raise ``DataError`` if ``df`` has zero rows."""
    if df.empty:
        raise DataError(f"{label} is empty (0 rows).")


def require_no_nan_in(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ``DataError`` if any of ``columns`` contain NaN."""
    missing = {c for c in columns if c in df.columns and df[c].isna().any()}
    if missing:
        raise DataError(
            f"Column(s) {sorted(missing)} contain NaN values. "
            "Use impute_values() or dropna() before calling this function."
        )


def require_no_duplicates(df: pd.DataFrame, col: str) -> None:
    """Raise ``DataError`` if ``col`` contains duplicate values."""
    if col in df.columns and df[col].duplicated().any():
        raise DataError(
            f"Column '{col}' contains duplicate values. "
            "Remove duplicates before calling this function."
        )
