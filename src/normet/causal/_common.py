# src/normet/causal/_common.py
"""Shared helpers for the synthetic-control estimators.

Keeps panel reshaping and donor-weight solving in one place so the individual
backends (``scm``, ``variants``, …) stay focused on their own algorithm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pivot_panel(
    df: pd.DataFrame,
    *,
    date_col: str,
    unit_col: str,
    outcome_col: str,
    treated_unit: str,
    donors: list[str] | None = None,
    parse_dates: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Reshape a long panel to wide (rows=date, cols=unit) and resolve donors.

    Parameters
    ----------
    parse_dates : bool, default True
        Parse/validate ``date_col`` as datetime first. Pass ``False`` when the
        caller has already parsed it to avoid redundant conversion.

    Returns
    -------
    (panel, donors)
        ``panel`` is the wide, date-sorted outcome matrix; ``donors`` is the
        validated donor pool (treated unit excluded, missing units dropped).

    Raises
    ------
    ValueError
        If ``date_col`` is missing/unparseable, the treated unit is absent, or
        no valid donors remain.
    """
    df = df.copy()
    if parse_dates:
        if date_col not in df.columns:
            raise ValueError(f"`date_col` '{date_col}' not found in df.")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].isna().any():
            n_bad = int(df[date_col].isna().sum())
            raise ValueError(f"{n_bad} rows have invalid {date_col} values.")

    panel = df.pivot_table(
        index=date_col, columns=unit_col, values=outcome_col, aggfunc="mean"
    ).sort_index()

    if treated_unit not in panel.columns:
        raise ValueError(f"Treated unit '{treated_unit}' not in panel.")
    if donors is None:
        donors = [u for u in panel.columns if u != treated_unit]
    else:
        donors = [u for u in donors if u in panel.columns and u != treated_unit]
    if not donors:
        raise ValueError("No valid donors after filtering.")
    return panel, donors


def solve_simplex_weights(
    X: np.ndarray,
    y: np.ndarray,
    *,
    allow_negative: bool = False,
) -> np.ndarray:
    """Solve ``min_w ||y - X w||²`` subject to ``sum(w) == 1``.

    With ``allow_negative=False`` (default) weights are additionally constrained
    to the simplex (``w >= 0``); otherwise only the sum-to-one equality holds.
    Falls back to uniform weights if the optimiser fails to converge.

    Parameters
    ----------
    X : ndarray, shape (n, J)
        Donor design matrix.
    y : ndarray, shape (n,)
        Target vector.
    """
    from scipy.optimize import Bounds, LinearConstraint, minimize

    J = X.shape[1]

    def obj(w: np.ndarray) -> float:
        return float(np.sum((y - X @ w) ** 2))

    def grad(w: np.ndarray) -> np.ndarray:
        return (2.0 * X.T @ (X @ w - y)).astype(float)

    Aeq = np.ones((1, J))
    beq = np.array([1.0])
    bounds = Bounds([-np.inf] * J, [np.inf] * J) if allow_negative else Bounds([0.0] * J, [1.0] * J)
    cons = [LinearConstraint(Aeq, beq, beq)]
    w0 = np.full(J, 1.0 / J)

    res = minimize(obj, w0, jac=grad, method="trust-constr", bounds=bounds, constraints=cons)
    w = res.x if res.success else w0
    if not allow_negative:
        w = np.maximum(w, 0.0)
        s = w.sum()
        w = w / s if s > 0 else w0
    return w
