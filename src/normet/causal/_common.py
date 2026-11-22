# src/normet/causal/_common.py
"""Shared helpers for the synthetic-control estimators.

Keeps panel reshaping and donor-weight solving in one place so the individual
backends (``scm``, ``variants``, …) stay focused on their own algorithm.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "newey_west_se_mean",
    "newey_west_se_slope",
    "pivot_panel",
    "project_simplex",
    "solve_simplex_weights",
    "solve_simplex_weights_fast",
]


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


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex ``{w >= 0, sum(w) == 1}``.

    Implements the exact :math:`O(m \\log m)` algorithm of Duchi et al. (2008).
    """
    arr = np.asarray(v, dtype=float).ravel()
    n = arr.size
    if n == 0:
        raise ValueError("Cannot project an empty vector onto the simplex.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input vector contains non-finite values.")
    if n == 1:
        return np.array([1.0], dtype=float)
    u = np.sort(arr)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1, dtype=float)
    cond = (u - cssv / ind) > 0.0
    nz = np.nonzero(cond)[0]
    rho = int(nz[-1]) if nz.size else 0
    theta = cssv[rho] / float(rho + 1)
    w = np.maximum(arr - theta, 0.0)
    s = float(w.sum())
    return w / s if s > 0.0 else np.full(n, 1.0 / n, dtype=float)


def solve_simplex_weights_fast(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2_penalty: float = 0.0,
    max_iter: int = 2000,
    tol: float = 1e-9,
) -> np.ndarray:
    """Fast FISTA solver with adaptive restart on the probability simplex.

    Minimises ``||y - X w||² + l2_penalty * ||w||²`` subject to
    ``w >= 0, sum(w) == 1``. Designed for both donor weights and high-dimensional
    time weights (e.g. Synthetic DiD).
    """
    if float(l2_penalty) < 0.0:
        raise ValueError(f"l2_penalty must be non-negative, got {l2_penalty}")
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X_arr.shape}.")
    n, J = X_arr.shape
    if J == 0:
        raise ValueError("Design matrix X has zero columns.")
    if J == 1:
        return np.array([1.0], dtype=float)

    lam2 = float(l2_penalty)
    s_vals = np.linalg.svd(X_arr, compute_uv=False)
    s_max2 = float(s_vals[0] ** 2) if s_vals.size else 0.0
    lip = 2.0 * (s_max2 + lam2)
    if lip <= 1e-12:
        return np.full(J, 1.0 / J, dtype=float)

    step = 1.0 / lip
    use_gram = n >= J
    if use_gram:
        gram = X_arr.T @ X_arr
        xty = X_arr.T @ y_arr
        yty = float(np.dot(y_arr, y_arr))

    w = np.full(J, 1.0 / J, dtype=float)
    z = w.copy()
    t = 1.0
    f_prev = float("inf")

    for it in range(int(max_iter)):
        if use_gram:
            grad_z = 2.0 * (gram @ z - xty + lam2 * z)
        else:
            grad_z = 2.0 * (X_arr.T @ (X_arr @ z - y_arr) + lam2 * z)

        w_next = project_simplex(z - step * grad_z)
        diff = float(np.linalg.norm(w_next - w, ord=np.inf))

        if diff <= tol * 0.1 or (diff <= tol and it % 5 == 0):
            if use_gram:
                f_cur = float(
                    yty
                    - 2.0 * np.dot(w_next, xty)
                    + np.dot(w_next, gram @ w_next)
                    + lam2 * np.dot(w_next, w_next)
                )
            else:
                r_next = X_arr @ w_next - y_arr
                f_cur = float(np.dot(r_next, r_next) + lam2 * np.dot(w_next, w_next))
            if abs(f_prev - f_cur) <= tol * max(1.0, abs(f_cur)) and diff <= 1e-7:
                w = w_next
                break
            f_prev = f_cur

        # O'Donoghue–Candès gradient-scheme adaptive restart.
        if float(np.dot(z - w_next, w_next - w)) > 0.0:
            z = w_next.copy()
            t = 1.0
        else:
            t_next = 0.5 * (1.0 + float(np.sqrt(1.0 + 4.0 * t * t)))
            beta = (t - 1.0) / t_next
            z = w_next + beta * (w_next - w)
            t = t_next
        w = w_next

    w = np.maximum(w, 0.0)
    s = float(w.sum())
    return w / s if s > 0.0 else np.full(J, 1.0 / J, dtype=float)


def solve_simplex_weights(
    X: np.ndarray,
    y: np.ndarray,
    *,
    allow_negative: bool = False,
    l2_penalty: float = 0.0,
    l1_penalty: float = 0.0,
    sum_to_one: bool = True,
) -> np.ndarray:
    """Solve regularised least squares with optional simplex / sum-to-one constraints.

    Minimises ``||y - X w||² + l2_penalty * ||w||² + l1_penalty * ||w||_1``
    subject to optional ``sum(w) == 1`` and (when ``allow_negative=False``)
    ``w >= 0``.

    With ``allow_negative=False, l2_penalty=0.0, l1_penalty=0.0, sum_to_one=True``
    this reproduces the classic Abadie simplex QP.
    """
    lam2 = float(l2_penalty)
    lam1 = float(l1_penalty)
    if lam2 < 0.0 or lam1 < 0.0:
        raise ValueError("l2_penalty and l1_penalty must be non-negative.")

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    J = X_arr.shape[1]
    if J == 0:
        raise ValueError("Design matrix X has zero columns.")

    # Pure closed-form OLS / Ridge when completely unconstrained.
    if allow_negative and not sum_to_one and lam1 == 0.0:
        if lam2 > 0.0:
            gram = X_arr.T @ X_arr + lam2 * np.eye(J, dtype=float)
            rhs = X_arr.T @ y_arr
            try:
                return np.linalg.solve(gram, rhs).astype(float)
            except np.linalg.LinAlgError:
                coef, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
                return coef.astype(float)
        coef, *_ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
        return coef.astype(float)

    from scipy.optimize import Bounds, LinearConstraint, minimize

    w0 = np.full(J, 1.0 / J, dtype=float)

    # Split-variable formulation for signed L1 + L2 Elastic Net.
    if allow_negative and lam1 > 0.0:
        if not sum_to_one:
            init_w = solve_simplex_weights(
                X_arr, y_arr, allow_negative=True, l2_penalty=lam2, l1_penalty=0.0, sum_to_one=False
            )
        else:
            init_w = w0
        u0 = np.r_[np.maximum(init_w, 0.0), np.maximum(-init_w, 0.0)]

        def obj_en(u: np.ndarray) -> float:
            w = u[:J] - u[J:]
            r = y_arr - X_arr @ w
            return float(np.dot(r, r) + lam2 * np.dot(w, w) + lam1 * np.sum(u))

        def grad_en(u: np.ndarray) -> np.ndarray:
            w = u[:J] - u[J:]
            gw = 2.0 * (X_arr.T @ (X_arr @ w - y_arr) + lam2 * w)
            return np.r_[gw + lam1, -gw + lam1].astype(float)

        bounds_en = [(0.0, None)] * (2 * J)
        if sum_to_one:
            aeq = np.r_[np.ones(J, dtype=float), -np.ones(J, dtype=float)]
            cons_en = [
                {
                    "type": "eq",
                    "fun": lambda u: float(np.sum(u[:J] - u[J:]) - 1.0),
                    "jac": lambda _u: aeq,
                }
            ]
            res = minimize(
                obj_en,
                u0,
                jac=grad_en,
                method="SLSQP",
                bounds=bounds_en,
                constraints=cons_en,
                options={"ftol": 1e-10, "maxiter": 1000},
            )
        else:
            res = minimize(
                obj_en,
                u0,
                jac=grad_en,
                method="L-BFGS-B",
                bounds=bounds_en,
                options={"ftol": 1e-10, "maxiter": 1000},
            )
        w = (res.x[:J] - res.x[J:]).astype(float) if res.success else init_w
        if sum_to_one:
            w = w - (float(w.sum()) - 1.0) / float(J)
        return w

    def obj(w: np.ndarray) -> float:
        r = y_arr - X_arr @ w
        val = float(np.dot(r, r))
        if lam2 > 0.0:
            val += lam2 * float(np.dot(w, w))
        if lam1 > 0.0 and not allow_negative and not sum_to_one:
            val += lam1 * float(np.sum(w))
        return val

    def grad(w: np.ndarray) -> np.ndarray:
        g = 2.0 * X_arr.T @ (X_arr @ w - y_arr)
        if lam2 > 0.0:
            g = g + 2.0 * lam2 * w
        if lam1 > 0.0 and not allow_negative and not sum_to_one:
            g = g + lam1
        return g.astype(float)

    if allow_negative:
        bounds = Bounds([-np.inf] * J, [np.inf] * J)
    elif sum_to_one:
        bounds = Bounds([0.0] * J, [1.0] * J)
    else:
        bounds = Bounds([0.0] * J, [np.inf] * J)

    if sum_to_one:
        Aeq = np.ones((1, J), dtype=float)
        beq = np.array([1.0], dtype=float)
        cons: list[Any] = [LinearConstraint(Aeq, beq, beq)]
        res = minimize(obj, w0, jac=grad, method="trust-constr", bounds=bounds, constraints=cons)
    else:
        # allow_negative=False, sum_to_one=False (NNLS / Elastic-Net NNLS)
        res = minimize(
            obj,
            w0,
            jac=grad,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * J,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

    w = res.x if res.success else w0
    if not allow_negative:
        w = np.maximum(w, 0.0)
        if sum_to_one:
            s = float(w.sum())
            w = w / s if s > 0 else w0
    elif sum_to_one and abs(float(w.sum()) - 1.0) > 1e-6:
        w = w - (float(w.sum()) - 1.0) / float(J)
    return w.astype(float)


def _default_hac_lags(n: int) -> int:
    """Newey–West (1994) automatic plug-in lag truncation ``floor(4*(n/100)^(2/9))``."""
    if n <= 2:
        return 0
    return min(max(0, int(n) - 2), int(np.floor(4.0 * (float(n) / 100.0) ** (2.0 / 9.0))))


def newey_west_se_mean(x: np.ndarray, *, maxlags: int | None = None) -> tuple[float, int]:
    """Zero-LAPACK Newey–West (1987) HAC standard error of the sample mean.

    Uses the Bartlett kernel with ``use_correction=True`` (``n / (n - 1)``),
    matching ``statsmodels`` OLS HAC on an intercept-only regression without
    calling LAPACK ``linalg.inv``.
    """
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return float("nan"), 0
    lags = _default_hac_lags(n) if maxlags is None else max(0, min(int(maxlags), n - 2))
    u = arr - float(np.mean(arr))
    meat = float(np.dot(u, u))
    for lag in range(1, lags + 1):
        w = 1.0 - float(lag) / float(lags + 1)
        meat += 2.0 * w * float(np.dot(u[lag:], u[:-lag]))
    # Sample variance of mean with finite-sample df correction n / (n - 1).
    var_mean = (float(n) / float(n - 1)) * max(meat, 0.0) / float(n * n)
    return float(np.sqrt(var_mean)), int(lags)


def newey_west_se_slope(e: np.ndarray, *, maxlags: int | None = None) -> tuple[float, float, int]:
    """Zero-LAPACK OLS slope and Newey–West (1987) HAC standard error for ``e_t ~ 1 + t``.

    Returns ``(slope, se_slope, lags_used)``. Because centered regressors
    ``[1, t - mean(t)]`` are mutually orthogonal, the slope variance depends
    exclusively on the centered score ``g_t = (t - mean(t)) * resid_t`` and
    requires no matrix inversion (safe inside macOS Qt worker threads).
    """
    arr = np.asarray(e, dtype=float).ravel()
    n = arr.size
    if n < 3:
        return float("nan"), float("nan"), 0
    t = np.arange(n, dtype=float)
    t_mean = 0.5 * float(n - 1)
    t_c = t - t_mean
    s_tt = float(np.dot(t_c, t_c))
    if s_tt <= 0.0:
        return float("nan"), float("nan"), 0
    e_mean = float(np.mean(arr))
    slope = float(np.dot(t_c, arr - e_mean) / s_tt)
    intercept = e_mean - slope * t_mean
    resid = arr - (intercept + slope * t)

    lags = _default_hac_lags(n) if maxlags is None else max(0, min(int(maxlags), n - 2))
    score = t_c * resid
    meat_11 = float(np.dot(score, score))
    for lag in range(1, lags + 1):
        w = 1.0 - float(lag) / float(lags + 1)
        meat_11 += 2.0 * w * float(np.dot(score[lag:], score[:-lag]))
    var_slope = (float(n) / float(n - 2)) * max(meat_11, 0.0) / (s_tt * s_tt)
    return slope, float(np.sqrt(var_slope)), int(lags)
