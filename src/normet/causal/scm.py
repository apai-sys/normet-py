# src/normet/causal/scm.py
"""Synthetic Control Method via ridge-augmented donor weights (the ``scm`` backend)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from ..utils.logging import get_logger
from ._common import pivot_panel, solve_simplex_weights

log = get_logger(__name__)

__all__ = ["scm"]


def _ridge_augment(
    Xd: np.ndarray,
    Xt: np.ndarray,
    Y: np.ndarray,
    alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched augmented-ridge predictions across all timestamps.

    Equivalent to fitting ``RidgeCV(alphas, fit_intercept=True)`` separately at
    every timestamp — regressing donor outcomes at time *t* on the fixed donor
    pre-period design ``Xd`` — but computed from a single SVD of ``Xd`` with
    exact leave-one-out alpha selection per timestamp, so the SVD is reused
    across all ``T`` targets instead of refit ``T`` times.

    Parameters
    ----------
    Xd : ndarray, shape (J, p)
        Donor design (pre-period profiles, optionally covariate-augmented).
    Xt : ndarray, shape (1, p)
        Treated design row.
    Y : ndarray, shape (T, J)
        Donor outcomes per timestamp; each row is one regression target. Must be
        finite — callers fall back to the per-timestamp loop otherwise.
    alphas : ndarray, shape (m,)
        Ridge penalty grid.

    Returns
    -------
    (alpha_per_t, m_treated, m_donors)
        ``alpha_per_t`` (T,), ``m_treated`` (T,), ``m_donors`` (T, J).
    """
    B = Y.T  # (J, T): samples = donors, targets = timestamps
    # Centre design and targets (mirrors RidgeCV(fit_intercept=True)).
    Xmean = Xd.mean(axis=0)
    Xc = Xd - Xmean
    ymean = B.mean(axis=0)  # (T,)
    Bc = B - ymean

    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)  # U (J,k), s (k,), Vt (k,p)
    n = U.shape[0]  # number of samples (donors)
    s2 = s**2
    UtB = U.T @ Bc  # (k, T)
    U2 = U**2  # (J, k)

    # --- Exact leave-one-out error for each alpha; pick the best per timestamp ---
    # The hat-matrix diagonal includes a 1/n term from the fitted intercept
    # (RidgeCV(fit_intercept=True) centres the data); omitting it biases alpha
    # selection toward the smallest value.
    eps = np.finfo(float).eps
    loo = np.empty((len(alphas), Y.shape[0]))  # (m, T)
    for ai, a in enumerate(alphas):
        d = s2 / (s2 + a)  # (k,)
        resid = Bc - U @ (d[:, None] * UtB)  # (J, T)
        denom = np.maximum(1.0 - 1.0 / n - (U2 @ d), eps)[:, None]  # (J, 1)
        loo[ai] = np.mean((resid / denom) ** 2, axis=0)
    alpha_sel = np.asarray(alphas, dtype=float)[np.argmin(loo, axis=0)]  # (T,)

    # --- Predictions at the selected per-timestamp alpha ---
    D = s2[:, None] / (s2[:, None] + alpha_sel[None, :])  # (k, T)
    m_donors = (U @ (D * UtB) + ymean).T  # (T, J) in-sample donor fits

    g = (Xt.ravel() - Xmean) @ Vt.T  # (k,)
    SR = s[:, None] / (s2[:, None] + alpha_sel[None, :])  # (k, T)
    m_treated = ymean + np.einsum("k,kt,kt->t", g, SR, UtB)  # (T,)

    return alpha_sel, m_treated, m_donors


def scm(
    df: pd.DataFrame,
    date_col: str = "date",
    unit_col: str = "code",
    outcome_col: str = "poll",
    treated_unit: str | None = None,
    cutoff_date: str | None = None,
    donors: list[str] | None = None,
    pre_covariates: list[str] | None = None,
    alphas: list[float] | None = None,
    allow_negative_weights: bool = False,
) -> dict[str, Any]:
    """
    Augmented Synthetic Control Method (SCM) for a single treated unit.

    Fits a ridge-augmented outcome model at each time point using pre-treatment
    information, then balances donor residuals to construct a synthetic counterfactual
    for the treated unit. Returns the treated series, synthetic series, and effect
    (observed − synthetic), plus donor weights and per-time ridge alphas.

    Parameters
    ----------
    df : pandas.DataFrame
        Long panel with columns at least `[date_col, unit_col, outcome_col]`.
    date_col : str, optional
        Name of the time index column. Default "date".
    unit_col : str, optional
        Name of the unit identifier column. Default "code".
    outcome_col : str, optional
        Name of the outcome variable column. Default "poll".
    treated_unit : str, optional
        Identifier of the treated unit. Required.
    cutoff_date : str, optional
        Treatment start date in "YYYY-MM-DD" format. Required.
    donors : List[str] | None, optional
        Donor pool. If None, uses all units except `treated_unit` that appear in `df`.
    pre_covariates : List[str] | None, optional
        Additional unit-level covariates to augment ridge features using *pre-period means*.
        If provided, rows with missing pre-period means are dropped for affected units.
    alphas : List[float] | None, optional
        RidgeCV alpha grid. Default is `[0.1, 0.2, ..., 10.0]`.
    allow_negative_weights : bool, optional
        If True, donor weights may be negative (sum-to-one constraint retained).
        If False (default), simplex constraints are enforced: w_j ≥ 0, ∑ w_j = 1.

    Returns
    -------
    dict
        A dictionary with the structure::

            {
              "synthetic": pandas.DataFrame,
                  # index: date
                  # columns: ["observed", "synthetic", "effect"]
              "weights": pandas.Series,
                  # donor weights indexed by donor unit (sum=1 if negatives not allowed)
              "alpha": dict,
                  # mapping timestamp -> chosen RidgeCV alpha (diagnostics)
            }

    Raises
    ------
    ValueError
        If required inputs are missing, the treated unit is not in the panel,
        no valid donors remain after filtering, or there are insufficient aligned
        pre-period rows to estimate weights.

    Notes
    -----
    - The ridge features are constructed from *pre-treatment* outcomes (and optional
      pre-period covariate means), held fixed across time when fitting/predicting.
    - Residual balancing solves a quadratic program with an equality constraint
      (weights sum to 1) and optional non-negativity bounds.
    - Very short pre-periods or heavy missingness can lead to unstable estimates.
    """
    t0 = time.time()

    if treated_unit is None or cutoff_date is None:
        raise ValueError("Both `treated_unit` and `cutoff_date` must be provided.")

    # --- Parse/validate dates without renaming the column ---
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"`date_col` '{date_col}' not found in df.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        n_bad = int(df[date_col].isna().sum())
        raise ValueError(f"{n_bad} rows have invalid {date_col} values.")
    cutoff_ts = pd.to_datetime(cutoff_date)

    # --- Pivot to wide panel: rows=time, cols=units (dates already parsed) ---
    panel, donors = pivot_panel(
        df,
        date_col=date_col,
        unit_col=unit_col,
        outcome_col=outcome_col,
        treated_unit=treated_unit,
        donors=donors,
        parse_dates=False,
    )

    # --- Pre/post masks & slices ---
    pre_idx = panel.index < cutoff_ts
    dates_pre = panel.index[pre_idx]
    if dates_pre.size < 3:
        log.warning(
            "Very short pre-period (%d timestamps); results may be unstable.", dates_pre.size
        )

    # --- Build ridge feature matrices using *pre* outcomes (+ covariates means if provided) ---
    Y_pre = panel.loc[dates_pre, donors + [treated_unit]]
    Y_pre = Y_pre.dropna(how="any")  # ensure aligned outcome vectors for all donors+treated in pre
    if Y_pre.empty or Y_pre.shape[0] < 3:
        raise ValueError("Not enough complete pre-treatment rows after dropping NaNs.")

    X_donors: np.ndarray = Y_pre[donors].T.to_numpy()  # shape: J x T_pre
    X_treated: np.ndarray = Y_pre[treated_unit].to_numpy().reshape(1, -1)  # shape: 1 x T_pre

    # Optional: augment with pre-period covariate means
    if pre_covariates:
        cov_df = (
            df.loc[df[date_col] < cutoff_ts, [unit_col] + pre_covariates]
            .groupby(unit_col, dropna=False)[pre_covariates]
            .mean()
        )
        cov_df = cov_df.reindex(donors + [treated_unit])
        if cov_df.isna().any().any():
            log.warning("Missing covariate means for some units; rows with NaN will be dropped.")
            cov_df = cov_df.dropna(how="any")
            # reindex donors list accordingly if needed
            valid_units = cov_df.index.tolist()
            donors = [u for u in donors if u in valid_units]
            if treated_unit not in valid_units or not donors:
                raise ValueError("Covariates removal left no valid donor/treated units.")
            # rebuild Y_pre / X matrices to match filtered units
            Y_pre = panel.loc[dates_pre, donors + [treated_unit]].dropna(how="any")
            X_donors = Y_pre[donors].T.to_numpy()
            X_treated = Y_pre[treated_unit].to_numpy().reshape(1, -1)

        X_donors = np.hstack([X_donors, cov_df.loc[donors].values])
        X_treated = np.hstack([X_treated, cov_df.loc[[treated_unit]].values])

    # --- Ridge helper over donors at each time t (using X from pre) ---
    if alphas is None:
        alphas = [i / 10 for i in range(1, 101)]  # 0.1..10

    n_donors = len(donors)

    def fit_ridge(y_donors: np.ndarray, Xd: np.ndarray, Xt: np.ndarray):
        """Fit RidgeCV (donor outcomes y at time t) ~ Xd; predict for treated Xt and donors Xd."""
        mask = np.isfinite(y_donors)
        if mask.sum() < 3:
            return np.nan, np.nan, np.full(n_donors, np.nan, dtype=float)
        mdl = RidgeCV(alphas=alphas, fit_intercept=True)
        mdl.fit(Xd[mask], y_donors[mask])
        return mdl.alpha_, float(mdl.predict(Xt)[0]), mdl.predict(Xd)

    # --- Augmented predictions per time t ---
    # X_donors / X_treated are fixed across time; only the target y_t varies.
    # When donor outcomes are fully observed we solve all timestamps from a
    # single SVD of X_donors (exact LOO alpha selection), which is equivalent to
    # — but far cheaper than — refitting RidgeCV at every timestamp. Any
    # missingness falls back to the exact per-timestamp loop.
    log.info("Fitting ridge augmentation across %d timestamps …", len(panel.index))
    Y = panel[donors].to_numpy()  # (n_times, n_donors)
    alphas_grid = np.asarray(alphas, dtype=float)

    if np.isfinite(Y).all() and n_donors >= 3:
        alphas_arr, m_treated_arr, m_donors_arr = _ridge_augment(
            X_donors, X_treated, Y, alphas_grid
        )
    else:
        fitted = [fit_ridge(Y[i], X_donors, X_treated) for i in range(len(panel.index))]
        alphas_arr = np.array([f[0] for f in fitted], dtype=float)
        m_treated_arr = np.array([f[1] for f in fitted], dtype=float)
        m_donors_arr = np.vstack([f[2] for f in fitted])  # (n_times, n_donors)

    alpha_map: dict[pd.Timestamp, float] = dict(zip(panel.index, alphas_arr, strict=False))
    m_treated = pd.Series(m_treated_arr, index=panel.index, dtype=float)
    m_donors = pd.DataFrame(m_donors_arr, index=panel.index, columns=donors, dtype=float)

    # --- Residuals (observed - augmented prediction) ---
    R_don = panel[donors] - m_donors
    r_treat = panel[treated_unit] - m_treated

    # Align pre-period residual matrices (avoid separate dropna that misaligns)
    pre_common = R_don.loc[dates_pre].dropna(how="any")
    r_pre = r_treat.loc[pre_common.index].to_numpy()
    R_pre = pre_common.to_numpy()

    if R_pre.shape[0] < 3:
        raise ValueError("Insufficient aligned pre-period residual rows to estimate weights.")

    # --- Solve donor weights: min || r_pre - R_pre w ||^2 s.t. sum(w)=1 (+ simplex) ---
    w = solve_simplex_weights(R_pre, r_pre, allow_negative=allow_negative_weights)
    weights = pd.Series(w, index=donors, name="weight")

    # --- Synthetic path over full horizon ---
    synth = m_treated + (R_don @ weights)
    out = pd.DataFrame({"observed": panel[treated_unit], "synthetic": synth})
    out["effect"] = out["observed"] - out["synthetic"]

    log.info(
        "SCM finished in %.2fs | pre T=%d | donors=%d",
        time.time() - t0,
        len(dates_pre),
        len(donors),
    )
    return {"synthetic": out, "weights": weights, "alpha": alpha_map}
