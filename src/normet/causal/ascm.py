# src/normet/causal/ascm.py
"""Ben-Michael, Feller & Rothstein (2021, JASA) Ridge Augmented Synthetic Control."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ._common import pivot_panel, solve_simplex_weights_fast

log = get_logger(__name__)

__all__ = ["scm_ascm"]


def _aug_weights_from_svd(
    X0: np.ndarray,
    x1: np.ndarray,
    w_scm: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Closed-form Ridge ASCM weights ``w_scm + delta(lam)`` via single SVD of centered ``X0``."""
    if np.isinf(lam):
        return w_scm.copy()
    x_mean = X0.mean(axis=0)
    Xc = X0 - x_mean
    x1c = x1 - x_mean
    imbalance = x1c - Xc.T @ w_scm
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    if s.size == 0:
        return w_scm.copy()
    shrink = np.where((s**2 + float(lam)) > 0.0, s / (s**2 + float(lam)), 0.0)
    delta = U @ (shrink * (Vt @ imbalance))
    return (w_scm + delta).astype(float)


def scm_ascm(
    df: pd.DataFrame,
    date_col: str = "date",
    unit_col: str = "code",
    outcome_col: str = "poll",
    treated_unit: str | None = None,
    cutoff_date: str | None = None,
    donors: list[str] | None = None,
    *,
    pre_covariates: list[str] | None = None,
    demean: bool = False,
    ridge_lambda: float | None = None,
    lambda_grid: Sequence[float] | None = None,
    cv_folds: int = 5,
    one_se: bool = True,
    scm_l2_penalty: float = 0.0,
) -> dict[str, Any]:
    """Ben-Michael, Feller & Rothstein (2021) Ridge Augmented Synthetic Control Method.

    Fits simplex base SCM weights ``w_scm`` on the pre-treatment period, then
    learns a single unique vector of ridge-augmented donor weights
    ``w_aug = w_scm + delta(lambda)`` with ``sum(w_aug) == 1`` where ``lambda``
    is chosen by contiguous **time-blocked cross-validation** strictly holding out
    pre-period blocks (zero self-regression leakage, unlike the legacy
    row-wise regression in :func:`normet.causal.scm.scm`).

    Parameters
    ----------
    df : pandas.DataFrame
        Long panel with at least ``[date_col, unit_col, outcome_col]``.
    date_col, unit_col, outcome_col : str
        Column names in ``df``.
    treated_unit : str
        Treated unit identifier. Required.
    cutoff_date : str
        Treatment cutoff date (``"YYYY-MM-DD"``). Required.
    donors : list of str, optional
        Donor pool. Defaults to all units other than ``treated_unit``.
    pre_covariates : list of str, optional
        Unit-level covariates averaged over the pre-period and standardised
        before augmenting the pre-treatment balance profile.
    demean : bool, default False
        Subtract each unit's pre-period mean before fitting and restore the
        level difference ``intercept = mean(y_pre) - w_aug @ mean(Y_0_pre)``
        (intercept-augmented ASCM / Doudchenko–Imbens style). Recommended when
        the treated site lies outside the donor convex hull (e.g., roadside vs.
        background stations).
    ridge_lambda : float, optional
        Explicit ridge penalty ``lambda >= 0`` (supports ``np.inf`` to recover
        pure base SCM). If ``None`` (default), selected via contiguous
        time-blocked cross-validation over ``lambda_grid``.
    lambda_grid : sequence of float, optional
        Candidate ridge penalties for CV (defaults to 24 log-spaced values from
        ``1e-3 * s_max²`` to ``1e3 * s_max²`` plus ``np.inf``).
    cv_folds : int, default 5
        Number of contiguous pre-period time blocks for leak-free CV.
    one_se : bool, default True
        Use the one-standard-error rule picking the largest ``lambda`` within
        1 SE of the minimum CV MSE (Ben-Michael et al. 2021, Section 4.2).
    scm_l2_penalty : float, default 0.0
        Optional quadratic dispersion penalty on the base simplex weights
        ``w_scm``.

    Returns
    -------
    dict
        ``{"synthetic": DataFrame[observed, synthetic, effect],
        "weights": Series(w_aug), "base_weights": Series(w_scm),
        "ridge_lambda": float, "intercept": float,
        "l2_imbalance_base": float, "l2_imbalance_aug": float,
        "cv_mse": Series}``
    """
    if treated_unit is None or cutoff_date is None:
        raise ValueError("Both `treated_unit` and `cutoff_date` must be provided.")
    if float(scm_l2_penalty) < 0.0:
        raise ValueError(f"scm_l2_penalty must be non-negative, got {scm_l2_penalty}")
    if ridge_lambda is not None and float(ridge_lambda) < 0.0:
        raise ValueError(f"ridge_lambda must be non-negative, got {ridge_lambda}")

    d = df.copy()
    if date_col not in d.columns:
        raise ValueError(f"`date_col` '{date_col}' not found in df.")
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    if d[date_col].isna().any():
        n_bad = int(d[date_col].isna().sum())
        raise ValueError(f"{n_bad} rows have invalid {date_col} values.")
    cutoff_ts = pd.to_datetime(cutoff_date)

    panel, donors = pivot_panel(
        d,
        date_col=date_col,
        unit_col=unit_col,
        outcome_col=outcome_col,
        treated_unit=treated_unit,
        donors=donors,
        parse_dates=False,
    )

    pre_idx = panel.index < cutoff_ts
    dates_pre = panel[pre_idx].index
    Y_pre = panel.loc[dates_pre, donors + [treated_unit]].dropna(how="any")
    T_pre = int(Y_pre.shape[0])
    if T_pre < 3:
        raise ValueError(f"Need at least 3 complete pre-treatment rows for scm_ascm, got {T_pre}.")

    cov_donors: np.ndarray | None = None
    cov_treated: np.ndarray | None = None
    if pre_covariates:
        cov_df = (
            d.loc[d[date_col] < cutoff_ts, [unit_col] + list(pre_covariates)]
            .groupby(unit_col, dropna=False)[list(pre_covariates)]
            .mean()
        )
        cov_df = cov_df.reindex(donors + [treated_unit])
        if cov_df.isna().any().any():
            log.warning("Missing covariate means for some units; rows with NaN will be dropped.")
            cov_df = cov_df.dropna(how="any")
            valid_units = cov_df.index.tolist()
            donors = [u for u in donors if u in valid_units]
            if treated_unit not in valid_units or not donors:
                raise ValueError("Covariates removal left no valid donor/treated units.")
            Y_pre = panel.loc[dates_pre, donors + [treated_unit]].dropna(how="any")
            T_pre = int(Y_pre.shape[0])
            if T_pre < 3:
                raise ValueError(
                    "Not enough complete pre-treatment rows after covariate filtering."
                )

        cov_raw = cov_df.loc[donors + [treated_unit]].to_numpy(dtype=float)
        y_scale = max(float(np.std(Y_pre.to_numpy(dtype=float))), 1e-8)
        cov_mu = cov_raw.mean(axis=0, keepdims=True)
        cov_sd = np.maximum(cov_raw.std(axis=0, keepdims=True), 1e-8)
        cov_scaled = ((cov_raw - cov_mu) / cov_sd) * y_scale
        cov_donors = cov_scaled[:-1]  # (J, n_cov)
        cov_treated = cov_scaled[-1]  # (n_cov,)

    Y0_pre = Y_pre[donors].to_numpy(dtype=float)  # (T_pre, J)
    y1_pre = Y_pre[treated_unit].to_numpy(dtype=float)  # (T_pre,)

    def _build_design(
        Y0_blk: np.ndarray, y1_blk: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        if demean:
            y1_mu = float(np.mean(y1_blk))
            Y0_mu = np.mean(Y0_blk, axis=0)
            Y0_c = Y0_blk - Y0_mu[None, :]
            y1_c = y1_blk - y1_mu
        else:
            y1_mu = 0.0
            Y0_mu = np.zeros(Y0_blk.shape[1], dtype=float)
            Y0_c = Y0_blk
            y1_c = y1_blk

        if cov_donors is not None and cov_treated is not None:
            scm_X = np.vstack([Y0_c, cov_donors.T])
            scm_y = np.concatenate([y1_c, cov_treated])
            X0 = np.hstack([Y0_c.T, cov_donors])
            x1 = np.concatenate([y1_c, cov_treated])
        else:
            scm_X = Y0_c
            scm_y = y1_c
            X0 = Y0_c.T
            x1 = y1_c
        return scm_X, scm_y, X0, x1, y1_mu, Y0_mu

    scm_X_full, scm_y_full, X0_full, x1_full, y1_mu_full, Y0_mu_full = _build_design(Y0_pre, y1_pre)
    w_scm = solve_simplex_weights_fast(scm_X_full, scm_y_full, l2_penalty=float(scm_l2_penalty))

    # Build / validate candidate lambda grid and run leak-free blocked time CV if needed.
    if ridge_lambda is not None:
        lam_sel = float(ridge_lambda)
        cv_series = pd.Series(dtype=float, name="cv_mse")
    else:
        if lambda_grid is None:
            X0c = X0_full - X0_full.mean(axis=0)
            s_vals = np.linalg.svd(X0c, compute_uv=False)
            s2_max = float(s_vals[0] ** 2) if s_vals.size else 0.0
            if s2_max > 0.0:
                grid = np.r_[np.geomspace(1e-3 * s2_max, 1e3 * s2_max, 24), np.inf]
            else:
                grid = np.array([np.inf], dtype=float)
        else:
            grid = np.asarray(list(lambda_grid), dtype=float)
            if grid.size == 0:
                raise ValueError("lambda_grid must not be empty.")
            if np.any(grid < 0.0):
                raise ValueError("All entries in lambda_grid must be >= 0.")

        K = min(max(int(cv_folds), 2), T_pre)
        blocks = [b for b in np.array_split(np.arange(T_pre), K) if T_pre - len(b) >= 2]
        if len(blocks) < 2:
            lam_sel = float(grid[-1])
            cv_series = pd.Series(np.full(len(grid), np.nan), index=grid, name="cv_mse")
        else:
            fold_mses = np.empty((len(blocks), len(grid)), dtype=float)
            all_idx = np.arange(T_pre)
            for fi, val_idx in enumerate(blocks):
                tr_mask = np.ones(T_pre, dtype=bool)
                tr_mask[val_idx] = False
                tr_idx = all_idx[tr_mask]

                sX_k, sy_k, X0_k, x1_k, y1_mu_k, Y0_mu_k = _build_design(
                    Y0_pre[tr_idx], y1_pre[tr_idx]
                )
                w_base_k = solve_simplex_weights_fast(sX_k, sy_k, l2_penalty=float(scm_l2_penalty))
                X0_k_mean = X0_k.mean(axis=0)
                Xc_k = X0_k - X0_k_mean
                imb_k = (x1_k - X0_k_mean) - Xc_k.T @ w_base_k
                U_k, s_k, Vt_k = np.linalg.svd(Xc_k, full_matrices=False)
                Vt_imb_k = Vt_k @ imb_k if s_k.size else np.empty(0, dtype=float)

                Y0_val = Y0_pre[val_idx]
                y1_val = y1_pre[val_idx]
                for li, lam_cand in enumerate(grid):
                    if np.isinf(lam_cand) or s_k.size == 0:
                        w_cand = w_base_k
                    else:
                        sh = np.where((s_k**2 + lam_cand) > 0.0, s_k / (s_k**2 + lam_cand), 0.0)
                        w_cand = w_base_k + U_k @ (sh * Vt_imb_k)
                    int_k = float(y1_mu_k - np.dot(Y0_mu_k, w_cand)) if demean else 0.0
                    pred_val = int_k + Y0_val @ w_cand
                    fold_mses[fi, li] = float(np.mean((y1_val - pred_val) ** 2))

            cv_mean = fold_mses.mean(axis=0)
            cv_se = (
                fold_mses.std(axis=0, ddof=1) / np.sqrt(len(blocks))
                if len(blocks) >= 2
                else np.zeros_like(cv_mean)
            )
            best_idx = int(np.argmin(cv_mean))
            if one_se and len(blocks) >= 2:
                thresh = float(cv_mean[best_idx] + cv_se[best_idx])
                eligible = np.flatnonzero(cv_mean <= thresh)
                # Prefer the highest regularisation within 1 SE of the minimum.
                chosen_idx = int(eligible[np.argmax(grid[eligible])]) if eligible.size else best_idx
            else:
                chosen_idx = best_idx
            lam_sel = float(grid[chosen_idx])
            cv_series = pd.Series(cv_mean, index=grid, name="cv_mse")

    w_aug = _aug_weights_from_svd(X0_full, x1_full, w_scm, lam_sel)
    intercept = float(y1_mu_full - np.dot(Y0_mu_full, w_aug)) if demean else 0.0

    imb_base = float(np.linalg.norm(x1_full - X0_full.T @ w_scm))
    imb_aug = float(np.linalg.norm(x1_full - X0_full.T @ w_aug))

    syn_vals = intercept + panel[donors].to_numpy(dtype=float) @ w_aug
    out = pd.DataFrame(
        {"observed": panel[treated_unit].to_numpy(dtype=float), "synthetic": syn_vals},
        index=panel.index,
    )
    out["effect"] = out["observed"] - out["synthetic"]

    log.info(
        "ASCM finished | pre T=%d | donors=%d | lambda=%.4g | imb %.3f->%.3f",
        T_pre,
        len(donors),
        lam_sel,
        imb_base,
        imb_aug,
    )
    return {
        "synthetic": out,
        "weights": pd.Series(w_aug, index=donors, name="weight"),
        "base_weights": pd.Series(w_scm, index=donors, name="weight"),
        "ridge_lambda": lam_sel,
        "intercept": intercept,
        "l2_imbalance_base": imb_base,
        "l2_imbalance_aug": imb_aug,
        "cv_mse": cv_series,
    }
