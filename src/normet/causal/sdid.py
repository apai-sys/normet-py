# src/normet/causal/sdid.py
"""Synthetic Difference-in-Differences (Arkhangelsky et al. 2021, AER)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ._common import pivot_panel, solve_simplex_weights_fast

log = get_logger(__name__)

__all__ = ["sdid"]


def _sdid_core(
    Y_co_pre: np.ndarray,
    y_tr_pre: np.ndarray,
    Y_co_post: np.ndarray,
    y_tr_post: np.ndarray,
    *,
    zeta_omega: float | None = None,
    zeta_lambda: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Compute SDID unit weights ``omega``, time weights ``lam``, intercept, and ATT."""
    T_pre, J = Y_co_pre.shape
    T_post = max(1, int(Y_co_post.shape[0]))
    if J == 0 or T_pre < 2:
        raise ValueError("SDID requires at least 1 donor and 2 pre-period rows.")

    diffs = np.diff(Y_co_pre, axis=0)
    sigma = float(np.std(diffs, ddof=1)) if diffs.size >= 2 else float(np.std(Y_co_pre))
    sigma = max(sigma, 1e-8) if np.isfinite(sigma) else 1e-8

    z_omega = float(zeta_omega) if zeta_omega is not None else float((T_post**0.25) * sigma)
    z_lam = float(zeta_lambda) if zeta_lambda is not None else float(1e-6 * sigma)

    # 1. Unit weights: minimise || (y_tr_pre - Y_co_pre w) - w0 ||^2 + T_pre * zeta_omega^2 ||w||^2
    Y_co_c = Y_co_pre - Y_co_pre.mean(axis=0, keepdims=True)
    y_tr_c = y_tr_pre - float(np.mean(y_tr_pre))
    omega = solve_simplex_weights_fast(Y_co_c, y_tr_c, l2_penalty=float(T_pre) * (z_omega**2))

    # 2. Time weights: match pre-period donor profiles to post-period donor means.
    with np.errstate(invalid="ignore"):
        y_co_post_mean = np.nanmean(Y_co_post, axis=0)
    if not np.all(np.isfinite(y_co_post_mean)):
        col_means = Y_co_pre.mean(axis=0)
        y_co_post_mean = np.where(np.isfinite(y_co_post_mean), y_co_post_mean, col_means)

    X_t = Y_co_pre.T  # (J, T_pre)
    X_t_c = X_t - X_t.mean(axis=0, keepdims=True)
    y_post_c = y_co_post_mean - float(np.mean(y_co_post_mean))
    lam = solve_simplex_weights_fast(X_t_c, y_post_c, l2_penalty=float(J) * (z_lam**2))

    # 3. Time-weighted baseline offset mu ensures lam @ (y_tr_pre - Y_co_pre @ omega - mu) == 0.
    mu = float(np.dot(lam, y_tr_pre - Y_co_pre @ omega))
    post_diff = y_tr_post - (mu + Y_co_post @ omega)
    fin_post = post_diff[np.isfinite(post_diff)]
    att = float(np.mean(fin_post)) if fin_post.size else float("nan")
    return omega, lam, mu, att, z_omega, z_lam


def sdid(
    df: pd.DataFrame,
    date_col: str = "date",
    unit_col: str = "code",
    outcome_col: str = "poll",
    treated_unit: str | None = None,
    cutoff_date: str | None = None,
    donors: list[str] | None = None,
    *,
    zeta_omega: float | None = None,
    zeta_lambda: float | None = None,
    se_method: str = "none",
) -> dict[str, Any]:
    """Synthetic Difference-in-Differences (Arkhangelsky et al. 2021, AER).

    Jointly learns non-negative unit weights ``omega`` (simplex) and
    pre-treatment time weights ``lambda`` (simplex) to combine the strengths of
    Synthetic Control (local parallel trends) and Difference-in-Differences
    (invariant to constant level offsets and time-varying aggregate shocks).

    Parameters
    ----------
    df : pandas.DataFrame
        Long panel with at least ``[date_col, unit_col, outcome_col]``.
    date_col, unit_col, outcome_col : str
        Column names in ``df``.
    treated_unit : str
        Identifier of the treated unit. Required.
    cutoff_date : str
        Treatment cutoff date (``"YYYY-MM-DD"``). Required.
    donors : list of str, optional
        Donor pool. Defaults to all units other than ``treated_unit``.
    zeta_omega : float, optional
        L2 regularisation scale for unit weights ``omega``. Defaults to
        ``(T_post ** 0.25) * sigma`` where ``sigma`` is the SD of pre-period
        donor first differences.
    zeta_lambda : float, optional
        L2 regularisation scale for time weights ``lambda``. Defaults to
        ``1e-6 * sigma``.
    se_method : {"none", "placebo"}, default "none"
        Optional placebo variance estimation (Arkhangelsky et al. 2021
        Algorithm 4) when ``J >= 3`` donors are present.

    Returns
    -------
    dict
        ``{"synthetic": DataFrame[observed, synthetic, effect],
        "weights": Series(omega), "time_weights": Series(lambda),
        "att": float, "intercept": float, "zeta_omega": float,
        "zeta_lambda": float}`` (plus ``"att_se"`` and ``"p_value"`` when
        ``se_method="placebo"``).
    """
    if treated_unit is None or cutoff_date is None:
        raise ValueError("Both `treated_unit` and `cutoff_date` must be provided.")
    se_mode = (se_method or "none").lower()
    if se_mode not in {"none", "placebo"}:
        raise ValueError(f"se_method must be 'none' or 'placebo', got {se_method!r}")
    if zeta_omega is not None and float(zeta_omega) < 0.0:
        raise ValueError(f"zeta_omega must be non-negative, got {zeta_omega}")
    if zeta_lambda is not None and float(zeta_lambda) < 0.0:
        raise ValueError(f"zeta_lambda must be non-negative, got {zeta_lambda}")

    panel, donors = pivot_panel(
        df,
        date_col=date_col,
        unit_col=unit_col,
        outcome_col=outcome_col,
        treated_unit=treated_unit,
        donors=donors,
    )
    cutoff_ts = pd.to_datetime(cutoff_date)
    pre_mask = panel.index < cutoff_ts
    post_mask = ~pre_mask

    pre_df = panel.loc[pre_mask, donors + [treated_unit]].dropna(how="any")
    if pre_df.shape[0] < 2:
        raise ValueError("Need at least 2 complete pre-treatment rows for SDID.")
    post_df = panel.loc[post_mask, donors + [treated_unit]]
    if post_df.empty:
        raise ValueError("No post-treatment observations for SDID.")

    Y_co_pre = pre_df[donors].to_numpy(dtype=float)
    y_tr_pre = pre_df[treated_unit].to_numpy(dtype=float)
    Y_co_post = post_df[donors].to_numpy(dtype=float)
    y_tr_post = post_df[treated_unit].to_numpy(dtype=float)

    omega, lam, mu, att, z_omega, z_lam = _sdid_core(
        Y_co_pre,
        y_tr_pre,
        Y_co_post,
        y_tr_post,
        zeta_omega=zeta_omega,
        zeta_lambda=zeta_lambda,
    )

    syn_full = mu + panel[donors].to_numpy(dtype=float) @ omega
    out = pd.DataFrame(
        {"observed": panel[treated_unit].to_numpy(dtype=float), "synthetic": syn_full},
        index=panel.index,
    )
    out["effect"] = out["observed"] - out["synthetic"]
    post_eff = out.loc[post_mask, "effect"].to_numpy(dtype=float)
    fin = post_eff[np.isfinite(post_eff)]
    if fin.size:
        att = float(np.mean(fin))

    result: dict[str, Any] = {
        "synthetic": out,
        "weights": pd.Series(omega, index=donors, name="weight"),
        "time_weights": pd.Series(lam, index=pre_df.index, name="time_weight"),
        "att": att,
        "intercept": mu,
        "zeta_omega": z_omega,
        "zeta_lambda": z_lam,
    }

    if se_mode == "placebo":
        J = len(donors)
        if J < 3:
            log.warning("SDID placebo SE requires at least 3 donors; returning NaN.")
            result["att_se"] = float("nan")
            result["p_value"] = float("nan")
        else:
            plc_atts = []
            for j in range(J):
                keep = [k for k in range(J) if k != j]
                _, _, _, att_j, _, _ = _sdid_core(
                    Y_co_pre[:, keep],
                    Y_co_pre[:, j],
                    Y_co_post[:, keep],
                    Y_co_post[:, j],
                    zeta_omega=zeta_omega,
                    zeta_lambda=zeta_lambda,
                )
                if np.isfinite(att_j):
                    plc_atts.append(att_j)
            arr_p = np.asarray(plc_atts, dtype=float)
            result["att_se"] = float(np.std(arr_p, ddof=1)) if arr_p.size >= 2 else float("nan")
            result["p_value"] = (
                float((np.sum(np.abs(arr_p) >= abs(att)) + 1.0) / (arr_p.size + 1.0))
                if arr_p.size and np.isfinite(att)
                else float("nan")
            )

    log.info(
        "SDID finished | T_pre=%d | T_post=%d | donors=%d | ATT=%.4f",
        Y_co_pre.shape[0],
        Y_co_post.shape[0],
        len(donors),
        att if np.isfinite(att) else float("nan"),
    )
    return result
