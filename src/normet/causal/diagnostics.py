# src/normet/causal/diagnostics.py
"""
Diagnostic summaries for synthetic-control fits.

These helpers operate on the outputs of :func:`normet.causal.scm.scm` and
:func:`normet.causal.mlscm.mlscm` (or any compatible result that exposes a
``synthetic`` DataFrame with columns ``observed`` / ``synthetic`` / ``effect``
and an optional ``weights`` Series indexed by donor).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["scm_diagnostics", "loo_weight_stability"]


def _pre_period_fit(synthetic_df: pd.DataFrame, cutoff_ts: pd.Timestamp) -> dict[str, float]:
    """Pre-period observed-vs-synthetic fit metrics."""
    pre = synthetic_df.loc[synthetic_df.index < cutoff_ts]
    if pre.empty:
        return {
            "pre_n": 0,
            "pre_rmse": np.nan,
            "pre_mae": np.nan,
            "pre_mape": np.nan,
            "pre_r2": np.nan,
        }
    obs = pre["observed"].to_numpy(dtype=float)
    syn = pre["synthetic"].to_numpy(dtype=float)
    mask = np.isfinite(obs) & np.isfinite(syn)
    obs, syn = obs[mask], syn[mask]
    if obs.size == 0:
        return {
            "pre_n": 0,
            "pre_rmse": np.nan,
            "pre_mae": np.nan,
            "pre_mape": np.nan,
            "pre_r2": np.nan,
        }
    err = obs - syn
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    ss_res = float(np.sum(err**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    denom = np.abs(obs)
    mape = float(np.mean(np.where(denom > 1e-12, np.abs(err) / denom, np.nan)))
    return {
        "pre_n": int(obs.size),
        "pre_rmse": float(np.sqrt(np.mean(err**2))),
        "pre_mae": float(np.mean(np.abs(err))),
        "pre_mape": mape,
        "pre_r2": r2,
    }


def _post_period_effect(synthetic_df: pd.DataFrame, cutoff_ts: pd.Timestamp) -> dict[str, float]:
    post = synthetic_df.loc[synthetic_df.index >= cutoff_ts]
    if post.empty:
        return {"post_n": 0, "att": np.nan, "att_cum": np.nan, "post_rmse": np.nan}
    eff = post["effect"].to_numpy(dtype=float)
    eff = eff[np.isfinite(eff)]
    if eff.size == 0:
        return {"post_n": 0, "att": np.nan, "att_cum": np.nan, "post_rmse": np.nan}
    return {
        "post_n": int(eff.size),
        "att": float(np.mean(eff)),
        "att_cum": float(np.sum(eff)),
        "post_rmse": float(np.sqrt(np.mean(eff**2))),
    }


def _weight_concentration(weights: pd.Series | None, top_k: int = 5) -> dict[str, Any]:
    """Herfindahl, effective N, and top-k donor share."""
    if weights is None or len(weights) == 0:
        return {
            "hhi": np.nan,
            "effective_n_donors": np.nan,
            "top_donors": [],
            "top_donor_share": np.nan,
            "n_donors": 0,
        }
    w = pd.Series(weights, dtype=float).fillna(0.0)
    # Normalise just in case (sum-to-one is enforced by the fitter, but be safe)
    s = w.sum()
    w_norm = w / s if abs(s) > 1e-12 else w
    hhi = float(np.sum(w_norm**2))
    eff_n = 1.0 / hhi if hhi > 0 else np.nan
    sorted_w = w_norm.abs().sort_values(ascending=False)
    top = sorted_w.head(top_k)
    return {
        "hhi": hhi,
        "effective_n_donors": eff_n,
        "top_donors": [(idx, float(val)) for idx, val in top.items()],
        "top_donor_share": float(top.sum()),
        "n_donors": int(len(w)),
    }


def scm_diagnostics(
    scm_result: dict[str, Any] | pd.DataFrame,
    cutoff_date: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Summarise the quality of a synthetic-control fit.

    Parameters
    ----------
    scm_result : dict
        Output of :func:`scm` (with keys ``synthetic`` and ``weights``) or any
        dict that contains a ``synthetic`` DataFrame with ``observed``,
        ``synthetic``, ``effect`` columns and (optionally) a ``weights`` Series.
        A bare DataFrame (e.g., from :func:`mlscm`) is also accepted.
    cutoff_date : str
        Treatment cutoff. Parseable by ``pd.to_datetime``.
    top_k : int, default 5
        How many top donors (by absolute weight) to include in the summary.

    Returns
    -------
    dict
        Keys:
          - ``pre_n``, ``pre_rmse``, ``pre_mae``, ``pre_mape``, ``pre_r2``
          - ``post_n``, ``att``, ``att_cum``, ``post_rmse``
          - ``hhi``, ``effective_n_donors``, ``n_donors``
          - ``top_donors`` (list of (donor, weight) tuples)
          - ``top_donor_share`` (sum of ``|w|`` over top_k)
    """
    cutoff_ts = pd.to_datetime(cutoff_date)

    if isinstance(scm_result, pd.DataFrame):
        synthetic_df = scm_result
        weights = None
    else:
        synthetic_df = scm_result.get("synthetic") if isinstance(scm_result, dict) else None  # type: ignore[assignment]
        weights = scm_result.get("weights") if isinstance(scm_result, dict) else None
    if synthetic_df is None or "effect" not in getattr(synthetic_df, "columns", []):
        raise ValueError(
            "`scm_result` must contain a DataFrame with 'observed'/'synthetic'/'effect'."
        )

    out: dict[str, Any] = {}
    out.update(_pre_period_fit(synthetic_df, cutoff_ts))
    out.update(_post_period_effect(synthetic_df, cutoff_ts))
    out.update(_weight_concentration(weights, top_k=top_k))
    return out


def loo_weight_stability(
    df: pd.DataFrame,
    *,
    date_col: str,
    unit_col: str,
    outcome_col: str,
    treated_unit: str,
    cutoff_date: str,
    donors: list[str],
    **scm_kwargs,
) -> pd.DataFrame:
    """
    Leave-one-donor-out weight stability for classic SCM.

    Refits SCM with each donor held out in turn and reports the average and
    max absolute drift of the remaining weights from the full-pool baseline.

    Parameters
    ----------
    df, date_col, unit_col, outcome_col, treated_unit, cutoff_date, donors
        Same semantics as :func:`scm`.
    scm_kwargs :
        Forwarded to :func:`scm` (e.g., ``pre_covariates``, ``alphas``,
        ``allow_negative_weights``).

    Returns
    -------
    pandas.DataFrame
        One row per held-out donor with columns:
          - ``dropped_donor``
          - ``mean_abs_drift`` — average absolute change in remaining donor weights
          - ``max_abs_drift``  — worst-case change
          - ``effect_shift``   — change in post-period ATT vs. baseline
    """
    from .scm import scm  # avoid circular import

    base = scm(
        df=df,
        date_col=date_col,
        unit_col=unit_col,
        outcome_col=outcome_col,
        treated_unit=treated_unit,
        cutoff_date=cutoff_date,
        donors=donors,
        **scm_kwargs,
    )
    w_base = base["weights"]
    cutoff_ts = pd.to_datetime(cutoff_date)
    att_base = float(base["synthetic"].loc[base["synthetic"].index >= cutoff_ts, "effect"].mean())

    rows = []
    for d in donors:
        sub_donors = [u for u in donors if u != d]
        if len(sub_donors) < 2:
            continue
        try:
            r = scm(
                df=df,
                date_col=date_col,
                unit_col=unit_col,
                outcome_col=outcome_col,
                treated_unit=treated_unit,
                cutoff_date=cutoff_date,
                donors=sub_donors,
                **scm_kwargs,
            )
        except Exception as e:
            log.warning("LOO failed dropping donor %s: %s", d, e)
            continue
        w = r["weights"].reindex(sub_donors).fillna(0.0)
        drift = (w - w_base.reindex(sub_donors).fillna(0.0)).abs()
        att = float(r["synthetic"].loc[r["synthetic"].index >= cutoff_ts, "effect"].mean())
        rows.append(
            {
                "dropped_donor": d,
                "mean_abs_drift": float(drift.mean()),
                "max_abs_drift": float(drift.max()),
                "effect_shift": float(att - att_base),
            }
        )

    return pd.DataFrame(rows)
