# src/normet/causal/inference.py
"""
Inference layers for synthetic-control estimates.

Two complementary helpers:

- :func:`conformal_effect_interval` — finite-sample (sub-sampling) conformal
  prediction interval for the average post-period effect, in the spirit of
  Chernozhukov, Wüthrich & Zhu (2021).
- :func:`rmspe_ratio_test` — Abadie's classical placebo significance heuristic:
  compare the treated unit's ``post / pre`` RMSPE ratio against the placebo
  distribution.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["conformal_effect_interval", "rmspe_ratio_test"]


def _coerce_effect_series(scm_result: Any) -> pd.Series:
    """Pull the ``effect`` series out of a backend result (dict or DataFrame)."""
    if isinstance(scm_result, pd.DataFrame):
        synth = scm_result
    elif isinstance(scm_result, Mapping) and "synthetic" in scm_result:
        synth = scm_result["synthetic"]
    else:
        raise ValueError("Expected a DataFrame or dict with 'synthetic' key.")
    if "effect" not in synth.columns:
        raise ValueError("Effect column missing from synthetic-control result.")
    return synth["effect"]


def conformal_effect_interval(
    scm_result: Any,
    cutoff_date: str,
    *,
    n_perm: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 7_654_321,
) -> dict[str, float]:
    """
    Conformal inference for the average post-period effect.

    The idea: under the null that the post-period effect equals ``τ``, the
    residuals ``effect_t - 1[post]·τ`` should be exchangeable in t. We
    therefore construct a sub-sampling null distribution of the test statistic
    ``|mean(window)|`` over random length-``T_post`` blocks drawn from the full
    horizon, and invert the test to obtain an interval for the post-period
    ATT.

    Parameters
    ----------
    scm_result : DataFrame or dict
        Output of any backend in :mod:`normet.causal` — anything with a
        ``synthetic`` DataFrame carrying an ``effect`` column (or that DF directly).
    cutoff_date : str
        Treatment cutoff date.
    n_perm : int, default 1000
        Number of random sub-samples used to build the null distribution.
    ci_level : float, default 0.95
        Two-sided confidence level.
    random_state : int, default 7_654_321

    Returns
    -------
    dict
        - ``att``    : observed average post-period effect
        - ``low``    : lower confidence limit
        - ``high``   : upper confidence limit
        - ``p_value``: two-sided p-value for H0 (τ = 0)
        - ``n_post`` : number of post-period observations
        - ``n_perm`` : number of permutations actually used
    """
    eff = _coerce_effect_series(scm_result).dropna()
    cutoff_ts = pd.to_datetime(cutoff_date)
    post = eff.loc[eff.index >= cutoff_ts]
    n_post = len(post)
    if n_post == 0:
        raise ValueError("No post-period observations.")

    att = float(post.mean())
    all_vals = eff.to_numpy()
    n_total = len(all_vals)
    if n_total <= n_post:
        raise ValueError("Need pre-period observations to build a null distribution.")

    rng = np.random.default_rng(random_state)
    # Draw block starting indices over the *full* horizon (pre+post combined),
    # then re-center residuals by subtracting the post-mean so the null is
    # τ = 0. The conformalised statistic is |mean(window) − att|, but for the
    # two-sided p-value we use |mean(window)| ≥ |att|.
    starts = rng.integers(0, n_total - n_post + 1, size=int(n_perm))
    means = np.empty(starts.size, dtype=float)
    for k, s in enumerate(starts):
        means[k] = float(np.nanmean(all_vals[s : s + n_post]))

    # Two-sided p-value with the +1 small-sample correction.
    p_two_sided = (np.sum(np.abs(means) >= abs(att)) + 1.0) / (means.size + 1.0)

    # Invert the test: CI = att ± q where q is the (1-α) quantile of |perm mean|.
    alpha = 1.0 - float(ci_level)
    q = float(np.quantile(np.abs(means), 1.0 - alpha))
    return {
        "att": att,
        "low": att - q,
        "high": att + q,
        "p_value": float(p_two_sided),
        "n_post": int(n_post),
        "n_perm": int(means.size),
    }


def rmspe_ratio_test(
    placebo_space_out: Mapping[str, Any],
    cutoff_date: str,
) -> dict[str, Any]:
    """
    Abadie's RMSPE-ratio placebo test.

    For each candidate "treated" unit (the true treated + every placebo donor),
    compute ``post-RMSPE / pre-RMSPE``. Units with a *real* treatment effect
    should sit in the right tail of the placebo distribution.

    Parameters
    ----------
    placebo_space_out : mapping
        Output of :func:`placebo_in_space` — keys ``treated`` (DataFrame with
        ``effect``) and ``placebos`` (dict of cutoff → DataFrame).
    cutoff_date : str
        Treatment cutoff date.

    Returns
    -------
    dict
        - ``treated_ratio`` : RMSPE ratio for the treated unit
        - ``placebo_ratios``: pandas.Series of ratios for each placebo donor
        - ``p_value``       : (#{placebo ratio ≥ treated} + 1) / (n + 1)
        - ``rank``          : rank of treated ratio among placebos+treated (1-indexed,
                              from largest to smallest)
    """
    if "treated" not in placebo_space_out:
        raise ValueError("`placebo_space_out` must contain a 'treated' DataFrame.")
    cutoff_ts = pd.to_datetime(cutoff_date)

    def _ratio(eff_series: pd.Series) -> float:
        pre = eff_series.loc[eff_series.index < cutoff_ts].dropna()
        post = eff_series.loc[eff_series.index >= cutoff_ts].dropna()
        if pre.empty or post.empty:
            return np.nan
        pre_rmspe = float(np.sqrt(np.mean(pre.to_numpy() ** 2)))
        post_rmspe = float(np.sqrt(np.mean(post.to_numpy() ** 2)))
        return post_rmspe / pre_rmspe if pre_rmspe > 0 else np.nan

    treated_df = placebo_space_out["treated"]
    treated_ratio = _ratio(treated_df["effect"])

    placebo_ratios: dict[str, float] = {}
    for unit, frame in (placebo_space_out.get("placebos") or {}).items():
        # Each frame has either a single effect-labelled column or 'effect'
        if "effect" in frame.columns:
            s = frame["effect"]
        elif unit in frame.columns:
            s = frame[unit]
        else:
            s = frame.iloc[:, 0]
        placebo_ratios[unit] = _ratio(s)

    ratios = pd.Series(placebo_ratios, name="rmspe_ratio").dropna()
    if ratios.empty:
        return {
            "treated_ratio": treated_ratio,
            "placebo_ratios": ratios,
            "p_value": float("nan"),
            "rank": 1,
        }
    p_value = float((np.sum(ratios.to_numpy() >= treated_ratio) + 1.0) / (ratios.size + 1.0))
    # Rank (1 = largest)
    all_sorted = np.sort(np.concatenate([ratios.values, [treated_ratio]]))[::-1]
    rank = int(np.searchsorted(-all_sorted, -treated_ratio) + 1)
    return {
        "treated_ratio": float(treated_ratio),
        "placebo_ratios": ratios,
        "p_value": p_value,
        "rank": rank,
    }
