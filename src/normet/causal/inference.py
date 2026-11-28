# src/normet/causal/inference.py
"""
Inference layers for synthetic-control estimates.

Three complementary helpers:

- :func:`conformal_effect_interval` — circular (or random) block re-sampling
  standard error for the average post-period effect on an already-fitted
  result, turned into a Student-t interval (the cheap layer; see
  :func:`conformal_inference` for refitting inference).
- :func:`conformal_inference` — full refitting Chernozhukov, Wüthrich & Zhu
  (2021, JASA) finite-sample conformal inference via symmetric full-horizon
  estimation under ``H0: tau = tau_0`` and circular / i.i.d. permutations,
  inverting the test into ATT confidence bounds plus a pointwise band.
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
from ._common import newey_west_se_mean, pivot_panel, solve_simplex_weights_fast

log = get_logger(__name__)

__all__ = ["conformal_effect_interval", "conformal_inference", "rmspe_ratio_test"]


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
    circular: bool = True,
    random_state: int = 7_654_321,
) -> dict[str, float]:
    """
    Student-t interval for the average post-period effect, with a Newey-West
    (HAC) standard error.

    The interval is ``att ± t_{1-alpha/2, n_post-1} * se``, where ``se`` is the
    Newey-West (1987) heteroskedasticity- and autocorrelation-consistent
    standard error of the mean of the post-period effect path (Bartlett kernel,
    :func:`normet.causal._common.newey_west_se_mean`). The HAC variance is
    computed on the effect path itself, so shifting the post block by a
    constant does not change it; with ``n_post <= 2`` the classical
    ``std / sqrt(n_post)`` is used instead.

    Why the previous construction was too narrow
    --------------------------------------------
    The previous version built a null distribution by shifting the post block
    to zero mean (which is right: the path the null is read from must not carry
    the effect under test) and then took the empirical ``ci_level`` quantile of
    ``|block mean|`` over *overlapping* windows of ``n_post`` timestamps as the
    half-width, using the same window means for the p-value. An empirical
    quantile of the window means of a single realisation is not the quantile of
    the sampling distribution of the post-period mean: overlapping windows
    share ``n_post - 1`` of their ``n_post`` timestamps, so a record holds only
    ``n_total / n_post`` effectively independent windows, and the window that
    covers the post block exactly had been forced to sum to zero. The quantile
    consequently sat near 1.4 sd where 1.96 sd was needed, and the p-value
    counted against the same shrunken spread: nominal 95 % intervals covered
    85-88 % of i.i.d. null panels, and the null was called significant about
    three times too often.

    Why HAC
    -------
    Effect paths from a fitted SCM on air-quality series are serially
    correlated, and the classical ``std / sqrt(n_post)`` then understates the
    variance of the mean by roughly the number of independent blocks in the
    record. The Newey-West estimate absorbs that autocorrelation, while
    remaining calibrated (it reduces to the classical standard error when the
    path is i.i.d.), so it is used as the single ``se``.

    Parameters
    ----------
    scm_result : DataFrame or dict
        Output of any backend in :mod:`normet.causal` — anything with a
        ``synthetic`` DataFrame carrying an ``effect`` column (or that DF directly).
    cutoff_date : str
        Treatment cutoff date.
    n_perm : int, default 1000
        Retained for backwards compatibility of the signature and of the
        returned ``n_perm``: it is the number of block positions the earlier
        block re-sampling construction used, and sets the finite-sample floor
        ``1 / (n_perm + 1)`` of the p-value. It no longer changes the interval.
    ci_level : float, default 0.95
        Two-sided confidence level.
    circular : bool, default True
        Retained for backwards compatibility; the interval no longer depends on
        it (the HAC standard error uses the post-period path only).
    random_state : int, default 7_654_321
        Retained for backwards compatibility; no random numbers are drawn.

    Returns
    -------
    dict
        ``att``, ``low``, ``high``, ``p_value``, ``n_post``, ``n_perm``. The
        p-value is the two-sided Student-t tail of ``|att| / se``, so it is
        consistent with the interval (the interval excludes zero exactly when
        ``p_value < 1 - ci_level``); when the effect path shows no spread at
        all the interval collapses onto ``att`` and the p-value takes its
        finite-sample floor ``1 / (n_perm + 1)``.
    """
    eff = _coerce_effect_series(scm_result).dropna()
    cutoff_ts = pd.to_datetime(cutoff_date)
    post_mask = (eff.index.to_series() >= cutoff_ts).to_numpy()
    post = eff[post_mask]
    n_post = len(post)
    if n_post == 0:
        raise ValueError("No post-period observations.")

    att = float(post.mean())
    all_vals = eff.to_numpy(dtype=float)
    n_total = len(all_vals)
    if n_total <= n_post:
        raise ValueError("Need pre-period observations to build a null distribution.")

    post_vals = all_vals[post_mask]
    # n_perm is kept for the returned metadata and the p-value floor only.
    n_positions = max(1, min(int(n_perm), int(n_total)))

    # HAC (Newey-West) standard error of the post-period mean: absorbs the
    # serial correlation a plain std / sqrt(n_post) would miss.
    se = float(newey_west_se_mean(post_vals)[0])
    if n_post <= 2 or not np.isfinite(se):
        se = float(np.std(post_vals, ddof=1) / np.sqrt(n_post)) if n_post > 1 else 0.0

    # Imported here, not at module level: scipy is a heavy import the package
    # deliberately keeps out of `import normet` (see tests/test_startup_imports.py).
    from scipy import stats as _stats

    alpha = 1.0 - float(ci_level)
    dof = max(int(n_post) - 1, 1)
    if se > 0.0 and np.isfinite(se):
        q = float(_stats.t.ppf(1.0 - alpha / 2.0, dof)) * se
        p_two_sided = float(_stats.t.sf(abs(att) / se, dof)) * 2.0
        p_two_sided = min(max(p_two_sided, 0.0), 1.0)
    else:
        q = 0.0
        p_two_sided = 1.0 / (n_positions + 1.0)
    return {
        "att": att,
        "low": att - q,
        "high": att + q,
        "p_value": p_two_sided,
        "n_post": int(n_post),
        "n_perm": int(n_positions),
    }


def _stat_value(v: np.ndarray, statistic: str, q_norm: float) -> float:
    fin = v[np.isfinite(v)]
    if fin.size == 0:
        return float("nan")
    if statistic == "mean":
        return float(abs(np.mean(fin)))
    return float((np.mean(np.abs(fin) ** float(q_norm))) ** (1.0 / float(q_norm)))


def conformal_inference(
    df: pd.DataFrame,
    *,
    date_col: str,
    unit_col: str,
    outcome_col: str,
    treated_unit: str,
    cutoff_date: str,
    donors: list[str] | None = None,
    scm_backend: str = "ascm",
    null_effect: float = 0.0,
    ci_level: float = 0.95,
    statistic: str = "mean",
    q_norm: float = 1.0,
    permutation: str = "circular",
    n_perm: int | None = None,
    grid_points: int = 31,
    pointwise: bool = True,
    random_state: int = 7_654_321,
    **scm_kwargs: Any,
) -> dict[str, Any]:
    """Chernozhukov, Wüthrich & Zhu (2021, JASA) refitting conformal inference.

    Unlike :func:`conformal_effect_interval` (which holds fitted weights fixed),
    this function refits the synthetic-control estimator **symmetrically across
    the full pre + post horizon** after subtracting each candidate null effect
    ``tau_0`` from the treated unit's post-period outcomes, accounting for
    estimation error in the weights and ensuring exact finite-sample validity
    under stationary / circularly exchangeable errors.

    Parameters
    ----------
    df : pandas.DataFrame
        Long panel with ``[date_col, unit_col, outcome_col]``.
    date_col, unit_col, outcome_col : str
        Column names in ``df``.
    treated_unit, cutoff_date : str
        Treated unit identifier and intervention date.
    donors : list of str, optional
        Donor pool.
    scm_backend : str, default "ascm"
        Estimator passed to :func:`~normet.causal.run_scm.run_scm` (supports
        ``"ascm"``, ``"sdid"``, ``"abadie"``, ``"did"``, ``"scm"``,
        ``"robust"``, ``"mcnnm"``, ``"mlscm"``).
    null_effect : float, default 0.0
        Null ATT hypothesis ``H0: tau == null_effect`` for the reported ``p_value``.
    ci_level : float, default 0.95
        Confidence level in ``(0, 1)`` for the inverted CI and pointwise band.
    statistic : {"mean", "lp"}, default "mean"
        Test statistic on the post-period residuals: ``"mean"`` tests
        ``|mean(u_post)|`` (sharp for average treatment effect shifts);
        ``"lp"`` tests the ``L_q`` norm ``(mean(|u_post|^q))^(1/q)``.
    q_norm : float, default 1.0
        Exponent ``q > 0`` when ``statistic="lp"``.
    permutation : {"circular", "iid"}, default "circular"
        Permutation scheme across time indices (CWZ 2021 Algorithm 1).
    n_perm : int, optional
        Number of permutations (defaults to all ``T`` circular shifts for
        ``permutation="circular"``, or 500 draws for ``permutation="iid"``).
    grid_points : int, default 31
        Initial grid density used to bracket the inverted confidence interval
        before bisection refinement.
    pointwise : bool, default True
        Whether to compute a post-period pointwise conformal band around the
        unrestricted effect path from the full-horizon refit at ``tau = att``.
    random_state : int, default 7_654_321
    **scm_kwargs
        Forwarded to the chosen backend.

    Returns
    -------
    dict
        ``att``, ``low``, ``high``, ``p_value``, ``null_effect``, ``ci_level``,
        ``n_pre``, ``n_post``, ``n_perm``, ``permutation``, ``refit`` (``True``),
        ``grid`` (DataFrame of ``tau`` vs. ``p_value``), and ``pointwise``
        (post-period DataFrame with ``effect``, ``lower``, ``upper`` or ``None``).
    """
    from .run_scm import run_scm

    stat_mode = (statistic or "mean").lower()
    if stat_mode not in {"mean", "lp"}:
        raise ValueError(f"statistic must be 'mean' or 'lp', got {statistic!r}")
    if float(q_norm) <= 0.0:
        raise ValueError(f"q_norm must be positive, got {q_norm}")
    perm_mode = (permutation or "circular").lower()
    if perm_mode not in {"circular", "iid"}:
        raise ValueError(f"permutation must be 'circular' or 'iid', got {permutation!r}")
    if not (0.0 < float(ci_level) < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    backend_eff = (scm_backend or "ascm").lower()

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    if d[date_col].isna().any():
        raise ValueError(f"Invalid timestamps in `{date_col}`.")
    cutoff_ts = pd.to_datetime(cutoff_date)

    # 1. Unrestricted point fit to obtain ATT and scale.
    base_kw = dict(scm_kwargs)
    zeta_omega_sdid: float | None = None
    if backend_eff == "sdid":
        from .sdid import sdid

        sdid_out = sdid(
            d,
            date_col=date_col,
            unit_col=unit_col,
            outcome_col=outcome_col,
            treated_unit=treated_unit,
            cutoff_date=cutoff_ts.strftime("%Y-%m-%d"),
            donors=donors,
            **base_kw,
        )
        unrest_df = sdid_out["synthetic"]
        zeta_omega_sdid = float(sdid_out["zeta_omega"])
    else:
        unrest_df = run_scm(
            df=d,
            date_col=date_col,
            unit_col=unit_col,
            outcome_col=outcome_col,
            treated_unit=treated_unit,
            cutoff_date=cutoff_ts.strftime("%Y-%m-%d"),
            donors=donors,
            scm_backend=backend_eff,
            **base_kw,
        )

    eff_unrest = unrest_df["effect"].sort_index()
    dates_all = eff_unrest.index
    pre_mask = np.asarray(pd.to_datetime(dates_all) < cutoff_ts, dtype=bool)
    post_mask = ~pre_mask
    n_pre = int(pre_mask.sum())
    n_post = int(post_mask.sum())
    if n_pre < 3 or n_post < 1:
        raise ValueError(
            f"conformal_inference requires >=3 pre and >=1 post rows, got ({n_pre}, {n_post})."
        )

    eff_pre = eff_unrest.to_numpy(dtype=float)[pre_mask]
    eff_post = eff_unrest.to_numpy(dtype=float)[post_mask]
    att = float(np.nanmean(eff_post))
    pre_rmse = float(np.sqrt(np.nanmean(eff_pre**2)))
    se_init, _ = newey_west_se_mean(eff_post)
    if not np.isfinite(se_init) or se_init <= 0.0:
        se_init = float(np.nanstd(eff_pre, ddof=1) / np.sqrt(max(n_post, 1)))

    # Symmetric "all-pre" cutoff strictly past the final timestamp so backends
    # estimate on the entire horizon under H0 without treatment-boundary asymmetry.
    max_dt = pd.to_datetime(d[date_col]).max()
    sym_cutoff_str = (pd.Timestamp(max_dt).normalize() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    treated_mask = (d[unit_col] == treated_unit) & (d[date_col] >= cutoff_ts)
    y_treated_post = d.loc[treated_mask, outcome_col].to_numpy(dtype=float)

    # Pre-resolve SDID panel geometry once when backend is "sdid".
    if backend_eff == "sdid":
        panel_all, donors_eff = pivot_panel(
            d,
            date_col=date_col,
            unit_col=unit_col,
            outcome_col=outcome_col,
            treated_unit=treated_unit,
            donors=donors,
            parse_dates=False,
        )
        panel_all = panel_all.reindex(dates_all)
        Y_co_all = panel_all[donors_eff].to_numpy(dtype=float)
        y_tr_all_orig = panel_all[treated_unit].to_numpy(dtype=float)
        finite_all = np.all(np.isfinite(Y_co_all), axis=1) & np.isfinite(y_tr_all_orig)
        Y_co_c_all = Y_co_all[finite_all] - Y_co_all[finite_all].mean(axis=0, keepdims=True)
        # ``zeta_omega=None`` is sdid's documented "auto" rule, so an explicit
        # None in the forwarded kwargs must not shadow the fitted value: a plain
        # dict.get default is only used when the key is absent.
        zeta_kw = base_kw.get("zeta_omega")
        zeta_eff = float(zeta_omega_sdid or 0.0) if zeta_kw is None else float(zeta_kw)
        sdid_l2 = float(finite_all.sum()) * (zeta_eff**2)

    T = len(dates_all)
    post_pos = np.flatnonzero(post_mask)
    rng = np.random.default_rng(random_state)

    shifts: np.ndarray[Any, Any]
    if perm_mode == "circular":
        if n_perm is None or int(n_perm) >= T:
            shifts = np.arange(T, dtype=int)
        elif int(n_perm) <= 1:
            shifts = np.array([0], dtype=int)
        else:
            other = rng.choice(np.arange(1, T, dtype=int), size=int(n_perm) - 1, replace=False)
            shifts = np.concatenate([np.array([0], dtype=int), other])
        n_perm_eff = int(shifts.size)
        iid_perms: list[np.ndarray] | None = None
    else:
        n_perm_eff = max(2, int(n_perm) if n_perm is not None else 500)
        iid_perms = [np.arange(T, dtype=int)] + [rng.permutation(T) for _ in range(n_perm_eff - 1)]
        shifts = np.empty(0, dtype=int)

    def _refit_null_residuals(tau_0: float) -> np.ndarray:
        if backend_eff == "sdid":
            y_null = y_tr_all_orig.copy()
            y_null[post_mask] -= float(tau_0)
            y_c = y_null[finite_all] - float(np.mean(y_null[finite_all]))
            omega = solve_simplex_weights_fast(Y_co_c_all, y_c, l2_penalty=sdid_l2)
            mu = float(np.mean(y_null[finite_all] - Y_co_all[finite_all] @ omega))
            return y_null - (mu + Y_co_all @ omega)

        d_null = d.copy()
        d_null.loc[treated_mask, outcome_col] = y_treated_post - float(tau_0)
        syn_null = run_scm(
            df=d_null,
            date_col=date_col,
            unit_col=unit_col,
            outcome_col=outcome_col,
            treated_unit=treated_unit,
            cutoff_date=sym_cutoff_str,
            donors=donors,
            scm_backend=backend_eff,
            **base_kw,
        )
        return syn_null["effect"].reindex(dates_all).to_numpy(dtype=float)

    def _eval_null(tau_0: float) -> tuple[float, np.ndarray]:
        u = _refit_null_residuals(tau_0)
        s_obs = _stat_value(u[post_pos], stat_mode, q_norm)
        if not np.isfinite(s_obs):
            return float("nan"), u

        if perm_mode == "circular":
            u2 = np.concatenate([u, u])
            stats_arr = np.empty(shifts.size, dtype=float)
            for k, sh in enumerate(shifts):
                stats_arr[k] = _stat_value(u2[post_pos + int(sh)], stat_mode, q_norm)
        else:
            assert iid_perms is not None
            stats_arr = np.empty(len(iid_perms), dtype=float)
            for k, p_idx in enumerate(iid_perms):
                stats_arr[k] = _stat_value(u[p_idx][post_pos], stat_mode, q_norm)

        fin = stats_arr[np.isfinite(stats_arr)]
        if fin.size == 0:
            return float("nan"), u
        p_val = float(np.mean(fin >= (s_obs - 1e-12)))
        return p_val, u

    p_null, _ = _eval_null(float(null_effect))

    # Invert the conformal p-value function to obtain [low, high].
    alpha = 1.0 - float(ci_level)
    scale = max(se_init, pre_rmse / np.sqrt(max(n_post, 1)), 0.1 * max(abs(att), 1.0), 1e-2)
    half_span = 4.0 * scale

    n_pts = max(9, int(grid_points))
    cache: dict[float, float] = {}

    def _p_cached(tau_val: float) -> float:
        key = round(float(tau_val), 12)
        if key not in cache:
            p_v, _ = _eval_null(float(tau_val))
            cache[key] = p_v
        return cache[key]

    taus = np.linspace(att - half_span, att + half_span, n_pts)
    pvals = np.array([_p_cached(t) for t in taus], dtype=float)

    # Expand outward if either endpoint is still accepted (p > alpha) and rejection is possible.
    if n_perm_eff > int(np.ceil(1.0 / max(alpha, 1e-12))):
        for _ in range(4):
            expand_left = np.isfinite(pvals[0]) and pvals[0] > alpha
            expand_right = np.isfinite(pvals[-1]) and pvals[-1] > alpha
            if not (expand_left or expand_right):
                break
            half_span *= 2.0
            taus = np.linspace(att - half_span, att + half_span, n_pts)
            pvals = np.array([_p_cached(t) for t in taus], dtype=float)

    accepted = np.flatnonzero(np.isfinite(pvals) & (pvals > alpha))
    if accepted.size == 0:
        idx_max = int(np.nanargmax(pvals)) if np.any(np.isfinite(pvals)) else n_pts // 2
        ci_low = float(taus[idx_max])
        ci_high = float(taus[idx_max])
    else:
        lo_i = int(accepted[0])
        hi_i = int(accepted[-1])
        ci_low = float(taus[lo_i])
        ci_high = float(taus[hi_i])

        if lo_i > 0:
            a, b = float(taus[lo_i - 1]), float(taus[lo_i])
            for _ in range(8):
                mid = 0.5 * (a + b)
                if _p_cached(mid) > alpha:
                    b = mid
                else:
                    a = mid
            ci_low = float(b)

        if hi_i < len(taus) - 1:
            a, b = float(taus[hi_i]), float(taus[hi_i + 1])
            for _ in range(8):
                mid = 0.5 * (a + b)
                if _p_cached(mid) > alpha:
                    a = mid
                else:
                    b = mid
            ci_high = float(a)

    grid_df = pd.DataFrame({"tau": taus, "p_value": pvals})

    pointwise_df: pd.DataFrame | None = None
    if pointwise:
        _, u_att = _eval_null(att)
        fin_u = np.abs(u_att[np.isfinite(u_att)])
        post_index = dates_all[post_mask]
        pointwise_df = pd.DataFrame(index=post_index)
        pointwise_df["effect"] = eff_post
        if fin_u.size > 0:
            q_level = min(1.0, float(np.ceil((1.0 - alpha) * (fin_u.size + 1))) / float(fin_u.size))
            q_pt = float(np.quantile(fin_u, q_level))
            pointwise_df["lower"] = eff_post - q_pt
            pointwise_df["upper"] = eff_post + q_pt
        else:
            pointwise_df["lower"] = np.nan
            pointwise_df["upper"] = np.nan

    return {
        "att": att,
        "low": float(min(ci_low, ci_high)),
        "high": float(max(ci_low, ci_high)),
        "p_value": float(p_null),
        "null_effect": float(null_effect),
        "ci_level": float(ci_level),
        "n_pre": n_pre,
        "n_post": n_post,
        "n_perm": n_perm_eff,
        "permutation": perm_mode,
        "refit": True,
        "grid": grid_df,
        "pointwise": pointwise_df,
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
                              from largest to smallest); NaN when the treated
                              ratio itself is undefined
    """
    if "treated" not in placebo_space_out:
        raise ValueError("`placebo_space_out` must contain a 'treated' DataFrame.")
    cutoff_ts = pd.to_datetime(cutoff_date)

    def _ratio(eff_series: pd.Series) -> float:
        pre = eff_series[eff_series.index.to_series() < cutoff_ts].dropna()
        post = eff_series[eff_series.index.to_series() >= cutoff_ts].dropna()
        if pre.empty or post.empty:
            return np.nan
        pre_rmspe = float(np.sqrt(np.mean(pre.to_numpy(dtype=float) ** 2)))
        post_rmspe = float(np.sqrt(np.mean(post.to_numpy(dtype=float) ** 2)))
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
    if ratios.empty or not np.isfinite(treated_ratio):
        # A treated ratio that is NaN or infinite (no pre- or post-period
        # variation, a zero pre-RMSPE) makes every ``>=`` comparison False,
        # which used to return p = 1/(n+1) -- the strongest possible
        # significance -- from a test that never ran. With no placebo ratios
        # at all the treated unit is still the only entry, so its rank is 1.
        return {
            "treated_ratio": treated_ratio,
            "placebo_ratios": ratios,
            "p_value": float("nan"),
            "rank": (1 if ratios.empty and np.isfinite(treated_ratio) else float("nan")),
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
