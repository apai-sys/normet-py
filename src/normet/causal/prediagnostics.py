# src/normet/causal/prediagnostics.py
"""Pre-trend diagnostics for synthetic-control fits.

The first question a reviewer asks of an SCM is whether the pre-period was
parallel: the whole identification argument rests on the synthetic unit
tracking the treated unit *before* the intervention, so a pre-period residual
that drifts away from zero is a parallel-trends violation wearing a good
pre-RMSE costume. ``scm_diagnostics`` reports how close the fit is on average
in the pre-period, but not whether the fit is *getting* closer or further as
the intervention approaches -- which is exactly what a violation looks like
when it is about to happen.

:func:`pre_trend_test` answers it with two complementary numbers on the
pre-period residuals ``e_t = observed_t - synthetic_t``:

- the OLS slope of ``e_t`` against the period index, with its classical
  two-sided t-test; and
- ``rmpe_drift``, the RMSPE of the final third of the pre-period divided by
  the RMSPE of the first two thirds (the raw-residual quantity Abadie's
  post/pre RMSPE ratio is built from). One means a stable fit; a value above
  one means the residuals grow toward the cutoff.

The two numbers catch different failure modes: a slow linear drift has a
significant slope but a mild drift ratio; a late fit breakdown can push the
drift ratio up while the slope test, diluted over the whole pre-period,
stays quiet. The verdict reads them together as a traffic light -- and says
so, with the numbers, so a report can quote which light and why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ._common import newey_west_se_mean, newey_west_se_slope

log = get_logger(__name__)

__all__ = ["HonestDiDResult", "PreTrend", "honest_did_bounds", "pre_trend_test"]

#: Fewer pre-period points than this cannot support a slope test with two
#: parameters and a readable p-value.
_DEFAULT_MIN_PERIODS = 5


@dataclass(frozen=True)
class PreTrend:
    """The pre-trend verdict for one synthetic-control fit.

    ``slope`` is the OLS slope of the pre-period residuals (observed minus
    synthetic) against the period index, in outcome units per period;
    ``p_slope`` its two-sided t-test p-value (using Newey–West HAC SE by
    default). ``rmpe_drift`` compares the RMSPE of the final third of the
    pre-period with that of the first two thirds: 1 is a stable fit, above 1
    the residuals grow toward the intervention. ``verdict`` is the traffic
    light with its numbers, and ``str()`` is the verdict itself.
    """

    slope: float
    p_slope: float
    rmpe_drift: float
    n_periods: int
    verdict: str
    se_slope: float = float("nan")
    se_type: str = "hac"
    hac_lags: int = 0

    def __str__(self) -> str:  # pragma: no cover - the result's own words
        return self.verdict


@dataclass(frozen=True)
class HonestDiDResult:
    """Rambachan & Roth (2023) sensitivity bounds for parallel-trend violations.

    Attributes
    ----------
    table : pandas.DataFrame
        Sensitivity grid with columns ``["M", "id_low", "id_high", "ci_low",
        "ci_high", "contains_zero_id", "contains_zero_ci"]``.
    att : float
        Baseline post-period average treatment effect.
    se : float
        Autocorrelation-robust standard error of ``att``.
    method : str
        Sensitivity class: ``"relative_magnitude"``, ``"level_magnitude"``, or
        ``"smoothness"``.
    ci_level : float
        Confidence level of the Imbens–Manski robust confidence intervals.
    breakdown_m : float
        Smallest ``M >= 0`` at which 0 enters the robust confidence interval
        (``0.0`` if already insignificant at ``M = 0``).
    breakdown_m_id : float
        Smallest ``M >= 0`` at which 0 enters the identified set.
    reference_scale : float
        Pre-period empirical scale used to interpret ``M``.
    pre_slope : float
        Pre-period linear residual slope (non-zero under ``method="smoothness"``).
    n_pre, n_post : int
        Number of finite pre- and post-period observations used.
    """

    table: pd.DataFrame
    att: float
    se: float
    method: str
    ci_level: float
    breakdown_m: float
    breakdown_m_id: float
    reference_scale: float
    pre_slope: float
    n_pre: int
    n_post: int


def _synthetic_frame(scm_result: Any) -> pd.DataFrame:
    """Pull the synthetic frame out of a backend result (dict or DataFrame)."""
    if isinstance(scm_result, pd.DataFrame):
        synth = scm_result
    elif isinstance(scm_result, dict) and "synthetic" in scm_result:
        synth = scm_result["synthetic"]
    else:
        raise ValueError("Expected a DataFrame or a dict with a 'synthetic' key.")
    if not isinstance(synth, pd.DataFrame):
        raise ValueError("The 'synthetic' entry must be a DataFrame.")
    for col in ("observed", "synthetic"):
        if col not in synth.columns:
            raise ValueError(f"Column '{col}' missing from the synthetic-control result.")
    return synth


def _insufficient(n: int, min_periods: int, se_type: str = "hac") -> PreTrend:
    return PreTrend(
        slope=float("nan"),
        p_slope=float("nan"),
        rmpe_drift=float("nan"),
        n_periods=int(n),
        verdict=f"insufficient: {n} pre-period point{'s' if n != 1 else ''}, "
        f"{min_periods} are needed for a trend test",
        se_slope=float("nan"),
        se_type=se_type,
        hac_lags=0,
    )


def _rmpe_drift(e: np.ndarray) -> float:
    """RMSPE of the final third of ``e`` divided by that of the first two thirds."""
    n = e.size
    third = n // 3
    if third < 1:
        return float("nan")
    early = float(np.sqrt(np.mean(e[: n - third] ** 2)))
    late = float(np.sqrt(np.mean(e[n - third :] ** 2)))
    if early <= 0.0:
        return float("inf") if late > 0.0 else float("nan")
    return late / early


def pre_trend_test(
    scm_result: dict[str, Any] | pd.DataFrame,
    cutoff_date: str,
    *,
    alpha: float = 0.05,
    min_periods: int = _DEFAULT_MIN_PERIODS,
    se_type: str = "hac",
    maxlags: int | None = None,
) -> PreTrend:
    """Test whether the pre-period of a synthetic-control fit was parallel.

    Parameters
    ----------
    scm_result : dict or pandas.DataFrame
        Output of any backend in :mod:`normet.causal` -- anything with a
        ``synthetic`` DataFrame carrying ``observed`` and ``synthetic``
        columns (or that DataFrame directly, e.g. the return of
        :func:`normet.run_scm`).
    cutoff_date : str
        Treatment cutoff. Parseable by ``pd.to_datetime``.
    alpha : float, default 0.05
        Significance level for the slope test in the verdict.
    min_periods : int, default 5
        Fewer finite pre-period points than this cannot support the test; the
        result is then all-NaN with an ``insufficient`` verdict rather than a
        p-value computed from two degrees of freedom.
    se_type : {"hac", "ols"}, default "hac"
        Standard-error estimator for the pre-period residual trend. ``"hac"``
        (default) uses Newey–West (1987) Bartlett-kernel HAC standard errors so
        serially correlated air-quality residuals do not inflate false positives;
        ``"ols"`` uses classical i.i.d. OLS standard errors.
    maxlags : int, optional
        Lag truncation parameter for Newey–West HAC (defaults to the
        Newey–West 1994 automatic rule ``floor(4 * (n / 100)^(2/9))``).

    Returns
    -------
    PreTrend
        ``slope``, ``p_slope``, ``rmpe_drift``, ``n_periods``, ``verdict``,
        ``se_slope``, ``se_type``, and ``hac_lags`` (see :class:`PreTrend`).
    """
    se_mode = (se_type or "hac").lower()
    if se_mode not in {"hac", "ols"}:
        raise ValueError(f"se_type must be 'hac' or 'ols', got {se_type!r}")

    synth = _synthetic_frame(scm_result)
    cutoff_ts = pd.to_datetime(cutoff_date)
    pre = synth[synth.index < cutoff_ts]
    if pre.empty:
        return _insufficient(0, min_periods, se_mode)

    obs = pre["observed"].to_numpy(dtype=float)
    syn = pre["synthetic"].to_numpy(dtype=float)
    mask = np.isfinite(obs) & np.isfinite(syn)
    e = obs[mask] - syn[mask]
    n = int(e.size)
    if n < max(min_periods, 3):
        return _insufficient(n, min_periods, se_mode)

    dof = n - 2
    if se_mode == "hac":
        slope, se_slope, lags_used = newey_west_se_slope(e, maxlags=maxlags)
    else:
        lags_used = 0
        t = np.arange(n, dtype=float)
        x = np.vstack([np.ones(n), t]).T
        beta, _, _, _ = np.linalg.lstsq(x, e, rcond=None)
        resid = e - x @ beta
        s2 = float(np.sum(resid**2)) / dof
        # Closed-form 2x2 inverse of ``x.T @ x`` (avoids LAPACK SIGBUS in Qt worker threads).
        n_f = float(n)
        t_sum = n_f * (n_f - 1.0) / 2.0
        t_sq_sum = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0
        det = n_f * t_sq_sum - t_sum * t_sum
        se_slope = float(np.sqrt(max(s2, 0.0) * n_f / det)) if det > 0.0 else float("nan")
        slope = float(beta[1])

    if se_slope > 0.0 and np.isfinite(se_slope):
        from scipy import stats  # lazy: mirrors the trend module's pattern

        t_stat = float(slope) / se_slope
        p = float(2.0 * stats.t.sf(abs(t_stat), dof))
    elif np.isfinite(slope) and se_slope == 0.0:
        p = 0.0 if abs(slope) > 0.0 else 1.0
    else:
        p = float("nan")

    drift = _rmpe_drift(e)

    if not (np.isfinite(p) and np.isfinite(drift)):
        verdict = f"indeterminate: the test could not be computed (p={p:.3g}, drift={drift:.3g})"
    elif p < 0.01:
        verdict = (
            f"fail: significant pre-trend (p={p:.3g}, drift={drift:.2f}) -- "
            "the pre-period was not parallel"
        )
    elif p < alpha:
        verdict = f"caution: borderline pre-trend (p={p:.3g}, drift={drift:.2f})"
    elif drift > 1.25:
        verdict = (
            f"caution: the pre-period fit deteriorates toward the cutoff "
            f"(p={p:.3g}, drift={drift:.2f})"
        )
    else:
        verdict = f"ok: no significant pre-trend (p={p:.3g}, drift={drift:.2f})"

    log.info("pre-trend (%s) on %d pre-period points: %s", se_mode, n, verdict)
    return PreTrend(
        slope=slope,
        p_slope=p,
        rmpe_drift=drift,
        n_periods=n,
        verdict=verdict,
        se_slope=se_slope,
        se_type=se_mode,
        hac_lags=int(lags_used),
    )


def _imbens_manski_cv(half_width: float, se: float, alpha: float) -> float:
    """Critical value solving ``Phi(c + 2*hw/se) - Phi(-c) == 1 - alpha``."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    z_one = float(norm.ppf(1.0 - alpha))
    z_two = float(norm.ppf(1.0 - 0.5 * alpha))
    if not np.isfinite(se) or se <= 0.0 or half_width <= 0.0:
        return z_two
    delta = 2.0 * float(half_width) / float(se)
    if delta > 38.0:
        return z_one

    def _fn(c: float) -> float:
        return float(norm.cdf(c + delta) - norm.cdf(-c) - (1.0 - alpha))

    fa = _fn(z_one)
    if fa >= 0.0:
        return z_one
    fb = _fn(z_two)
    if fb <= 0.0:
        return z_two
    return float(brentq(_fn, z_one, z_two))


def honest_did_bounds(
    scm_result: dict[str, Any] | pd.DataFrame,
    cutoff_date: str,
    *,
    method: str = "relative_magnitude",
    m_grid: list[float] | np.ndarray | None = None,
    ci_level: float = 0.95,
    se: float | None = None,
    maxlags: int | None = None,
    horizon: int | None = None,
) -> HonestDiDResult:
    """Compute Rambachan & Roth (2023) sensitivity bounds for parallel-trend violations.

    Evaluates how much post-treatment departure from parallel trends (measured by
    parameter ``M >= 0``) the estimated ATT can tolerate before becoming
    statistically indistinguishable from zero, using HAC standard errors and
    Imbens & Manski (2004) uniform confidence intervals for the partially
    identified parameter.

    Parameters
    ----------
    scm_result : dict or pandas.DataFrame
        Fitted synthetic-control result (DataFrame or dict with ``"synthetic"``).
    cutoff_date : str
        Treatment cutoff date.
    method : {"relative_magnitude", "level_magnitude", "smoothness"}, default "relative_magnitude"
        Restriction class on post-treatment trend violations ``delta``:
        - ``"relative_magnitude"``: post-period first differences bounded by
          ``M * max_{t in pre} |delta e_t|``.
        - ``"level_magnitude"``: post-period level biases bounded by
          ``M * max_{t in pre} |e_t|``.
        - ``"smoothness"``: second differences bounded by ``M``, anchored around
          extrapolation of the pre-period residual slope.
    m_grid : sequence of float, optional
        Grid of ``M`` values to evaluate. Sensible defaults depend on ``method``.
    ci_level : float, default 0.95
        Confidence level in ``(0, 1)``.
    se : float, optional
        Override standard error of ``att``. By default computed using
        :func:`~normet.causal._common.newey_west_se_mean` on post-period residuals.
    maxlags : int, optional
        Optional HAC truncation lag count when estimating ``se``.
    horizon : int, optional
        Restrict post-period evaluation to the first ``horizon`` post dates.
    """
    method_eff = (method or "relative_magnitude").lower()
    if method_eff not in {"relative_magnitude", "level_magnitude", "smoothness"}:
        raise ValueError(
            "method must be 'relative_magnitude', 'level_magnitude', or 'smoothness', "
            f"got {method!r}"
        )
    if not (0.0 < float(ci_level) < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")

    synth = _synthetic_frame(scm_result)
    cutoff_ts = pd.to_datetime(cutoff_date)
    pre = synth[synth.index < cutoff_ts]
    post = synth[synth.index >= cutoff_ts]
    if horizon is not None:
        if int(horizon) < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        post = post.iloc[: int(horizon)]

    e_pre = (pre["observed"] - pre["synthetic"]).to_numpy(dtype=float)
    e_pre = e_pre[np.isfinite(e_pre)]
    e_post = (post["observed"] - post["synthetic"]).to_numpy(dtype=float)
    e_post = e_post[np.isfinite(e_post)]

    n_pre = int(e_pre.size)
    n_post = int(e_post.size)
    if n_pre < 3:
        raise ValueError(f"Need at least 3 finite pre-period residuals, got {n_pre}.")
    if n_post < 1:
        raise ValueError("No finite post-period observations.")

    att = float(np.mean(e_post))
    if se is None:
        if n_post >= 3:
            se_eff, _ = newey_west_se_mean(e_post, maxlags=maxlags)
        elif n_post == 2:
            se_eff = float(np.std(e_post, ddof=1) / np.sqrt(2.0))
        else:
            se_eff = float(np.std(e_pre, ddof=1))
    else:
        se_eff = float(se)
    if not np.isfinite(se_eff) or se_eff < 0.0:
        se_eff = 0.0

    pre_slope = 0.0
    bias_center = 0.0
    if method_eff == "relative_magnitude":
        ref = float(np.max(np.abs(np.diff(e_pre))))
        hw_per_m = 0.5 * float(n_post + 1) * ref
        default_grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    elif method_eff == "level_magnitude":
        ref = float(np.max(np.abs(e_pre)))
        hw_per_m = ref
        default_grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    else:  # smoothness
        t_pre = np.arange(n_pre, dtype=float)
        t_c = t_pre - float(np.mean(t_pre))
        s_tt = float(np.dot(t_c, t_c))
        pre_slope = float(np.dot(t_c, e_pre - float(np.mean(e_pre))) / s_tt) if s_tt > 0 else 0.0
        bias_center = 0.5 * float(n_post + 1) * pre_slope
        hw_per_m = float((n_post + 1) * (n_post + 2)) / 6.0
        if n_pre >= 4:
            ref = float(np.std(np.diff(e_pre, n=2), ddof=1))
        else:
            diffs1 = np.diff(e_pre)
            ref = float(np.std(diffs1, ddof=1)) if diffs1.size >= 2 else float(np.std(diffs1))
        if ref > 0.0 and np.isfinite(ref):
            default_grid = [0.0, 0.1 * ref, 0.25 * ref, 0.5 * ref, 1.0 * ref, 2.0 * ref]
        else:
            default_grid = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5]

    grid_arr = (
        np.asarray(default_grid, dtype=float)
        if m_grid is None
        else np.asarray(list(m_grid), dtype=float)
    )
    if np.any(grid_arr < 0.0) or not np.all(np.isfinite(grid_arr)):
        raise ValueError("All values in m_grid must be finite and non-negative.")

    alpha = 1.0 - float(ci_level)
    center = att - bias_center
    rows: list[dict[str, Any]] = []
    for m_val in grid_arr:
        hw = float(m_val) * hw_per_m
        id_lo = center - hw
        id_hi = center + hw
        c_im = _imbens_manski_cv(hw, se_eff, alpha)
        ci_lo = id_lo - c_im * se_eff
        ci_hi = id_hi + c_im * se_eff
        rows.append(
            {
                "M": float(m_val),
                "id_low": float(id_lo),
                "id_high": float(id_hi),
                "ci_low": float(ci_lo),
                "ci_high": float(ci_hi),
                "contains_zero_id": bool(id_lo <= 0.0 <= id_hi),
                "contains_zero_ci": bool(ci_lo <= 0.0 <= ci_hi),
            }
        )

    # Compute exact breakdown M where 0 enters the identified set and the robust CI.
    if abs(center) == 0.0:
        breakdown_id = 0.0
    elif hw_per_m <= 0.0:
        breakdown_id = float("inf")
    else:
        breakdown_id = float(abs(center) / hw_per_m)

    c0 = _imbens_manski_cv(0.0, se_eff, alpha)
    if abs(center) <= c0 * se_eff:
        breakdown_ci = 0.0
    elif hw_per_m <= 0.0:
        breakdown_ci = float("inf")
    else:
        from scipy.optimize import brentq

        def _margin(m_val: float) -> float:
            hw = float(m_val) * hw_per_m
            return hw + _imbens_manski_cv(hw, se_eff, alpha) * se_eff - abs(center)

        m_hi = abs(center) / hw_per_m + 1.0
        breakdown_ci = float(brentq(_margin, 0.0, m_hi))

    return HonestDiDResult(
        table=pd.DataFrame(rows),
        att=att,
        se=se_eff,
        method=method_eff,
        ci_level=float(ci_level),
        breakdown_m=breakdown_ci,
        breakdown_m_id=breakdown_id,
        reference_scale=ref,
        pre_slope=pre_slope,
        n_pre=n_pre,
        n_post=n_post,
    )
