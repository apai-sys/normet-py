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

log = get_logger(__name__)

__all__ = ["PreTrend", "pre_trend_test"]

#: Fewer pre-period points than this cannot support a slope test with two
#: parameters and a readable p-value.
_DEFAULT_MIN_PERIODS = 5


@dataclass(frozen=True)
class PreTrend:
    """The pre-trend verdict for one synthetic-control fit.

    ``slope`` is the OLS slope of the pre-period residuals (observed minus
    synthetic) against the period index, in outcome units per period;
    ``p_slope`` its two-sided t-test p-value. ``rmpe_drift`` compares the
    RMSPE of the final third of the pre-period with that of the first two
    thirds: 1 is a stable fit, above 1 the residuals grow toward the
    intervention. ``verdict`` is the traffic light with its numbers, and
    ``str()`` is the verdict itself.
    """

    slope: float
    p_slope: float
    rmpe_drift: float
    n_periods: int
    verdict: str

    def __str__(self) -> str:  # pragma: no cover - the result's own words
        return self.verdict


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


def _insufficient(n: int, min_periods: int) -> PreTrend:
    return PreTrend(
        slope=float("nan"),
        p_slope=float("nan"),
        rmpe_drift=float("nan"),
        n_periods=int(n),
        verdict=f"insufficient: {n} pre-period point{'s' if n != 1 else ''}, "
        f"{min_periods} are needed for a trend test",
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

    Returns
    -------
    PreTrend
        ``slope``, ``p_slope``, ``rmpe_drift``, ``n_periods`` and ``verdict``
        (see :class:`PreTrend`). The verdict is a traffic light: ``fail``
        when the slope is significant at the 1 % level -- a pre-period that
        is trending is not parallel, whatever the drift ratio says;
        ``caution`` when the slope is significant at ``alpha`` but not at 1 %,
        or when the drift ratio exceeds 1.25 (the fit deteriorates toward the
        cutoff without a resolvable trend); ``ok`` otherwise.

    Notes
    -----
    The t-test is the classical OLS one (it assumes the residuals are
    independent); pre-period SCM residuals at daily-or-slower frequency are
    usually close enough to that, and a violation that is *systematic* --
    which is what the test hunts for -- shows up in the drift ratio as well,
    which makes no independence assumption at all.
    """
    synth = _synthetic_frame(scm_result)
    cutoff_ts = pd.to_datetime(cutoff_date)
    pre = synth[synth.index < cutoff_ts]
    if pre.empty:
        return _insufficient(0, min_periods)

    obs = pre["observed"].to_numpy(dtype=float)
    syn = pre["synthetic"].to_numpy(dtype=float)
    mask = np.isfinite(obs) & np.isfinite(syn)
    e = obs[mask] - syn[mask]
    n = int(e.size)
    if n < max(min_periods, 3):
        return _insufficient(n, min_periods)

    t = np.arange(n, dtype=float)
    x = np.vstack([np.ones(n), t]).T
    beta, _, _, _ = np.linalg.lstsq(x, e, rcond=None)
    resid = e - x @ beta
    dof = n - 2
    s2 = float(np.sum(resid**2)) / dof
    xt_x_inv = np.linalg.inv(x.T @ x)
    se_slope = float(np.sqrt(max(s2, 0.0) * xt_x_inv[1, 1]))
    if se_slope > 0.0 and np.isfinite(se_slope):
        from scipy import stats  # lazy: mirrors the trend module's pattern

        t_stat = float(beta[1]) / se_slope
        p = float(2.0 * stats.t.sf(abs(t_stat), dof))
    else:
        p = float("nan")

    drift = _rmpe_drift(e)
    slope = float(beta[1])

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

    log.info("pre-trend on %d pre-period points: %s", n, verdict)
    return PreTrend(
        slope=slope,
        p_slope=p,
        rmpe_drift=drift,
        n_periods=n,
        verdict=verdict,
    )
