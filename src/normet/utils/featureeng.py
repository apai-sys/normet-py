# src/normet/utils/featureeng.py
"""
Feature-engineering helpers for environmental time-series.

These utilities operate on tidy long DataFrames with a ``date`` column
(or DatetimeIndex). They never modify the input in place.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ._lazy import require
from .logging import get_logger

log = get_logger(__name__)

__all__ = [
    "add_lag_features",
    "add_rolling_features",
    "analyze_lag",
    "cyclical_encode",
    "wind_to_uv",
    "LagDiagnostics",
]


def _ensure_sorted_by_date(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Return a copy sorted by date_col; require the column to be present and datetime-like."""
    if date_col not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": date_col})
        else:
            raise ValueError(f"`{date_col}` column not found and index is not DatetimeIndex.")
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(f"`{date_col}` must be datetime-like.")
    return df.sort_values(date_col).reset_index(drop=True)


def add_lag_features(
    df: pd.DataFrame,
    cols: Sequence[str],
    lags: Sequence[int],
    *,
    group_col: str | None = None,
    date_col: str = "date",
    suffix: str = "_lag",
) -> pd.DataFrame:
    """
    Add lagged copies of ``cols`` shifted by each value in ``lags`` (rows).

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format dataset with a datetime column.
    cols : sequence of str
        Columns to lag.
    lags : sequence of int
        Positive integers; ``lag=1`` looks one row back.
    group_col : str, optional
        If provided, lags are computed within each group (e.g., per station).
    date_col : str, default "date"
        Datetime column used to sort before lagging.
    suffix : str, default "_lag"
        New columns are named ``f"{col}{suffix}{k}"``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` (sorted by date) with new lag columns appended.
    """
    if not cols or not lags:
        return df.copy()
    bad = [c for c in cols if c not in df.columns]
    if bad:
        raise ValueError(f"Columns not in df: {bad}")
    if any(int(k) <= 0 for k in lags):
        raise ValueError("`lags` must contain positive integers.")

    out = _ensure_sorted_by_date(df, date_col=date_col)
    grouped = out.groupby(group_col, sort=False) if group_col else None

    for c in cols:
        for k in lags:
            name = f"{c}{suffix}{int(k)}"
            out[name] = grouped[c].shift(int(k)) if grouped is not None else out[c].shift(int(k))
    return out


def add_rolling_features(
    df: pd.DataFrame,
    cols: Sequence[str],
    windows: Sequence[int],
    *,
    aggs: Sequence[str] = ("mean",),
    min_periods: int | None = None,
    group_col: str | None = None,
    date_col: str = "date",
    suffix: str = "_roll",
    causal: bool = True,
) -> pd.DataFrame:
    """
    Add rolling-window statistics for ``cols``.

    Parameters
    ----------
    df : pandas.DataFrame
    cols : sequence of str
        Columns to roll over.
    windows : sequence of int
        Window sizes in rows.
    aggs : sequence of {"mean","std","min","max","median","sum"}, default ("mean",)
    min_periods : int, optional
        Forwarded to pandas rolling; defaults to the window size.
    group_col : str, optional
        Roll within each group (e.g., per station) if provided.
    date_col : str, default "date"
        Datetime column used to sort before rolling.
    suffix : str, default "_roll"
        New columns are named ``f"{col}{suffix}{w}_{agg}"``.
    causal : bool, default True
        If True, use trailing windows (no look-ahead). If False, the window is
        centered (``center=True``) — only appropriate for diagnostics, not
        for predictors used in forecasting tasks.

    Returns
    -------
    pandas.DataFrame
        Copy with rolling-statistics columns appended.
    """
    if not cols or not windows:
        return df.copy()

    allowed = {"mean", "std", "min", "max", "median", "sum"}
    bad_aggs = [a for a in aggs if a not in allowed]
    if bad_aggs:
        raise ValueError(f"Unsupported aggs: {bad_aggs}. Choose from {sorted(allowed)}.")
    bad_cols = [c for c in cols if c not in df.columns]
    if bad_cols:
        raise ValueError(f"Columns not in df: {bad_cols}")
    if any(int(w) <= 0 for w in windows):
        raise ValueError("`windows` must contain positive integers.")

    out = _ensure_sorted_by_date(df, date_col=date_col)
    grouped = out.groupby(group_col, sort=False) if group_col else None

    for c in cols:
        for w in windows:
            w_int = int(w)
            mp = w_int if min_periods is None else int(min_periods)
            if grouped is not None:
                r = grouped[c].rolling(window=w_int, min_periods=mp, center=not causal)
            else:
                r = out[c].rolling(window=w_int, min_periods=mp, center=not causal)  # type: ignore[assignment]
            for agg in aggs:
                name = f"{c}{suffix}{w_int}_{agg}"
                vals = getattr(r, agg)()
                # When grouped, rolling returns a multi-index Series; align back.
                if grouped is not None:
                    vals = vals.reset_index(level=0, drop=True).sort_index()
                out[name] = vals.to_numpy() if hasattr(vals, "to_numpy") else np.asarray(vals)
    return out


def cyclical_encode(
    df: pd.DataFrame,
    col: str,
    period: float,
    *,
    drop: bool = False,
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Replace a periodic feature with its sine/cosine encoding.

    Useful for hour-of-day (period=24), day-of-week (period=7),
    day-of-year (period=365.25), month (period=12), wind direction (period=360).

    Parameters
    ----------
    df : pandas.DataFrame
    col : str
        Column to encode. Must be numeric and roughly in ``[0, period)``.
    period : float
        Cycle length (e.g., 24 for hour).
    drop : bool, default False
        If True, drop the original column.
    prefix : str, optional
        Prefix for the new columns; defaults to ``col``. Produces
        ``f"{prefix}_sin"`` and ``f"{prefix}_cos"``.

    Returns
    -------
    pandas.DataFrame
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not in df.")
    if period <= 0:
        raise ValueError("`period` must be positive.")

    out = df.copy()
    base = prefix or col
    theta = 2.0 * np.pi * pd.to_numeric(out[col], errors="coerce") / float(period)
    out[f"{base}_sin"] = np.sin(theta)
    out[f"{base}_cos"] = np.cos(theta)
    if drop:
        out = out.drop(columns=[col])
    return out


def wind_to_uv(
    speed: pd.Series | np.ndarray | Iterable[float],
    direction_deg: pd.Series | np.ndarray | Iterable[float],
    *,
    convention: str = "meteorological",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose wind speed + direction into orthogonal components (u, v).

    Parameters
    ----------
    speed : array-like
        Wind speed (any consistent unit; output uses the same unit).
    direction_deg : array-like
        Wind direction in degrees.
    convention : {"meteorological","oceanographic"}, default "meteorological"
        - "meteorological": direction is the angle the wind is *from*,
          measured clockwise from North. u is eastward, v is northward.
        - "oceanographic": direction is the angle the wind is *toward*.

    Returns
    -------
    (u, v) : tuple of numpy.ndarray
        Zonal (east-west) and meridional (north-south) components.
    """
    s = np.asarray(speed, dtype=float)
    d = np.asarray(direction_deg, dtype=float)
    if s.shape != d.shape:
        raise ValueError(f"`speed` and `direction_deg` shape mismatch: {s.shape} vs {d.shape}")

    rad = np.deg2rad(d)
    if convention == "meteorological":
        u = -s * np.sin(rad)
        v = -s * np.cos(rad)
    elif convention == "oceanographic":
        u = s * np.sin(rad)
        v = s * np.cos(rad)
    else:
        raise ValueError("`convention` must be 'meteorological' or 'oceanographic'.")
    return u, v


# ---------------------------------------------------------------------------
# Lag-structure diagnostics (ACF / PACF / pre-whitened CCF)
# ---------------------------------------------------------------------------


@dataclass
class LagDiagnostics:
    """Result of :func:`analyze_lag`.

    Attributes
    ----------
    target : str
        Name of the response series (e.g. the pollutant).
    driver : str or None
        Name of the driver series (e.g. a meteorological variable). ``None``
        when only the target's own lag structure was requested.
    n : int
        Effective number of observations used for the significance bands.
    alpha : float
        Significance level used for the two-sided bands.
    band : float
        White-noise significance threshold ``z_{1-alpha/2} / sqrt(n)``. A
        coefficient is flagged "significant" when ``abs(value) > band``.
    acf, pacf : pandas.DataFrame
        Columns ``lag`` (>= 0) and ``value`` for the target's autocorrelation
        and partial autocorrelation.
    ccf : pandas.DataFrame or None
        Columns ``lag`` and ``value`` for the cross-correlation between driver
        and target. ``lag = k > 0`` means the driver *leads* the target by
        ``k`` rows, i.e. ``corr(driver.shift(k), target)`` — the orientation
        you want for predictive lag features. ``None`` when ``driver`` is
        ``None``.
    target_ar_lags : list of int
        Significant PACF lags (>= 1) — suggested autoregressive lags of the
        target itself.
    driver_lags : list of int
        Significant CCF lags (>= 0) where the driver leads the target —
        suggested lags to feed to :func:`add_lag_features`.
    peak_lag : int or None
        Driver-leading lag (>= 0) with the largest absolute CCF.
    prewhitened : bool
        Whether the CCF was computed on pre-whitened series.
    """

    target: str
    driver: str | None
    n: int
    alpha: float
    band: float
    acf: pd.DataFrame
    pacf: pd.DataFrame
    ccf: pd.DataFrame | None = None
    target_ar_lags: list[int] = field(default_factory=list)
    driver_lags: list[int] = field(default_factory=list)
    peak_lag: int | None = None
    prewhitened: bool = False

    def summary(self) -> str:
        """Return a short human-readable summary of the recommended lags."""
        lines = [
            f"Lag diagnostics for target='{self.target}'"
            + (f", driver='{self.driver}'" if self.driver else ""),
            f"  n={self.n}, significance band=±{self.band:.3f} (alpha={self.alpha})",
            f"  suggested target AR lags (PACF): {self.target_ar_lags or '—'}",
        ]
        if self.driver is not None:
            tag = "pre-whitened" if self.prewhitened else "raw"
            lines.append(f"  suggested driver lags (CCF, {tag}): {self.driver_lags or '—'}")
            lines.append(f"  peak driver-leading lag: {self.peak_lag}")
        return "\n".join(lines)

    def plot(self, ax: Any = None):  # pragma: no cover - visual helper
        """Stem-plot the ACF, PACF and (if present) CCF with significance bands.

        Requires matplotlib. Returns the array of axes.
        """
        plt = require("matplotlib.pyplot", hint="pip install matplotlib")
        npanels = 3 if self.ccf is not None else 2
        if ax is None:
            _, axes = plt.subplots(npanels, 1, figsize=(8, 2.4 * npanels), sharex=False)
        else:
            axes = np.atleast_1d(ax)
        axes = np.atleast_1d(axes)

        def _stem(a, frame, title):
            a.stem(frame["lag"].to_numpy(), frame["value"].to_numpy(), basefmt=" ")
            a.axhline(0.0, color="0.5", lw=0.8)
            a.axhline(self.band, color="crimson", ls="--", lw=0.8)
            a.axhline(-self.band, color="crimson", ls="--", lw=0.8)
            a.set_title(title)
            a.set_ylabel("corr")

        _stem(axes[0], self.acf, f"ACF — {self.target}")
        _stem(axes[1], self.pacf, f"PACF — {self.target}")
        if self.ccf is not None:
            tag = "pre-whitened" if self.prewhitened else "raw"
            _stem(axes[2], self.ccf, f"CCF ({tag}) — {self.driver} → {self.target}")
            axes[2].set_xlabel("lag (driver leads target →)")
        axes[-1].set_xlabel(axes[-1].get_xlabel() or "lag")
        return axes


def _z_value(alpha: float) -> float:
    """Two-sided normal quantile ``z_{1-alpha/2}`` (uses SciPy if available)."""
    try:
        stats = require("scipy.stats")
        return float(stats.norm.ppf(1.0 - alpha / 2.0))
    except ImportError:
        # Fall back to the common 95% value; good enough for a rule-of-thumb band.
        return 1.959963984540054


def _clean_target(df: pd.DataFrame, col: str, date_col: str) -> pd.Series:
    """Sort by date and return ``col`` as a float Series indexed 0..n-1."""
    out = _ensure_sorted_by_date(df, date_col=date_col)
    if col not in out.columns:
        raise ValueError(f"Column '{col}' not in df.")
    return pd.to_numeric(out[col], errors="coerce").reset_index(drop=True)


def _ccf_at_lags(driver: pd.Series, target: pd.Series, lags: Sequence[int]) -> np.ndarray:
    """Pearson cross-correlation; lag k>0 => corr(driver.shift(k), target)."""
    vals = []
    for k in lags:
        pair = pd.concat([driver.shift(k), target], axis=1).dropna()
        vals.append(pair.iloc[:, 0].corr(pair.iloc[:, 1]) if len(pair) >= 3 else np.nan)
    return np.asarray(vals, dtype=float)


def _prewhiten(
    driver: pd.Series, target: pd.Series, max_ar: int
) -> tuple[pd.Series, pd.Series, int]:
    """Box-Jenkins pre-whitening.

    Fit an AR(p) to the driver (p chosen by AIC up to ``max_ar``), take its
    innovations as the white driver, then filter the (centred) target with the
    same AR polynomial. Returns ``(driver_white, target_filtered, p)`` aligned
    on a common index.
    """
    ar_model = require("statsmodels.tsa.ar_model", hint="pip install statsmodels")
    # AutoReg warns when the index is not a supported (datetime/period) index; we
    # only need residuals here, not forecasts, so the warning is noise. The index
    # is preserved (not reset) so the filtered driver stays aligned with target.
    d = driver.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = ar_model.ar_select_order(d, maxlag=max(1, int(max_ar)), ic="aic", old_names=False)
        p = max(sel.ar_lags) if sel.ar_lags else 1
        res = ar_model.AutoReg(d, lags=p, old_names=False).fit()
    phi = np.asarray(res.params[1 : p + 1], dtype=float)  # skip the constant

    driver_white = res.resid  # AR innovations of the driver (index p..end of d)

    # Apply the same AR polynomial (1 - phi(B)) to the centred target.
    yc = target - target.mean()
    y_f = yc.copy()
    for i in range(1, p + 1):
        y_f = y_f - phi[i - 1] * yc.shift(i)
    y_f = y_f.iloc[p:]

    common = driver_white.index.intersection(y_f.index)
    return driver_white.loc[common], y_f.loc[common], p


def analyze_lag(
    df: pd.DataFrame,
    target: str,
    driver: str | None = None,
    *,
    max_lag: int = 48,
    date_col: str = "date",
    prewhiten: bool = True,
    max_ar: int = 24,
    alpha: float = 0.05,
) -> LagDiagnostics:
    """Diagnose the lag structure of a target and its lead-lag with a driver.

    Computes the target's autocorrelation (ACF) and partial autocorrelation
    (PACF) to suggest autoregressive lags, and — when ``driver`` is given — the
    cross-correlation (CCF) between driver and target to suggest predictive
    driver lags. By default the CCF is computed on *pre-whitened* series so that
    shared seasonality / autocorrelation does not produce spurious peaks
    (Box-Jenkins pre-whitening).

    The series is sorted by ``date_col`` and assumed to be regularly spaced;
    non-finite values are dropped pairwise. For multi-site panels, call this
    once per site (pass a single-site slice).

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format dataset with a datetime column.
    target : str
        Response column (e.g. the pollutant).
    driver : str, optional
        Driver column (e.g. a meteorological variable). If omitted, only ACF
        and PACF of the target are returned.
    max_lag : int, default 48
        Maximum lag (rows) for ACF/PACF and the positive/negative CCF range.
    date_col : str, default "date"
        Datetime column used to sort before the analysis.
    prewhiten : bool, default True
        Pre-whiten before the CCF. Strongly recommended for autocorrelated,
        seasonal environmental series. Ignored when ``driver`` is ``None``.
    max_ar : int, default 24
        Maximum AR order considered for the pre-whitening filter (AIC-selected).
    alpha : float, default 0.05
        Two-sided significance level for the white-noise bands.

    Returns
    -------
    LagDiagnostics
        Tables of ACF/PACF/CCF plus the suggested lags. ``CCF`` lag ``k > 0``
        means the driver leads the target by ``k`` rows, matching the sign of
        ``add_lag_features(..., lags=[k])``.
    """
    if max_lag < 1:
        raise ValueError("`max_lag` must be >= 1.")
    stattools = require("statsmodels.tsa.stattools", hint="pip install statsmodels")

    y = _clean_target(df, target, date_col=date_col)
    z = _z_value(alpha)

    nlags = min(int(max_lag), max(1, len(y.dropna()) - 2))
    acf_vals = stattools.acf(y, nlags=nlags, missing="drop", fft=True)
    pacf_vals = stattools.pacf(y.dropna(), nlags=min(nlags, len(y.dropna()) // 2 - 1))

    acf_df = pd.DataFrame({"lag": np.arange(len(acf_vals)), "value": acf_vals})
    pacf_df = pd.DataFrame({"lag": np.arange(len(pacf_vals)), "value": pacf_vals})

    n_target = int(y.notna().sum())
    band = z / np.sqrt(max(n_target, 1))

    target_ar_lags = [
        int(k)
        for k, v in zip(pacf_df["lag"], pacf_df["value"], strict=False)
        if k >= 1 and np.isfinite(v) and abs(v) > band
    ]

    if driver is None:
        return LagDiagnostics(
            target=target,
            driver=None,
            n=n_target,
            alpha=alpha,
            band=float(band),
            acf=acf_df,
            pacf=pacf_df,
            target_ar_lags=target_ar_lags,
        )

    x = _clean_target(df, driver, date_col=date_col)
    prewhitened = bool(prewhiten)
    if prewhitened:
        try:
            x_use, y_use, _p = _prewhiten(x, y, max_ar=max_ar)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully to raw CCF
            log.warning("Pre-whitening failed (%s); falling back to raw CCF.", exc)
            prewhitened = False
            x_use, y_use = x, y
    else:
        x_use, y_use = x, y

    lags = list(range(-int(max_lag), int(max_lag) + 1))
    ccf_vals = _ccf_at_lags(x_use, y_use, lags)
    ccf_df = pd.DataFrame({"lag": lags, "value": ccf_vals})

    n_ccf = int(pd.concat([x_use, y_use], axis=1).dropna().shape[0])
    band_ccf = z / np.sqrt(max(n_ccf, 1))

    lead = ccf_df[(ccf_df["lag"] >= 0) & np.isfinite(ccf_df["value"])]
    driver_lags = [
        int(k) for k, v in zip(lead["lag"], lead["value"], strict=False) if abs(v) > band_ccf
    ]
    peak_lag = int(lead.loc[lead["value"].abs().idxmax(), "lag"]) if not lead.empty else None

    return LagDiagnostics(
        target=target,
        driver=driver,
        n=n_ccf,
        alpha=alpha,
        band=float(band_ccf),
        acf=acf_df,
        pacf=pacf_df,
        ccf=ccf_df,
        target_ar_lags=target_ar_lags,
        driver_lags=driver_lags,
        peak_lag=peak_lag,
        prewhitened=prewhitened,
    )
