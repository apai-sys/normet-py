# src/normet/benchmarks.py
"""Synthetic ground-truth benchmarks for meteorological normalisation.

Provides controlled air-quality time-series generators with known analytical
emission trajectories, meteorological confounding, step events, photochemical
titration (:math:`\\mathrm{O_x\\text{–}NO_x}`), non-stationary trends, and
upwind transport episodes so estimators can be evaluated against exact ground
truth:

- ``F1_additive`` — additive meteorological response + trend + step event.
- ``F2_multiplicative`` — multiplicative boundary-layer / ventilation dilution.
- ``F3_titration`` — nonlinear :math:`\\mathrm{NO\\text{–}NO_2\\text{–}O_3}`
  Leighton photochemical titration where :math:`\\mathrm{NO_x}` cuts alter
  :math:`\\mathrm{O_3}` partitioning at fixed :math:`\\mathrm{O_x}`.
- ``F4_nonstationary`` — piecewise trend break + seasonal modulation.
- ``F5_transport`` — local emissions + intermittent upwind transport plume.
- ``F6_interaction_dial`` — smooth interpolation between additive (``lambda=0``)
  and multiplicative (``lambda=1``) action-weather coupling with optional floor clipping.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "SyntheticBenchmark",
    "make_synthetic_benchmark",
    "benchmark_interaction_diagnostics",
    "evaluate_benchmark",
]

_VALID_FAMILIES = frozenset(
    {
        "F1_additive",
        "F2_multiplicative",
        "F3_titration",
        "F4_nonstationary",
        "F5_transport",
        "F6_interaction_dial",
    }
)


@dataclass
class SyntheticBenchmark:
    """Container for one synthetic ground-truth benchmark dataset."""

    family: str
    df: pd.DataFrame
    truth: pd.DataFrame
    covariates: list[str]
    met_covariates: list[str]
    target: str
    event_start: pd.Timestamp
    event_end: pd.Timestamp
    metadata: dict[str, Any] = field(default_factory=dict)


def _generate_synoptic_meteorology(
    dates: pd.DatetimeIndex, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Generate physically plausible hourly meteorology with diurnal, synoptic, and seasonal cycles."""
    n = len(dates)
    hr = dates.hour.to_numpy(dtype=np.float64)
    doy = dates.dayofyear.to_numpy(dtype=np.float64)
    dt_h = max(float((dates[1] - dates[0]).total_seconds() / 3600.0), 1.0) if n > 1 else 1.0
    phi_ar = float(np.exp(-dt_h / 72.0))  # ~3-day synoptic persistence

    def _ar1(scale: float) -> np.ndarray:
        x = np.empty(n, dtype=np.float64)
        cur = float(rng.normal(0.0, scale))
        inn_sd = scale * float(np.sqrt(max(1.0 - phi_ar**2, 1e-6)))
        for t in range(n):
            cur = phi_ar * cur + float(rng.normal(0.0, inn_sd))
            x[t] = cur
        return x

    syn_wind = _ar1(1.4)
    syn_temp = _ar1(3.2)
    syn_pres = _ar1(8.0)

    ws = np.clip(
        4.2 + 1.1 * np.sin(2.0 * np.pi * (hr - 9.0) / 24.0) + syn_wind + rng.normal(0.0, 0.4, n),
        0.4,
        18.0,
    )
    wd_raw = (210.0 + 45.0 * np.sin(2.0 * np.pi * doy / 365.25) + _ar1(45.0)) % 360.0
    t2m = (
        12.0
        + 7.5 * np.sin(2.0 * np.pi * (doy - 105.0) / 365.25)
        + 4.0 * np.sin(2.0 * np.pi * (hr - 9.0) / 24.0)
        + syn_temp
    )
    rh = np.clip(
        72.0
        - 14.0 * np.sin(2.0 * np.pi * (hr - 9.0) / 24.0)
        - 0.8 * syn_temp
        + rng.normal(0.0, 4.0, n),
        18.0,
        98.0,
    )
    blh = np.clip(
        450.0
        + 480.0 * np.maximum(0.0, np.sin(np.pi * np.clip((hr - 6.0) / 13.0, 0.0, 1.0)))
        + 90.0 * (ws - 4.2)
        + rng.normal(0.0, 45.0, n),
        80.0,
        2400.0,
    )
    ssrd = np.clip(
        650.0
        * np.maximum(0.0, np.sin(np.pi * np.clip((hr - 6.0) / 12.0, 0.0, 1.0))) ** 1.3
        * (0.7 + 0.3 * np.sin(2.0 * np.pi * (doy - 80.0) / 365.25))
        * np.clip(1.0 - 0.004 * np.maximum(rh - 55.0, 0.0), 0.25, 1.0),
        0.0,
        950.0,
    )
    sp = 1013.25 + syn_pres
    rain_latent = -1.2 + 0.025 * (rh - 70.0) - 0.05 * syn_pres + rng.normal(0.0, 0.6, n)
    tp = np.where(rain_latent > 0.0, 0.4 * rain_latent**2, 0.0)
    return {
        "ws": ws,
        "wd": wd_raw,
        "t2m": t2m,
        "rh": rh,
        "blh": blh,
        "ssrd": ssrd,
        "sp": sp,
        "tp": tp,
    }


def _compute_weather_channels(
    met: Mapping[str, np.ndarray] | pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute zero-mean additive weather anomaly and unit-mean dilution factor."""
    ws = np.asarray(met["ws"], dtype=np.float64)
    blh = np.asarray(met["blh"], dtype=np.float64)
    t2m = np.asarray(met["t2m"], dtype=np.float64)
    tp = np.asarray(met["tp"], dtype=np.float64)

    w_raw = (
        18.0 * np.exp(-ws / 3.2)
        + 12.0 * (350.0 / np.maximum(blh, 80.0))
        + 0.35 * (t2m - 12.0)
        - 4.5 * np.log1p(tp)
    )
    weather_add = w_raw - float(np.mean(w_raw))

    dil_raw = (
        (3.8 / np.maximum(ws, 0.6)) ** 0.45
        * (550.0 / np.maximum(blh, 80.0)) ** 0.35
        * np.exp(-0.08 * tp)
    )
    phi_dil = dil_raw / float(np.mean(dil_raw))
    return weather_add, phi_dil


def make_synthetic_benchmark(
    family: str = "F1_additive",
    *,
    n_days: int = 180,
    freq: str = "h",
    start_date: str = "2023-01-01",
    event_start_day: int | None = None,
    event_duration_days: int | None = None,
    event_reduction_pct: float = -30.0,
    trend_pct_per_year: float = -5.0,
    noise_sd: float = 1.5,
    interaction_lambda: float = 0.5,
    floor_clip_quantile: float = 0.0,
    seed: int = 42,
) -> SyntheticBenchmark:
    """Create a synthetic air-quality benchmark with exact ground-truth components.

    Parameters
    ----------
    family : {"F1_additive", "F2_multiplicative", "F3_titration", "F4_nonstationary", "F5_transport"}
        Data-generating mechanism.
    n_days : int, default 180
        Total record length in days.
    freq : str, default "h"
        Sampling frequency (e.g. ``"h"`` or ``"D"``).
    start_date : str, default "2023-01-01"
        Starting timestamp.
    event_start_day : int, optional
        Day offset when the step intervention begins. Defaults to
        ``max(1, n_days // 2)`` (day 90 when ``n_days=180``).
    event_duration_days : int, optional
        Duration of the step intervention in days. Defaults to
        ``min(14, max(2, n_days // 6))`` (14 days when ``n_days=180``).
    event_reduction_pct : float, default -30.0
        Percentage emission shift during the step event (e.g. ``-30.0`` = 30% drop).
    trend_pct_per_year : float, default -5.0
        Annual emission trend percentage.
    noise_sd : float, default 1.5
        Irreducible observation noise standard deviation.
    seed : int, default 42
        RNG seed for reproducibility.

    Returns
    -------
    SyntheticBenchmark
    """
    if family not in _VALID_FAMILIES:
        raise ValueError(f"Unknown family {family!r}; expected one of {sorted(_VALID_FAMILIES)}")
    if n_days < 7:
        raise ValueError(f"`n_days` must be >= 7, got {n_days}")
    if not (0.0 <= float(interaction_lambda) <= 1.0):
        raise ValueError(f"`interaction_lambda` must lie in [0, 1], got {interaction_lambda!r}")
    if not (0.0 <= float(floor_clip_quantile) < 1.0):
        raise ValueError(f"`floor_clip_quantile` must lie in [0, 1), got {floor_clip_quantile!r}")

    rng = np.random.default_rng(int(seed))
    t0 = pd.Timestamp(start_date)
    # The record ends one *sampling step* before t0 + n_days, so the row count,
    # the span and ``metadata["n_days"]`` agree for every frequency. A fixed
    # 24-h deduction dropped a whole day for any sub-daily `freq` other than "h"
    # (e.g. "6h" produced 37 rows where 10 days holds 40).
    try:
        step = pd.tseries.frequencies.to_offset(freq)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid `freq` {freq!r}: {exc}") from exc
    t_end = t0 + pd.Timedelta(days=int(n_days)) - step
    dates = pd.date_range(t0, t_end, freq=freq)
    n = len(dates)

    met = _generate_synoptic_meteorology(dates, rng)
    t_yr = (dates - t0).total_seconds().to_numpy(dtype=np.float64) / (365.25 * 86400.0)
    hr = dates.hour.to_numpy(dtype=np.float64)
    dw = dates.dayofweek.to_numpy(dtype=np.float64)
    doy = dates.dayofyear.to_numpy(dtype=np.float64)

    ev_start_d = int(event_start_day) if event_start_day is not None else max(1, int(n_days) // 2)
    ev_dur_d = (
        int(event_duration_days)
        if event_duration_days is not None
        else min(14, max(2, int(n_days) // 6))
    )
    ev_start = t0 + pd.Timedelta(days=ev_start_d)
    ev_end = ev_start + pd.Timedelta(days=ev_dur_d)
    ev_mask = (dates >= ev_start) & (dates < ev_end)
    ev_mult = np.where(ev_mask, 1.0 + float(event_reduction_pct) / 100.0, 1.0)

    diurnal = 5.0 * np.sin(2.0 * np.pi * (hr - 6.0) / 24.0) - 3.0 * np.cos(4.0 * np.pi * hr / 24.0)
    weekly = np.where(dw < 5, 2.5, -4.5)
    seasonal = 4.0 * np.cos(2.0 * np.pi * (doy - 15.0) / 365.25)

    if family == "F4_nonstationary":
        mid_yr = 0.5 * float(n_days) / 365.25
        slope_1 = float(trend_pct_per_year) / 100.0
        slope_2 = -2.0 * slope_1
        trend_factor = np.where(
            t_yr <= mid_yr,
            1.0 + slope_1 * t_yr,
            1.0 + slope_1 * mid_yr + slope_2 * (t_yr - mid_yr),
        )
        seas_amp = (1.0 + 0.35 * t_yr) * seasonal
    else:
        trend_factor = 1.0 + (float(trend_pct_per_year) / 100.0) * t_yr
        seas_amp = seasonal

    baseline_true = np.maximum(5.0, (42.0 + seas_amp + diurnal + weekly) * trend_factor)
    emission_true = baseline_true * ev_mult

    # Zero-mean additive weather anomaly & positive unit-mean dilution factor.
    weather_add, phi_dil = _compute_weather_channels(met)

    df_out = pd.DataFrame({"date": dates, **met})
    df_out["date_unix"] = dates.astype("int64") // 10**9
    df_out["day_julian"] = dates.dayofyear
    df_out["weekday"] = dates.dayofweek + 1
    df_out["hour"] = dates.hour

    met_covs = ["ws", "wd", "t2m", "rh", "blh", "ssrd", "sp", "tp"]
    eps = rng.normal(0.0, float(noise_sd), n)

    if family == "F1_additive":
        weather_true = weather_add
        obs = np.maximum(0.5, emission_true + weather_true + eps)
        df_out["value"] = obs
        truth = pd.DataFrame(
            {
                "observed": obs,
                "emission_true": emission_true,
                "baseline_true": baseline_true,
                "weather_true": weather_true,
                "event_mask": ev_mask,
            },
            index=dates,
        )
        target_col = "value"
    elif family == "F2_multiplicative":
        obs = np.maximum(0.5, emission_true * phi_dil + eps)
        weather_true = emission_true * (phi_dil - 1.0)
        df_out["value"] = obs
        truth = pd.DataFrame(
            {
                "observed": obs,
                "emission_true": emission_true,
                "baseline_true": baseline_true,
                "weather_true": weather_true,
                "event_mask": ev_mask,
            },
            index=dates,
        )
        target_col = "value"
    elif family == "F3_titration":
        # Molar units (ppb-equivalent): conserved Ox = O3 + NO2, NOx = NO + NO2.
        mw_o3, mw_no2, mw_no = 48.0000, 46.0055, 30.0061
        ox_emi_mol = np.maximum(
            18.0,
            (46.0 + 10.0 * np.sin(2.0 * np.pi * (doy - 100.0) / 365.25) + 0.4 * diurnal)
            * (1.0 + 0.25 * (float(trend_pct_per_year) / 100.0) * t_yr),
        )
        nox_emi_mol = np.maximum(
            4.0,
            (38.0 + 0.6 * diurnal + weekly) * trend_factor * ev_mult,
        )

        # Apply meteorological dilution to total Ox and NOx. Observation noise
        # enters the conserved totals, so the species still add up exactly:
        # `noise_sd` is documented in target units (O3 ug/m3), and the same
        # absolute noise on NOx is converted with its own molar mass. The old
        # hard-coded 0.5 ppb ignored `noise_sd` entirely.
        noise_ox_mol = eps / mw_o3
        noise_nox_mol = rng.normal(0.0, float(noise_sd) / mw_no2, n)
        ox_mol_obs = np.maximum(2.0, ox_emi_mol * (0.75 + 0.25 * phi_dil) + noise_ox_mol)
        nox_mol_obs = np.maximum(1.0, nox_emi_mol * phi_dil + noise_nox_mol)

        def _f_no2(nox_v: np.ndarray, ssrd_v: np.ndarray, t_v: np.ndarray) -> np.ndarray:
            logit = -0.35 + 0.85 * np.log1p(nox_v / 15.0) - 0.0022 * ssrd_v - 0.025 * (t_v - 12.0)
            return 1.0 / (1.0 + np.exp(-np.clip(logit, -10.0, 10.0)))

        f_obs = _f_no2(nox_mol_obs, met["ssrd"], met["t2m"])
        no2_mol_obs = np.clip(f_obs * ox_mol_obs, 0.0, np.minimum(ox_mol_obs, nox_mol_obs))
        o3_mol_obs = np.maximum(0.0, ox_mol_obs - no2_mol_obs)
        no_mol_obs = np.maximum(0.0, nox_mol_obs - no2_mol_obs)

        # Ground truth is the estimator's own estimand: the pool expectation
        # E[O3 | emissions], i.e. the response averaged over resampled
        # meteorology. The old plug-in at mean met / mean dilution is not the
        # same quantity -- f_NO2 is nonlinear in NOx and downwelling radiation
        # and correlated with the dilution factor, so Jensen and covariance
        # terms shift it. Monte Carlo over the record's own met pool matches
        # what `normalise` marginalises over. Observation noise is mean-zero
        # and excluded (the model estimates its conditional mean).
        truth_draws = 512
        truth_rng = np.random.default_rng(int(seed) + 991)
        pool_pos = truth_rng.integers(0, n, size=truth_draws)
        phi_t = phi_dil[pool_pos]
        ssrd_t = met["ssrd"][pool_pos]
        t2m_t = met["t2m"][pool_pos]
        ox_t = ox_emi_mol[:, None] * (0.75 + 0.25 * phi_t[None, :])
        nox_t = nox_emi_mol[:, None] * phi_t[None, :]
        f_t = _f_no2(nox_t, ssrd_t[None, :], t2m_t[None, :])
        no2_t = np.clip(f_t * ox_t, 0.0, np.minimum(ox_t, nox_t))
        o3_t = np.maximum(0.0, ox_t - no2_t)
        o3_mol_true = o3_t.mean(axis=1)
        no2_mol_true = no2_t.mean(axis=1)
        no_mol_true = (nox_t - no2_t).mean(axis=1)

        # Convert ppb-equivalent -> ug/m3 (standard 48.0, 46.0055, 30.0061 g/mol)
        df_out["O3"] = o3_mol_obs * mw_o3
        df_out["NO2"] = no2_mol_obs * mw_no2
        df_out["NO"] = no_mol_obs * mw_no
        df_out["Ox"] = ox_mol_obs * mw_o3
        df_out["NOx"] = nox_mol_obs * mw_no2
        df_out["value"] = df_out["O3"]

        truth = pd.DataFrame(
            {
                "observed": df_out["O3"].to_numpy(dtype=float),
                "emission_true": o3_mol_true * mw_o3,
                "O3_true_deweathered": o3_mol_true * mw_o3,
                "NO2_true_deweathered": no2_mol_true * mw_no2,
                "NO_true_deweathered": no_mol_true * mw_no,
                "Ox_true_deweathered": ox_emi_mol * mw_o3,
                "NOx_true_deweathered": nox_emi_mol * mw_no2,
                "weather_true": (o3_mol_obs - o3_mol_true) * mw_o3,
                "event_mask": ev_mask,
            },
            index=dates,
        )
        target_col = "O3"
    elif family == "F4_nonstationary":
        weather_true = weather_add
        obs = np.maximum(0.5, emission_true + weather_true + eps)
        df_out["value"] = obs
        truth = pd.DataFrame(
            {
                "observed": obs,
                "emission_true": emission_true,
                "baseline_true": baseline_true,
                "weather_true": weather_true,
                "event_mask": ev_mask,
            },
            index=dates,
        )
        target_col = "value"
        floor_value = 0.5
    elif family == "F5_transport":
        # Intermittent upwind transport plume correlated with westerly winds (180..300 deg)
        wd_rad = np.deg2rad(met["wd"] - 240.0)
        upwind_gate = np.maximum(0.0, np.cos(wd_rad)) ** 2
        fp_upwind = np.clip(upwind_gate * (0.4 + 0.6 * rng.exponential(0.5, n)), 0.0, 2.5)
        df_out["traj_footprint_upwind"] = fp_upwind
        met_covs.append("traj_footprint_upwind")
        transport_contrib = 14.0 * (fp_upwind - float(np.mean(fp_upwind)))
        weather_true = weather_add + transport_contrib
        obs = np.maximum(0.5, emission_true + weather_true + eps)
        df_out["value"] = obs
        truth = pd.DataFrame(
            {
                "observed": obs,
                "emission_true": emission_true,
                "baseline_true": baseline_true,
                "weather_true": weather_true,
                "transport_true": transport_contrib,
                "event_mask": ev_mask,
            },
            index=dates,
        )
        target_col = "value"
        floor_value = 0.5
    else:  # F6_interaction_dial
        lam = float(interaction_lambda)
        y_unclipped = (
            (1.0 - lam) * (emission_true + weather_add) + lam * (emission_true * phi_dil) + eps
        )
        floor_value = (
            max(0.5, float(np.quantile(y_unclipped, float(floor_clip_quantile))))
            if float(floor_clip_quantile) > 0.0
            else 0.5
        )
        obs = np.maximum(floor_value, y_unclipped)
        weather_true = (1.0 - lam) * weather_add + lam * emission_true * (phi_dil - 1.0)
        df_out["value"] = obs
        truth = pd.DataFrame(
            {
                "observed": obs,
                "emission_true": emission_true,
                "baseline_true": baseline_true,
                "weather_true": weather_true,
                "event_mask": ev_mask,
            },
            index=dates,
        )
        target_col = "value"

    if family not in ("F5_transport", "F6_interaction_dial"):
        floor_value = 0.5

    covs = [*met_covs, "date_unix", "day_julian", "weekday", "hour"]
    meta: dict[str, Any] = {
        "n_days": n_days,
        "freq": freq,
        "event_reduction_pct": event_reduction_pct,
        "trend_pct_per_year": trend_pct_per_year,
        "seed": seed,
    }
    if family == "F6_interaction_dial":
        meta["interaction_lambda"] = float(interaction_lambda)
        meta["floor_clip_quantile"] = float(floor_clip_quantile)
        meta["floor_value"] = float(floor_value)
    return SyntheticBenchmark(
        family=family,
        df=df_out,
        truth=truth,
        covariates=covs,
        met_covariates=met_covs,
        target=target_col,
        event_start=ev_start,
        event_end=ev_end,
        metadata=meta,
    )


def _linear_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Ridge-stabilized linear R^2 bounded below at 0."""
    if len(y) < 5 or X.shape[1] == 0:
        return 0.0
    mu = X.mean(axis=0)
    sd = np.where(X.std(axis=0) > 1e-12, X.std(axis=0), 1.0)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    reg = np.eye(Z.shape[1]) * 1e-4
    reg[0, 0] = 0.0
    coef = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
    pred = Z @ coef
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if sst <= 1e-12:
        return 0.0
    sse = float(np.sum((y - pred) ** 2))
    return float(max(0.0, 1.0 - sse / sst))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(m.sum()) < 2:
        return float("nan")
    yt = y_true[m]
    yp = y_pred[m]
    var_t = float(np.var(yt))
    if var_t <= 1e-12:
        return float("nan")
    mse = float(np.mean((yt - yp) ** 2))
    return float(1.0 - mse / var_t)


def _anova_interaction_ratio(Z: np.ndarray) -> float:
    """Fraction of non-weather outcome variance due to action-weather interaction."""
    col_mean = Z.mean(axis=0, keepdims=True)
    denom = float(np.mean((Z - col_mean) ** 2))
    if denom <= 1e-12:
        return 0.0
    grand = float(Z.mean())
    row_mean = Z.mean(axis=1, keepdims=True)
    inter = Z - row_mean - col_mean + grand
    return float(np.clip(np.mean(inter**2) / denom, 0.0, 1.0))


def benchmark_interaction_diagnostics(
    benchmark: SyntheticBenchmark,
    *,
    n_draws: int = 256,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute exact simulator-level separability and interaction diagnostics.

    Uses the known synthetic mechanism to measure ground-truth action-weather
    coupling ``rho_raw`` / ``rho_log``, no-outcome oracle R^2, factual-copy R^2,
    ideal residual-reuse R^2 in raw and ``log1p`` coordinates, and lower-floor
    saturation rate.
    """
    df = benchmark.df
    truth = benchmark.truth
    n = len(df)
    if n < 5:
        raise ValueError("Need at least 5 timestamps for benchmark_interaction_diagnostics.")

    weather_add, phi_dil = _compute_weather_channels(df)
    rng = np.random.default_rng(int(seed))
    pool_idx = rng.integers(0, n, size=max(16, int(n_draws)))

    obs = pd.to_numeric(truth["observed"], errors="coerce").to_numpy(dtype=float)
    emi = pd.to_numeric(truth["emission_true"], errors="coerce").to_numpy(dtype=float)
    fam = benchmark.family

    if fam in ("F1_additive", "F4_nonstationary"):
        base = pd.to_numeric(truth["baseline_true"], errors="coerce").to_numpy(dtype=float)
        Z = np.maximum(0.5, emi[:, None] + weather_add[None, pool_idx])
        mu_base_draws = np.maximum(0.5, base[:, None] + weather_add[None, pool_idx])
        mu_base_fact = np.maximum(0.5, base + weather_add)
    elif fam == "F2_multiplicative":
        base = pd.to_numeric(truth["baseline_true"], errors="coerce").to_numpy(dtype=float)
        Z = np.maximum(0.5, emi[:, None] * phi_dil[None, pool_idx])
        mu_base_draws = np.maximum(0.5, base[:, None] * phi_dil[None, pool_idx])
        mu_base_fact = np.maximum(0.5, base * phi_dil)
    elif fam == "F5_transport":
        base = pd.to_numeric(truth["baseline_true"], errors="coerce").to_numpy(dtype=float)
        trans = pd.to_numeric(truth["transport_true"], errors="coerce").to_numpy(dtype=float)
        w_eff = weather_add + trans
        Z = np.maximum(0.5, emi[:, None] + w_eff[None, pool_idx])
        mu_base_draws = np.maximum(0.5, base[:, None] + w_eff[None, pool_idx])
        mu_base_fact = np.maximum(0.5, base + w_eff)
    elif fam == "F6_interaction_dial":
        base = pd.to_numeric(truth["baseline_true"], errors="coerce").to_numpy(dtype=float)
        lam = float(benchmark.metadata.get("interaction_lambda", 0.5))
        floor_v = float(benchmark.metadata.get("floor_value", 0.5))
        Z = np.maximum(
            floor_v,
            (1.0 - lam) * (emi[:, None] + weather_add[None, pool_idx])
            + lam * (emi[:, None] * phi_dil[None, pool_idx]),
        )
        mu_base_draws = np.maximum(
            floor_v,
            (1.0 - lam) * (base[:, None] + weather_add[None, pool_idx])
            + lam * (base[:, None] * phi_dil[None, pool_idx]),
        )
        mu_base_fact = np.maximum(
            floor_v,
            (1.0 - lam) * (base + weather_add) + lam * (base * phi_dil),
        )
    elif fam == "F3_titration":
        mw_o3, mw_no2 = 48.0000, 46.0055
        ox_emi_mol = (
            pd.to_numeric(truth["Ox_true_deweathered"], errors="coerce").to_numpy(dtype=float)
            / mw_o3
        )
        nox_emi_mol = (
            pd.to_numeric(truth["NOx_true_deweathered"], errors="coerce").to_numpy(dtype=float)
            / mw_no2
        )
        ev_mask = truth["event_mask"].to_numpy(dtype=bool)
        ev_red = float(benchmark.metadata.get("event_reduction_pct", -30.0))
        ev_mult = np.where(ev_mask, 1.0 + ev_red / 100.0, 1.0)
        nox_base_mol = np.where(np.abs(ev_mult) > 1e-6, nox_emi_mol / ev_mult, nox_emi_mol)

        ssrd_arr = df["ssrd"].to_numpy(dtype=float)
        t2m_arr = df["t2m"].to_numpy(dtype=float)

        def _f_no2(nox_v: np.ndarray, ssrd_v: np.ndarray, t_v: np.ndarray) -> np.ndarray:
            logit = -0.35 + 0.85 * np.log1p(nox_v / 15.0) - 0.0022 * ssrd_v - 0.025 * (t_v - 12.0)
            return 1.0 / (1.0 + np.exp(-np.clip(logit, -10.0, 10.0)))

        def _o3_grid(
            ox_v: np.ndarray, nox_v: np.ndarray, phi_v: np.ndarray, s_v: np.ndarray, tv: np.ndarray
        ) -> np.ndarray:
            ox_g = ox_v * (0.75 + 0.25 * phi_v)
            nox_g = nox_v * phi_v
            f_g = _f_no2(nox_g, s_v, tv)
            no2_g = np.clip(f_g * ox_g, 0.0, np.minimum(ox_g, nox_g))
            return np.maximum(0.0, ox_g - no2_g) * mw_o3

        Z = _o3_grid(
            ox_emi_mol[:, None],
            nox_emi_mol[:, None],
            phi_dil[None, pool_idx],
            ssrd_arr[None, pool_idx],
            t2m_arr[None, pool_idx],
        )
        mu_base_draws = _o3_grid(
            ox_emi_mol[:, None],
            nox_base_mol[:, None],
            phi_dil[None, pool_idx],
            ssrd_arr[None, pool_idx],
            t2m_arr[None, pool_idx],
        )
        mu_base_fact = _o3_grid(ox_emi_mol, nox_base_mol, phi_dil, ssrd_arr, t2m_arr)
    else:
        raise ValueError(f"Unsupported benchmark family: {fam!r}")

    rho_raw = _anova_interaction_ratio(Z)
    rho_log = _anova_interaction_ratio(np.log1p(np.maximum(Z, 0.0)))

    mu_base_cf = mu_base_draws.mean(axis=1)
    oracle_no_outcome_r2 = _r2_score(emi, mu_base_cf)
    factual_copy_r2 = _r2_score(emi, obs)
    ideal_res_raw = mu_base_cf + (obs - mu_base_fact)
    ideal_residual_r2_raw = _r2_score(emi, ideal_res_raw)

    z_cf = np.log1p(np.maximum(mu_base_cf, 0.0))
    z_obs = np.log1p(np.maximum(obs, 0.0))
    z_fact = np.log1p(np.maximum(mu_base_fact, 0.0))
    ideal_res_log = np.maximum(np.expm1(z_cf + (z_obs - z_fact)), 0.0)
    ideal_residual_r2_log = _r2_score(emi, ideal_res_log)

    lo = float(np.nanmin(obs))
    hi = float(np.nanmax(obs))
    rng_y = hi - lo
    med_abs = float(np.median(np.abs(obs)))
    tol_zero = max(1e-6, 0.01 * med_abs) if med_abs > 1e-6 else 1e-6
    floor_rate = float(np.mean(obs <= tol_zero))
    if floor_rate == 0.0 and rng_y > 1e-12:
        tol_lo = max(1e-7, 1e-5 * rng_y)
        n_lo = int(np.sum(obs <= lo + tol_lo))
        if n_lo >= max(3, int(0.02 * n)):
            floor_rate = float(n_lo / n)

    return {
        "family": fam,
        "rho_raw": float(rho_raw),
        "rho_log": float(rho_log),
        "oracle_no_outcome_r2": float(oracle_no_outcome_r2),
        "factual_copy_r2": float(factual_copy_r2),
        "ideal_residual_r2_raw": float(ideal_residual_r2_raw),
        "ideal_residual_r2_log": float(ideal_residual_r2_log),
        "floor_saturation_rate": float(floor_rate),
    }


def evaluate_benchmark(
    benchmark: SyntheticBenchmark,
    result: pd.DataFrame,
    *,
    col: str = "normalised",
    target_truth_col: str = "emission_true",
    shuffle_control: bool = False,
    seed: int = 0,
) -> dict[str, float]:
    """Score a normalised series against known synthetic benchmark ground truth."""
    if col not in result.columns:
        raise KeyError(f"Column {col!r} not found in `result`.")
    if target_truth_col not in benchmark.truth.columns:
        raise KeyError(f"Column {target_truth_col!r} not found in `benchmark.truth`.")

    res_s = pd.to_numeric(result[col], errors="coerce")
    if not isinstance(res_s.index, pd.DatetimeIndex):
        if "date" in result.columns:
            res_s = pd.Series(res_s.to_numpy(dtype=float), index=pd.to_datetime(result["date"]))
        else:
            raise ValueError("`result` must have a DatetimeIndex or 'date' column.")

    common = benchmark.truth.index.intersection(res_s.index)
    if common.empty:
        raise ValueError("`result` and `benchmark.truth` share no overlapping timestamps.")

    est = res_s.reindex(common).to_numpy(dtype=float)
    truth = pd.to_numeric(benchmark.truth.loc[common, target_truth_col], errors="coerce").to_numpy(
        dtype=float
    )
    obs = pd.to_numeric(benchmark.truth.loc[common, "observed"], errors="coerce").to_numpy(
        dtype=float
    )
    ev_m = benchmark.truth.loc[common, "event_mask"].to_numpy(dtype=bool)

    fin = np.isfinite(est) & np.isfinite(truth) & np.isfinite(obs)
    if int(fin.sum()) < 5:
        raise ValueError("Fewer than 5 overlapping finite points to evaluate benchmark.")

    est_f, truth_f, obs_f, ev_f = est[fin], truth[fin], obs[fin], ev_m[fin]
    mean_truth = max(float(np.mean(truth_f)), 1e-12)
    rmse = float(np.sqrt(np.mean((est_f - truth_f) ** 2)))
    nrmse_pct = 100.0 * rmse / mean_truth
    bias_pct = 100.0 * float(np.mean(est_f - truth_f)) / mean_truth

    if float(np.std(est_f)) > 1e-12 and float(np.std(truth_f)) > 1e-12:
        pearson_r = float(np.corrcoef(est_f, truth_f)[0, 1])
    else:
        pearson_r = float("nan")

    if np.any(ev_f) and np.any(~ev_f):
        true_jump = float(np.mean(truth_f[ev_f]) - np.mean(truth_f[~ev_f]))
        est_jump = float(np.mean(est_f[ev_f]) - np.mean(est_f[~ev_f]))
        if abs(true_jump) > 1e-6:
            event_recovery_pct = 100.0 * (est_jump / true_jump)
            absorption_pct = 100.0 - event_recovery_pct
        else:
            event_recovery_pct = float("nan")
            absorption_pct = float("nan")
    else:
        event_recovery_pct = float("nan")
        absorption_pct = float("nan")

    met_df = benchmark.df.set_index("date").reindex(common).iloc[fin][benchmark.met_covariates]
    X_met = met_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    err_obs = obs_f - truth_f
    err_est = est_f - truth_f
    if "weather_true" in benchmark.truth.columns:
        w_true = pd.to_numeric(
            benchmark.truth.loc[common, "weather_true"], errors="coerce"
        ).to_numpy(dtype=float)[fin]
        X_w = np.column_stack([X_met, np.nan_to_num(w_true, nan=0.0)])
    else:
        X_w = X_met
    v_obs = _linear_r2(X_w, err_obs) * float(np.var(err_obs))
    v_est = _linear_r2(X_w, err_est) * float(np.var(err_est))
    wrf_linear = float(max(0.0, 1.0 - v_est / v_obs)) if v_obs > 1e-9 else float("nan")
    cf_r2 = _r2_score(truth_f, est_f)
    factual_copy_r2 = _r2_score(truth_f, obs_f)

    out: dict[str, float] = {
        "nrmse_pct": float(nrmse_pct),
        "bias_pct": float(bias_pct),
        "event_recovery_pct": float(event_recovery_pct),
        "absorption_pct": float(absorption_pct),
        "wrf_linear": float(wrf_linear),
        "pearson_r": float(pearson_r),
        "cf_r2": float(cf_r2),
        "factual_copy_r2": float(factual_copy_r2),
    }
    if shuffle_control:
        rng_shuf = np.random.default_rng(int(seed))
        perm = rng_shuf.permutation(len(est_f))
        out["shuffle_control_r2"] = float(_r2_score(truth_f, est_f[perm]))
    return out
