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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "SyntheticBenchmark",
    "make_synthetic_benchmark",
    "evaluate_benchmark",
]

_VALID_FAMILIES = frozenset(
    {
        "F1_additive",
        "F2_multiplicative",
        "F3_titration",
        "F4_nonstationary",
        "F5_transport",
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

    rng = np.random.default_rng(int(seed))
    t0 = pd.Timestamp(start_date)
    t_end = t0 + pd.Timedelta(days=int(n_days)) - pd.Timedelta(hours=1 if freq == "h" else 24)
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
    w_raw = (
        18.0 * np.exp(-met["ws"] / 3.2)
        + 12.0 * (350.0 / np.maximum(met["blh"], 80.0))
        + 0.35 * (met["t2m"] - 12.0)
        - 4.5 * np.log1p(met["tp"])
    )
    weather_add = w_raw - float(np.mean(w_raw))

    dil_raw = (
        (3.8 / np.maximum(met["ws"], 0.6)) ** 0.45
        * (550.0 / np.maximum(met["blh"], 80.0)) ** 0.35
        * np.exp(-0.08 * met["tp"])
    )
    phi_dil = dil_raw / float(np.mean(dil_raw))

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
        ox_emi_mol = np.maximum(
            18.0,
            (46.0 + 10.0 * np.sin(2.0 * np.pi * (doy - 100.0) / 365.25) + 0.4 * diurnal)
            * (1.0 + 0.25 * (float(trend_pct_per_year) / 100.0) * t_yr),
        )
        nox_emi_mol = np.maximum(
            4.0,
            (38.0 + 0.6 * diurnal + weekly) * trend_factor * ev_mult,
        )

        # Apply meteorological dilution to total Ox and NOx.
        ox_mol_obs = np.maximum(2.0, ox_emi_mol * (0.75 + 0.25 * phi_dil) + rng.normal(0.0, 0.5, n))
        nox_mol_obs = np.maximum(1.0, nox_emi_mol * phi_dil + rng.normal(0.0, 0.5, n))

        def _f_no2(nox_v: np.ndarray, ssrd_v: np.ndarray, t_v: np.ndarray) -> np.ndarray:
            logit = -0.35 + 0.85 * np.log1p(nox_v / 15.0) - 0.0022 * ssrd_v - 0.025 * (t_v - 12.0)
            return 1.0 / (1.0 + np.exp(-np.clip(logit, -10.0, 10.0)))

        f_obs = _f_no2(nox_mol_obs, met["ssrd"], met["t2m"])
        no2_mol_obs = np.clip(f_obs * ox_mol_obs, 0.0, np.minimum(ox_mol_obs, nox_mol_obs))
        o3_mol_obs = np.maximum(0.0, ox_mol_obs - no2_mol_obs)
        no_mol_obs = np.maximum(0.0, nox_mol_obs - no2_mol_obs)

        # True deweathered species under average meteorology (phi_dil = 1, mean met)
        # evaluated at true emission totals (ox_emi_mol, nox_emi_mol).
        f_dew_true = _f_no2(
            nox_emi_mol,
            np.full(n, float(np.mean(met["ssrd"]))),
            np.full(n, float(np.mean(met["t2m"]))),
        )
        no2_mol_true = np.clip(f_dew_true * ox_emi_mol, 0.0, np.minimum(ox_emi_mol, nox_emi_mol))
        o3_mol_true = np.maximum(0.0, ox_emi_mol - no2_mol_true)
        no_mol_true = np.maximum(0.0, nox_emi_mol - no2_mol_true)

        # Convert ppb-equivalent -> ug/m3 (standard 48.0, 46.0055, 30.0061 g/mol)
        mw_o3, mw_no2, mw_no = 48.0000, 46.0055, 30.0061
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
    else:  # F5_transport
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

    covs = [*met_covs, "date_unix", "day_julian", "weekday", "hour"]
    return SyntheticBenchmark(
        family=family,
        df=df_out,
        truth=truth,
        covariates=covs,
        met_covariates=met_covs,
        target=target_col,
        event_start=ev_start,
        event_end=ev_end,
        metadata={
            "n_days": n_days,
            "freq": freq,
            "event_reduction_pct": event_reduction_pct,
            "trend_pct_per_year": trend_pct_per_year,
            "seed": seed,
        },
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


def evaluate_benchmark(
    benchmark: SyntheticBenchmark,
    result: pd.DataFrame,
    *,
    col: str = "normalised",
    target_truth_col: str = "emission_true",
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

    return {
        "nrmse_pct": float(nrmse_pct),
        "bias_pct": float(bias_pct),
        "event_recovery_pct": float(event_recovery_pct),
        "absorption_pct": float(absorption_pct),
        "wrf_linear": float(wrf_linear),
        "pearson_r": float(pearson_r),
    }
