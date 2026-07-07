"""Bundled example data for the GUI, generated on the fly.

Both generators are adapted from the tutorial notebooks' synthetic data
(`notebooks/_synth.py`) so the GUI ships no CSV files: "Load example" is a
one-click way to try every workflow immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_deweather_example(
    n_days: int = 365, start: str = "2020-01-01", seed: int = 42
) -> pd.DataFrame:
    """Hourly single-site air-quality + ERA5-style meteorology time series.

    The pollutant signals mix seasonal, rush-hour and wind-speed dependence,
    so training/normalising on them produces meaningful pictures.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days * 24, freq="h")
    n = len(dates)

    hour = dates.hour.to_numpy(dtype=float)
    doy = dates.dayofyear.to_numpy(dtype=float)

    season = np.sin(2 * np.pi * (doy - 80) / 365)  # peaks ~June
    rush = np.sin(2 * np.pi * (hour - 8) / 24)
    solar = np.maximum(0, np.sin(2 * np.pi * (hour - 6) / 24))

    t2m = 285 + 10 * season + 3 * solar + rng.normal(0, 1, n)
    u10 = rng.normal(-1.0, 2.0, n)
    v10 = rng.normal(0.0, 2.0, n)
    ws = np.sqrt(u10**2 + v10**2)
    blh = np.clip(300 + 1500 * solar + rng.normal(0, 80, n), 50, 3500)
    sp = 101325 + 400 * season + rng.normal(0, 200, n)
    ssrd = np.maximum(0, 2.5e6 * solar * (0.7 + 0.3 * season) + rng.normal(0, 5e4, n))
    tcc = np.clip(0.5 - 0.1 * season + rng.normal(0, 0.15, n), 0.0, 1.0)
    tp = np.maximum(0, rng.exponential(5e-5, n) * (rng.random(n) < 0.04))
    rh2m = np.clip(80 - 5 * season - 3 * solar + rng.normal(0, 4, n), 30, 100)

    pm25 = np.clip(
        20 - 6 * season + 8 * np.maximum(0, rush) - 1.5 * ws + rng.normal(0, 5, n), 1, 120
    )
    no2 = np.clip(
        30 + 10 * np.maximum(0, rush) - 4 * season - 1.0 * ws + rng.normal(0, 5, n), 0, 120
    )
    o3 = np.clip(30 + 20 * season + 10 * solar - 0.3 * no2 + rng.normal(0, 5, n), 0, 120)
    so2 = np.clip(5 - 2 * season + rng.normal(0, 2, n), 0, 30)
    co = np.clip(0.5 + 0.2 * np.maximum(0, rush) + rng.normal(0, 0.1, n), 0.1, 3.0)
    pm10 = np.clip(pm25 * 1.5 + rng.normal(0, 5, n), 0, 200)

    return pd.DataFrame(
        {
            "date": dates,
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "O3": o3,
            "SO2": so2,
            "CO": co,
            "u10": u10,
            "v10": v10,
            "t2m": t2m,
            "d2m": t2m - 8 + rng.normal(0, 1, n),
            "blh": blh,
            "sp": sp,
            "ssrd": ssrd,
            "tcc": tcc,
            "tp": tp,
            "rh2m": rh2m,
            "ws": ws,
            "wd": (np.degrees(np.arctan2(v10, u10)) + 360) % 360,
        }
    )


SCM_EXAMPLE_TREATED = "2+26 cities"
SCM_EXAMPLE_CUTOFF = "2015-10-23"
SCM_EXAMPLE_OUTCOME = "SO2wn"


def make_scm_example(
    start: str = "2015-01-01",
    end: str = "2016-07-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Weekly long panel for synthetic control (date, ID, pollutant columns).

    The treated unit "2+26 cities" shows a ~35 % step-down in ``SO2wn`` from
    2015-10-23; every other unit is a clean donor.
    """
    rng = np.random.default_rng(seed)

    donors = [
        "Dongguan",
        "Zhongshan",
        "Foshan",
        "Beihai",
        "Nanning",
        "Nanchang",
        "Xiamen",
        "Taizhou",
        "Ningbo",
        "Guangzhou",
        "Huizhou",
        "Hangzhou",
        "Liuzhou",
        "Shantou",
        "Jiangmen",
        "Heyuan",
        "Quanzhou",
        "Haikou",
        "Shenzhen",
        "Wenzhou",
        "Huzhou",
        "Zhuhai",
        "Fuzhou",
        "Shaoxing",
        "Zhaoqing",
        "Zhoushan",
        "Quzhou",
        "Jinhua",
        "Shaoguan",
        "Sanya",
    ]
    all_ids = [SCM_EXAMPLE_TREATED] + donors

    cutoff = pd.Timestamp(SCM_EXAMPLE_CUTOFF)
    dates = pd.date_range(start, end, freq="W-SUN")
    n_weeks = len(dates)

    common = 60 + 5 * np.sin(2 * np.pi * np.arange(n_weeks) / 52) + rng.normal(0, 3, n_weeks)

    frames = []
    for unit in all_ids:
        scale = rng.uniform(0.6, 1.4)
        so2wn = common * scale + rng.normal(0, 4, n_weeks)
        if unit == SCM_EXAMPLE_TREATED:
            so2wn[dates >= cutoff] *= 0.65
        so2wn = np.maximum(so2wn, 1.0)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ID": unit,
                    "SO2": so2wn + rng.normal(0, 2, n_weeks),
                    "SO2wn": so2wn,
                    "NO2wn": so2wn * 0.58 + rng.normal(0, 3, n_weeks),
                    "PM2.5wn": so2wn * 1.05 + rng.normal(0, 6, n_weeks),
                    "PM10wn": so2wn * 1.75 + rng.normal(0, 10, n_weeks),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    for col in df.columns:
        if col not in ("date", "ID"):
            df[col] = df[col].clip(lower=0)
    return df
