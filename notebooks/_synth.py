"""Synthetic data generators for normet tutorial notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_my1_data(n_days: int = 365, start: str = "2020-01-01", seed: int = 42) -> pd.DataFrame:
    """Return a MY1-like hourly air quality + ERA5 meteorology DataFrame.

    Covers *n_days* days starting from *start*.  All pollutant and met columns
    that appear in the original MY1 dataset are present so existing notebook
    code that inspects ``df.columns`` continues to work.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days * 24, freq="h")
    n = len(dates)

    hour = dates.hour.to_numpy(dtype=float)
    doy = dates.dayofyear.to_numpy(dtype=float)

    # Seasonal (summer peak) and diurnal signals
    season = np.sin(2 * np.pi * (doy - 80) / 365)   # peaks ~June
    rush = np.sin(2 * np.pi * (hour - 8) / 24)       # peaks ~14 h
    solar_angle = np.maximum(0, np.sin(2 * np.pi * (hour - 6) / 24))

    # ── ERA5 met ──────────────────────────────────────────────────────────────
    t2m = 285 + 10 * season + 3 * solar_angle + rng.normal(0, 1, n)
    d2m = t2m - 8 + rng.normal(0, 1, n)
    u10 = rng.normal(-1.0, 2.0, n)
    v10 = rng.normal(0.0, 2.0, n)
    ws_arr = np.sqrt(u10 ** 2 + v10 ** 2)
    blh = np.clip(
        300 + 1500 * solar_angle + rng.normal(0, 80, n),
        50, 3500,
    )
    sp = 101325 + 400 * season + rng.normal(0, 200, n)
    ssrd = np.maximum(0, 2.5e6 * solar_angle * (0.7 + 0.3 * season) + rng.normal(0, 5e4, n))
    tcc = np.clip(0.5 - 0.1 * season + rng.normal(0, 0.15, n), 0.0, 1.0)
    tp_arr = np.maximum(0, rng.exponential(5e-5, n) * (rng.random(n) < 0.04))
    rh2m = np.clip(80 - 5 * season - 3 * solar_angle + rng.normal(0, 4, n), 30, 100)

    # ── Pollutants ────────────────────────────────────────────────────────────
    pm25 = np.clip(
        20 - 6 * season + 8 * np.maximum(0, rush) - 1.5 * ws_arr + rng.normal(0, 5, n),
        1, 120,
    )
    no = np.clip(25 + 15 * np.maximum(0, rush) - 5 * season + rng.normal(0, 6, n), 0, 200)
    no2 = np.clip(30 + 10 * np.maximum(0, rush) - 4 * season + rng.normal(0, 5, n), 0, 120)
    nox = no + no2 * 46 / 30          # approximate NOX as NO
    o3 = np.clip(30 + 20 * season + 10 * solar_angle - 0.3 * no2 + rng.normal(0, 5, n), 0, 120)
    ox = o3 + no2
    so2 = np.clip(5 - 2 * season + rng.normal(0, 2, n), 0, 30)
    co = np.clip(0.5 + 0.2 * np.maximum(0, rush) + rng.normal(0, 0.1, n), 0.1, 3.0)
    pm10 = np.clip(pm25 * 1.5 + rng.normal(0, 5, n), 0, 200)
    nv10 = pm10 * 0.85
    v10_pm = pm10 * 0.15
    pm25v = pm25 * 0.12
    pm25nv = pm25 * 0.88

    # ── VOC species (small realistic values) ─────────────────────────────────
    def _voc(base, scale=1.0):
        return np.clip(base + scale * rng.exponential(base * 0.5, n), 0, base * 10)

    ethane = _voc(2.0)
    ethene = _voc(1.2)
    ethyne = _voc(0.8)
    propane = _voc(3.0)
    propene = _voc(0.9)
    ibutane = _voc(1.5)
    nbutane = _voc(2.2)
    butene1 = _voc(0.15)
    t2butene = _voc(0.07)
    c2butene = _voc(0.05)
    ipentane = _voc(0.9)
    npentane = _voc(0.5)
    t2penten = _voc(0.04)
    penten1 = _voc(0.08)
    mepent2 = _voc(0.2)
    isoprene = _voc(0.03)
    nhexane = _voc(0.2)
    nheptane = _voc(0.12)
    ioctane = _voc(0.18)
    noctane = _voc(0.07)
    benzene = _voc(0.7)
    toluene = _voc(1.0)
    ethbenz = _voc(0.2)
    mpxylene = _voc(0.4)
    oxylene = _voc(0.2)
    tmb124 = _voc(0.15)
    tmb135 = _voc(0.05)

    # ── Surface met ───────────────────────────────────────────────────────────
    wd = (np.degrees(np.arctan2(v10, u10)) + 360) % 360
    temp = t2m - 273.15
    at10 = np.clip(pm10 * 0.3 + rng.normal(0, 1, n), 0, 30)
    ap10 = sp / 100 + rng.normal(0, 0.5, n)
    at25 = at10 * 0.9
    ap25 = ap10 - 1 + rng.normal(0, 0.2, n)

    df = pd.DataFrame(
        {
            "date": dates,
            "O3": o3,
            "NO": no,
            "NO2": no2,
            "NOXasNO2": nox,
            "SO2": so2,
            "CO": co,
            "PM10": pm10,
            "NV10": nv10,
            "V10": v10_pm,
            "PM2.5": pm25,
            "NV2.5": pm25nv,
            "V2.5": pm25v,
            "ETHANE": ethane,
            "ETHENE": ethene,
            "ETHYNE": ethyne,
            "PROPANE": propane,
            "PROPENE": propene,
            "iBUTANE": ibutane,
            "nBUTANE": nbutane,
            "1BUTENE": butene1,
            "t2BUTENE": t2butene,
            "c2BUTENE": c2butene,
            "iPENTANE": ipentane,
            "nPENTANE": npentane,
            "t2PENTEN": t2penten,
            "1PENTEN": penten1,
            "2MEPENT": mepent2,
            "ISOPRENE": isoprene,
            "nHEXANE": nhexane,
            "nHEPTANE": nheptane,
            "iOCTANE": ioctane,
            "nOCTANE": noctane,
            "BENZENE": benzene,
            "TOLUENE": toluene,
            "ETHBENZ": ethbenz,
            "mpXYLENE": mpxylene,
            "oXYLENE": oxylene,
            "124TMB": tmb124,
            "135TMB": tmb135,
            "wd": wd,
            "ws": ws_arr,
            "temp": temp,
            "AT10": at10,
            "AP10": ap10,
            "AT2.5": at25,
            "AP2.5": ap25,
            "site": "London Marylebone Road",
            "code": "MY1",
            "latitude": 51.52253,
            "longitude": -0.154611,
            "location_type": "Urban Traffic",
            "Ox": ox,
            "NOx": nox,
            "u10": u10,
            "v10": v10,
            "d2m": d2m,
            "t2m": t2m,
            "blh": blh,
            "sp": sp,
            "ssrd": ssrd,
            "tcc": tcc,
            "tp": tp_arr,
            "rh2m": rh2m,
            "lat": 51.52253,
            "lon": -0.154611,
        }
    )
    return df


def make_aq_weekly(
    start: str = "2015-01-01",
    end: str = "2016-07-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Return a synthetic weekly air-quality panel for the SCM tutorial.

    Covers the full donor pool used in notebook 4.  The treated unit
    "2+26 cities" shows a step-down reduction in SO2wn starting
    2015-10-23 (the intervention date used in the notebook).
    """
    rng = np.random.default_rng(seed)

    treated = "2+26 cities"
    donors = [
        "Dongguan", "Zhongshan", "Foshan", "Beihai", "Nanning", "Nanchang",
        "Xiamen", "Taizhou", "Ningbo", "Guangzhou", "Huizhou", "Hangzhou",
        "Liuzhou", "Shantou", "Jiangmen", "Heyuan", "Quanzhou", "Haikou",
        "Shenzhen", "Wenzhou", "Huzhou", "Zhuhai", "Fuzhou", "Shaoxing",
        "Zhaoqing", "Zhoushan", "Quzhou", "Jinhua", "Shaoguan", "Sanya",
        "Jieyang", "Meizhou", "Shanwei", "Zhanjiang", "Chaozhou", "Maoming",
        "Yangjiang",
    ]
    all_ids = [treated] + donors
    n_donors = len(donors)

    cutoff = pd.Timestamp("2015-10-23")
    dates = pd.date_range(start, end, freq="W-SUN")
    n_weeks = len(dates)

    # Common baseline factor (shared trend + weekly noise)
    common = 60 + 5 * np.sin(2 * np.pi * np.arange(n_weeks) / 52) + rng.normal(0, 3, n_weeks)

    rows = []
    for unit in all_ids:
        # Each unit = common * scale + idiosyncratic noise
        scale = rng.uniform(0.6, 1.4)
        idio = rng.normal(0, 4, n_weeks)
        so2wn = common * scale + idio

        # Treated unit: reduction after cutoff
        if unit == treated:
            post = dates >= cutoff
            so2wn[post] *= 0.65   # ~35% reduction

        for i, dt in enumerate(dates):
            v = max(so2wn[i], 1.0)
            rows.append(
                {
                    "date": dt,
                    "ID": unit,
                    "CO": v * 0.025 + rng.normal(0, 0.2),
                    "COwn": v * 0.024 + rng.normal(0, 0.2),
                    "NO2": v * 0.6 + rng.normal(0, 3),
                    "NO2wn": v * 0.58 + rng.normal(0, 3),
                    "O3": max(35 - v * 0.1 + rng.normal(0, 5), 5),
                    "O3_8h": max(40 - v * 0.1 + rng.normal(0, 5), 5),
                    "O3_8hwn": max(38 - v * 0.1 + rng.normal(0, 5), 5),
                    "O3wn": max(36 - v * 0.1 + rng.normal(0, 5), 5),
                    "Ox": v * 0.5 + rng.normal(0, 4),
                    "Oxwn": v * 0.48 + rng.normal(0, 4),
                    "PM10": v * 1.8 + rng.normal(0, 10),
                    "PM10wn": v * 1.75 + rng.normal(0, 10),
                    "PM2.5": v * 1.1 + rng.normal(0, 6),
                    "PM2.5wn": v * 1.05 + rng.normal(0, 6),
                    "SO2": v + rng.normal(0, 2),
                    "SO2wn": v,
                }
            )

    df = pd.DataFrame(rows)
    # Clip negatives for pollutant columns
    for col in df.columns:
        if col not in ("date", "ID"):
            df[col] = df[col].clip(lower=0)
    return df
