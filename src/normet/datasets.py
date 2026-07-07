"""Bundled example datasets from the normet model-description paper.

All four datasets come from the paper's case studies around the London
Marylebone Road (MY1) kerbside monitoring station:

- :func:`load_my1` — hourly NO2 + ERA5 meteorology, Jan–Aug 2020
  (deweathering case; the window spans the UK COVID-19 lockdown).
- :func:`load_scm` — monthly deweathered NO2 panel for 104 UK sites,
  2016–2021 (ULEZ Synthetic Control case).
- :func:`load_my1_pm25` — hourly PM2.5 + meteorology at MY1, Jan–Aug 2020
  (transport-aware normalisation case).
- :func:`load_traj_my1` — 6-hourly HYSPLIT back-trajectory features
  arriving at MY1, Jan–Aug 2020 (same case). :func:`example_traj_dir`
  points at a two-day sample of the raw HYSPLIT ``tdump`` output the
  features were derived from.

Sources: UK AURN (Defra, Open Government Licence v3.0); ERA5 (Copernicus
Climate Change Service); HYSPLIT (NOAA ARL) driven by GDAS 1° meteorology.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pandas as pd

__all__ = [
    "load_my1",
    "load_scm",
    "load_my1_pm25",
    "load_traj_my1",
    "example_traj_dir",
]


def _read(name: str, **kwargs) -> pd.DataFrame:
    with files("normet.data").joinpath(f"{name}.csv.gz").open("rb") as fh:
        return pd.read_csv(fh, compression="gzip", parse_dates=["date"], **kwargs)


def load_my1() -> pd.DataFrame:
    """Hourly NO2 and meteorology at London Marylebone Road, Jan–Aug 2020.

    Real AURN NO2 observations merged with ERA5 single-level meteorology
    (5 793 rows × 11 columns: ``date``, ``NO2``, ``ws``, ``wd``, ``temp``,
    ``RH``, ``atmos_pres``, ``blh``, ``tcc``, ``tp``, ``ssrd``). The window
    spans the UK COVID-19 lockdown (from 2020-03-23), making it a compact
    real-world deweathering example.
    """
    return _read("my1")


def load_scm() -> pd.DataFrame:
    """Monthly deweathered NO2 panel for UK sites (ULEZ SCM case study).

    Observed and weather-normalised monthly NO2 for 104 UK AURN sites,
    2016–2021 (7 378 rows × 6 columns: ``date``, ``NO2_obs``, ``NO2_dw``,
    ``NO2_dw_common``, ``code``, ``type``). The London kerbside site
    ``"MY1"`` is the unit treated by the Ultra Low Emission Zone (launched
    2019-04-08); non-London sites of matching ``type`` form the donor pool.
    """
    return _read("scm")


def load_my1_pm25() -> pd.DataFrame:
    """Hourly PM2.5 and meteorology at MY1, Jan–Aug 2020.

    Companion dataset to :func:`load_traj_my1` for the transport-aware
    normalisation case study (5 856 rows × 17 columns, mixing observed
    ``ws``/``wd``/``temp`` with ERA5 fields and ERA5-derived
    ``ws_era5``/``wd_era5``/``temp_era5``).
    """
    return _read("my1_pm25")


def load_traj_my1() -> pd.DataFrame:
    """HYSPLIT back-trajectory features arriving at MY1, Jan–Aug 2020.

    Summary features of 72-hour back-trajectories arriving 6-hourly
    (976 rows × 12 columns; ``traj_``-prefixed distance, speed, inflow
    bearing, height, and source-region residence-time fractions). Join on
    ``date`` with :func:`load_my1_pm25` and include the ``traj_`` columns
    among the resampled variables for transport-aware normalisation.
    """
    return _read("traj_my1")


def example_traj_dir() -> Path:
    """Directory containing a two-day sample of raw HYSPLIT ``tdump`` files.

    Eight 72-hour back-trajectory endpoint files (2020-01-01 00 UTC to
    2020-01-02 18 UTC, 6-hourly) for the trajectory-reader examples, e.g.
    :func:`normet.io.trajectory.read_trajectory_tdump`.
    """
    return Path(str(files("normet.data").joinpath("traj")))
