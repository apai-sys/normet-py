"""I/O adapters for non-tabular data sources."""

from .defra import AURN_POLLUTANT_CODES, fetch_aurn_measurements, list_aurn_stations
from .eea import EEA_POLLUTANT_CODES, fetch_eea_data
from .era5 import ERA5_AQ_VARIABLES_DEFAULT, fetch_era5_timeseries
from .gdas import ARL_GDAS1_BASE_URL, fetch_gdas1, gdas1_filenames
from .openaq import fetch_openaq_measurements, openaq_locations, openaq_sensors
from .trajectory import (
    build_trajectory_features,
    read_trajectory_tdump,
    run_back_trajectories,
    trajectory_features,
)

__all__ = [
    "fetch_openaq_measurements",
    "openaq_locations",
    "openaq_sensors",
    "fetch_era5_timeseries",
    "ERA5_AQ_VARIABLES_DEFAULT",
    "EEA_POLLUTANT_CODES",
    "fetch_eea_data",
    "AURN_POLLUTANT_CODES",
    "fetch_aurn_measurements",
    "list_aurn_stations",
    "read_trajectory_tdump",
    "trajectory_features",
    "build_trajectory_features",
    "run_back_trajectories",
    "ARL_GDAS1_BASE_URL",
    "gdas1_filenames",
    "fetch_gdas1",
]
