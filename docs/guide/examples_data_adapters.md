# Data Adapters

`normet` includes built-in adapters to pull air quality observations and gridded meteorological datasets from open platforms, converting them directly into long-format DataFrames suitable for analysis.

## OpenAQ (Air Quality Observations)

OpenAQ provides global, real-time and historical air quality data. The `normet` adapter integrates with the OpenAQ v3 API.

### Authentication

Get a free API key at [explore.openaq.org/register](https://explore.openaq.org/register), then set the `OPENAQ_API_KEY` environment variable or pass `api_key` directly.

### 1. Discover Monitoring Locations

Find active stations matching criteria such as country, city, bounding box, and target pollutant. Bounding box coordinates are formatted as `(min_lon, min_lat, max_lon, max_lat)`:

```python
import normet as nm

# Find PM2.5 locations in London
df_locs = nm.io.openaq_locations(
    country="GB",
    city="London",
    bbox=(-0.5, 51.3, 0.2, 51.7),
    parameter="pm25",
    limit=10,
)
print(df_locs[["id", "name", "lat", "lon", "parameters"]])
```

Each location in the returned DataFrame also contains a `sensors` column which lists the exact active sensors (including their parameter IDs and units).

### 2. Discover Active Sensors for a Location

To list active sensors at a specific monitoring location with their identifier and metadata:

```python
df_sensors = nm.io.openaq_sensors(location_id=12345)
print(df_sensors[["id", "name", "parameter_name", "parameter_units"]])
```

### 3. Fetch Measurements

Fetch historical hourly measurements for one or more location IDs and a specific parameter slug (e.g. `"pm25"`, `"no2"`, `"o3"`):

```python
df_aq = nm.io.fetch_openaq_measurements(
    location_id=12345,
    parameter="pm25",
    date_from="2024-01-01",
    date_to="2024-01-07",
)
# Returns a DataFrame with columns: [date, site, parameter, value, unit, lat, lon]
```

---

## ERA5 (Reanalysis Meteorology)

The ERA5 reanalysis product from Copernicus (ECMWF) provides globally comprehensive weather parameters. `normet` integrates with the Copernicus Climate Data Store (CDS) to download ERA5 meteorology directly as station time-series.

### Prerequisites

1. Set up a free Copernicus CDS account.
2. Store your credential in `~/.cdsapirc` as per the CDS API instructions:
   ```ini
   url: https://cds.climate.copernicus.eu/api
   key: <YOUR-PERSONAL-ACCESS-TOKEN>
   ```
3. Install the CDS client: `pip install cdsapi`

### Fetch Point Time-series (recommended)

`fetch_era5_timeseries` queries the CDS `reanalysis-era5-single-levels-timeseries` dataset, which returns pre-interpolated single-point CSVs for each site. It downloads only the variables you ask for — no gridded NetCDF, and no `xarray` / `netCDF4` dependency:

```python
# Pass sites as a dict/mapping of {site_name: (lat, lon)}
sites = {
    "London-West": (51.5, -0.1),
    "London-South": (51.3, -0.2),
}

df_met = nm.io.fetch_era5_timeseries(
    sites=sites,
    date_from="2024-01-01",
    date_to="2024-01-31",
    variables=["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"],
    cache_dir=".era5_cache",   # optional: reuse downloaded CSVs
)
# Returns long-format columns: [site, date, lat, lon, t2m, u10, v10]
```

> **Note:** this endpoint returns **short** variable names (`t2m`, `u10`, `v10`,
> `d2m`, `sp`, `blh`, `tcc`, `tp`, `ssrd`), regardless of the long CDS names you
> request — use those short names as your `variables_resample` downstream.

---

## EEA (European Environment Agency)

For European users, `normet` provides an adapter to download historical air quality datasets directly from the EEA:

```python
df_eea = nm.io.fetch_eea_data(
    country="FR",
    pollutant="PM2.5",
    year_from=2023,
    year_to=2023,
    station="FR0101A",
)
```

---

## AURN (UK Automatic Urban and Rural Network)

The AURN adapter fetches hourly pollutant measurements from the DEFRA UK-AIR
Sensor Observation Service (52°North SOS) REST API, using the same EIONET
pollutant codes as the EEA adapter. No API key is required.

### 1. List Stations

```python
# All stations
df_stations = nm.io.list_aurn_stations()

# Filtered by pollutant
df_pm25 = nm.io.list_aurn_stations(pollutant="PM2.5")
# Columns: [id, label, lat, lon]
```

### 2. Fetch Measurements

```python
df_aurn = nm.io.fetch_aurn_measurements(
    station="London North Kensington",  # station ID or label substring
    pollutant="NO2",
    date_from="2024-01-01",
    date_to="2024-01-07",
)
# Columns: [date, site, station_id, pollutant, value, unit, lat, lon]
```

Filter by label substring instead of exact station:

```python
df_london = nm.io.fetch_aurn_measurements(
    station_label="London",
    pollutant="PM2.5",
    date_from="2024-01-01",
    date_to="2024-01-07",
)
```

### Supported Pollutants

`AURN_POLLUTANT_CODES` maps names to EIONET numeric codes:

| Pollutant | Code |
|-----------|------|
| PM2.5     | 6001 |
| PM10      | 5    |
| NO2       | 8    |
| NOX       | 9    |
| NO        | 20   |
| O3        | 7    |
| SO2       | 1    |
| CO        | 10   |
| BENZENE   | 24   |

---

## HYSPLIT Back-Trajectories

Same-timestamp meteorology ignores where the air *came from*. The trajectory
adapter turns HYSPLIT back-trajectory output into per-receptor transport
features (inflow direction, transport distance/speed, trajectory height,
along-path rainfall/boundary-layer height, and residence-time fractions over
source regions) that you can feed to the models as extra predictors.

This module **consumes** trajectory output — it does not run HYSPLIT. Generate
the `tdump` files separately (e.g. with `pysplit`, `splitr`, or the HYSPLIT
`hyts_std` executable), one back-trajectory run per receptor time, then:

```python
import normet as nm
import normet.io as nio

# 1. Reduce a directory of tdump files to a receptor-time feature table
traj = nio.build_trajectory_features(
    "traj/tdump_*",
    source_regions={
        # name: (lon_min, lat_min, lon_max, lat_max)
        "industrial_NE": (116.0, 39.0, 120.0, 42.0),
        "marine_SW":     (-12.0, 45.0, -2.0, 52.0),
    },
)
# traj is indexed by receptor timestamp with columns: traj_dist_km,
# traj_pathlen_km, traj_speed_kmh, traj_inflow_deg, traj_height_mean,
# traj_height_min, traj_resid_<region>, plus (only if the tdump run wrote
# them) traj_rain_sum, traj_blh_mean, traj_rh_mean, traj_pressure_mean,
# traj_temp_mean

# 2. Join onto the AQ + local-met panel (trajectory features are slow-varying)
df = df.join(traj).ffill(limit=8)

# 3. Use them as transport-aware predictors. Because they are
#    transport/meteorology, list them in variables_resample too so the
#    Monte-Carlo normalisation deweathers them along with local met.
traj_feats = [c for c in df.columns if c.startswith("traj_")]
local_met = ["t2m", "blh", "u10", "v10", "ssrd", "tp"]

out, model, df_prep = nm.do_all(
    df=df, value="PM2.5", backend="flaml",
    feature_names=local_met + traj_feats + ["date_unix", "day_julian", "weekday", "hour"],
    variables_resample=local_met + traj_feats,
    n_samples=300,
)
```

To **keep** the transport signal and normalise only local weather, leave
`traj_*` out of `variables_resample` (they stay fixed at observed values).

Lower-level helpers are also exported: `nm.io.read_trajectory_tdump(path)`
parses a single `tdump` file, and `nm.io.trajectory_features(traj, ...)`
reduces one trajectory DataFrame to a feature dict.

### Downloading GDAS1 met (when you have none)

HYSPLIT needs ARL-format meteorology. If you don't have it locally,
`fetch_gdas1` pulls the weekly GDAS1 (1°) files from NOAA ARL's archive and
caches them (each is ~570 MB, so existing files are never re-fetched):

```python
# Remember to cover the full BACKWARD window: earliest receptor minus hours_back
met = nm.io.fetch_gdas1("2020-04-01", "2020-04-30", cache_dir=".gdas_cache")
# -> ['.../gdas1.apr20.w1', '.../gdas1.apr20.w2', ...]   (ARL files, chronological)
```

`nm.io.gdas1_filenames(date_from, date_to)` returns the same filenames without
downloading, if you just want to see what's needed.

### Running HYSPLIT end-to-end

With a built `hyts_std` and met files, `run_back_trajectories` does the whole
loop — writes a `CONTROL` per receptor time, runs HYSPLIT, and reduces the
output to the feature table:

```python
met = nm.io.fetch_gdas1("2020-04-01", "2020-04-30", cache_dir=".gdas_cache")

traj = nm.run_back_trajectories(
    times=df.index,                       # receptor (arrival) timestamps
    lat=51.52, lon=-0.13,
    met_files=met,
    hysplit_exec="~/hysplit-5.4.2/exec/hyts_std",
    hours_back=72,
    source_regions={"industrial_NE": (116.0, 39.0, 120.0, 42.0)},
)
df = df.join(traj).ffill(limit=8)         # then pass traj_* to nm.do_all
```

> **macOS:** HYSPLIT ships `x86_64` binaries via a disk image (run under Rosetta
> on Apple Silicon), so they carry a Gatekeeper quarantine flag. If `hyts_std`
> is killed (exit 137), clear it once:
> `xattr -dr com.apple.quarantine /path/to/hysplit`.
