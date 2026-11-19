# Example: Real-world Weather Normalisation with OpenAQ and ERA5

This recipe demonstrates the end-to-end workflow of pulling ground-level air quality observations from the **OpenAQ** platform, fetching corresponding meteorological fields from **ERA5** (Copernicus CDS), compiling them into a long-format panel, and performing weather normalisation.

---

## 1. Setup & Credentials

Before starting, ensure that:
1. You have an **OpenAQ API Key**. Get a free key at [explore.openaq.org/register](https://explore.openaq.org/register).
2. You have a **Copernicus CDS account** and your credentials are saved in `~/.cdsapirc` as per [Copernicus instructions](https://cds.climate.copernicus.eu/api).
3. The required dependencies are installed:
   ```bash
   pip install cdsapi requests pandas
   ```

Set the API keys in your environment:
```python
import os
os.environ["OPENAQ_API_KEY"] = "your_openaq_api_key_here"
```

---

## 2. Discover Locations and Active Sensors

First, we will query OpenAQ for monitoring locations in a target area (e.g., London) measuring PM2.5, and check what active sensors are available on the selected station.

```python
import normet as nm

# 1. Search for PM2.5 monitoring stations in London
df_locs = nm.openaq_locations(
    country="GB",
    city="London",
    bbox=(-0.5, 51.3, 0.2, 51.7),
    parameter="pm25",
    limit=5
)

# Display discovered stations
print("Discovered Locations:")
print(df_locs[["id", "name", "lat", "lon"]])

# Let's select the first location ID to query its active sensors
target_location_id = df_locs.iloc[0]["id"]
print(f"\nQuerying active sensors at location {target_location_id}:")

df_sensors = nm.openaq_sensors(location_id=target_location_id)
print(df_sensors[["id", "name", "parameter_name", "parameter_units"]])
```

---

## 3. Fetch Historical Hourly Measurements

Using the target location ID, we fetch historical hourly PM2.5 measurements for our analysis window (e.g., the first week of 2024).

```python
# Fetch hourly PM2.5 observations
df_aq = nm.fetch_openaq_measurements(
    location_id=target_location_id,
    parameter="pm25",
    date_from="2024-01-01",
    date_to="2024-01-07"
)

# The returned DataFrame matches the long-format panel requirements:
# [date, site, parameter, value, unit, lat, lon]
print(df_aq.head())
```

---

## 4. Fetch Corresponding ERA5 Meteorology

Next, we download the corresponding meteorological parameters at the exact coordinates of our target monitoring station. `fetch_era5_timeseries` pulls pre-interpolated single-point time-series straight from the CDS — no gridded NetCDF or `xarray` needed.

```python
# Map the station name to its coordinates
station_name = df_aq.iloc[0]["site"]
sites = {
    station_name: (df_aq.iloc[0]["lat"], df_aq.iloc[0]["lon"])
}

# Fetch surface temperature and wind components as point time-series
df_met = nm.fetch_era5_timeseries(
    sites=sites,
    date_from="2024-01-01",
    date_to="2024-01-07",
    variables=["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"],
    cache_dir=".era5_cache",
)

# The endpoint returns SHORT variable names, so columns are:
# [site, date, lat, lon, t2m, u10, v10]
print(df_met.head())
```

---

## 5. Merge and Preprocess Panel Data

With both air quality and meteorology datasets fetched, we merge them on `site` and `date` to construct our final modeling panel.

```python
import pandas as pd

# Standardize date formats to datetime
df_aq["date"] = pd.to_datetime(df_aq["date"])
df_met["date"] = pd.to_datetime(df_met["date"])

# Merge datasets
panel = pd.merge(df_aq, df_met, on=["site", "date"], how="inner")

# Inspect the merged panel dataset
print(panel.head())
```

---

## 6. Weather Normalisation & HTML Report

We can now run weather normalisation using `normet`'s AutoML engine. `nm.do_all`
runs the full pipeline — prepare → train → Monte-Carlo normalise — in one call,
then we bundle the result with provenance metadata and export an HTML report.

```python
# Meteorological resample variables (short ERA5 names) + time features
# engineered by prepare_data
met_vars = ["t2m", "u10", "v10"]
predictors = met_vars + ["date_unix", "day_julian", "weekday", "hour"]

# Full pipeline: prepare -> train (AutoML) -> normalise
out, model, df_prep = nm.do_all(
    df=panel,
    target="value",            # the PM2.5 column returned by OpenAQ
    backend="flaml",
    covariates=predictors,
    variables_resample=met_vars,
    n_samples=50,
)

print(out.head())             # columns: observed, normalised

# Bundle the result with provenance, then render a single-file HTML report.
# generate_html_report writes the file itself and returns the path.
run = nm.make_run(
    result=out,
    kind="do_all",
    model=model,
    df_prep=df_prep,
    df=panel,
    config={"value": "value", "backend": "flaml"},
)
report_path = nm.generate_html_report(
    run,
    "london_normalisation_report.html",
    title="London PM2.5 Weather Normalisation",
)

print(f"Weather normalisation complete! Report saved to '{report_path}'.")
```
