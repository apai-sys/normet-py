# Example: Comprehensive Visualisation and Plotting Suite

This recipe provides a complete showcase of the visualisations available in the `normet.plotting` suite. These plotting functions are designed to create publication-quality figures for environmental research. Every function shown here is exported at the top level, so `nm.<name>` works after `import normet as nm`.

---

## 1. Wind Analysis with Polar Plots (`polar_plot`)

Polar plots display concentrations as a function of wind speed and wind direction, helping researchers identify local vs. regional emission sources.

```python
import numpy as np
import pandas as pd
import normet as nm
import matplotlib.pyplot as plt

# Generate synthetic wind speed (ws), wind direction (wd) and concentration (val) data
np.random.seed(42)
n_points = 500
df_wind = pd.DataFrame({
    "ws": np.random.gamma(2, 2, n_points),
    "wd": np.random.uniform(0, 360, n_points),
    "val": np.random.uniform(5.0, 45.0, n_points),
})

# Let's add an emission source from the Northeast at high wind speeds:
mask = (df_wind["wd"] > 30) & (df_wind["wd"] < 60) & (df_wind["ws"] > 4)
df_wind.loc[mask, "val"] += 25.0

# Generate polar plot (needs a polar projection axis)
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
nm.polar_plot(
    df=df_wind,
    target="val",
    ws_col="ws",
    wd_col="wd",
    statistic="mean",   # "mean" | "median" | "max" | "sum" | "p95"
    cmap="viridis",
    title="London PM2.5 Polar Plot",
    ax=ax,
)
plt.savefig("polar_plot.png", dpi=150)
plt.show()
```

---

## 2. Time Series with Uncertainty Bands (`time_series_plot`)

A general-purpose time series plotting function supporting uncertainty bounds (confidence/quantile bands) and an optional resampling rule (e.g. converting daily to weekly).

```python
# Create a daily time series, indexed by date
dates = pd.date_range("2024-01-01", periods=100, freq="D")
df_ts = pd.DataFrame({
    "observed": np.sin(np.linspace(0, 10, 100)) * 5.0 + 15.0 + np.random.normal(0, 1.0, 100),
    "low": np.sin(np.linspace(0, 10, 100)) * 5.0 + 12.0,
    "high": np.sin(np.linspace(0, 10, 100)) * 5.0 + 18.0,
}, index=dates)

# Plot with uncertainty bands and a weekly-mean resample
fig, ax = plt.subplots(figsize=(10, 5))
nm.time_series_plot(
    df=df_ts,
    target="observed",
    ci_low="low",
    ci_high="high",
    resample="W",   # pandas resample rule; resamples using the mean
    title="Weekly Averaged Concentration with Uncertainty Bands",
    ylabel="PM2.5 (ug/m3)",
    ax=ax,
)
plt.savefig("time_series_uncertainty.png", dpi=150)
plt.show()
```

---

## 3. Weather Normalisation Trends (`normalise_plot`)

Compare observed pollutant concentrations with deweathered (weather-normalised) trends, complete with a shaded band from the quantile columns produced by `nm.normalise(..., return_quantiles=...)`.

```python
# Mirror the output of nm.normalise / nm.do_all: observed + normalised (+ quantile cols)
df_norm = pd.DataFrame({
    "observed": 20.0 + np.sin(np.linspace(0, 5, 50)) * 4.0 + np.random.normal(0, 1.5, 50),
    "normalised": 19.5 + np.sin(np.linspace(0, 5, 50)) * 4.0,
    "q025": 17.5 + np.sin(np.linspace(0, 5, 50)) * 4.0,
    "q975": 21.5 + np.sin(np.linspace(0, 5, 50)) * 4.0,
}, index=pd.date_range("2024-01-01", periods=50, freq="D"))

# Plot weather-normalisation results, overlaying the quantile band
fig, ax = plt.subplots(figsize=(10, 5))
nm.normalise_plot(
    result_df=df_norm,
    ci_low="q025",
    ci_high="q975",
    title="PM2.5 Observed vs. Weather-Normalised Trend",
    ylabel="Concentration (ug/m3)",
    ax=ax,
)
plt.savefig("normalisation_trend.png", dpi=150)
plt.show()
```

---

## 4. Partial Dependence Profiles (`pdp_grid`)

Display the isolated response of a pollutant concentration relative to individual
meteorological features. `pdp_grid` consumes the long-format DataFrame returned by
`nm.pdp` — columns `["variable", "value", "pdp_mean", "pdp_std"]`:

```python
# In practice: pdp_df = nm.pdp(df_prep, model, feature_names=["t2m", "ws"])
# Here we build a DataFrame with the same schema for illustration.
temp_grid = np.linspace(-5, 30, 20)
ws_grid = np.linspace(0, 12, 20)

pdp_df = pd.concat([
    pd.DataFrame({
        "variable": "t2m",
        "value": temp_grid,
        "pdp_mean": 25.0 - 0.2 * temp_grid,
        "pdp_std": 1.0,
    }),
    pd.DataFrame({
        "variable": "ws",
        "value": ws_grid,
        "pdp_mean": 30.0 / (1.0 + ws_grid),
        "pdp_std": 1.5,
    }),
], ignore_index=True)

# Faceted grid of PDP curves (shaded ±1 std band)
fig = nm.pdp_grid(pdp_df, cols=2, title="Partial Dependence Profiles")
fig.savefig("pdp_grid.png", dpi=150)
```

---

## 5. Time Series Decomposition Stack (`decomposition_stack`)

Stacked-area view of an additive decomposition. `decomposition_stack` plots every
column except the observed series and any names listed in `exclude`, so it pairs
naturally with the output of `nm.decompose`.

```python
# Mirror a decomposition output: observed + additive components
dates = pd.date_range("2024-01-01", periods=100, freq="D")
df_decom = pd.DataFrame({
    "observed": 15.0 + np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.5, 100),
    "trend": np.linspace(14.0, 16.0, 100),
    "seasonal": np.sin(np.linspace(0, 10, 100)),
    "meteorological": np.random.normal(0, 0.3, 100),
    "residual": np.random.normal(0, 0.2, 100),
}, index=dates)

# Stack the components (keep observed as the overlaid reference line)
fig, ax = plt.subplots(figsize=(10, 8))
nm.decomposition_stack(
    decomp_df=df_decom,
    exclude=("observed", "residual"),
    title="Additive decomposition of PM2.5",
    ax=ax,
)
plt.savefig("decomposition_stack.png", dpi=150)
plt.show()
```
