# Multi-site batch pipelines

When you have one long-format DataFrame with many stations, use the multisite
drivers:

```python
df_norm = nm.do_all_multisite(
    df, site_col="station_id", value="PM2.5",
    feature_names=feats, backend="flaml",
    n_samples=300, n_cores=8,
)

df_decomp = nm.decompose_multisite(
    df, site_col="station_id", value="PM2.5",
    method="emission", feature_names=feats, n_cores=8,
)
```

Both return long-format DataFrames keyed by the original site column, so you
can plot/group like any other tidy table. Failed sites are warned and skipped,
not raised.

For other per-site operations, the generic helper `nm.multisite_apply`
parallelises any callable that takes a per-site DataFrame and returns a
DataFrame.
