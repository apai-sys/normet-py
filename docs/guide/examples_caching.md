# Caching and provenance

## Opt-in disk cache

`normet` ships with a thin wrapper around `joblib.Memory`:

```python
import normet as nm

memory = nm.make_memory(".normet_cache")

@memory.cache
def train(df_hash, **cfg):
    df = pd.read_parquet("data.parquet")
    return nm.train_model(df, **cfg)
```

The cache key is whatever you pass to `train(...)` — so use
`nm.dataframe_hash(df)` and `nm.config_hash(cfg)` to derive deterministic
cache keys without pickling huge DataFrames into the key.

## Run archives

For reproducibility, wrap a result with provenance metadata and save it:

```python
out, model, df_prep = nm.do_all(df, target="PM2.5", ...)
run = nm.make_run(
    result=out, model=model, df_prep=df_prep, df=df,
    kind="do_all", config={...}, seed=42,
)
paths = nm.save_run(run, "results/run_2024_q1")
# writes results/run_2024_q1.joblib (artifact)
#    and results/run_2024_q1.meta.json (sidecar metadata)

# Later, in a clean kernel:
back = nm.load_run("results/run_2024_q1")
back.metadata          # full provenance dict
back.result.head()
```

The JSON sidecar always contains:

- `normet_version`, `python_version`, `platform`, `host`, `user`, `timestamp`
- `seed`
- `data_hash`, `data_shape` (when `df` provided)
- `config`, `config_hash` (when `config` provided)
- `extra` (free-form, optional)
