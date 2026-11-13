# Command-line interface

The `cli` extra installs a `normet` command that wraps the most common library entry
points, so a pipeline can run from a shell script or a batch job without writing
Python:

```bash
pip install "normet[cli]"
normet --help
```

| Command | What it runs |
|---|---|
| `normet do-all INPUT` | Prepare → train → normalise (`nm.do_all`); `--backend chronos-2` for the zero-shot path |
| `normet decompose INPUT` | Emission or meteorology decomposition (`nm.decompose`) |
| `normet deweather INPUT` | Zero-shot meteorological normalisation with Chronos-2 (needs `normet[foundation]`) |
| `normet scm INPUT` | Synthetic control on a panel table (`nm.run_scm`) |
| `normet cv INPUT` | Walk-forward cross-validation (`nm.cv_score`) |
| `normet report RUN_PATH` | Render a saved run archive as an HTML or Markdown report |
| `normet info` | Show the installed version and the available backends |

`INPUT` is a table on disk (CSV, Parquet or Feather), and every command except `info`
writes its result to the path given by `--out`. `normet <command> --help` lists each
command's options.

## Examples

```bash
# One-shot normalisation; --run-path also saves a run archive (.joblib + .meta.json)
normet do-all my_site.csv --target PM2.5 \
  --covariates ws,wd,temp,RH,blh,date_unix,day_julian,weekday,hour \
  --resample-vars ws,wd,temp,RH,blh \
  --backend flaml --n-samples 300 --out normalised.csv --run-path runs/my_site

# Decomposition
normet decompose my_site.csv --target PM2.5 \
  --covariates ws,wd,temp,RH,blh,date_unix,day_julian,weekday,hour \
  --method emission --out emission.csv

# SCM
normet scm panel.csv --treated MY1 --cutoff 2019-04-01 \
  --date-col date --unit-col code --outcome-col NO2_dw --out scm.csv

# Zero-shot de-weathering with Chronos-2 (needs normet[foundation])
normet deweather my_site.csv \
  --target PM2.5 --met-vars t2m,blh,u10,v10 --out normalised.csv

# A report from the saved run
normet report runs/my_site.joblib --out report.html

# Show installed version and available backends
normet info
```

## Configuration files

Any subcommand can read its options from a YAML file via `--config`; flags given on
the command line take precedence over the file's values. The keys are the option
names with underscores, and list-valued options may be written as YAML lists:

```yaml
# config.yaml
target: PM2.5
covariates: [ws, wd, temp, RH, blh, date_unix, day_julian, weekday, hour]
resample_vars: [ws, wd, temp, RH, blh]
backend: flaml
n_samples: 300
split_method: random
train_fraction: 0.75
```

```bash
normet do-all my_site.csv --config config.yaml --out normalised.csv
```
