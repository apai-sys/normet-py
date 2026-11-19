# Conda recipe

`meta.yaml` packages the PyPI sdist of the same version (version, URL and
sha256 change together on a release). It is written for a `noarch: python`
build, so one artefact serves Linux, macOS and Windows.

## Build locally

```bash
conda install -n base conda-build
conda build packaging/conda
```

## Upload to a personal channel

```bash
anaconda login
anaconda upload -u dsncas <build-dir>/normet-<version>-py_0.conda
# users: conda install -c dsncas normet
```

## conda-forge

Copy `meta.yaml` to `recipes/normet/meta.yaml` in a fork of
`conda-forge/staged-recipes` and open a PR. Once the feedstock exists, the
autotick bot proposes later PyPI releases automatically.

## Optional dependencies

Conda has no "extras": the recipe carries the core dependencies only. Install
the matching conda-forge packages for an extra:

| extra | conda-forge packages |
|-------|----------------------|
| `flaml` | `flaml xgboost` |
| `lgb` | `lightgbm` |
| `dask` | `dask` |
| `data` | `requests cdsapi rdata pyarrow` |
| `cli` | `click pyyaml` |
| `gui` | `pyside6 pyarrow openpyxl xlrd` |
| `geo` | `shapely pyshp` |
| `stats` | `statsmodels` |
| `foundation` / `finetune` | `chronos-forecasting torch accelerate peft` |
| `bayes` | `pymc arviz` |
| `mcp` | `mcp` |
| `docs` | `sphinx sphinx-rtd-theme myst-parser` |
