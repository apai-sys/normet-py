# Installation

`normet` supports Python 3.10 and later on macOS, Windows and Linux.

## From PyPI

```bash
pip install normet
```

The core package is deliberately small (see {ref}`dependencies`). It needs at
least one model backend to train anything, so most users install it with an extra:

```bash
pip install "normet[flaml]"      # AutoML backend (recommended)
```

## Development version

The documentation tracks the `main` branch on GitHub. If a feature described here is
missing from the release you have installed, install the development version:

```bash
pip install git+https://github.com/apai-sys/normet-py.git
```

## Optional extras

Every heavy or niche dependency is optional and imported lazily, so `import normet`
never fails because an extra is missing; the function that needs it raises an error
naming the extra to install.

```bash
# AutoML backend (recommended)
pip install "normet[flaml]"

# Lightweight LightGBM tuner
pip install "normet[lgb]"

# Dask support for large datasets
pip install "normet[dask]"

# ERA5 / EEA / OpenAQ / UK air quality (ukaq) data adapters
pip install "normet[data]"

# CLI entry point
pip install "normet[cli]"

# Chronos-2 foundation model (zero-shot de-weathering, station embeddings)
pip install "normet[foundation]"

# MCP server for AI assistants (Claude Code, Claude Desktop, ...)
pip install "normet[mcp]"

# Everything
pip install "normet[all]"
```

The full list, as declared in `pyproject.toml`:

| Extra | Installs | Used for |
|---|---|---|
| `flaml` | `flaml[automl]` | AutoML backend (`backend="flaml"`) |
| `lgb` | `lightgbm` | LightGBM backend (`backend="lightgbm"`) |
| `dask` | `dask[dataframe]` | Large datasets |
| `data` | `requests`, `cdsapi`, `rdata`, `pyarrow` | The `normet.io` data adapters (ERA5, EEA, OpenAQ, UK networks) |
| `cli` | `click`, `pyyaml` | The `normet` command-line tool |
| `gui` | `PySide6`, `pyarrow`, `openpyxl`, `xlrd` | The desktop app (`normet-gui`) |
| `geo` | `shapely`, `pyshp` | Polygon and shapefile source regions for the HYSPLIT trajectory features |
| `stats` | `statsmodels` | STL-based event detection (`method="stl_residual"`) |
| `foundation` | `chronos-forecasting`, `torch` | Chronos-2 zero-shot de-weathering and embeddings |
| `finetune` | `chronos-forecasting`, `torch`, `accelerate`, `peft` | Fine-tuning the Chronos-2 checkpoint (`peft` only for LoRA) |
| `bayes` | `pymc`, `arviz` | Bayesian synthetic control |
| `mcp` | `mcp` | The `normet-mcp` server for AI assistants |
| `all` | all of the above | Everything except the `dev` and `docs` tooling |
| `dev` | `pytest`, `ruff`, `mypy`, ... | Development (see {doc}`../development/contributing`) |
| `docs` | `sphinx`, `sphinx-rtd-theme`, `myst-parser` | Building this site |

Extras combine: `pip install "normet[gui,flaml]"`.

## Desktop app

The desktop GUI is part of the same package:

```bash
pip install "normet[gui,flaml]"   # or normet[gui,lgb]
normet-gui                        # optionally: normet-gui mydata.csv
```

Ready-made installers are also built: `.dmg` (macOS), `.exe` (Windows, Inno Setup)
and `.AppImage` (Linux), produced by the
[`build-gui.yml`](https://github.com/apai-sys/normet-py/actions/workflows/build-gui.yml)
workflow's 3-OS matrix. Grab the latest from the workflow's **Actions → Artifacts**, or
build one locally with `packaging/macos/build_dmg.sh` (macOS) or
`pyinstaller packaging/normet_gui.spec` (any OS — see
[`packaging/README.md`](https://github.com/apai-sys/normet-py/blob/main/packaging/README.md)).
See {doc}`../guide/gui` for a tour of the window.

## From source

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/apai-sys/normet-py
cd normet-py
pip install -e ".[all,dev,docs]"
```

{doc}`../development/contributing` covers the test, lint and docs commands.

## Checking the installation

With the `cli` extra, `normet info` prints the installed version and the backends that
are available:

```bash
normet info
```

(dependencies)=
## Dependencies

| Group | Packages |
|---|---|
| Core | `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`, `matplotlib` |
| AutoML | `flaml` (recommended) or `lightgbm` |
| I/O (data adapters) | `requests`, `cdsapi`, `rdata`, `pyarrow` |
| Parallel/large data | `dask` |
| CLI | `click`, `pyyaml` |
| GUI | `PySide6`, `pyarrow`, `openpyxl`, `xlrd` |
