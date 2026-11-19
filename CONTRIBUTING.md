# Contributing to normet

Thanks for considering a contribution! This project welcomes bug reports,
feature requests, documentation fixes and pull requests.

## Development setup

`normet` needs Python ≥ 3.10. Clone the repository and install it in editable mode
with the development tools, in a virtual environment or a conda environment:

```bash
git clone https://github.com/apai-sys/normet-py
cd normet-py

# either a venv ...
python -m venv .venv && source .venv/bin/activate
# ... or conda
conda create -n normet python=3.12 && conda activate normet

pip install -e ".[all,dev,docs]"
pre-commit install
```

`all` installs every optional extra, so no test skips for a missing import; this is
what CI installs too (`.[all,dev]`). A lighter `pip install -e ".[flaml,dev]"` is
enough for most library work — tests whose optional dependency is absent skip
themselves.

## Development workflow

1. **Branch off `main`**, one feature per branch, with a short descriptive name:
   `feature/...`, `fix/...`, `gui/...` or `docs/...`
   (`fix/normalise-quantiles`, `feature/bayesian-scm`).
2. **Add tests** for any new behaviour.
3. **Run the checks** below before pushing.
4. **Update the CHANGELOG**: add a bullet to the `[Unreleased]` section under the
   appropriate heading (`Added` / `Changed` / `Fixed` / `Removed`). Every
   user-visible change gets one. New capabilities also go into the "Core
   capabilities" table in `README.md`, and into the user guide under `docs/`.
5. **Open a pull request** with a clear description of the change and link any
   related issue.

### Checks

CI (`.github/workflows/ci.yml`) runs these on Python 3.10–3.13; run the same locally:

```bash
ruff check src tests
ruff format --check src tests
mypy
pytest -q
```

CI runs the full suite with coverage, gated at 70 %:

```bash
pytest --cov=normet --cov-report=term-missing --cov-fail-under=70
```

Tests marked `slow` load the real Chronos-2 checkpoint and take minutes;
`pytest -m "not slow"` skips them.

**Known local hang on macOS.** In one process, the LightGBM tests
(`test_lgb_backend.py`, `test_assistant.py`, ...) followed by the Chronos-2 tests
(`test_foundation_embedder.py`, `test_foundation_estimator.py`) can hang or segfault,
because the two libomp runtimes collide. This is not caused by your change. For a full
local run, deselect those two foundation files and run them in a separate process:

```bash
pytest -q --ignore=tests/test_foundation_embedder.py \
          --ignore=tests/test_foundation_estimator.py
pytest -q tests/test_foundation_embedder.py tests/test_foundation_estimator.py
```

### Documentation

The documentation site is built with Sphinx and MyST from `docs/`; CI builds it with
warnings treated as errors:

```bash
pip install -e ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

The API reference is generated from the docstrings (`docs/api.rst`). The PDF user
manuals in `docs/manual/` are LaTeX sources: `docs/manual/make_figures.py` regenerates
their screenshots of the desktop app, and `docs/manual/build.sh` builds the PDFs
(`pdflatex` for English, `xelatex` with `ctex` for Chinese).

## Code style

- Python ≥ 3.10 with `X | None` / `list[X]` annotations; ruff (line length 100)
  formats and lints the code.
- mypy runs in `--strict` mode except `warn_return_any` and
  `disallow_any_generics`; the desktop app (`normet.gui.*`) is exempt.
- Public functions get a NumPy-style docstring. Docstrings explain *why*: the failure
  modes and measured facts, not only what the code does.
- New optional dependencies must be lazy-imported via
  `normet.utils._lazy.require` so importing `normet` never fails when a
  niche extra is absent.
- Existing results should stay bit-identical: add a new code path or option rather
  than silently changing the numbers an existing one returns.
- Code, docstrings, the CHANGELOG and commit messages are in English.

## Commit messages

Commit subjects are a plain sentence about the behaviour, optionally with an area
prefix, as in the existing log:

```text
Tables, normalise and PDP do less work per call, and `import normet` is lighter
gui: the window's start skips PySide6's source scan
Test: the Qt import-hook test can fail again
```

## Filing bugs

- Search [existing issues](https://github.com/apai-sys/normet-py/issues)
  first.
- Include a minimal reproducer, your `normet` and Python versions
  (`normet info`), and the full traceback.
- Report security problems privately, as described in the
  [security policy](https://github.com/apai-sys/normet-py/blob/main/SECURITY.md),
  not in a public issue.

## Proposing features

For non-trivial changes, please open an issue first to discuss the design.
This avoids wasted work when an idea overlaps with planned work or doesn't
fit the project's scope.

## Code of Conduct

By participating you agree to abide by the [Contributor Covenant Code of
Conduct](https://github.com/apai-sys/normet-py/blob/main/CODE_OF_CONDUCT.md).
