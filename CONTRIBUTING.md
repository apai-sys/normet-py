# Contributing to normet

Thanks for considering a contribution! This project welcomes bug reports,
feature requests, and pull requests.

## Quick start

```bash
git clone https://github.com/normet-dev/normet-py
cd normet-py
python -m venv .venv && source .venv/bin/activate
pip install -e ".[flaml,dev]"
pre-commit install
```

## Development workflow

1. **Branch off `main`** for any change. Use a short, descriptive name
   (`fix/normalise-quantiles`, `feat/bayesian-scm`).
2. **Run the test suite locally** before pushing:
   ```bash
   pytest -q
   ```
   The smoke tests that depend on FLAML auto-skip when the backend is not
   importable, so a clean run can have a few skipped items.
3. **Run lint + format** (pre-commit does this automatically):
   ```bash
   ruff check src tests
   ruff format src tests
   ```
4. **Add tests** for any new behaviour. Coverage is currently gated at 50%
   in CI; we are raising this number gradually.
5. **Update the CHANGELOG**. Add a bullet to the `[Unreleased]` section
   under the appropriate heading (`Added` / `Changed` / `Fixed` / `Removed`).
6. **Open a pull request** with a clear description of the change and link
   any related issue.

## Code style

- Python ≥ 3.10 is required. New code may use `X | None` / `list[X]`
  annotations. Existing modules still use `Optional[X]` / `List[X]` from
  `typing`; both are fine until the planned `ruff --fix` UP* migration
  modernises them in one pass.
- Public functions get a NumPy-style docstring.
- New optional dependencies must be lazy-imported via
  `normet.utils._lazy.require` so importing `normet` never fails when a
  niche extra is absent.

## Filing bugs

- Search [existing issues](https://github.com/normet-dev/normet-py/issues)
  first.
- Include a minimal reproducer, your `normet` and Python versions
  (`python -m normet info` or `normet info`), and the full traceback.

## Proposing features

For non-trivial changes, please open an issue first to discuss the design.
This avoids wasted work when an idea overlaps with planned work or doesn't
fit the project's scope.

## Code of Conduct

By participating you agree to abide by the [Contributor Covenant Code of
Conduct](./CODE_OF_CONDUCT.md).
