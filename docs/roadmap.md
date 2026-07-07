# Roadmap to 1.0

`normet` is currently at version **0.4.x**. The path to 1.0 is about
*hardening* — not adding more features — and committing to API stability for
downstream users.

## What 1.0 means

When we cut `1.0.0`:

- **Public API is stable.** Anything exported from the top-level `normet`
  package will follow [Semantic Versioning](https://semver.org). Breaking
  changes require a major version bump and a deprecation window of at least
  one minor release.
- **Internal modules** (anything imported via `normet._something` or
  `normet.subpkg.module.private_helper`) are explicitly *not* part of the
  contract.
- **Optional dependencies** stay optional; importing `normet` itself will
  never require any of `flaml`, `lightgbm`, `xarray`, `dask`, `click`,
  `pyyaml`, `cdsapi`, `pymc`.

## Pre-1.0 milestones

| Status | Item |
|:------:|------|
| ✅ | Core normalisation / decomposition / SCM pipelines |
| ✅ | Walk-forward CV + per-time-bucket diagnostics |
| ✅ | Multiple SCM backends + diagnostics + inference |
| ✅ | Multi-site batch drivers |
| ✅ | Provenance archives (`NormetRun`) |
| ✅ | xarray ingestion |
| ✅ | CLI |
| ✅ | Sphinx docs |
| ✅ | Real-world data adapters (OpenAQ / ERA5 / EEA / AURN) |
| ✅ | Plotting suite (polar / PDP grid / decomposition / SCM dashboard) |
| ✅ | HTML / Markdown report generator |
| ✅ | Bayesian SCM + event detection |
| ⏳ | mypy strict on the public surface (currently lenient) |
| ⏳ | Test coverage ≥ 80% (currently 70% threshold) |
| ⏳ | A v1.0 deprecation policy in writing |

## Deprecation policy (proposed)

Starting at 1.0:

- A symbol marked for removal will:
  1. Emit a `DeprecationWarning` for at least one minor release.
  2. Be documented as deprecated in the docstring and CHANGELOG.
  3. Have a replacement linked from the warning message.
- Symbols are only removed in a major release.

## Out of scope for 1.0

- A GUI / Streamlit dashboard — community-contributed examples welcome but
  not part of the library proper.
