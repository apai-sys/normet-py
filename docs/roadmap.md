# Roadmap

`normet` is at version **1.0.0**. 1.0 was about *hardening* rather than adding
features, and it commits the project to API stability for downstream users.

## What 1.0 means

- **Public API is stable.** Anything exported from the top-level `normet`
  package follows [Semantic Versioning](https://semver.org). Breaking changes
  require a major version bump and a deprecation window of at least one minor
  release.
- **Internal modules** (anything imported via `normet._something` or
  `normet.subpkg.module.private_helper`) are explicitly *not* part of the
  contract.
- **Optional dependencies** stay optional; importing `normet` itself will never
  require any of `flaml`, `lightgbm`, `xarray`, `dask`, `click`, `pyyaml`,
  `cdsapi`, `pymc`, `torch`, `chronos-forecasting`, `PySide6`.

## Shipped

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
| ✅ | Qt desktop GUI (`normet-gui`) with packaged installers |
| ✅ | Chronos-2 foundation model: zero-shot de-weathering, counterfactuals, station embeddings |
| ✅ | Physics-informed graph models (PI-STGNN, advection-diffusion-reaction loss) |

## Open

These were listed as 1.0 gates and 1.0 shipped without them, so they are now
ordinary follow-up work rather than release blockers.

| Status | Item |
|:------:|------|
| ⏳ | mypy strict on the public surface (currently lenient, but clean) |
| ⏳ | Test coverage ≥ 80% (currently 78%, with the CI gate at 70%) |
| ⏳ | Ratify the deprecation policy below |
| ⏳ | A zero-shot path through `do_all` / `pipeline`, which is shaped as train → normalise and so has nothing for Chronos-2 to train; `normet deweather` covers this from the CLI in the meantime |
| ⏳ | Connect `ChronosEmbedder.embed_stations` to `normet.multisite` — the 768-D station vectors and the multi-site pipeline currently have no meeting point |

## Deprecation policy (proposed, not yet ratified)

- A symbol marked for removal will:
  1. Emit a `DeprecationWarning` for at least one minor release.
  2. Be documented as deprecated in the docstring and CHANGELOG.
  3. Have a replacement linked from the warning message.
- Symbols are only removed in a major release.

This is still the *proposed* wording rather than a commitment in force: 1.0
shipped before it was ratified, so nothing has yet been deprecated under it.

## Out of scope

- A Streamlit or web dashboard. The desktop GUI covers the no-code workflow;
  a hosted equivalent is a separate project, and community-contributed
  examples are welcome but not part of the library proper.
