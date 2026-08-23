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
| ✅ | Zero-shot paths through `do_all` and `decompose(method="meteorology")` |
| ✅ | `embed_multisite`: station embeddings meet the multi-site drivers |
| ✅ | mypy strict on the public surface (`warn_return_any` and `disallow_any_generics` deliberately off; `disallow_untyped_decorators` off for `normet.cli` alone -- see CHANGELOG for why) |
| ✅ | Test coverage ≥ 80% (80% measured; the CI gate stays at 70% so an unlucky branch does not block a merge) |

## Open

These were listed as 1.0 gates and 1.0 shipped without them, so they are now
ordinary follow-up work rather than release blockers.

| Status | Item |
|:------:|------|
| ⏳ | Fine-tuning via `Chronos2Pipeline.fit` (full or LoRA). Everything here is zero-shot; adapting the checkpoint to a site's own record is the largest unused capability, and the one plausible route to making the model attend to calendar covariates it currently ignores |
| ⏳ | `cross_learning=True` for joint multi-site prediction — the upstream docs say it helps most where individual series have little history, which is exactly a newly commissioned station, but also that it does not always help and must be tested per use case |
| ⏳ | Multivariate targets: several species at one site, or neighbouring stations as variates, which is where `normet.physics`'s graphs would meet the foundation model |
| ⏳ | Categorical covariates (site type, wind sector), which Chronos-2 encodes natively and normet currently has to one-hot |

## Deprecation policy

In force. It covers the public API as defined under *What 1.0 means* above --
anything exported from the top-level `normet` package -- and nothing else.

- A symbol marked for removal will:
  1. Emit a `DeprecationWarning` for at least one minor release.
  2. Be documented as deprecated in the docstring and CHANGELOG.
  3. Have a replacement linked from the warning message.
- Symbols are only removed in a major release.
- A symbol that has never appeared in a release is not covered and may be
  changed or withdrawn outright: no downstream code can yet depend on it, so a
  warning cycle would protect nobody and would ship the mistake instead of
  fixing it. `cluster_multisite` was withdrawn on those grounds before it ever
  reached a release.

Nothing has been deprecated under this policy so far, so no removal is due.

## Out of scope

- A Streamlit or web dashboard. The desktop GUI covers the no-code workflow;
  a hosted equivalent is a separate project, and community-contributed
  examples are welcome but not part of the library proper.
