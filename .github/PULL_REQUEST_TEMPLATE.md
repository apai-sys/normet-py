## Summary

<!-- 1-3 bullets describing the change. -->

## Related issue

<!-- "Fixes #123" / "Refs #456" -->

## Checklist

- [ ] Added or updated tests
- [ ] Added an entry to `CHANGELOG.md` under `[Unreleased]`
- [ ] `pytest` passes locally
- [ ] `ruff check src tests` and `ruff format --check src tests` are clean
- [ ] `mypy` is clean
- [ ] New capabilities are in the README's "Core capabilities" table and the user guide
- [ ] Docs changes build with `sphinx-build -W -b html docs docs/_build/html`
- [ ] Public API additions follow the docstring style of neighbouring code
- [ ] No new unconditional optional-dependency imports (`from xarray import ...`
      at module top); use lazy `require()` instead

## Notes for reviewers

<!-- Anything subtle, performance tradeoffs, follow-up work, etc. -->
