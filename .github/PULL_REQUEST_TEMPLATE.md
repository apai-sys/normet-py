## Summary

<!-- 1-3 bullets describing the change. -->

## Related issue

<!-- "Fixes #123" / "Refs #456" -->

## Checklist

- [ ] Added or updated tests
- [ ] Added an entry to `CHANGELOG.md` under `[Unreleased]`
- [ ] `pytest` passes locally
- [ ] `ruff check src tests` is clean
- [ ] Public API additions follow the docstring style of neighbouring code
- [ ] No new unconditional optional-dependency imports (`from xarray import ...`
      at module top); use lazy `require()` instead

## Notes for reviewers

<!-- Anything subtle, performance tradeoffs, follow-up work, etc. -->
