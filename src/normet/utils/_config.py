from __future__ import annotations

import warnings
from dataclasses import fields, replace
from typing import TypeVar

T = TypeVar("T")

#: Default random seed used across modelling, resampling, and causal routines.
#: Centralised here so a single change propagates to every public default.
DEFAULT_SEED = 7654321


def resolve_config(cls: type[T], config: T | None = None, **kwargs) -> T:
    """Build a dataclass config from an optional instance and keyword overrides.

    Unknown kwargs emit a UserWarning; None-valued kwargs do not override an
    existing config field (so callers can pass optional params without clobbering
    explicit config values).
    """
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(kwargs) - known
    if unknown:
        warnings.warn(
            f"Unknown config field(s): {sorted(unknown)}. Known fields: {sorted(known)}",
            UserWarning,
            stacklevel=3,
        )
    if config is None:
        return cls(**{k: v for k, v in kwargs.items() if k in known})
    overrides = {k: v for k, v in kwargs.items() if k in known and v is not None}
    return replace(config, **overrides) if overrides else config  # type: ignore[type-var]
