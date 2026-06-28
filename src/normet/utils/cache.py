# src/normet/utils/cache.py
"""
Optional on-disk caching for expensive ``normet`` calls.

``normet`` does *not* enable caching by default. Opt-in by creating a
:class:`joblib.Memory` instance and wrapping the function(s) you want to
memoize:

.. code-block:: python

    import normet as nm
    memory = nm.make_memory(".normet_cache")

    @memory.cache
    def train_cached(df_hash, **cfg):
        # df_hash drives cache key; the real DataFrame must be loaded inside
        df = pd.read_parquet("data.parquet")
        return nm.train_model(df, **cfg)

For a quick "hash this DataFrame" key, use :func:`dataframe_hash`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ._lazy import require
from .logging import get_logger

log = get_logger(__name__)

__all__ = ["make_memory", "dataframe_hash", "config_hash"]


def make_memory(
    location: str | Path = ".normet_cache",
    *,
    verbose: int = 0,
    bytes_limit: int | None = None,
) -> Any:
    """
    Create a :class:`joblib.Memory` cache bound to a directory.

    Parameters
    ----------
    location : str | Path, default ".normet_cache"
        Disk path for cached artefacts. Created if missing.
    verbose : int, default 0
        Forwarded to ``joblib.Memory``.
    bytes_limit : int, optional
        Soft size cap; pass through to ``joblib.Memory(bytes_limit=...)`` if
        supported by your joblib version.

    Returns
    -------
    joblib.Memory
        Use the returned object's ``.cache`` decorator to memoize callables.
    """
    joblib = require("joblib", hint="pip install joblib")
    Memory = joblib.Memory
    p = Path(location)
    p.mkdir(parents=True, exist_ok=True)
    kwargs = {"location": str(p), "verbose": verbose}
    if bytes_limit is not None:
        kwargs["bytes_limit"] = bytes_limit
    mem = Memory(**kwargs)
    log.info("normet cache directory: %s", p.resolve())
    return mem


def dataframe_hash(
    df: pd.DataFrame,
    *,
    cols: Sequence[str] | None = None,
    include_index: bool = True,
) -> str:
    """Compute a content hash of a DataFrame.

    The hash is stable across runs; row order only matters if you don't
    sort first.

    Parameters
    ----------
    df : pandas.DataFrame
    cols : sequence of str, optional
        Restrict hashing to a subset of columns.
    include_index : bool, default True
        Hash the index alongside the values.

    Returns
    -------
    str
        Hex digest (SHA-1 of pandas' row-hash array).
    """
    sub = df[list(cols)] if cols else df
    row_hash = pd.util.hash_pandas_object(sub, index=include_index, encoding="utf8")
    digest = hashlib.sha1(row_hash.values.tobytes()).hexdigest()  # type: ignore[union-attr]
    return digest


def config_hash(*objs: Any) -> str:
    """
    SHA-1 hash of the repr of arbitrary config objects.

    Suitable for cache keys built from a mix of small primitives, dicts,
    and dataclass-like config. Don't pass huge DataFrames here — use
    :func:`dataframe_hash` for those.

    Parameters
    ----------
    *objs : object
        One or more configuration objects whose ``repr`` will be hashed.

    Returns
    -------
    str
        Hex digest of the SHA-1 hash.
    """
    parts: Iterable[str] = (repr(o) for o in objs)
    blob = "||".join(parts).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()
