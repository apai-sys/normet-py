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
import inspect
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from ._lazy import require
from .logging import get_logger

log = get_logger(__name__)

__all__ = [
    "make_memory",
    "dataframe_hash",
    "config_hash",
    "model_hash",
    "package_version",
    "dependency_versions",
    "callable_hash",
]

#: The packages whose code can change a cached result: the model backends
#: (a fit or a prediction can move with a release) and the numeric/data
#: libraries the resampling runs through. Deliberately not every installed
#: package: each name here must earn its cache invalidation.
_RESULT_PACKAGES = (
    "lightgbm",
    "flaml",
    "xgboost",
    "scikit-learn",
    "numpy",
    "pandas",
    "scipy",
)


def package_version() -> str:
    """The installed normet version, for cache keys.

    A cache entry is valid only for the code that wrote it, and the failure
    it guards against is silent: upgrade the package, rerun the script with
    the same cache directory, and the old result is served with no sign that
    the code changed. Dependency upgrades are covered by
    :func:`dependency_versions`; a source change that keeps the version (an
    editable checkout) is not covered -- clear the directory for one.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("normet")
    except PackageNotFoundError:  # a source tree with no installed metadata
        return "unknown"


@lru_cache(maxsize=1)
def dependency_versions() -> tuple[tuple[str, str], ...]:
    """Installed versions of the packages that can change a cached result.

    A new LightGBM/FLAML/numpy can change a fit or a prediction without the
    caller's data or configuration changing, so an entry written under one
    version set must miss under another. Only installed packages are listed
    -- installing one of the optional backends later changes the tuple, and
    with it every key. Cached for the process: versions do not change under
    a running interpreter.
    """
    from importlib.metadata import PackageNotFoundError, version

    found = []
    for name in _RESULT_PACKAGES:
        try:
            found.append((name, version(name)))
        except PackageNotFoundError:
            continue
    return tuple(found)


def _cloudpickle() -> Any:
    """The cloudpickle module: installed, else joblib's bundled copy."""
    try:
        import cloudpickle

        return cloudpickle
    except ImportError:  # joblib bundles it for its process backend
        from joblib.externals import cloudpickle

        return cloudpickle


def callable_hash(fn: Any) -> str:
    """A content fingerprint of a callable, for a cache key.

    ``repr(fn)`` names the function object's address. It is therefore tied
    to one process -- a fresh run never hits the entry -- and, worse, a
    closure's captured value can change without the repr changing: the same
    function object then selects a different row set while the key stays,
    and the cache serves the draws of the old filter. Cloudpickle serialises
    the code and the captured state (a closure's cells, a bound method's
    instance, a partial's arguments), and its bytes are stable across
    processes for the same definition. A function imported from a module is
    stored by reference, so its source text joins the fingerprint; a
    callable cloudpickle cannot serialise falls back to ``repr`` -- the old,
    per-process behaviour.
    """
    try:
        blob = _cloudpickle().dumps(fn)
    except Exception:  # a C callable, or state cloudpickle cannot reach
        return config_hash("repr", repr(fn))
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):  # defined in an interpreter, or a builtin
        source = ""
    digest = hashlib.sha1(blob)
    digest.update(b"\0")
    digest.update(source.encode("utf-8", "replace"))
    return digest.hexdigest()


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
        Soft size cap; passed to ``Memory.reduce_size`` once, which deletes
        the oldest entries until the store fits. ``joblib.Memory`` takes no
        size cap in its constructor and never prunes on its own, so this is
        applied whenever a handle is created -- not on every write. A single
        item larger than the cap leaves the store slightly above it.

    Returns
    -------
    joblib.Memory
        Use the returned object's ``.cache`` decorator to memoize callables.
    """
    joblib = require("joblib", hint="pip install joblib")
    Memory = joblib.Memory
    p = Path(location)
    p.mkdir(parents=True, exist_ok=True)
    mem = Memory(location=str(p), verbose=verbose)
    if bytes_limit is not None:
        # The constructor has taken no size limit in any joblib release; the
        # pruning lives on the instance. Passing it to ``Memory`` itself --
        # what this function used to do -- raised TypeError on joblib 1.5.
        mem.reduce_size(bytes_limit=int(bytes_limit))
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
    # The row hash covers values only, so a frame whose columns were renamed
    # (or whose columns changed dtype -- bool vs int rows hash alike) would
    # collide with the original and a cache entry or provenance ``data_hash``
    # could not tell them apart. Fold the names and dtypes in as well.
    digest = hashlib.sha1()
    digest.update("|".join(f"{name}:{dtype}" for name, dtype in sub.dtypes.items()).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(row_hash.values.tobytes())  # type: ignore[union-attr]
    return digest.hexdigest()


def model_hash(model: Any) -> str:
    """
    Content hash of an arbitrary (typically trained-model) object.

    Unlike :func:`dataframe_hash`, which uses a fast pandas-native row hash,
    this delegates to ``joblib.hash`` (pickle-based) since trained model
    objects have no cheaper stable fingerprint. Intended for cache keys that
    need to invalidate when the model itself changes -- e.g. re-running
    :func:`normet.normalise` with a re-fit model on otherwise identical data
    and config should not silently reuse a stale cached result.

    Parameters
    ----------
    model : object
        Any picklable object, typically a trained model.

    Returns
    -------
    str
        Hex digest from ``joblib.hash``.
    """
    joblib = require("joblib", hint="pip install joblib")
    return str(joblib.hash(model))


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
