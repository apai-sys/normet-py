"""Cache helpers — content-hash stability and joblib Memory wiring."""

import numpy as np
import pandas as pd

from normet.analysis.normalise import normalise
from normet.backends import backend_registry
from normet.model.train import train_model
from normet.utils.cache import config_hash, dataframe_hash, make_memory


def test_dataframe_hash_stable(synthetic_aq):
    h1 = dataframe_hash(synthetic_aq)
    h2 = dataframe_hash(synthetic_aq.copy())
    assert h1 == h2


def test_dataframe_hash_changes_on_edit(synthetic_aq):
    df = synthetic_aq.copy()
    h1 = dataframe_hash(df)
    df.iloc[0, df.columns.get_loc("PM2.5")] += 1.0
    h2 = dataframe_hash(df)
    assert h1 != h2


def test_config_hash_consistent():
    assert config_hash({"a": 1, "b": [2, 3]}) == config_hash({"a": 1, "b": [2, 3]})
    assert config_hash(1) != config_hash(2)


def test_make_memory_round_trip(tmp_path):
    mem = make_memory(tmp_path / "cache")

    calls = {"n": 0}

    @mem.cache
    def expensive(x):
        calls["n"] += 1
        return x * 2

    assert expensive(3) == 6
    assert expensive(3) == 6  # second call hits the cache
    assert calls["n"] == 1


def test_make_memory_bytes_limit_prunes_the_store(tmp_path):
    """``bytes_limit`` is a cap on the existing store, applied by ``reduce_size``.

    ``joblib.Memory`` has no size limit in its constructor (the argument this
    helper used to pass raised ``TypeError`` on joblib 1.5) and never prunes
    on its own; the bound is applied through ``reduce_size`` when the handle
    is created, deleting the oldest entries until the store fits.
    """
    mem = make_memory(tmp_path / "cache")

    @mem.cache
    def big(i):
        return np.zeros(100_000)  # ~800 KB per entry, uncompressed

    for i in range(3):
        big(i)
    store = tmp_path / "cache"
    before = len(list(store.glob("joblib/**/output.pkl")))
    assert before == 3

    make_memory(store, bytes_limit=1_000_000)
    after = len(list(store.glob("joblib/**/output.pkl")))
    assert after < before, "the size cap did not prune the over-limit store"


def test_callable_hash_follows_captured_state():
    """A closure's captured value is part of what the filter selects.

    ``repr`` names the function object's address and does not change when
    the closure's state does: the same object could select a different row
    set while the cache key stayed, and the cache served the old draws.
    """
    from normet.utils.cache import callable_hash

    state = [10]

    def keep(x):
        return x > state[0]

    first = callable_hash(keep)
    assert callable_hash(keep) == first  # unchanged state, unchanged key
    state[0] = 20
    assert callable_hash(keep) != first


def test_callable_hash_is_stable_across_definitions():
    """Two runs defining the same filter must share a key, across processes.

    The old ``repr`` key was tied to one process's function address, so a
    rerun never hit; content hashing makes the same definition reuse the
    entry.
    """
    from normet.utils.cache import callable_hash

    def make(threshold):
        return lambda x: x > threshold

    assert callable_hash(make(10)) == callable_hash(make(10))
    assert callable_hash(make(10)) != callable_hash(make(20))


def test_dependency_versions_lists_the_installed_result_packages():
    from normet.utils.cache import dependency_versions

    versions = dict(dependency_versions())
    assert {"numpy", "pandas", "scipy", "scikit-learn"} <= set(versions)
    assert all(isinstance(v, str) and v for v in versions.values())


class _CountingBackend:
    """Minimal Backend that records how many times train() is invoked."""

    name = "counting"

    def __init__(self):
        self.train_calls = 0

    def train(self, df, **kwargs):
        self.train_calls += 1
        return {"trained": True, "seed": kwargs.get("seed"), "n": self.train_calls}

    def save(self, model, path=".", filename="automl.joblib"):  # pragma: no cover
        return ""

    def load(self, path=".", filename=None):  # pragma: no cover
        return {}


def test_train_model_cache_hits_on_repeat(tmp_path):
    backend = _CountingBackend()
    backend_registry.register(backend)

    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]})
    kw = dict(target="value", backend="counting", covariates=["x"], cache=str(tmp_path / "c"))

    m1 = train_model(df, **kw)
    m2 = train_model(df, **kw)  # identical data + config → served from disk
    assert backend.train_calls == 1
    assert m1 == m2

    # Different data content → cache miss → backend retrains.
    df2 = df.copy()
    df2.loc[0, "value"] = 99.0
    train_model(df2, **kw)
    assert backend.train_calls == 2


class _StubBackend:
    """Registry entry so ``ml_predict`` accepts ``_StubPredictor``."""

    name = "stub"

    def train(self, df, **kwargs):  # pragma: no cover
        return None

    def save(self, model, path=".", filename=""):  # pragma: no cover
        return ""

    def load(self, path=".", filename=None):  # pragma: no cover
        return {}


class _StubPredictor:
    """Constant predictor whose call count lives on the *class*.

    Both ``calls`` and ``backend`` are class attributes on purpose: anything in
    the instance ``__dict__`` would change ``model_hash(model)`` between calls,
    defeating the cache and making the test below vacuously pass.
    """

    calls = 0
    backend = "stub"

    def predict(self, X):
        type(self).calls += 1
        return np.zeros(len(X))


def test_normalise_cache_key_ignores_resample_order(tmp_path, synthetic_aq):
    """Same resampled *set* in a different order must hit the cache.

    Regression guard for the key built in ``normalise``: ``variables_resample``
    used to be hashed as given, so callers that reach the same set by different
    routes -- ``decompose()``'s shrinking sublist of a feature-importance or
    permutation order -- recomputed every time.
    """
    backend_registry.register(_StubBackend())
    df = synthetic_aq.rename(columns={"PM2.5": "value"})
    kw = dict(covariates=["t2m", "blh", "u10"], n_samples=2, seed=1, cache=str(tmp_path / "c"))
    model = _StubPredictor()
    _StubPredictor.calls = 0

    normalise(df, model, variables_resample=["t2m", "blh", "u10"], **kw)
    first = _StubPredictor.calls
    assert first > 0

    normalise(df, model, variables_resample=["blh", "u10", "t2m"], **kw)
    assert _StubPredictor.calls == first, "same set, different order should be a cache hit"

    # A genuinely different set must still miss -- otherwise the assert above
    # would pass for the wrong reason (e.g. the set dropping out of the key).
    normalise(df, model, variables_resample=["t2m", "blh"], **kw)
    assert _StubPredictor.calls > first


def test_normalise_cache_key_sees_the_conditional_filter_column(tmp_path, synthetic_aq):
    """The rows ``conditional_on`` keeps depend on its column's values.

    A caller's ``resample_df`` may carry that column without it being a
    covariate or a resampled variable ("season", a wind sector), and only
    the resampled columns were hashed: two pools differing only in the
    filter column shared a cache entry, so the second call was served the
    first pool's draws.
    """
    backend_registry.register(_StubBackend())
    df = synthetic_aq.rename(columns={"PM2.5": "value"}).copy()
    pool = df.copy()
    pool["sector"] = ["N", "S"] * (len(pool) // 2) + ["N"] * (len(pool) % 2)
    kw = dict(
        covariates=["t2m", "blh", "u10"],
        variables_resample=["t2m", "blh", "u10"],
        conditional_on={"sector": ["N"]},
        resample_df=pool,
        n_samples=2,
        seed=1,
        cache=str(tmp_path / "c"),
    )
    model = _StubPredictor()
    _StubPredictor.calls = 0

    normalise(df, model, **kw)
    first = _StubPredictor.calls
    assert first > 0

    # The same resampled columns, but the filter now keeps the other half of
    # the pool: the cached draws are of the wrong row set.
    flipped_pool = pool.copy()
    flipped_pool["sector"] = pool["sector"].map({"N": "S", "S": "N"})
    normalise(df, model, **{**kw, "resample_df": flipped_pool})
    assert _StubPredictor.calls > first


def test_normalise_cache_key_sees_a_pools_served_columns(tmp_path, synthetic_aq):
    """A pool serves the ``variables_resample`` columns it carries.

    Those are normally covariates, but ``variables_resample`` is not
    validated against ``covariates``: a pool may carry a resampled column
    the covariate list does not name. Hashing only the covariates let two
    pools differing solely in that column share a cache entry.
    """
    backend_registry.register(_StubBackend())
    df = synthetic_aq.rename(columns={"PM2.5": "value"}).copy()
    pool = df[["t2m", "blh"]].copy()
    kw = dict(
        covariates=["t2m"],
        variables_resample=["t2m", "blh"],
        resample_pools={"p": pool},
        n_samples=2,
        seed=1,
        cache=str(tmp_path / "c"),
    )
    model = _StubPredictor()
    _StubPredictor.calls = 0

    normalise(df, model, **kw)
    first = _StubPredictor.calls
    assert first > 0

    flipped = pool.copy()
    flipped["blh"] = flipped["blh"] + 100.0
    normalise(df, model, **{**kw, "resample_pools": {"p": flipped}})
    assert _StubPredictor.calls > first


def test_train_model_no_cache_always_trains(tmp_path):
    backend = _CountingBackend()
    backend_registry.register(backend)
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0]})
    kw = dict(target="value", backend="counting", covariates=["x"])
    train_model(df, **kw)
    train_model(df, **kw)
    assert backend.train_calls == 2  # no cache → trains every time


def test_train_model_cache_key_carries_the_package_version(tmp_path, monkeypatch):
    """An upgraded package must retrain, not serve the old code's model.

    The failure is silent: same data and config, a new normet, and the cache
    answers with a model the previous release fitted.
    """
    backend = _CountingBackend()
    backend_registry.register(backend)
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0]})
    kw = dict(target="value", backend="counting", covariates=["x"], cache=str(tmp_path / "c"))

    import normet.utils.cache as cache_mod

    monkeypatch.setattr(cache_mod, "package_version", lambda: "1")
    train_model(df, **kw)
    assert backend.train_calls == 1
    train_model(df, **kw)  # same version → still a cache hit
    assert backend.train_calls == 1

    monkeypatch.setattr(cache_mod, "package_version", lambda: "2")
    train_model(df, **kw)
    assert backend.train_calls == 2


def test_normalise_cache_key_carries_the_package_version(tmp_path, synthetic_aq, monkeypatch):
    """Same guard on the draw-level cache: a release change must miss."""
    backend_registry.register(_StubBackend())
    df = synthetic_aq.rename(columns={"PM2.5": "value"})
    kw = dict(covariates=["t2m", "blh", "u10"], n_samples=2, seed=1, cache=str(tmp_path / "c"))
    model = _StubPredictor()
    _StubPredictor.calls = 0

    import normet.utils.cache as cache_mod

    monkeypatch.setattr(cache_mod, "package_version", lambda: "1")
    normalise(df, model, **kw)
    first = _StubPredictor.calls
    assert first > 0

    monkeypatch.setattr(cache_mod, "package_version", lambda: "2")
    normalise(df, model, **kw)
    assert _StubPredictor.calls > first


def test_cache_keys_carry_the_dependency_versions(tmp_path, synthetic_aq, monkeypatch):
    """A new backend/numeric release can change a fit or a prediction.

    The caller's data and config are unchanged, so without the versions the
    cache would serve a result the old libraries computed.
    """
    backend_registry.register(_StubBackend())
    backend_registry.register(_CountingBackend())
    df = synthetic_aq.rename(columns={"PM2.5": "value"})
    model = _StubPredictor()
    _StubPredictor.calls = 0
    nkw = dict(covariates=["t2m", "blh", "u10"], n_samples=2, seed=1, cache=str(tmp_path / "c"))
    tr = pd.DataFrame({"value": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0]})
    tkw = dict(target="value", backend="counting", covariates=["x"], cache=str(tmp_path / "c2"))

    import normet.utils.cache as cache_mod

    monkeypatch.setattr(cache_mod, "dependency_versions", lambda: (("lightgbm", "4.0"),))
    normalise(df, model, **nkw)
    first = _StubPredictor.calls
    assert first > 0
    train_model(tr, **tkw)
    assert backend_registry.get("counting").train_calls == 1  # type: ignore[attr-defined]

    monkeypatch.setattr(cache_mod, "dependency_versions", lambda: (("lightgbm", "5.0"),))
    normalise(df, model, **nkw)
    assert _StubPredictor.calls > first
    train_model(tr, **tkw)
    assert backend_registry.get("counting").train_calls == 2  # type: ignore[attr-defined]


def test_normalise_cache_key_sees_a_callables_captured_state(tmp_path, synthetic_aq):
    """The same function object can select different rows after its state moves."""
    backend_registry.register(_StubBackend())
    df = synthetic_aq.rename(columns={"PM2.5": "value"})
    state = [0.0]

    def keep(x):
        return x >= state[0]

    kw = dict(
        covariates=["t2m", "blh", "u10"],
        variables_resample=["t2m", "blh", "u10"],
        conditional_on={"t2m": keep},
        n_samples=2,
        seed=1,
        cache=str(tmp_path / "c"),
    )
    model = _StubPredictor()
    _StubPredictor.calls = 0

    normalise(df, model, **kw)
    first = _StubPredictor.calls
    assert first > 0

    state[0] = 15.0  # same callable, different filter
    normalise(df, model, **kw)
    assert _StubPredictor.calls > first


def test_normalise_cache_key_hashes_a_long_condition_by_content(tmp_path, synthetic_aq):
    """pandas/NumPy abbreviate long reprs: two filters could share a key.

    ``repr(np.zeros(20_000))`` and the same array with one middle element
    changed print identically, so the old key could not tell the filters
    apart (and the cache served the other row set).
    """
    backend_registry.register(_StubBackend())
    df = synthetic_aq.rename(columns={"PM2.5": "value"}).copy()
    pool = df.copy()
    pool["cluster"] = np.arange(len(pool)) % 24
    allowed = np.zeros(20_000, dtype=int)
    other = allowed.copy()
    other[10_000] = 1
    assert repr(allowed) == repr(other)  # the elided middle hides the difference

    kw = dict(
        covariates=["t2m", "blh", "u10"],
        variables_resample=["t2m", "blh", "u10"],
        conditional_on={"cluster": allowed},
        resample_df=pool,
        n_samples=2,
        seed=1,
        cache=str(tmp_path / "c"),
    )
    model = _StubPredictor()
    _StubPredictor.calls = 0

    normalise(df, model, **kw)
    first = _StubPredictor.calls
    assert first > 0

    normalise(df, model, **{**kw, "conditional_on": {"cluster": other}})
    assert _StubPredictor.calls > first
