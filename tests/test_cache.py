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


def test_train_model_no_cache_always_trains(tmp_path):
    backend = _CountingBackend()
    backend_registry.register(backend)
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0]})
    kw = dict(target="value", backend="counting", covariates=["x"])
    train_model(df, **kw)
    train_model(df, **kw)
    assert backend.train_calls == 2  # no cache → trains every time
