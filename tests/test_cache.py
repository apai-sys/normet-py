"""Cache helpers — content-hash stability and joblib Memory wiring."""

import pandas as pd

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
    kw = dict(value="value", backend="counting", feature_names=["x"], cache=str(tmp_path / "c"))

    m1 = train_model(df, **kw)
    m2 = train_model(df, **kw)  # identical data + config → served from disk
    assert backend.train_calls == 1
    assert m1 == m2

    # Different data content → cache miss → backend retrains.
    df2 = df.copy()
    df2.loc[0, "value"] = 99.0
    train_model(df2, **kw)
    assert backend.train_calls == 2


def test_train_model_no_cache_always_trains(tmp_path):
    backend = _CountingBackend()
    backend_registry.register(backend)
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0]})
    kw = dict(value="value", backend="counting", feature_names=["x"])
    train_model(df, **kw)
    train_model(df, **kw)
    assert backend.train_calls == 2  # no cache → trains every time
