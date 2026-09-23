"""normalise(resample_pools=...): variables drawn from their own pools.

The model predicts ``a + 1000 * b``, and ``a`` stays below 1000, so every
per-draw prediction (``aggregate=False``) can be decoded back into the ``a`` and
``b`` values that draw used -- which shows exactly which pool each came from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from normet import normalise
from normet.analysis.normalise import generate_resampled
from normet.exceptions import ConfigError, DataError

N = 40
COVS = ["a", "b", "hour"]


class Model:
    backend = "flaml"
    feature_names_in_ = np.array(COVS)
    feature_importances_ = np.ones(len(COVS))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["a"].to_numpy(float) + 1000.0 * X["b"].to_numpy(float)


def _frame() -> pd.DataFrame:
    t = np.arange(N)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=N, freq="h"),
            "value": 1.0,
            "a": t.astype(float),  # 0..39
            "b": 1.0 + (t % 3),  # 1, 2, 3
            "hour": t % 24,
        }
    )


CLEAN = pd.DataFrame({"b": [7.0, 8.0]})


def _draws(res: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    wide = res.drop(columns="observed").to_numpy(float)
    b = np.floor(wide / 1000.0)
    return wide - 1000.0 * b, b


def _run(**kw) -> pd.DataFrame:
    base = {"covariates": COVS, "n_samples": 12, "seed": 3, "n_cores": 1, "batch_size": 0}
    return normalise(_frame(), Model(), **{**base, **kw})


def test_pool_variables_come_from_the_pool_and_the_rest_from_the_record():
    a, b = _draws(
        _run(variables_resample=["a", "b"], resample_pools={"clean": CLEAN}, aggregate=False)
    )

    assert set(np.unique(b)) == {7.0, 8.0}
    assert set(np.unique(a)) <= set(_frame()["a"])
    assert len(np.unique(a)) > 10


def test_without_pools_both_come_from_the_record():
    _, b = _draws(_run(variables_resample=["a", "b"], aggregate=False))
    assert set(np.unique(b)) == {1.0, 2.0, 3.0}


def test_a_pool_keeps_its_rows_together():
    pool = pd.DataFrame({"a": [100.0, 200.0, 300.0], "b": [1.0, 2.0, 3.0]})
    a, b = _draws(_run(variables_resample=["a", "b"], resample_pools={"p": pool}, aggregate=False))

    np.testing.assert_array_equal(a, 100.0 * b)


def test_adding_a_pool_leaves_the_other_variables_draws_alone():
    """The record's stream is untouched: a is drawn identically with or without
    a pool for b."""
    a_plain, _ = _draws(_run(variables_resample=["a", "b"], aggregate=False))
    a_pooled, _ = _draws(
        _run(variables_resample=["a", "b"], resample_pools={"clean": CLEAN}, aggregate=False)
    )
    np.testing.assert_array_equal(a_plain, a_pooled)


def test_a_pooled_variable_draws_do_not_move_when_others_are_fixed():
    """Paired differences need common random numbers: b's draws must not depend
    on whether a is resampled in the same call."""
    _, b_both = _draws(
        _run(variables_resample=["a", "b"], resample_pools={"clean": CLEAN}, aggregate=False)
    )
    _, b_alone = _draws(
        _run(variables_resample=["b"], resample_pools={"clean": CLEAN}, aggregate=False)
    )
    np.testing.assert_array_equal(b_both, b_alone)


def test_a_pool_whose_variables_are_all_fixed_is_ignored():
    fixed = _run(variables_resample=["a"], resample_pools={"clean": CLEAN})
    plain = _run(variables_resample=["a"])
    pd.testing.assert_frame_equal(fixed, plain)


def test_vectorised_and_batched_paths_agree():
    kw = {"variables_resample": ["a", "b"], "resample_pools": {"clean": CLEAN}}
    pd.testing.assert_frame_equal(_run(batch_size=0, **kw), _run(batch_size=5, **kw))


def test_memory_save_path_draws_from_the_pool_too():
    res = _run(
        variables_resample=["a", "b"],
        resample_pools={"clean": CLEAN},
        memory_save=True,
        batch_size=None,
        aggregate=False,
    )
    _, b = _draws(res)
    assert set(np.unique(b)) <= {7.0, 8.0}


def test_generate_resampled_uses_the_pool():
    out = generate_resampled(_frame(), ["a", "b"], True, 11, _frame(), resample_pools={"c": CLEAN})
    assert set(out["b"]) <= {7.0, 8.0}
    assert set(out["a"]) <= set(_frame()["a"])


def test_conditional_on_filters_the_record_not_the_pools():
    a, b = _draws(
        _run(
            variables_resample=["a", "b"],
            resample_pools={"clean": CLEAN},
            conditional_on={"hour": [0, 1]},
            aggregate=False,
        )
    )
    assert set(np.unique(a)) <= {0.0, 1.0, 24.0, 25.0}
    assert set(np.unique(b)) <= {7.0, 8.0}


@pytest.mark.parametrize(
    ("pools", "error", "match"),
    [
        ({"p": CLEAN, "q": pd.DataFrame({"b": [9.0]})}, ConfigError, "two resample pools"),
        ({"p": CLEAN.iloc[:0]}, DataError, "no rows"),
        ({"p": pd.DataFrame({"traj_resid_contnent": [0.5]})}, ConfigError, "no covariate"),
        ({"p": [1, 2, 3]}, ConfigError, "must be a DataFrame"),
    ],
)
def test_bad_pools_are_refused(pools, error, match):
    with pytest.raises(error, match=match):
        _run(variables_resample=["a", "b"], resample_pools=pools)


def test_cache_tells_pools_apart(tmp_path):
    kw = {"variables_resample": ["b"], "cache": str(tmp_path)}
    first = _run(resample_pools={"clean": CLEAN}, **kw)
    other = _run(resample_pools={"clean": pd.DataFrame({"b": [5.0]})}, **kw)
    again = _run(resample_pools={"clean": CLEAN}, **kw)

    assert (other["normalised"] != first["normalised"]).all()
    pd.testing.assert_frame_equal(first, again)
