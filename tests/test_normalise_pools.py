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


def test_a_pool_draws_do_not_depend_on_the_other_pools():
    """A pool's stream comes from its name, so adding a pool that sorts before
    it does not move its draws."""
    _, b_one = _draws(
        _run(variables_resample=["a", "b"], resample_pools={"clean": CLEAN}, aggregate=False)
    )
    _, b_two = _draws(
        _run(
            variables_resample=["a", "b"],
            resample_pools={"a-pool": pd.DataFrame({"a": [100.0, 200.0]}), "clean": CLEAN},
            aggregate=False,
        )
    )
    np.testing.assert_array_equal(b_one, b_two)


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


@pytest.mark.parametrize("cached", [False, True])
def test_pools_must_be_a_mapping(tmp_path, cached):
    with pytest.raises(ConfigError, match="mapping of name -> DataFrame"):
        _run(
            variables_resample=["a", "b"],
            resample_pools=[("clean", CLEAN)],
            cache=str(tmp_path) if cached else None,
        )


def test_cache_tells_pools_apart(tmp_path):
    kw = {"variables_resample": ["b"], "cache": str(tmp_path)}
    first = _run(resample_pools={"clean": CLEAN}, **kw)
    other = _run(resample_pools={"clean": pd.DataFrame({"b": [5.0]})}, **kw)
    again = _run(resample_pools={"clean": CLEAN}, **kw)

    assert (other["normalised"] != first["normalised"]).all()
    pd.testing.assert_frame_equal(first, again)


# ---------------------------------------------------------------------------
# log_transform: df and model on the log1p scale, result back in original units
# ---------------------------------------------------------------------------


class _LogModel:
    """Small log-scale predictions, so expm1 stays in a sane range."""

    backend = "flaml"
    feature_names_in_ = np.array(["w"])
    feature_importances_ = np.ones(1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return 0.5 + 0.2 * X["w"].to_numpy(float)


def _log_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(original-unit frame, its log1p-space twin) for the log tests."""
    n = 20
    df_orig = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "value": 1.0 + np.arange(n) / 10.0,  # 1.0 .. 2.9 in original units
            "w": np.tile([0.0, 1.0, 2.0], n // 3 + 1)[:n],
        }
    )
    df_log = df_orig.copy()
    df_log["value"] = np.log1p(df_orig["value"])  # the log-space frame do_all hands over
    return df_orig, df_log


def _log_run(**kw) -> pd.DataFrame:
    kw = {"covariates": ["w"], "n_samples": 12, "seed": 3, "n_cores": 1, **kw}
    _, df_log = _log_frame()
    return normalise(df_log, _LogModel(), **kw)


def test_log_transform_inverts_point_and_quantiles():
    """The log path = the linear path run on the same log-space frame, with
    every column inverted: observed, normalised and the quantiles."""
    df_orig, _ = _log_frame()
    lin = _log_run(return_quantiles=[0.1, 0.5, 0.9])
    log = _log_run(log_transform=True, return_quantiles=[0.1, 0.5, 0.9])
    assert np.allclose(np.expm1(lin["normalised"]), log["normalised"])
    for q in ("q100", "q500", "q900"):
        assert np.allclose(np.expm1(lin[q]), log[q])
    # observed comes back in the original units the user gave
    assert np.allclose(log["observed"], df_orig["value"])
    assert (log["normalised"] >= 0).all()


def test_log_transform_default_is_bit_identical():
    plain = _run()
    explicit = _run(log_transform=False)
    pd.testing.assert_frame_equal(plain, explicit)


def test_log_transform_aggregate_false_wide_table_inverts():
    lin = _log_run(aggregate=False)
    log = _log_run(aggregate=False, log_transform=True)
    seed_cols = [c for c in lin.columns if c != "observed"]
    assert set(seed_cols) == {c for c in log.columns if c != "observed"}
    for c in seed_cols:
        assert np.allclose(np.expm1(lin[c]), log[c])
