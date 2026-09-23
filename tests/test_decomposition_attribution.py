"""decom_met attribution: sequential vs Shapley, feature groups, forwarded pools.

A known coalition game stands in for the model. ``normalise`` is replaced by a
function whose "normalised" series depends only on which features are held at
their observed values, so the exact contribution each method must return can be
written down: three main effects plus an a-b interaction that only appears when
both a and b are held. The interaction is what separates the methods --
sequential fixing credits it to whichever of a and b comes second, Shapley
splits it evenly.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

import normet.analysis.decomposition  # noqa: F401  (registers the module)
from normet import decompose
from normet.analysis.decomposition import decom_emi, decom_met
from normet.exceptions import ConfigError

dmod = sys.modules["normet.analysis.decomposition"]

needs_lgb = pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None, reason="lightgbm not installed"
)

N = 48
T = np.arange(N, dtype=float)
MAIN = {"a": np.sin(T / 5.0), "b": 0.5 * np.cos(T / 3.0), "c": 0.1 * T}
AB = 0.3 + 0.2 * np.sin(T / 7.0)
FEATS = ["a", "b", "c", "hour"]


class Model:
    """Feature names and importances are all extract_features needs."""

    backend = "flaml"

    def __init__(self, features: list[str], importances: list[float]) -> None:
        self.feature_names_in_ = np.array(features)
        self.feature_importances_ = np.asarray(importances, dtype=float)


def _model() -> Model:
    # Importance order a, b, c: the default sequential order.
    return Model(FEATS, [3.0, 2.0, 1.0, 0.5])


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=N, freq="h"),
            "value": 12.0 + np.cos(T / 4.0),
            "a": 1.0,
            "b": 2.0,
            "c": 3.0,
            "hour": T % 24,
        }
    )


def _game(fixed: set[str], rows: np.ndarray | slice = slice(None)) -> np.ndarray:
    out = 10.0 + sum((MAIN[f][rows] for f in fixed), np.zeros(N)[rows])
    return out + AB[rows] if {"a", "b"} <= fixed else out


@pytest.fixture()
def calls(monkeypatch):
    """Replace normalise with the game; record every call's keyword arguments."""
    seen: list[dict] = []

    def fake_normalise(df, model, *, covariates, variables_resample, **kw):
        seen.append({"resample": sorted(variables_resample), **kw})
        fixed = {f for f in MAIN if f in covariates and f not in variables_resample}
        d = df.set_index("date")
        rows = ((d.index - d.index[0].normalize()) // pd.Timedelta("1h")).to_numpy()
        return pd.DataFrame(
            {"observed": d["value"], "normalised": _game(fixed, rows)}, index=d.index
        )

    monkeypatch.setattr(dmod, "normalise", fake_normalise)
    return seen


def _assert_closes(res: pd.DataFrame, columns: list[str]) -> None:
    """Contributions add up to prediction - emi_total; met_noise is the residual."""
    everything = _game({"a", "b", "c"})
    np.testing.assert_allclose(res[columns].sum(axis=1), everything - res["emi_total"], atol=1e-12)
    np.testing.assert_allclose(
        res["met_noise"],
        (res["observed"] - everything) - res["met_base"],
        atol=1e-12,
    )


# ------------------------------------------------------------------ sequential


def test_sequential_is_the_default_and_credits_the_interaction_to_the_later_feature(calls):
    res = decom_met(_frame(), _model(), n_samples=2)

    assert list(res.columns) == [
        "observed",
        "emi_total",
        "a",
        "b",
        "c",
        "met_total",
        "met_base",
        "met_noise",
    ]
    np.testing.assert_allclose(res["emi_total"], 10.0)
    np.testing.assert_allclose(res["a"], MAIN["a"])
    np.testing.assert_allclose(res["b"], MAIN["b"] + AB)
    np.testing.assert_allclose(res["c"], MAIN["c"])
    _assert_closes(res, ["a", "b", "c"])
    # k + 1 normalisations, time variables never resampled.
    assert [c["resample"] for c in calls] == [["a", "b", "c"], ["b", "c"], ["c"], []]


def test_sequential_split_moves_with_the_order(calls):
    res = decom_met(_frame(), _model(), n_samples=2, variable_order=["b", "a", "c"])

    np.testing.assert_allclose(res["a"], MAIN["a"] + AB)
    np.testing.assert_allclose(res["b"], MAIN["b"])


# --------------------------------------------------------------------- shapley


def test_shapley_splits_the_interaction_evenly(calls):
    res = decom_met(_frame(), _model(), n_samples=2, attribution="shapley")

    np.testing.assert_allclose(res["a"], MAIN["a"] + AB / 2)
    np.testing.assert_allclose(res["b"], MAIN["b"] + AB / 2)
    np.testing.assert_allclose(res["c"], MAIN["c"])
    _assert_closes(res, ["a", "b", "c"])
    assert len(calls) == 2**3  # every coalition, each once


def test_shapley_does_not_depend_on_importance(calls):
    first = decom_met(_frame(), _model(), n_samples=2, attribution="shapley")
    reordered = decom_met(
        _frame(), Model(FEATS, [1.0, 2.0, 3.0, 0.5]), n_samples=2, attribution="shapley"
    )

    pd.testing.assert_frame_equal(first[["a", "b", "c"]], reordered[["a", "b", "c"]])


def test_sampled_shapley_is_exact_for_pairwise_games(calls):
    """An order and its reverse flip every pair, so one antithetic pair already
    splits a pairwise interaction evenly -- and it costs 6 of the 8 coalitions."""
    res = decom_met(_frame(), _model(), n_samples=2, attribution="shapley", n_permutations=2)

    np.testing.assert_allclose(res["a"], MAIN["a"] + AB / 2)
    np.testing.assert_allclose(res["b"], MAIN["b"] + AB / 2)
    _assert_closes(res, ["a", "b", "c"])
    assert len(calls) == 6


def test_sampling_budget_covering_every_coalition_computes_the_exact_values(calls):
    decom_met(_frame(), _model(), n_samples=2, attribution="shapley", n_permutations=4)
    assert len(calls) == 2**3


# ---------------------------------------------------------------------- groups


def test_groups_default_to_shapley_over_the_groups(calls):
    res = decom_met(_frame(), _model(), n_samples=2, groups={"g1": ["a"], "g2": ["b", "c"]})

    assert list(res.columns) == [
        "observed",
        "emi_total",
        "g1",
        "g2",
        "met_total",
        "met_base",
        "met_noise",
    ]
    np.testing.assert_allclose(res["g1"], MAIN["a"] + AB / 2)
    np.testing.assert_allclose(res["g2"], MAIN["b"] + MAIN["c"] + AB / 2)
    _assert_closes(res, ["g1", "g2"])
    assert len(calls) == 2**2


def test_a_group_carries_the_interactions_inside_it(calls):
    res = decom_met(_frame(), _model(), n_samples=2, groups={"ab": ["a", "b"], "rest": "c"})

    np.testing.assert_allclose(res["ab"], MAIN["a"] + MAIN["b"] + AB)
    np.testing.assert_allclose(res["rest"], MAIN["c"])


def test_sequential_groups_follow_the_listed_order(calls):
    res = decom_met(
        _frame(),
        _model(),
        n_samples=2,
        attribution="sequential",
        groups={"g1": ["a"], "g2": ["b", "c"]},
    )

    np.testing.assert_allclose(res["g1"], MAIN["a"])
    np.testing.assert_allclose(res["g2"], MAIN["b"] + MAIN["c"] + AB)
    assert len(calls) == 3


def test_decompose_forwards_groups(calls):
    res = decompose(
        _frame(), _model(), method="meteorology", n_samples=2, groups={"x": ["a", "b", "c"]}
    )
    np.testing.assert_allclose(res["x"], _game({"a", "b", "c"}) - 10.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"groups": {"g": ["a", "b"]}}, "not in any"),
        ({"groups": {"g": ["a", "b"], "h": ["b", "c"]}}, "in two groups"),
        ({"groups": {"g": ["a", "a", "b", "c"]}}, "twice in group"),
        ({"groups": {"g": ["a", "b", "c", "zzz"]}}, "not a meteorological"),
        ({"groups": {"g": ["a", "b", "c", "hour"]}}, "time variable"),
        ({"groups": {"g": ["a", "b"], "met_total": ["c"]}}, "clashes"),
        ({"groups": {"g": ["a", "b", "c"], "h": []}}, "empty"),
        ({"groups": {}}, "non-empty mapping"),
        ({"groups": {"g": ["a", "b", "c"]}, "variable_order": ["a", "b", "c"]}, "groups"),
        ({"attribution": "shapley", "variable_order": ["a", "b", "c"]}, "no effect"),
        ({"n_permutations": 4}, "only applies"),
        ({"attribution": "shapley", "n_permutations": 0}, "at least 1"),
        ({"attribution": "banzhaf"}, "'sequential' or 'shapley'"),
        ({"variable_order": ["a", "b", "b", "c"]}, "more than once"),
        ({"variable_order": ["a", "b"]}, "variable_order"),
    ],
)
def test_inconsistent_options_are_refused(calls, kwargs, match):
    with pytest.raises(ConfigError, match=match):
        decom_met(_frame(), _model(), n_samples=2, **kwargs)
    assert not calls  # refused before any normalisation


def test_exact_shapley_over_too_many_features_is_refused(calls):
    feats = [f"f{i}" for i in range(11)]
    df = _frame().assign(**{f: 1.0 for f in feats})
    model = Model(feats, list(range(11, 0, -1)))

    with pytest.raises(ConfigError, match="n_permutations"):
        decom_met(df, model, n_samples=2, attribution="shapley")
    assert not calls


def test_decom_emi_refuses_the_met_only_options(calls):
    with pytest.raises(ConfigError, match="decom_met"):
        decom_emi(_frame(), _model(), n_samples=2, groups={"g": ["a", "b", "c"]})
    with pytest.raises(ConfigError, match="decom_met"):
        decom_emi(_frame(), _model(), n_samples=2, attribution="shapley")


# ---------------------------------------------------------- forwarded options


def test_pools_and_filters_reach_every_normalise_call(calls):
    pool = pd.DataFrame({"c": [0.0, 1.0]})
    ref = _frame().iloc[::2]
    decom_met(
        _frame(),
        _model(),
        n_samples=2,
        groups={"ab": ["a", "b"], "c": ["c"]},
        resample_df=ref,
        resample_pools={"clean": pool},
        conditional_on={"hour": [0, 1]},
    )

    assert len(calls) == 4
    for c in calls:
        assert c["resample_df"] is ref
        assert c["resample_pools"] == {"clean": pool}
        assert c["conditional_on"] == {"hour": [0, 1]}


def test_decom_emi_forwards_pools_too(calls):
    pool = pd.DataFrame({"c": [0.0, 1.0]})
    decom_emi(_frame(), _model(), n_samples=2, resample_pools={"clean": pool})
    assert calls and all(c["resample_pools"] == {"clean": pool} for c in calls)


def test_zero_shot_backend_refuses_pools_before_loading_anything():
    df = _frame().rename(columns={"value": "PM2.5"})
    with pytest.raises(ConfigError, match="not available on the chronos-2 backend"):
        decompose(
            df,
            target="PM2.5",
            method="meteorology",
            backend="chronos-2",
            covariates=["a", "b"],
            resample_pools={"clean": df[["a"]]},
        )


# ----------------------------------------------------------- missing targets


def test_a_model_trained_here_decomposes_the_rows_it_kept(calls, monkeypatch):
    """build_model drops rows with a missing target; observed must follow suit."""
    df = _frame()
    df.loc[::5, "value"] = np.nan

    def fake_build_model(df, **kwargs):
        return df[df["value"].notna()].reset_index(drop=True), _model()

    monkeypatch.setattr(dmod, "build_model", fake_build_model)
    res = decom_met(df, None, covariates=FEATS, backend="lightgbm", n_samples=2)

    assert len(res) == int(df["value"].notna().sum())
    assert res["observed"].notna().all()
    np.testing.assert_array_equal(res.index, df.loc[df["value"].notna(), "date"])


@needs_lgb
def test_missing_targets_with_a_real_model():
    """The reported crash: 'All arrays must be of the same length'."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "t2m": rng.normal(10, 3, n),
            "blh": rng.uniform(200, 1200, n),
        }
    )
    df["pm"] = 30 - 0.01 * df["blh"] + 0.5 * df["t2m"] + rng.normal(0, 1, n)
    df.loc[::13, "pm"] = np.nan

    res = decom_met(
        df,
        None,
        target="pm",
        covariates=["t2m", "blh", "hour"],
        backend="lightgbm",
        model_config={"n_trials": 1, "cv_folds": 2, "nrounds": 20},
        n_samples=3,
        n_cores=1,
        groups={"local": ["t2m", "blh"]},
    )

    assert len(res) == int(df["pm"].notna().sum())
    np.testing.assert_allclose(res["met_total"], res["observed"] - res["emi_total"])
