"""Regression tests for the model/eval/metrics API findings F1-F8 + grouped modStats.

One test per finding. They use LightGBM directly with one CV fold/trial so the
file stays fast (no AutoML search, no FLAML).
"""

from __future__ import annotations

import inspect
import logging
import re

import numpy as np
import pandas as pd
import pytest

from normet.eval.benchmarks import make_synthetic_benchmark
from normet.model.io import load_model, save_model
from normet.model.predict import ml_predict
from normet.model.train import build_model, train_model
from normet.utils.features import permutation_importance
from normet.utils.metrics import modStats

TINY_CONFIG = {"n_trials": 1, "cv_folds": 2, "nrounds": 10}


def _frame(n: int = 40) -> pd.DataFrame:
    """Small hourly frame with a weather-driven, strictly positive target."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    t2m = 10.0 + 6.0 * np.sin(2.0 * np.pi * np.arange(n) / 24.0) + rng.normal(0, 0.5, n)
    ws = np.abs(rng.normal(3.0, 1.0, n))
    value = 20.0 - 1.2 * t2m + 2.0 * ws + rng.normal(0, 0.5, n)
    return pd.DataFrame({"date": dates, "t2m": t2m, "ws": ws, "value": value})


# ---------------------------------------------------------------------------
# F1 - make_synthetic_benchmark: record length follows `freq`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("freq", "expected_rows"),
    [("h", 240), ("6h", 40), ("30min", 480), ("12h", 20), ("D", 10)],
)
def test_benchmark_record_length_matches_frequency(freq: str, expected_rows: int) -> None:
    bench = make_synthetic_benchmark(n_days=10, freq=freq, seed=1)
    dates = bench.df["date"]
    step = pd.tseries.frequencies.to_offset(freq)

    assert bench.metadata["n_days"] == 10
    assert len(bench.df) == expected_rows
    assert len(bench.truth) == expected_rows
    # Span in data == the requested n_days minus exactly one sampling step.
    assert dates.iloc[-1] - dates.iloc[0] == pd.Timedelta(days=10) - step


def test_benchmark_rejects_invalid_frequency() -> None:
    with pytest.raises(ValueError, match="Invalid `freq`"):
        make_synthetic_benchmark(n_days=7, freq="not-a-frequency")


# ---------------------------------------------------------------------------
# F2 - load_model(<directory>) finds a model saved under a custom filename
# ---------------------------------------------------------------------------


def test_load_model_directory_discovers_custom_filename(tmp_path) -> None:
    pytest.importorskip("lightgbm")
    from normet.backends.lgb_backend import train_lgb

    df = _frame()
    model = train_lgb(
        df, target="value", covariates=["t2m", "ws"], model_config=TINY_CONFIG, seed=3
    )
    saved = save_model(
        model,
        path=str(tmp_path),
        filename="custom.joblib",
        metadata={"log_transform": False},
    )
    assert saved.endswith("custom.joblib")

    # Loading the directory used to demand <dir>/automl.joblib and fail.
    loaded = load_model(str(tmp_path), backend="lightgbm")
    np.testing.assert_array_equal(ml_predict(model, df), ml_predict(loaded, df))
    # The explicit-file form still works.
    np.testing.assert_array_equal(
        ml_predict(model, df), ml_predict(load_model(saved, backend="lightgbm"), df)
    )
    # ... and the metadata sidecar follows the discovered file.
    _, meta = load_model(str(tmp_path), backend="lightgbm", return_metadata=True)
    assert meta is not None
    assert meta["features"] == ["t2m", "ws"]

    # Several candidates without the requested filename: ask, do not guess.
    save_model(model, path=str(tmp_path), filename="second.joblib")
    with pytest.raises(ValueError, match="Several model files"):
        load_model(str(tmp_path), backend="lightgbm")
    chosen = load_model(str(tmp_path), backend="lightgbm", filename="second.joblib")
    assert chosen.backend == "lightgbm"

    # No candidate at all: discovery is additive, so the backend's own error
    # names the requested default file (it used to be a silent wrong file).
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="automl.joblib"):
        load_model(str(empty), backend="lightgbm")


# ---------------------------------------------------------------------------
# F3 - non-finite sample weights warn and are dropped (weight 0), not imputed
# ---------------------------------------------------------------------------


def test_non_finite_sample_weight_warns_and_is_dropped(monkeypatch) -> None:
    pytest.importorskip("lightgbm")
    from normet.backends import lgb_backend

    df = _frame()
    weights = np.linspace(0.5, 2.0, len(df))
    weights[::7] = np.nan
    n_nan = int(np.isnan(weights).sum())

    seen: list[np.ndarray] = []
    real_resolve = lgb_backend.resolve_sample_weight

    def _recording_resolve(sw, df_full, df_sub, train_mask):
        arr = real_resolve(sw, df_full, df_sub, train_mask)
        if arr is not None:
            seen.append(np.asarray(arr, dtype=float).copy())
        return arr

    monkeypatch.setattr(lgb_backend, "resolve_sample_weight", _recording_resolve)

    with pytest.warns(UserWarning, match="non-finite"):
        _, model = build_model(
            df,
            target="value",
            covariates=["t2m", "ws"],
            backend="lightgbm",
            split_method="ts",
            model_config=TINY_CONFIG,
            sample_weight=weights,
            seed=11,
        )
    assert model.backend == "lightgbm"
    assert seen, "backend never received sample weights"
    arr = seen[0]
    assert np.isfinite(arr).all()
    # Median imputation would have filled the NaNs with a usable weight; the
    # rows whose weight was dropped carry exactly weight 0.
    expected = np.where(np.isnan(weights[: len(arr)]), 0.0, weights[: len(arr)])
    np.testing.assert_allclose(arr, expected, rtol=0.0, atol=0.0)
    assert int(np.sum(arr == 0.0)) == int(np.isnan(weights[: len(arr)]).sum()) > 0
    assert n_nan >= 1

    # The direct training entry point agrees: warn, do not raise.
    with pytest.warns(UserWarning, match="non-finite"):
        from normet.backends.lgb_backend import train_lgb

        train_lgb(
            df,
            target="value",
            covariates=["t2m", "ws"],
            model_config=TINY_CONFIG,
            sample_weight=weights,
            seed=11,
        )
    with pytest.raises(ValueError, match="non-negative"):
        train_lgb(
            df,
            target="value",
            covariates=["t2m", "ws"],
            model_config=TINY_CONFIG,
            sample_weight=-np.ones(len(df)),
            seed=11,
        )


# ---------------------------------------------------------------------------
# F4 - cv_folds larger than the training record: clear ValueError
# ---------------------------------------------------------------------------


def test_short_record_cv_folds_raises_clear_value_error() -> None:
    pytest.importorskip("lightgbm")
    from normet.backends.lgb_backend import train_lgb

    df = _frame(3)
    with pytest.raises(ValueError, match=r"`cv_folds`=5.*3 training row"):
        train_lgb(
            df,
            target="value",
            covariates=["t2m", "ws"],
            model_config={"n_trials": 1, "cv_folds": 5, "nrounds": 5},
            seed=1,
        )
    # A record that can fill the folds is unaffected.
    model = train_lgb(
        _frame(20),
        target="value",
        covariates=["t2m", "ws"],
        model_config=TINY_CONFIG,
        seed=1,
    )
    assert model.backend == "lightgbm"


# ---------------------------------------------------------------------------
# F5 - "geometric" back_transform is documented as agreeing with "arithmetic"
# ---------------------------------------------------------------------------


def test_geometric_back_transform_matches_arithmetic() -> None:
    pytest.importorskip("lightgbm")
    from normet.utils.cv import cv_score

    n = 60
    rng = np.random.default_rng(0)
    y = 10.0 + 5.0 * np.sin(np.arange(n) / 6.0) + rng.normal(0.0, 0.5, n)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "value": np.log1p(y),
            "pred_log": np.log1p(np.clip(y - 1.0, 1e-3, None)),
            "set": np.where(np.arange(n) < 45, "training", "testing"),
        }
    )

    def predictor(model, frame):
        return frame["pred_log"].to_numpy()

    kw = {"model": None, "predictor": predictor, "log_transform": True}
    tbl_arith = modStats(df, **kw, back_transform="arithmetic")
    tbl_geom = modStats(df, **kw, back_transform="geometric")
    tbl_smear = modStats(df, **kw, back_transform="smearing", smearing_factor=1.5)
    pd.testing.assert_frame_equal(tbl_arith, tbl_geom)
    assert not np.allclose(tbl_arith["RMSE"].to_numpy(), tbl_smear["RMSE"].to_numpy())
    # The docstrings now say the two coincide for per-row predictions.
    assert 'coincides with ``"arithmetic"``' in (modStats.__doc__ or "")

    # cv_score: same documented coincidence (per-row predictions).
    small = _frame(120)
    cv_kw = dict(target="value", covariates=["t2m", "ws"], backend="lightgbm", n_splits=2)
    model_config = {"n_trials": 1, "cv_folds": 2, "nrounds": 40}
    arith = cv_score(
        small, **cv_kw, log_transform=True, back_transform="arithmetic", model_config=model_config
    )
    geom = cv_score(
        small, **cv_kw, log_transform=True, back_transform="geometric", model_config=model_config
    )
    pd.testing.assert_frame_equal(arith, geom)
    assert 'coincides with ``"arithmetic"``' in (cv_score.__doc__ or "")


# ---------------------------------------------------------------------------
# F6 - permutation_importance(metric=None) raises its own ValueError
# ---------------------------------------------------------------------------


def test_permutation_importance_non_string_metric_raises_value_error() -> None:
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="Unsupported `metric`"):
        permutation_importance(df, object(), metric=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported `metric`"):
        permutation_importance(df, object(), metric=3)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# F7 - native_nan is a documented config key, not an unsupported one
# ---------------------------------------------------------------------------


def test_native_nan_config_key_is_known(caplog) -> None:
    pytest.importorskip("lightgbm")
    from normet.backends.lgb_backend import train_lgb

    df = _frame()
    with caplog.at_level(logging.WARNING, logger="normet.backends.lgb_backend"):
        model = train_lgb(
            df,
            target="value",
            covariates=["t2m", "ws"],
            model_config={**TINY_CONFIG, "native_nan": True, "bogus_key": 1},
            seed=2,
        )
    assert model.native_nan is True
    assert "native_nan" not in caplog.text
    assert "bogus_key" in caplog.text
    assert "unsupported model_config" in caplog.text


# ---------------------------------------------------------------------------
# F8 - documented defaults agree with the signatures
# ---------------------------------------------------------------------------


def test_train_docstrings_match_signatures() -> None:
    assert inspect.signature(build_model).parameters["verbose"].default is False
    assert inspect.signature(train_model).parameters["verbose"].default is False

    for fn in (build_model, train_model):
        doc = inspect.getdoc(fn) or ""
        doc = re.sub(r"\s+", " ", doc)
        verbose_line = re.search(r"verbose : bool, default=(True|False)", doc)
        assert verbose_line is not None
        assert verbose_line.group(1) == "False"
        cov_line = re.search(r"covariates : .*?(?= split_method :| model_config :)", doc)
        assert cov_line is not None
        assert "optional" not in cov_line.group(0)

    # No documented RuntimeError that the code never raises: backend errors
    # propagate unchanged.
    doc = inspect.getdoc(train_model) or ""
    raises = doc.split("Raises", 1)[1].split("Notes", 1)[0]
    assert "RuntimeError" not in raises
    assert "propagate" in raises


# ---------------------------------------------------------------------------
# Grouped modStats - `subset` selects the rows, not only the aggregation
# ---------------------------------------------------------------------------


def test_modstats_by_honours_subset() -> None:
    n = 60
    rng = np.random.default_rng(1)
    y = 20.0 + rng.normal(0.0, 1.0, n)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "value": y,
            "pred": y + rng.normal(0.0, 0.5, n),
            "set": np.where(np.arange(n) < 45, "training", "testing"),
        }
    )

    def predictor(model, frame):
        return frame["pred"].to_numpy()

    out = modStats(df, model=None, predictor=predictor, subset="testing", by="hour")
    group_rows = out.loc[out["hour"] != "all"]
    all_row = out.loc[out["hour"] == "all"].iloc[0]
    assert int(group_rows["n"].sum()) == 15
    assert int(all_row["n"]) == 15

    # Without a subset the grouped path still covers every row.
    every = modStats(df, model=None, predictor=predictor, by="hour")
    assert int(every.loc[every["hour"] != "all", "n"].sum()) == 60
    assert int(every.loc[every["hour"] == "all", "n"].iloc[0]) == 60

    # The legacy "subset without a 'set' column" validation is kept for `by`.
    with pytest.raises(ValueError, match="no 'set' column"):
        modStats(
            df.drop(columns="set"),
            model=None,
            predictor=predictor,
            subset="testing",
            by="hour",
        )
