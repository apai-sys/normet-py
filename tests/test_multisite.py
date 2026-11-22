"""Multisite driver — exercise multisite_apply without a real model."""

import pandas as pd
import pytest

from normet.pipeline.multisite import (
    decompose_multisite,
    do_all_multisite,
    multisite_apply,
)


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"mean_pm": [df["PM2.5"].mean()]})


def _summarise_with_kwarg(df: pd.DataFrame, site: str = "") -> pd.DataFrame:
    return pd.DataFrame({"mean_pm": [df["PM2.5"].mean()], "site_arg": [site]})


def test_multisite_apply_loops_over_sites(synthetic_aq):
    aq = synthetic_aq.copy()
    aq["site"] = "A"
    aq2 = synthetic_aq.copy().assign(site="B")
    df = pd.concat([aq, aq2], ignore_index=True)
    out = multisite_apply(df, site_col="site", func=_summarise, n_cores=1)
    assert {"site", "mean_pm"} <= set(out.columns)
    assert set(out["site"].unique()) == {"A", "B"}


def test_multisite_apply_raises_on_missing_col(synthetic_aq):
    with pytest.raises(ValueError):
        multisite_apply(synthetic_aq, site_col="not_a_col", func=_summarise)


def test_multisite_apply_empty_sites_returns_empty(synthetic_aq):
    df = synthetic_aq.copy()
    df["site"] = ""
    # Filter so there are zero sites with data
    out = multisite_apply(df.iloc[:0], site_col="site", func=_summarise)
    assert isinstance(out, pd.DataFrame) and out.empty


def test_multisite_apply_site_kwarg(synthetic_aq):
    aq = synthetic_aq.copy()
    aq["site"] = "A"
    out = multisite_apply(
        aq,
        site_col="site",
        func=_summarise_with_kwarg,
        n_cores=1,
        site_kwarg="site",
    )
    assert "site_arg" in out.columns
    assert out["site_arg"].iloc[0] == "A"


def test_multisite_apply_forwards_inner_n_cores_only_when_given(synthetic_aq):
    """Each worker inherits the whole allocation, so a callee sizing its own
    pool from ``n_cores=None`` opens one full pool per site. ``inner_n_cores``
    is forwarded as ``n_cores`` -- opt-in, because the generic ``func`` need
    not accept that keyword at all (``_summarise`` above does not).
    """
    seen: dict[str, object] = {}

    def record(df, n_cores=None):
        seen["n_cores"] = n_cores
        return pd.DataFrame({"n": [len(df)]})

    aq = synthetic_aq.assign(site="A")
    multisite_apply(aq, site_col="site", func=record, n_cores=1)
    assert seen["n_cores"] is None, "an omitted inner_n_cores must not be injected"

    multisite_apply(aq, site_col="site", func=record, n_cores=1, inner_n_cores=1)
    assert seen["n_cores"] == 1


def test_multisite_apply_all_fail_raises(synthetic_aq):
    def _failing(_df):
        raise RuntimeError("boom")

    aq = synthetic_aq.copy()
    aq["site"] = "X"
    with pytest.raises(RuntimeError, match="All per-site runs failed"):
        multisite_apply(aq, site_col="site", func=_failing, n_cores=1)


def test_multisite_apply_returns_empty_df(synthetic_aq):
    def _empty(_df):
        return pd.DataFrame()

    aq = synthetic_aq.copy()
    aq["site"] = "A"
    with pytest.raises(RuntimeError, match="All per-site"):
        multisite_apply(aq, site_col="site", func=_empty, n_cores=1)


def test_multisite_apply_func_returns_tuple(synthetic_aq):
    def _tuple(df):
        return (df[["PM2.5"]].head(1),)

    aq = synthetic_aq.copy()
    aq["site"] = "A"
    out = multisite_apply(aq, site_col="site", func=_tuple, n_cores=1)
    assert "site" in out.columns


def test_multisite_apply_dispatches_one_batch_per_worker(monkeypatch, synthetic_aq):
    """The process pool pickles a task's closure once per dispatch.

    ``_run`` closes over the whole frame, so task-per-task dispatch sent a
    copy of it to every site -- on a 48 MB frame, 1.9 GB of pickling over
    40 tasks. A batch size derived from the worker count shares one copy
    per worker.
    """
    import joblib

    seen: dict[str, object] = {}

    class FakeParallel:
        def __init__(self, n_jobs=None, batch_size=None):
            seen["n_jobs"] = n_jobs
            seen["batch_size"] = batch_size

        def __call__(self, tasks):
            return [fn(*args, **kwargs) for fn, args, kwargs in tasks]

    monkeypatch.setattr(joblib, "Parallel", FakeParallel)
    four = pd.concat([synthetic_aq.assign(site=s) for s in ("A", "B", "C", "D")], ignore_index=True)
    out = multisite_apply(four, site_col="site", func=_summarise, n_cores=2)
    assert seen == {"n_jobs": 2, "batch_size": 2}
    assert len(out) == 4


def test_do_all_multisite_with_sklearn(synthetic_aq):
    from normet.backends import backend_registry

    if not backend_registry.has("sklearn"):
        pytest.skip("sklearn backend not registered")
    aq = synthetic_aq.copy()
    aq["site"] = "X"
    result = do_all_multisite(
        aq,
        site_col="site",
        target="PM2.5",
        covariates=["t2m", "blh"],
        backend="sklearn",
        n_cores=1,
        n_samples=3,
    )
    assert "date" in result.columns
    assert "observed" in result.columns


def test_do_all_multisite_raises_on_missing_col(synthetic_aq):
    with pytest.raises((ValueError, KeyError)):
        do_all_multisite(
            synthetic_aq,
            site_col="site",
            target="PM2.5",
            covariates=["t2m", "blh"],
            backend="flaml",
            n_cores=1,
        )


def test_decompose_multisite_raises_on_missing_col(synthetic_aq):
    with pytest.raises(ValueError, match="not in df"):
        decompose_multisite(
            synthetic_aq,
            site_col="not_a_col",
            target="PM2.5",
            covariates=["t2m", "blh"],
            backend="flaml",
            n_cores=1,
        )


def test_decompose_multisite_raises_on_missing_col(synthetic_aq):
    with pytest.raises(ValueError, match="not in df"):
        decompose_multisite(
            synthetic_aq,
            site_col="site",
            target="PM2.5",
            covariates=["t2m", "blh"],
            backend="flaml",
            n_cores=1,
        )


def test_do_all_multisite_runs_one_site_thread_per_worker(monkeypatch, synthetic_aq):
    """The outer process pool is the parallelism; a site must not open its own.

    Each loky worker resolves ``n_cores=None`` to the whole allocation, so
    two sites used to run two full LightGBM/resampling pools inside one
    allocation. With several sites the per-site call now asks for one
    thread; a single site keeps ``do_all``'s own default.
    """
    seen: list[int | None] = []

    def fake_do_all(df, target, **kwargs):
        seen.append(kwargs.get("n_cores"))
        return (
            pd.DataFrame({"observed": 1.0}, index=df["date"].rename("date")),
            object(),
            None,
        )

    # ``normet.pipeline.do_all`` is the re-exported *function*; the driver
    # imports the module and the patch has to hit the module attribute.
    import sys

    do_all_mod = sys.modules["normet.pipeline.do_all"]
    monkeypatch.setattr(do_all_mod, "do_all", fake_do_all)
    both = pd.concat(
        [synthetic_aq.assign(site="A"), synthetic_aq.assign(site="B")], ignore_index=True
    )
    do_all_multisite(both, site_col="site", target="PM2.5", covariates=["t2m"], n_cores=1)
    assert seen == [1, 1]

    single = do_all_multisite(
        synthetic_aq.assign(site="A"),
        site_col="site",
        target="PM2.5",
        covariates=["t2m"],
        n_cores=1,
    )
    assert seen[-1] is None
    assert "date" in single.columns


def test_decompose_multisite_runs_one_site_thread_per_worker(monkeypatch, synthetic_aq):
    """Same nesting rule for decompose(): its normalise calls stay per-site."""
    seen: list[object] = []

    def fake_decompose(df, **kwargs):
        seen.append(kwargs.get("n_cores"))
        index = pd.DatetimeIndex(df["date"].iloc[:2], name="date")
        return pd.DataFrame({"emission": [1.0, 2.0]}, index=index)

    monkeypatch.setattr("normet.analysis.decomposition.decompose", fake_decompose)
    both = pd.concat(
        [synthetic_aq.assign(site="A"), synthetic_aq.assign(site="B")], ignore_index=True
    )
    decompose_multisite(both, site_col="site", target="PM2.5", covariates=["t2m"], n_cores=1)
    assert seen == [1, 1]

    decompose_multisite(
        synthetic_aq.assign(site="A"),
        site_col="site",
        target="PM2.5",
        covariates=["t2m"],
        n_cores=1,
    )
    assert seen[-1] is None


def test_wide_by_site_pivots_a_long_frame_onto_a_shared_time_axis():
    """The long-to-wide pivot is the whole bridge to ChronosEmbedder.

    embed_stations wants one column per site; multi-site frames in this package
    are long, which is why the two never met. No model or extra is needed to
    check the reshape itself.
    """
    import numpy as np
    import pandas as pd

    from normet.pipeline.multisite import _wide_by_site

    dates = pd.date_range("2024-01-01", periods=6, freq="h")
    df = pd.DataFrame(
        {
            "date": list(dates[:6]) + list(dates[:4]),
            "site": ["A"] * 6 + ["B"] * 4,
            "PM2.5": list(range(6)) + list(range(10, 14)),
        }
    )

    wide = _wide_by_site(df, "site", "PM2.5", "date")
    assert list(wide.columns) == ["A", "B"]
    assert wide.index.equals(dates)
    assert wide["A"].tolist() == [0, 1, 2, 3, 4, 5]
    # B is short: the missing hours must be NaN holes, not a shifted series.
    assert wide["B"].tolist()[:4] == [10, 11, 12, 13]
    assert np.isnan(wide["B"].tolist()[4:]).all()


def test_wide_by_site_rejects_a_missing_column():
    import pandas as pd
    import pytest

    from normet.pipeline.multisite import _wide_by_site

    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, freq="h"), "site": "A"})
    with pytest.raises(ValueError, match="PM2.5"):
        _wide_by_site(df, "site", "PM2.5", "date")


def test_do_all_multisite_chronos_single_checkpoint_deweather_and_icn(monkeypatch):
    """backend='chronos-2' loads one checkpoint in-process for both deweather and cross-learning ICN."""
    import numpy as np

    import normet.foundation.estimator as est_mod

    levels = [0.1, 0.5, 0.9]
    load_count = 0
    seen_cross: list[bool] = []

    class _FakePipe:
        quantiles = levels

        def predict(self, inputs, prediction_length, cross_learning=False, **kwargs):
            seen_cross.append(bool(cross_learning))
            out = []
            for item in inputs:
                hist = np.asarray(item["target"], dtype=float)
                med = np.full(prediction_length, float(np.nanmedian(hist)))
                out.append(np.stack([med - 1.0, med, med + 1.0])[None, ...])
            return out

    def _fake_load(name, dev, auto):
        nonlocal load_count
        load_count += 1
        return _FakePipe(), dev or "cpu"

    monkeypatch.setattr(est_mod, "_load_pipeline_on", _fake_load)
    monkeypatch.setattr(est_mod, "resolve_device", lambda req=None: req or "cpu")

    def _fail_parallel(*a, **k):
        raise AssertionError("chronos-2 fast path should not spawn joblib.Parallel")

    import joblib

    monkeypatch.setattr(joblib, "Parallel", _fail_parallel)

    dates = pd.date_range("2024-01-01", periods=96, freq="h")
    df = pd.concat(
        [
            pd.DataFrame(
                {"date": dates, "site": s, "PM2.5": np.linspace(15.0, 25.0, 96), "ws": 2.0}
            )
            for s in ("A", "B")
        ],
        ignore_index=True,
    )

    dew_out, dew_models = do_all_multisite(
        df,
        site_col="site",
        target="PM2.5",
        covariates=["ws"],
        backend="chronos-2",
        n_samples=2,
        return_models=True,
        model_config={"context_length": 24, "prediction_length": 12, "device": "cpu"},
    )
    assert load_count == 1
    assert set(dew_out["site"]) == {"A", "B"}
    assert dew_models["A"] is dew_models["B"]

    seen_cross.clear()
    icn_out, icn_models = do_all_multisite(
        df,
        site_col="site",
        target="PM2.5",
        covariates=["ws"],
        backend="chronos-2",
        n_samples=2,
        return_models=True,
        model_config={
            "method": "icn",
            "estimate": "paired",
            "context_length": 24,
            "prediction_length": 12,
            "cross_learning": True,
            "device": "cpu",
        },
    )
    assert load_count == 2  # one load per do_all_multisite call (not 2 per site)
    assert any(seen_cross)
    assert set(icn_out["site"]) == {"A", "B"}
    assert {"observed", "normalised", "q100", "q500", "q900"} <= set(icn_out.columns)
    assert icn_models["A"] is icn_models["B"]


def test_decompose_multisite_chronos_single_checkpoint(monkeypatch):
    """decompose_multisite(..., backend='chronos-2') loads once and avoids joblib processes."""
    import joblib
    import numpy as np

    import normet.foundation.estimator as est_mod

    levels = [0.1, 0.5, 0.9]
    load_count = 0

    class _FakePipe:
        quantiles = levels

        def predict(self, inputs, prediction_length, **kwargs):
            out = []
            for item in inputs:
                w = np.asarray(item["future_covariates"]["ws"], dtype=float)
                med = 20.0 + 2.0 * w
                out.append(np.stack([med - 1.0, med, med + 1.0])[None, ...])
            return out

    def _fake_load(name, dev, auto):
        nonlocal load_count
        load_count += 1
        return _FakePipe(), dev or "cpu"

    monkeypatch.setattr(est_mod, "_load_pipeline_on", _fake_load)
    monkeypatch.setattr(est_mod, "resolve_device", lambda req=None: req or "cpu")
    monkeypatch.setattr(
        joblib,
        "Parallel",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("chronos-2 decompose_multisite should not spawn joblib")
        ),
    )

    dates = pd.date_range("2024-01-01", periods=96, freq="h")
    ws = np.linspace(1.0, 5.0, 96)
    df = pd.concat(
        [
            pd.DataFrame({"date": dates, "site": s, "PM2.5": 20.0 + 2.0 * ws, "ws": ws})
            for s in ("A", "B")
        ],
        ignore_index=True,
    )
    out = decompose_multisite(
        df,
        site_col="site",
        target="PM2.5",
        method="meteorology",
        covariates=["ws"],
        backend="chronos-2",
        model_config={
            "method": "icn",
            "context_length": 24,
            "prediction_length": 12,
            "device": "cpu",
        },
        n_samples=2,
    )
    assert load_count == 1
    assert set(out["site"]) == {"A", "B"}
    assert {"date", "site", "observed", "emi_total", "met_total", "ws"} <= set(out.columns)
