"""Tests for normet.analysis.normalise helper functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from normet.analysis.normalise import (
    _apply_conditional_filter,
    _draw_rows,
    _format_quantile_name,
    _quantile_column_names,
    generate_resampled,
)
from normet.core.exceptions import ConfigError, DataError

# ---------------------------------------------------------------------------
# _format_quantile_name
# ---------------------------------------------------------------------------


def test_format_quantile_name_values():
    assert _format_quantile_name(0.025) == "q025"
    assert _format_quantile_name(0.5) == "q500"
    assert _format_quantile_name(0.975) == "q975"


def test_format_quantile_name_out_of_range():
    with pytest.raises(ValueError):
        _format_quantile_name(1.1)


# ---------------------------------------------------------------------------
# _apply_conditional_filter
# ---------------------------------------------------------------------------


def _simple_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7]})


def test_apply_conditional_filter_scalar():
    """Scalar value → exact-match filter."""
    df = _simple_df()
    result = _apply_conditional_filter(df, {"x": 2})
    assert list(result["x"]) == [2]


def test_apply_conditional_filter_iterable():
    """List of values → isin semantics."""
    df = _simple_df()
    result = _apply_conditional_filter(df, {"x": [1, 3]})
    assert len(result) == 2
    assert set(result["x"]) == {1, 3}


def test_apply_conditional_filter_callable():
    """Callable → boolean mask applied per column."""
    df = _simple_df()
    result = _apply_conditional_filter(df, {"x": lambda v: v > 5})
    assert list(result["x"]) == [6, 7]


def test_apply_conditional_filter_missing_col():
    """Raises ValueError for a condition on a column that does not exist."""
    df = _simple_df()
    with pytest.raises(ValueError, match="not found"):
        _apply_conditional_filter(df, {"nonexistent": 1})


# ---------------------------------------------------------------------------
# generate_resampled
# ---------------------------------------------------------------------------


def _weather_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "value": rng.normal(20.0, 2.0, n),
            "ws": rng.uniform(0.5, 10.0, n),
            "wd": rng.uniform(0.0, 360.0, n),
        }
    )


def test_generate_resampled_shape():
    """Output has same shape as input and includes a 'seed' column."""
    df = _weather_df()
    result = generate_resampled(
        df,
        variables_resample=["ws", "wd"],
        replace=True,
        seed=42,
        resample_df=df,
    )
    assert result.shape == df.shape or "seed" in result.columns
    assert result.shape[0] == df.shape[0]
    assert "seed" in result.columns
    assert result["seed"].iloc[0] == 42


def test_generate_resampled_missing_col():
    """Raises ValueError when resample_df is missing a required column."""
    df = _weather_df()
    pool = df.drop(columns=["ws"])
    with pytest.raises(ValueError, match="missing columns"):
        generate_resampled(
            df,
            variables_resample=["ws", "wd"],
            replace=True,
            seed=0,
            resample_df=pool,
        )


def test_generate_resampled_replaces_values():
    """Resampled columns should differ from original (with high probability)."""
    rng = np.random.default_rng(99)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    ws_original = np.arange(n, dtype=float)  # deterministic, easy to verify change
    df = pd.DataFrame({"date": dates, "value": np.ones(n), "ws": ws_original, "wd": np.zeros(n)})

    # Use a pool with completely different ws values
    pool = df.copy()
    pool["ws"] = rng.uniform(100.0, 200.0, n)

    result = generate_resampled(
        df,
        variables_resample=["ws"],
        replace=True,
        seed=7,
        resample_df=pool,
    )

    # All resampled ws values should come from pool range [100, 200]
    assert (result["ws"] >= 100.0).all()
    # Original df ws values were 0..99, so they should not match
    assert not (result["ws"] == ws_original).all()


# ---------------------------------------------------------------------------
# normalise() Function Tests
# ---------------------------------------------------------------------------


def test_batch_path_passes_n_cores_as_predict_threads(monkeypatch):
    """The batch-reduce path has no worker pool; ``n_cores`` sizes the predictor.

    LightGBM reads its thread count from the OpenMP environment at first use,
    which HPC harnesses pin to 1; without this hop ``n_cores`` would be ignored
    on the path most users take (aggregate=True, no quantiles).
    """
    import sys

    from normet.analysis.normalise import normalise

    # `normet.analysis.normalise` as an attribute of the package is the
    # re-exported *function*; the monkeypatch has to hit the module.
    normalise_mod = sys.modules["normet.analysis.normalise"]
    seen: dict = {}

    def mock_ml_predict(model, df, **kwargs):
        seen.update(kwargs)
        return np.full(len(df), 10.0)

    monkeypatch.setattr(normalise_mod, "ml_predict", mock_ml_predict)
    common = dict(
        df=_weather_df(n=30),
        model="dummy-model",
        covariates=["ws", "wd"],
        variables_resample=["ws", "wd"],
        n_samples=5,
        n_cores=3,
        verbose=False,
    )
    normalise(**common, aggregate=True)
    assert seen == {"num_threads": 3}
    # The vectorised path (quantiles / wide table) has no pool either.
    normalise(**common, aggregate=True, return_quantiles=[0.025, 0.975])
    assert seen == {"num_threads": 3}


def test_threaded_path_gives_each_worker_one_predict_thread(monkeypatch):
    """``memory_save=True`` already runs one worker per seed.

    Each worker used to call the predictor with no thread count, so every
    LightGBM call opened its own full-width OpenMP pool -- ``n_cores`` pools
    of ``n_cores`` threads. Prediction is bit-identical across thread counts,
    so each worker now asks for one thread.
    """
    import sys

    from normet.analysis.normalise import normalise

    normalise_mod = sys.modules["normet.analysis.normalise"]
    calls: list[dict] = []

    def mock_ml_predict(model, df, **kwargs):
        calls.append(dict(kwargs))
        return np.full(len(df), 10.0)

    monkeypatch.setattr(normalise_mod, "ml_predict", mock_ml_predict)
    result = normalise(
        df=_weather_df(n=30),
        model="dummy-model",
        covariates=["ws", "wd"],
        variables_resample=["ws", "wd"],
        n_samples=4,
        aggregate=True,
        memory_save=True,
        n_cores=2,
        verbose=False,
    )
    assert "normalised" in result.columns
    assert calls and all(c == {"num_threads": 1} for c in calls)


def test_normalise_core_runs(monkeypatch):
    import sys

    from normet.analysis.normalise import normalise

    normalise_mod = sys.modules["normet.analysis.normalise"]

    # Dummy predictor returning 10.0 for every sample
    def mock_ml_predict(model, df, **kwargs):
        return np.full(len(df), 10.0)

    monkeypatch.setattr(normalise_mod, "ml_predict", mock_ml_predict)

    # 1. Test aggregate=True with quantiles
    df = _weather_df(n=30)
    result = normalise(
        df=df,
        model="dummy-model",
        covariates=["ws", "wd"],
        variables_resample=["ws", "wd"],
        n_samples=5,
        aggregate=True,
        return_quantiles=[0.025, 0.975],
        n_cores=1,
        verbose=False,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.index.name == "date"
    assert "observed" in result.columns
    assert "normalised" in result.columns
    assert "q025" in result.columns
    assert "q975" in result.columns
    assert (result["normalised"] == 10.0).all()

    # 2. Test aggregate=False (wide seed table)
    result_wide = normalise(
        df=df,
        model="dummy-model",
        covariates=["ws", "wd"],
        variables_resample=["ws", "wd"],
        n_samples=5,
        aggregate=False,
        n_cores=1,
        verbose=False,
    )
    assert isinstance(result_wide, pd.DataFrame)
    assert "observed" in result_wide.columns
    # Should have 5 seed columns (not starting with 'q')
    seed_cols = [c for c in result_wide.columns if isinstance(c, int | np.integer)]
    assert len(seed_cols) == 5

    # 3. Test memory_save=True (ThreadPoolExecutor branch)
    result_mem = normalise(
        df=df,
        model="dummy-model",
        covariates=["ws", "wd"],
        variables_resample=["ws", "wd"],
        n_samples=5,
        aggregate=True,
        memory_save=True,
        n_cores=1,
        verbose=False,
    )
    assert "normalised" in result_mem.columns

    # 4. Test conditional_on filtering
    result_filtered = normalise(
        df=df,
        model="dummy-model",
        covariates=["ws", "wd"],
        variables_resample=["ws", "wd"],
        n_samples=5,
        aggregate=True,
        conditional_on={"ws": lambda v: v > 0.0},
        n_cores=1,
        verbose=False,
    )
    assert "normalised" in result_filtered.columns

    # 5. Test filter empty raises error
    with pytest.raises(DataError, match="left no rows"):
        normalise(
            df=df,
            model="dummy-model",
            covariates=["ws", "wd"],
            variables_resample=["ws", "wd"],
            n_samples=5,
            aggregate=True,
            conditional_on={"ws": lambda v: v > 1000.0},
            n_cores=1,
            verbose=False,
        )

    # 6. Test missing resample columns raises error
    with pytest.raises(DataError, match="missing columns"):
        normalise(
            df=df,
            model="dummy-model",
            covariates=["ws", "wd"],
            variables_resample=["missing_col"],
            n_samples=5,
            aggregate=True,
            n_cores=1,
            verbose=False,
        )


def test_a_model_feature_missing_from_the_frame_is_refused(monkeypatch):
    """``check_data`` drops absent covariates, and an AutoML backend then
    predicts them as zero; the covariates list alone cannot catch it.

    The caller can pass the full feature list (so the superset check passes)
    while the frame lacks one of the columns -- a normalise on a frame that
    never went through ``add_date_variables``, say.
    """
    import sys

    from normet.analysis.normalise import normalise
    from normet.core.exceptions import ConfigError

    normalise_mod = sys.modules["normet.analysis.normalise"]
    monkeypatch.setattr(normalise_mod, "ml_predict", lambda model, df, **kw: np.full(len(df), 1.0))

    class _Model:
        feature_names_in_ = np.array(["ws", "wd", "hour"])

    with pytest.raises(ConfigError, match="missing"):
        normalise(
            df=_weather_df(n=20),  # no hour column
            model=_Model(),
            covariates=["ws", "wd", "hour"],
            variables_resample=["ws", "wd"],
            n_samples=3,
            n_cores=1,
            verbose=False,
        )


def test_duplicate_timestamps_give_one_row_per_date(monkeypatch):
    """Every aggregation path groups by date; the wide table used to raise
    "cannot reshape" and the batch path returned one row per input row."""
    import sys

    from normet.analysis.normalise import normalise

    normalise_mod = sys.modules["normet.analysis.normalise"]
    monkeypatch.setattr(normalise_mod, "ml_predict", lambda model, df, **kw: np.full(len(df), 10.0))
    df = _weather_df(n=10)
    dup = pd.concat([df, df], ignore_index=True)
    common = dict(
        df=dup,
        model="dummy-model",
        covariates=["ws", "wd"],
        variables_resample=["ws", "wd"],
        n_samples=3,
        n_cores=1,
        verbose=False,
    )

    wide = normalise(**common, aggregate=False)
    seed_cols = [c for c in wide.columns if isinstance(c, int | np.integer)]
    assert len(seed_cols) == 3
    assert len(wide) == 10 and wide.index.is_unique

    batched = normalise(**common, aggregate=True)  # auto batch path
    vectorised = normalise(**common, aggregate=True, batch_size=0)
    assert len(batched) == len(vectorised) == 10
    np.testing.assert_allclose(batched["normalised"], vectorised["normalised"])
    np.testing.assert_allclose(batched["observed"], vectorised["observed"])


# ---------------------------------------------------------------------------
# Stratified draws: cached plan parity and stratum membership
# ---------------------------------------------------------------------------


def _strata_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n_rows, n_pool = 180, 240
    df = pd.DataFrame({"month": rng.integers(1, 7, n_rows), "hour": rng.integers(0, 8, n_rows)})
    pool = pd.DataFrame({"month": rng.integers(1, 7, n_pool), "hour": rng.integers(0, 8, n_pool)})
    return df, pool


def test_stratified_draws_are_unchanged_by_the_shared_plan_cache():
    """The cached stratum grouping must not move a single draw.

    The grouping depends only on the two label arrays, so it is built once
    per resampling call instead of inside every draw (``_stratified_draw_plan``).
    The RNG call sequence must stay exactly as before: a draw planned cold,
    a draw planned into a warm cache, and repeated draws sharing one cache
    all have to agree bit for bit, and every drawn pool row must carry the
    target row's stratum.
    """
    from normet.analysis.normalise import (
        _draw_rows_stratified,
        _draw_rows_stratified_balanced,
        _extract_strata,
    )

    df, pool = _strata_frames()
    df_strata = _extract_strata(df, ("month", "hour"), label="df")
    pool_strata = _extract_strata(pool, ("month", "hour"), label="pool")
    cache: dict = {}

    for seed in (3, 4, 5):
        cold = _draw_rows_stratified(0, seed, df_strata, pool_strata, True)
        warm = _draw_rows_stratified(0, seed, df_strata, pool_strata, True, cycle_cache=cache)
        np.testing.assert_array_equal(cold, warm)
        np.testing.assert_array_equal(pool_strata[cold], df_strata)

    for pos in (0, 1, 7):
        cold = _draw_rows_stratified_balanced(0, 11, pos, df_strata, pool_strata)
        warm = _draw_rows_stratified_balanced(0, 11, pos, df_strata, pool_strata, cycle_cache=cache)
        np.testing.assert_array_equal(cold, warm)
        np.testing.assert_array_equal(pool_strata[warm], df_strata)


def test_stratified_draw_plan_keeps_strata_apart_and_falls_back_for_missing():
    """A stratum absent from the pool still draws (from the full pool) and is reported."""
    from normet.analysis.normalise import (
        _draw_rows_stratified,
        _draw_rows_stratified_balanced,
        _extract_strata,
        _stratified_draw_plan,
    )

    df = pd.DataFrame({"month": [1, 1, 2, 2, 13], "hour": [0, 1, 0, 1, 5]})
    pool = pd.DataFrame({"month": [1, 2, 2, 1], "hour": [0, 1, 0, 1]})
    df_strata = _extract_strata(df, ("month", "hour"), label="df")
    pool_strata = _extract_strata(pool, ("month", "hour"), label="pool")

    plan = _stratified_draw_plan(0, df_strata, pool_strata)
    # Five month x hour strata in the target frame; the 13/5 one has no pool
    # rows and falls back to the full pool, the other four keep their subset.
    assert len(plan) == 5
    fallbacks = [cands for _s_idx, _pos, cands in plan if len(cands) == len(pool_strata)]
    assert len(fallbacks) == 1
    np.testing.assert_array_equal(fallbacks[0], np.arange(len(pool_strata)))
    for _s_idx, pos, cands in plan:
        if len(cands) < len(pool_strata):
            np.testing.assert_array_equal(pool_strata[cands], df_strata[pos])

    draw = _draw_rows_stratified(0, 21, df_strata, pool_strata, True)
    # Rows of the first two strata keep to their own stratum.
    np.testing.assert_array_equal(pool_strata[draw[:4]], df_strata[:4])
    balanced = _draw_rows_stratified_balanced(0, 21, 0, df_strata, pool_strata)
    np.testing.assert_array_equal(pool_strata[balanced[:4]], df_strata[:4])


# ---------------------------------------------------------------------------
# Regression: distinct quantiles that format to one column name (finding 2)
# ---------------------------------------------------------------------------


def test_quantile_column_names_keep_the_documented_convention():
    """The common quantiles keep their q025/q500/q975 names, one column each."""
    assert _quantile_column_names([0.025, 0.5, 0.975]) == ["q025", "q500", "q975"]
    assert _quantile_column_names([0.05, 0.06]) == ["q050", "q060"]
    # Exact repeats are one request, not a collision.
    assert _quantile_column_names([0.1, 0.1]) == ["q100", "q100"]


def test_two_quantiles_that_share_a_column_name_are_refused(monkeypatch):
    """0.05 and 0.0501 both round to ``q050``; the run used to return one band.

    ``_format_quantile_name`` keeps three decimals of a percent, so the two
    requests collapsed into a single column -- the second overwrote the first
    on the per-seed path and duplicated it on the quantile path. It now fails
    before any resampling work instead of silently dropping a requested band.
    """
    import sys

    from normet.analysis.normalise import normalise

    normalise_mod = sys.modules["normet.analysis.normalise"]
    calls: list[int] = []

    def mock_ml_predict(model, df, **kwargs):
        calls.append(len(df))
        return np.full(len(df), 10.0)

    monkeypatch.setattr(normalise_mod, "ml_predict", mock_ml_predict)
    with pytest.raises(ValueError, match="both format to the column name"):
        normalise(
            df=_weather_df(n=20),
            model="dummy-model",
            covariates=["ws", "wd"],
            variables_resample=["ws", "wd"],
            n_samples=3,
            n_cores=1,
            return_quantiles=[0.05, 0.0501],
        )
    assert calls == [], "a refused quantile request should not reach the model"

    # The documented case still produces one column per quantile.
    out = normalise(
        df=_weather_df(n=20),
        model="dummy-model",
        covariates=["ws", "wd"],
        variables_resample=["ws", "wd"],
        n_samples=3,
        n_cores=1,
        return_quantiles=[0.025, 0.975],
    )
    assert {"q025", "q975"} <= set(out.columns)


# ---------------------------------------------------------------------------
# Regression: conditional_on on a column that is not a covariate (finding 3)
# ---------------------------------------------------------------------------


def test_conditional_on_accepts_a_column_that_is_not_a_covariate(monkeypatch):
    """``{"site": ...}`` filters the pool although ``site`` is not a model feature.

    The default pool is ``check_data(df, covariates, "value")`` -- covariates +
    ``date`` + the target only -- so a filter on any other column of ``df``
    used to fail with "not found in resample pool columns", even though the
    column was right there in the caller's frame.
    """
    import sys

    from normet.analysis.normalise import normalise

    normalise_mod = sys.modules["normet.analysis.normalise"]
    drawn: list[np.ndarray] = []

    def mock_ml_predict(model, df, **kwargs):
        drawn.append(df["ws"].to_numpy(dtype=float))
        return np.full(len(df), 10.0)

    monkeypatch.setattr(normalise_mod, "ml_predict", mock_ml_predict)
    rng = np.random.default_rng(5)
    n = 40
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "value": rng.normal(20.0, 5.0, n),
            "ws": rng.uniform(0.5, 10.0, n),
            "site": np.where(np.arange(n) % 2 == 0, "urban", "rural"),
        }
    )
    urban_pool = {round(float(v), 12) for v in df.loc[df["site"] == "urban", "ws"]}

    out = normalise(
        df=df,
        model="dummy-model",
        covariates=["ws"],
        variables_resample=["ws"],
        n_samples=4,
        n_cores=1,
        conditional_on={"site": "urban"},
    )
    assert "normalised" in out.columns
    values = {round(float(v), 12) for v in np.concatenate(drawn)}
    assert values <= urban_pool, "resampled ws came from a row the filter excluded"
    assert len(values) > 1

    # A column in neither the frame nor the pool still raises, and says where
    # the lookup happened.
    with pytest.raises(ValueError, match="not found in the resampling pool"):
        normalise(
            df=df,
            model="dummy-model",
            covariates=["ws"],
            variables_resample=["ws"],
            n_samples=3,
            n_cores=1,
            conditional_on={"country": "FR"},
        )


# ---------------------------------------------------------------------------
# Regression: replace=False on a pool smaller than the frame (finding 4)
# ---------------------------------------------------------------------------


def test_draw_rows_replace_false_needs_a_pool_of_the_frames_size():
    """The message names ``replace=False``, the pool size and the sample size."""
    with pytest.raises(ConfigError, match="the pool has 5 row"):
        _draw_rows(0, 1, n_pool=5, n_rows=6, replace=False)
    with pytest.raises(ConfigError, match="6 row"):
        _draw_rows(0, 1, n_pool=5, n_rows=6, replace=False)
    idx = _draw_rows(0, 1, n_pool=6, n_rows=6, replace=False)
    assert sorted(int(i) for i in idx) == list(range(6))


def test_normalise_replace_false_with_a_shrunken_pool_explains_itself(monkeypatch):
    """A ``resample_df`` subset smaller than ``df`` used to raise numpy's message."""
    import sys

    from normet.analysis.normalise import normalise

    normalise_mod = sys.modules["normet.analysis.normalise"]
    monkeypatch.setattr(normalise_mod, "ml_predict", lambda model, df, **kw: np.full(len(df), 10.0))
    df = _weather_df(n=30)
    common = dict(
        df=df,
        model="dummy-model",
        covariates=["ws", "wd"],
        variables_resample=["ws", "wd"],
        n_samples=3,
        n_cores=1,
        replace=False,
        verbose=False,
    )
    with pytest.raises(ConfigError, match="the pool has 10 row.*30 row"):
        normalise(**common, resample_df=df.head(10))
    # A pool as long as the frame is what replace=False documents; it works.
    out = normalise(**common, resample_df=df)
    assert "normalised" in out.columns


# ---------------------------------------------------------------------------
# Regression: the Ox conservation diagnostic is falsifiable (finding 1)
# ---------------------------------------------------------------------------


def test_ox_conservation_error_flags_an_o3_equivalent_ox_column():
    """Ox reported as O3-*equivalent* mass is a real gap, not ``2e-16``.

    The old diagnostic was algebraically identical to the repartition that
    built the columns, so it read 0 for every input while the reported
    ``Ox_normalised`` sat ~4 % (about 0.9 ug m^-3 here) away from
    ``O3_normalised + NO2_normalised``. Measuring the frame the caller gets
    turns that into a number.
    """
    from normet.analysis.chemistry import ox_conservation_error

    o3 = np.full(4, 40.0)
    no2 = np.full(4, 20.0)
    ox_as_o3_equivalent = (o3 / 48.0 + no2 / 46.0055) * 48.0
    frame = pd.DataFrame(
        {
            "O3_normalised": o3,
            "NO2_normalised": no2,
            "Ox_normalised": ox_as_o3_equivalent,
        }
    )
    gap_ug = float(np.abs(ox_as_o3_equivalent[0] - (o3[0] + no2[0])))
    assert gap_ug > 0.5
    err = ox_conservation_error(frame, unit="ug_m3")
    np.testing.assert_allclose(err, gap_ug / 48.0e6, rtol=1e-12)
    assert err > 1e-8  # mol m^-3

    # The constituent-sum convention the fix reports closes it to round-off.
    frame["Ox_normalised"] = o3 + no2
    assert ox_conservation_error(frame, unit="ug_m3") == 0.0
    # A perturbed constituent column registers, however.
    frame["O3_normalised"] = frame["O3_normalised"] * 1.01
    assert ox_conservation_error(frame, unit="ug_m3") > 1e-9


@pytest.fixture(scope="module")
def ox_deweather_result():
    """One F3 deweathering used by the two Ox-column tests below."""
    from normet.analysis.chemistry import normalise_ox_invariant

    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "O3": 40.0 + 10.0 * rng.normal(size=n).clip(-2, 4),
            "NO2": 20.0 + 8.0 * rng.normal(size=n).clip(-2, 3),
            "NO": 10.0 + 5.0 * rng.normal(size=n).clip(-2, 3),
            "ws": rng.gamma(2, 2, n),
            "blh": rng.gamma(3, 200, n),
            "t2m": rng.normal(10, 5, n),
        }
    )
    return normalise_ox_invariant(
        df,
        covariates=["ws", "blh", "t2m"],
        method="standard",
        n_samples=4,
        n_cores=1,
    )


def test_ox_columns_are_constituent_sums(ox_deweather_result):
    """``Ox_*``/``NOx_*`` are the sums of their species, each in its own mass."""
    sp = ox_deweather_result.species
    np.testing.assert_allclose(
        sp["Ox_observed"], sp["O3_observed"] + sp["NO2_observed"], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        sp["Ox_normalised"], sp["O3_normalised"] + sp["NO2_normalised"], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        sp["NOx_observed"], sp["NO2_observed"] + sp["NO_observed"], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        sp["NOx_normalised"], sp["NO2_normalised"] + sp["NO_normalised"], rtol=0, atol=1e-12
    )
    # ... which is *not* the O3-equivalent mass of the molar total: the two
    # conventions differ by several per cent of the Ox level.
    o3_equivalent = (sp["O3_normalised"] / 48.0 + sp["NO2_normalised"] / 46.0055) * 48.0
    assert float((o3_equivalent - sp["Ox_normalised"]).abs().max()) > 0.1
    # The molar deweathering frames are unchanged: Ox is still there in
    # molar units, so the O3-equivalent convention stays available.
    assert set(ox_deweather_result.ox_result.columns) >= {"observed", "normalised"}


def test_conservation_max_err_is_roundoff_and_grows_when_perturbed(ox_deweather_result):
    """A consistent fit leaves round-off; perturbing a column raises it."""
    from normet.analysis.chemistry import ox_conservation_error

    sp = ox_deweather_result.species
    assert 0.0 <= ox_deweather_result.conservation_max_err < 1e-12
    assert ox_conservation_error(sp, unit=ox_deweather_result.unit) == 0.0

    for column in ("O3_normalised", "NO2_normalised"):
        perturbed = sp.copy()
        perturbed[column] = perturbed[column] * 1.01
        grown = ox_conservation_error(perturbed, unit=ox_deweather_result.unit)
        assert grown > 1e-10, f"1 % on {column} did not register ({grown:g})"

    # The model-side closure is the part the frame alone cannot fake: shift the
    # whole NO2 leg (and the totals with it, so the frame still adds up) and
    # the residual against the deweathered molar totals registers.
    idx = sp.index
    ox_mol = ox_deweather_result.ox_result["paired"].reindex(idx).to_numpy(dtype=float)
    nox_mol = ox_deweather_result.nox_result["paired"].reindex(idx).to_numpy(dtype=float)
    micro_to_mol = 1e-6 if ox_deweather_result.unit == "ug_m3" else 1.0
    assert (
        ox_conservation_error(
            sp,
            unit=ox_deweather_result.unit,
            ox_mol=ox_mol * micro_to_mol,
            nox_mol=nox_mol * micro_to_mol,
        )
        < 1e-12
    )
    shifted = sp.copy()
    shifted["NO2_normalised"] = shifted["NO2_normalised"] * 1.05
    shifted["Ox_normalised"] = shifted["O3_normalised"] + shifted["NO2_normalised"]
    shifted["NOx_normalised"] = shifted["NO2_normalised"] + shifted["NO_normalised"]
    assert (
        ox_conservation_error(
            shifted,
            unit=ox_deweather_result.unit,
            ox_mol=ox_mol * micro_to_mol,
            nox_mol=nox_mol * micro_to_mol,
        )
        > 1e-10
    )
