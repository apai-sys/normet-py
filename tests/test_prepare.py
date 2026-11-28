import numpy as np
import pandas as pd
import pytest

from normet.utils.prepare import (
    add_date_variables,
    check_data,
    impute_values,
    prepare_data,
    process_date,
    split_into_sets,
)


def test_process_date_from_datetime_index():
    n = 10
    df = pd.DataFrame({"x": range(n)}, index=pd.date_range("2024-01-01", periods=n, freq="h"))
    out = process_date(df)
    assert "date" in out.columns
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_process_date_coerces_string_column():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "x": [1, 2, 3],
        }
    )
    out = process_date(df)
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_process_date_raises_when_no_datetime():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        process_date(df)


def test_check_data_renames_target_to_value():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="h"),
            "PM2.5": [1.0, 2.0, 3.0, 4.0],
            "t2m": [10, 11, 12, 13],
        }
    )
    out = check_data(df, covariates=["t2m"], target="PM2.5")
    assert "value" in out.columns
    assert "PM2.5" not in out.columns


def test_impute_values_fills_numeric_with_median():
    df = pd.DataFrame(
        {
            "value": [1.0, 2.0, np.nan, 4.0],
            "x": [10.0, np.nan, np.nan, 40.0],
        }
    )
    out = impute_values(df, dropna=True)
    # NaN in target → row dropped
    assert len(out) == 3
    # NaN in feature → filled with median (which is 25.0 after target row drop)
    assert not out["x"].isna().any()


def test_add_date_variables_creates_time_features():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="h"),
            "value": [1.0, 2.0, 3.0],
        }
    )
    out = add_date_variables(df)
    for col in ("date_unix", "day_julian", "weekday", "hour"):
        assert col in out.columns


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_add_date_variables_date_unix_is_epoch_seconds_at_any_resolution(unit):
    """`date_unix` is seconds since the epoch whatever the column's unit.

    pandas >= 2.0 keeps a non-nanosecond resolution (a ``datetime64[s]``
    column survives ``read_parquet`` as seconds), and ``astype("int64")``
    counts in that resolution. Without the ns conversion the trend feature
    came out as ~1 instead of ~1.7e9 for such a column -- a silent loss of
    9 orders of magnitude, and no error anywhere.
    """
    date = pd.Series(
        np.array(["2024-01-01T00:00:00", "2024-01-01T01:00:00"], dtype=f"datetime64[{unit}]")
    )
    out = add_date_variables(pd.DataFrame({"date": date, "value": [1.0, 2.0]}))
    assert out["date_unix"].tolist() == [1_704_067_200, 1_704_070_800]
    assert out["hour"].tolist() == [0, 1]


def test_split_into_sets_partitions_all_rows():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100, freq="h"),
            "value": np.arange(100, dtype=float),
        }
    )
    for method in ("random", "ts", "month_ts", "season_ts"):
        out = split_into_sets(df, split_method=method, train_fraction=0.75, seed=0)
        assert set(out["set"].unique()) <= {"training", "testing"}
        assert len(out) == 100


def test_prepare_data_full_pipeline(synthetic_aq):
    out = prepare_data(
        synthetic_aq.copy(),
        target="PM2.5",
        covariates=["t2m", "blh", "u10", "v10"],
        split_method="ts",
        train_fraction=0.8,
    )
    assert "value" in out.columns
    assert "set" in out.columns
    for col in ("date_unix", "day_julian", "weekday", "hour"):
        assert col in out.columns
    # Time-ordered split: training rows precede testing rows in time.
    train_max = out.loc[out["set"] == "training", "date"].max()
    test_min = out.loc[out["set"] == "testing", "date"].min()
    assert train_max <= test_min


# ---------------------------------------------------------------------------
# log1p_target / expm1_columns (the log-scale fit's transform pair)
# ---------------------------------------------------------------------------


def test_log1p_target_transforms_only_the_named_column():
    from normet.utils.prepare import log1p_target

    df = pd.DataFrame({"value": [0.0, 1.0, 10.0], "other": [5.0, 5.0, 5.0]})
    out = log1p_target(df, "value")
    assert np.allclose(out["value"], np.log1p(df["value"].to_numpy()))
    assert np.allclose(out["other"], 5.0)
    # the input is not modified
    assert np.allclose(df["value"], [0.0, 1.0, 10.0])


def test_log1p_target_rejects_negative_values():
    from normet.utils.prepare import log1p_target

    df = pd.DataFrame({"value": [0.0, -0.5, 3.0]})
    with pytest.raises(ValueError, match="non-negative"):
        log1p_target(df, "value")
    # the message names how bad it is
    df = pd.DataFrame({"value": [0.0, -2.0, -0.5, 3.0]})
    with pytest.raises(ValueError, match="2 rows"):
        log1p_target(df, "value")


def test_log1p_target_rejects_a_missing_column():
    from normet.utils.prepare import log1p_target

    with pytest.raises(ValueError, match="not found"):
        log1p_target(pd.DataFrame({"value": [1.0]}), "nope")


def test_exm1_columns_inverts_log1p_target_on_all_numeric_columns():
    from normet.utils.prepare import expm1_columns, log1p_target

    df = pd.DataFrame({"value": [0.0, 1.0, 10.0], "q500": [0.0, 2.0, 8.0], "note": ["a", "b", "c"]})
    out = expm1_columns(log1p_target(df, "value"))
    assert np.allclose(out["value"], df["value"].to_numpy())
    assert np.allclose(out["q500"], np.expm1(df["q500"].to_numpy()))
    assert list(out["note"]) == ["a", "b", "c"]


def test_process_date_reads_a_column_spanning_a_daylight_saving_change():
    """Mixed UTC offsets have no one zone; pandas 2 left an object column
    (with a warning) and pandas 3 raises, so the column was unusable."""
    from normet.utils.prepare import process_date

    df = pd.DataFrame(
        {
            "date": ["2020-03-29T00:00+00:00", "2020-03-29T02:00+01:00"],
            "x": [1, 2],
        }
    )
    out = process_date(df)
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out["date"].isna().sum() == 0
    # Both are read on one clock: 00:00 UTC and 02:00+01:00 = 01:00 UTC.
    assert [str(t) for t in out["date"]] == [
        "2020-03-29 00:00:00+00:00",
        "2020-03-29 01:00:00+00:00",
    ]


def test_process_date_prefers_a_column_named_date_over_a_time_column():
    from normet.utils.prepare import process_date

    df = pd.DataFrame(
        {"time": ["13:00", "14:00"], "date": ["2020-01-31", "2020-02-01"], "x": [1, 2]}
    )
    out = process_date(df)
    assert out["date"].tolist() == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-01")]


def test_impute_values_fills_a_nullable_integer_with_a_fractional_median():
    """``fillna(1.5)`` on an Int64 column raises "Invalid value '1.5' for
    dtype 'Int64'" -- and whether the median was whole decided it."""
    from normet.utils.prepare import impute_values

    df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "x": pd.array([1, pd.NA, 2], dtype="Int64")})
    out = impute_values(df, dropna=True)
    assert out["x"].tolist() == [pytest.approx(1.0), pytest.approx(1.5), pytest.approx(2.0)]


# ---------------------------------------------------------------------------
# An embargo that purges the record, not just the block boundaries
# ---------------------------------------------------------------------------


def _embargo_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="h"),
            "x": rng.normal(size=n),
            "value": rng.normal(size=n) + 10.0,
        }
    )


def test_random_split_embargo_that_leaves_a_handful_of_rows_raises():
    """random + a wide embargo purges every test row's neighbours, and they overlap.

    At ``embargo=6`` on 300 rows only 11 training rows survived; the run then
    died inside LightGBM's fold builder with "range() arg 3 must not be zero",
    which names neither embargo nor the split.
    """
    from normet.utils.prepare import prepare_data

    with pytest.raises(ValueError, match="embargo=6.*leaves only"):
        prepare_data(
            _embargo_frame(),
            target="value",
            covariates=["x"],
            split_method="random",
            train_fraction=0.75,
            embargo=6,
        )


def test_random_split_embargo_that_eats_half_the_record_warns():
    from normet.utils.prepare import split_into_sets

    with pytest.warns(UserWarning, match="purges"):
        split_into_sets(_embargo_frame(), "random", 0.75, seed=1, embargo=1)


def test_a_contiguous_split_purges_only_its_boundaries():
    """The documented use of embargo: a block boundary, not every test row."""
    from normet.utils.prepare import split_into_sets

    out = split_into_sets(_embargo_frame(), "ts", 0.75, seed=1, embargo=6)
    counts = out["set"].value_counts().to_dict()
    assert counts["embargo"] == 6
    assert counts["training"] > 200


def test_zero_embargo_is_untouched():
    from normet.utils.prepare import split_into_sets

    out = split_into_sets(_embargo_frame(), "random", 0.75, seed=1, embargo=0)
    assert set(out["set"].unique()) == {"training", "testing"}


def _pandas_interp_reference(vals: np.ndarray, *, max_gap: int | None, is_circular: bool):
    """The previous implementation, kept as the oracle for the numpy rewrite.

    ``Series.interpolate(method="linear", limit_area="inside")`` fills every
    NaN in a series with the straight line through its neighbours; the
    segmented path then writes back only the eligible interior runs.
    """
    out_vals = vals.astype(float, copy=True)
    is_na = ~np.isfinite(out_vals)
    elig = np.zeros(len(is_na), dtype=bool)
    i = 0
    while i < len(is_na):
        if not is_na[i]:
            i += 1
            continue
        start = i
        while i < len(is_na) and is_na[i]:
            i += 1
        stop = i
        if start > 0 and stop < len(is_na) and (max_gap is None or stop - start <= max_gap):
            elig[start:stop] = True
    if not np.any(elig):
        return out_vals
    if is_circular:
        rad = np.deg2rad(out_vals)
        sin_raw = (
            pd.Series(np.sin(rad)).interpolate(method="linear", limit_area="inside").to_numpy()
        )
        cos_raw = (
            pd.Series(np.cos(rad)).interpolate(method="linear", limit_area="inside").to_numpy()
        )
        ok = (
            elig
            & np.isfinite(sin_raw)
            & np.isfinite(cos_raw)
            & (np.hypot(sin_raw, cos_raw) > 1e-12)
        )
        out_vals[ok] = np.rad2deg(np.arctan2(sin_raw[ok], cos_raw[ok])) % 360.0
    else:
        lin_raw = pd.Series(out_vals).interpolate(method="linear", limit_area="inside").to_numpy()
        out_vals[elig] = lin_raw[elig]
    return out_vals


def test_segment_interpolation_is_bit_identical_to_the_pandas_path():
    """The numpy rewrite must reproduce the old numbers exactly, not approximately.

    The interpolation runs once per column per split slice, so it is on the hot
    path of ``cv_score``/``prepare_data`` with ``impute="time_interp"``; the
    rewrite is several times cheaper, but any change in the filled values would
    change every downstream model.
    """
    from normet.utils.prepare import _interp_segment_1d

    rng = np.random.default_rng(11)
    for trial in range(120):
        n = int(rng.integers(1, 90))
        vals = rng.normal(size=n)
        for _ in range(int(rng.integers(0, 4))):
            start = int(rng.integers(0, n))
            vals[start : min(n, start + int(rng.integers(1, 6)))] = rng.choice(
                [np.nan, np.inf, -np.inf]
            )
        circular = trial % 3 == 0
        if circular:
            vals = np.abs(vals) * 360.0 / (np.nanmax(np.abs(vals)) + 1e-9)
        max_gap = [None, 1, 2, 6][trial % 4]
        expected = _pandas_interp_reference(vals, max_gap=max_gap, is_circular=circular)
        got = _interp_segment_1d(vals, max_gap=max_gap, is_circular=circular)
        assert np.array_equal(got, expected, equal_nan=True), (trial, n, circular, max_gap)
