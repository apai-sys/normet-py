# tests/test_calibration.py
"""Tests for quantile calibration (normet.foundation.calibration).

The estimator is duck-typed on what the calibration calls, so every test
runs against a deterministic stub -- no torch, no foundation extra, and
none of the libomp hangs the real checkpoint would bring into this file.
The record is a fixed cycle-plus-noise hourly series, so the same
"model" (context-median forecast, fixed band shape) miscalibrates the same
way every run and the coverage the table quotes is a property of the fit,
not of the draw.

What the tests pin down, in order of honesty:
  1. a well-calibrated model is left alone (the correction is near identity);
  2. an injected miscalibration is corrected (coverage comes back to target);
  3. a behaviour beyond the hold-out earns a CalibrationWarning;
  4. thin records fall back to the named constant fit instead of failing;
  5. the guards and the PAVA leg keep the correction honest.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from normet.exceptions import CalibrationWarning, ConfigError
from normet.foundation import (
    CalibratedEstimator,
    QuantileCalibration,
    calibrate_quantiles,
)
from normet.foundation import calibration as cal

# --------------------------------------------------------------------------
# The deterministic record and its "model"
# --------------------------------------------------------------------------


@pytest.fixture
def record() -> pd.DataFrame:
    """A fixed hourly record: a 24 h cycle plus seeded noise, stationary enough
    that a context-median forecast is honestly near-unbiased."""
    rng = np.random.default_rng(7)
    n = 2200
    t = np.arange(n)
    value = 40.0 + 5.0 * np.sin(2.0 * np.pi * t / 24.0) + 2.0 * rng.standard_normal(n)
    index = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({"value": value}, index=index)


def _z(q: float) -> float:
    """The standard normal's ``q`` quantile, from a small table.

    Interpolated to a part in a thousand -- enough for a stub whose job is
    to be a calibrated or miscalibrated band, not to be a distribution.
    """
    pts = (
        (0.0, 0.0),
        (0.025, 1.9600),
        (0.05, 1.6448),
        (0.1, 1.2816),
        (0.15, 1.0364),
        (0.2, 0.8416),
        (0.25, 0.6745),
        (0.3, 0.5244),
        (0.4, 0.2533),
        (0.5, 0.0),
    )
    p = min(q, 1.0 - q)
    v = float(np.interp(p, [a for a, _ in pts], [b for _, b in pts]))
    return v if q >= 0.5 else -v


class _StubEstimator:
    """A quantile-capable estimator in all the ways the calibration needs one.

    The forecast is the median of the context window's last horizon, the band
    is that median plus or minus the context's own spread at standard-normal
    multiples, and the spread is scaled by ``miscal`` -- the injected
    miscalibration. ``miscal=1.0`` is calibrated by construction; ``miscal<1``
    is a band too narrow to cover its nominal rate.
    """

    def __init__(
        self,
        context_length: int = 96,
        prediction_length: int = 24,
        miscal: float = 1.0,
        jump: tuple[int, float] | None = None,
    ) -> None:
        self.context_length = context_length
        self.prediction_length = prediction_length
        self._miscal = miscal
        self._jump = jump

    def predict_quantiles(
        self,
        df: pd.DataFrame,
        target: str,
        anchor: pd.Timestamp | str,
        *,
        horizon: int | None = None,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        covariates: list[str] | None = None,
    ) -> pd.DataFrame:
        horizon = int(horizon or self.prediction_length)
        anchor = pd.Timestamp(anchor)
        past = df.loc[df.index < anchor, target].to_numpy()
        window = past[-self.context_length :]
        jumped = self._jump is not None and (
            int(np.searchsorted(df.index.values, anchor.to_datetime64())) >= self._jump[0]
        )
        base = float(np.median(window[-horizon:])) + (self._jump[1] if jumped else 0.0)
        scale = float(np.std(window)) * self._miscal or 1.0
        future_index = df.index[df.index >= anchor][:horizon]
        data = {f"q{q}": np.full(horizon, base + scale * _z(q)) for q in quantiles}
        return pd.DataFrame(data, index=future_index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x = X.iloc[:, 0].to_numpy()
        return np.full(self.prediction_length, float(np.median(x[-self.context_length :])))


def _fit(
    record: pd.DataFrame,
    miscal: float = 1.0,
    n_blocks: int | None = None,
    jump: tuple[int, float] | None = None,
    method: str = "isotonic",
    anchor: pd.Timestamp | None = None,
) -> CalibratedEstimator:
    est = _StubEstimator(miscal=miscal, jump=jump)
    return calibrate_quantiles(
        est,
        record,
        "value",
        anchor or record.index[-1],
        n_blocks=n_blocks,
        method=method,
    )


# --------------------------------------------------------------------------
# 1. A well-calibrated model is left alone
# --------------------------------------------------------------------------


def test_a_calibrated_model_is_left_alone(record: pd.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", CalibrationWarning)
        cal_est = _fit(record, miscal=1.0)

    assert isinstance(cal_est, CalibratedEstimator)
    assert isinstance(cal_est.calibration, QuantileCalibration)
    table = cal_est.coverage
    assert list(table.columns) == list(cal.COVERAGE_COLUMNS)
    row = table[table["level"] == 0.9].iloc[0]
    assert row["nominal_coverage"] == pytest.approx(0.8)
    # The model's own band already covers at its nominal rate, on the hold-out.
    assert row["empirical_uncalibrated"] == pytest.approx(0.8, abs=0.1)

    # The correction is near identity: the band it prints is the band the
    # model made, within the fit's own noise.
    base = cal_est.base
    from_ts = record.index[-48]
    raw = base.predict_quantiles(record, "value", from_ts, horizon=24, quantiles=(0.1, 0.5, 0.9))
    fixed = cal_est.predict_quantiles(
        record, "value", from_ts, horizon=24, quantiles=(0.1, 0.5, 0.9)
    )
    assert list(fixed.columns) == [f"q{q}" for q in (0.1, 0.5, 0.9)]
    assert (fixed["q0.5"].to_numpy() == raw["q0.5"].to_numpy()).all()
    shift = float(np.median(np.abs(fixed["q0.9"].to_numpy() - raw["q0.9"].to_numpy())))
    assert shift < 0.5  # an eighth of the band's half-width, at most


# --------------------------------------------------------------------------
# 2. An injected miscalibration is corrected
# --------------------------------------------------------------------------


def test_injected_miscalibration_is_corrected(record: pd.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", CalibrationWarning)
        cal_est = _fit(record, miscal=0.6)

    table = cal_est.coverage
    row = table[table["level"] == 0.9].iloc[0]
    # Six-tenths of the spread: the model's own band covers well short of
    # its nominal 80 %, which is why the calibration was called.
    assert row["empirical_uncalibrated"] < 0.7
    # And the corrected band brings it back onto target, cross-validated.
    assert row["empirical_calibrated"] == pytest.approx(0.8, abs=0.1)
    assert row["abs_deviation"] <= cal.DEFAULT_TOLERANCE
    assert cal_est.calibration.warnings == ()
    assert cal_est.calibration.n_blocks == 7
    assert 6 * 24 <= cal_est.calibration.n_points <= 7 * 24

    # The correction acts on the band edges, not on the median.
    from_ts = record.index[-48]
    raw = cal_est.base.predict_quantiles(
        record, "value", from_ts, horizon=24, quantiles=(0.1, 0.5, 0.9)
    )
    fixed = cal_est.predict_quantiles(
        record, "value", from_ts, horizon=24, quantiles=(0.1, 0.5, 0.9)
    )
    assert (fixed["q0.9"].to_numpy() > raw["q0.9"].to_numpy()).all()
    assert (fixed["q0.1"].to_numpy() < raw["q0.1"].to_numpy()).all()
    assert (fixed["q0.5"].to_numpy() == raw["q0.5"].to_numpy()).all()


# --------------------------------------------------------------------------
# 3. Beyond the hold-out earns a warning
# --------------------------------------------------------------------------


def test_behaviour_beyond_the_holdout_warns(record: pd.DataFrame) -> None:
    # A level jump the hold-out never saw: it lands in the most recent held-out
    # block, so the fitted threshold cannot cover it and the table says so.
    n = len(record)
    jump_from = n - 24 - 1
    with pytest.warns(CalibrationWarning):
        cal_est = _fit(record, miscal=1.0, n_blocks=3, jump=(jump_from, 15.0))

    assert cal_est.calibration.warnings
    assert "q0.9" in cal_est.calibration.warnings[0]
    row = cal_est.coverage[cal_est.coverage["level"] == 0.9].iloc[0]
    assert row["abs_deviation"] > cal.DEFAULT_TOLERANCE


# --------------------------------------------------------------------------
# 4. Thin records take the named fallback
# --------------------------------------------------------------------------


def test_thin_holdout_falls_back_to_the_constant(record: pd.DataFrame) -> None:
    cal_est = _fit(record, miscal=0.6, n_blocks=1)
    assert cal_est.calibration.method == "constant"
    assert all(np.isfinite(cal_est.calibration.threshold))
    assert isinstance(cal_est.coverage, pd.DataFrame) and len(cal_est.coverage) == 1


def test_a_record_with_no_scored_errors_is_rejected(record: pd.DataFrame) -> None:
    est = _StubEstimator(context_length=2, prediction_length=2)
    tight2 = record.iloc[:7]
    with pytest.raises(ConfigError, match="no usable spread|scored errors"):
        calibrate_quantiles(est, tight2, "value", tight2.index[-1])


# --------------------------------------------------------------------------
# 5. Guards, the PAVA leg, and the wrapper's surface
# --------------------------------------------------------------------------


def test_guards_name_their_reason(record: pd.DataFrame) -> None:
    est = _StubEstimator()
    with pytest.raises(ConfigError, match="median"):
        calibrate_quantiles(est, record, "value", record.index[-1], quantiles=(0.1, 0.9))
    with pytest.raises(ConfigError, match="strictly increasing"):
        calibrate_quantiles(est, record, "value", record.index[-1], quantiles=(0.9, 0.1, 0.5))
    with pytest.raises(ConfigError, match="method"):
        calibrate_quantiles(est, record, "value", record.index[-1], method="gaussian")
    with pytest.raises(ConfigError, match="predict_quantiles"):
        calibrate_quantiles(object(), record, "value", record.index[-1])  # type: ignore[arg-type]
    unindexed = record.reset_index(drop=True)
    with pytest.raises(ConfigError, match="DatetimeIndex"):
        calibrate_quantiles(est, unindexed, "value", "2020-06-01")
    with pytest.raises(ConfigError, match="no column"):
        calibrate_quantiles(est, record, "nope", record.index[-1])
    tight = record.iloc[: 96 + 24]
    with pytest.raises(ConfigError, match="no room"):
        calibrate_quantiles(est, tight, "value", tight.index[-1])


def test_the_pava_leg_keeps_the_order_honest() -> None:
    # A crossing of the raw thresholds is resolved by pooling, and the
    # resolution is the non-decreasing one.
    out = cal._pava(np.array([2.0, 0.5]), np.array([1.0, 1.0]))
    assert (out >= 0.0).all()
    assert (np.diff(out) >= 0.0).all()
    assert out.min() == pytest.approx(1.25)

    # One step past the sample quantile: the order statistic, not an average.
    errs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert cal._conformal_threshold(errs, 0.5) == pytest.approx(3.0)
    assert cal._conformal_threshold(np.array([]), 0.8) == pytest.approx(1.0)

    # Two bands whose raw thresholds cross come back non-decreasing.
    rng = np.random.default_rng(11)
    wide = rng.uniform(0.0, 5.0, 40)
    narrow = rng.uniform(0.0, 1.0, 40)
    blocks = [np.column_stack([wide, narrow])]
    th = cal._fit_pass(blocks, [0.75, 0.9], set(), method="isotonic")
    assert th[0] <= th[1]
    assert th[0] > 0.0


def test_the_wrapper_covers_the_base_estimators_surface(record: pd.DataFrame) -> None:
    cal_est = _fit(record, miscal=1.0)
    assert cal_est.context_length == 96
    assert cal_est.prediction_length == 24
    assert cal_est.base is cal_est.__dict__["base"]
    assert cal_est.coverage is cal_est.calibration.coverage
    assert cal_est.calibration.levels == (0.9,)


def test_the_fit_is_deterministic(record: pd.DataFrame) -> None:
    a = _fit(record, miscal=0.6)
    b = _fit(record, miscal=0.6)
    assert a.calibration.threshold == b.calibration.threshold
    pd.testing.assert_frame_equal(a.coverage, b.coverage)
