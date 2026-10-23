"""Skill scores: the pinball loss, the CRPS exact against a reference integral, and
the point forecast's score on the spread scores' scale."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from normet.analysis.verification import (
    crps_from_quantiles,
    crps_from_samples,
    crps_observed,
    pinball_loss,
)
from normet.exceptions import DataError

# numpy 2.0 renamed `np.trapz` to `np.trapezoid` and removed the old name. The
# getattr must not take `np.trapz` as its default: that expression is evaluated
# eagerly and would itself raise on numpy 2.x, defeating the shim.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _reference(q, tau, y, n=2_000_001):
    """The CRPS of one row's quantile forecast by dense integration, as an
    independent check: the piecewise-linear CDF through the (value, level)
    points, flat 0 below the smallest and 1 above the largest, squared
    distance from the observation's step CDF, integrated over the span."""
    q = np.asarray(q, dtype=float)
    v, inv = np.unique(q, return_inverse=True)
    u = np.zeros(v.size)
    np.maximum.at(u, inv, np.asarray(tau, dtype=float))

    def F(x):
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        out[x <= v[0]] = 0.0
        out[x >= v[-1]] = 1.0
        mid = (x > v[0]) & (x < v[-1])
        i = np.clip(np.searchsorted(v, x[mid], side="right") - 1, 0, v.size - 2)
        t = (x[mid] - v[i]) / (v[i + 1] - v[i])
        out[mid] = u[i] + t * (u[i + 1] - u[i])
        return out

    lo = min(v.min(), y)
    hi = max(v.max(), y)
    xs = np.linspace(lo, hi, n)
    return float(_trapz((F(xs) - (xs >= y).astype(float)) ** 2, xs))


# --------------------------------------------------------------- pinball


def test_pinball_at_half_is_half_the_mae():
    pred = np.array([1.0, 3.0, 5.0])
    obs = np.array([2.0, 1.0, 6.0])
    assert pinball_loss(pred, obs, 0.5) == pytest.approx(0.5 * np.mean(np.abs(obs - pred)))


def test_pinball_costs_under_and_over_forecasts_by_the_level():
    # one row, so the mean is the row's loss
    assert pinball_loss(np.array([0.0]), np.array([1.0]), 0.25) == pytest.approx(0.25)
    assert pinball_loss(np.array([1.0]), np.array([0.0]), 0.25) == pytest.approx(0.75)
    assert pinball_loss(np.array([0.0]), np.array([1.0]), 0.75) == pytest.approx(0.75)
    assert pinball_loss(np.array([1.0]), np.array([0.0]), 0.75) == pytest.approx(0.25)


def test_pinball_of_the_true_quantile_beats_the_neighbours():
    # the pinball at q is lowest at the q-quantile among the candidates
    obs = np.arange(10.0)
    q = 0.4
    candidates = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    losses = [pinball_loss(np.full_like(obs, c), obs, q) for c in candidates]
    assert candidates[int(np.argmin(losses))] == pytest.approx(4.0)
    # and symmetrically below, at the 0.4-quantile of a 4-point record
    assert pinball_loss(np.full(4, 1.2), np.arange(4.0), q) <= pinball_loss(
        np.full(4, 2.4), np.arange(4.0), q
    )


def test_pinball_drops_only_the_non_finite_pairs():
    pred = np.array([1.0, np.nan, 3.0, np.inf])
    obs = np.array([1.0, 1.0, 2.0, 2.0])
    # finite pairs: (1,1) -> 0, (3,2) -> 0.5 * 1
    assert pinball_loss(pred, obs, 0.5) == pytest.approx(0.25)


def test_pinball_refuses_an_out_of_range_level():
    with pytest.raises(DataError):
        pinball_loss(np.array([1.0]), np.array([1.0]), 1.0)
    with pytest.raises(DataError):
        pinball_loss(np.array([1.0]), np.array([1.0]), -0.1)


# ------------------------------------------------- the quantile route
# the hand-integrated values below are exact: the integrand is a quadratic
# on each piece of the break-point grid (the quantile values and the
# observation), and those pieces add up as stated.


def test_the_simple_case_against_the_hand_integral():
    # q = (0, 1, 10) at (0.05, 0.5, 0.9), observed 3: pieces [0,1] 0.05->0.5,
    # [1,3] 0.5->0.5 + 2*(4/45), [3,10] -> 0.9, squared distance from the
    # step at 3
    assert crps_from_quantiles(
        np.array([[0.0, 1.0, 10.0]]), np.array([3.0]), quantiles=[0.05, 0.5, 0.9]
    ) == pytest.approx(1.20027778, abs=1e-6)


def test_the_observation_off_the_span_reads_the_flat_tails():
    # observed above the largest quantile and below the smallest: the
    # integrand is 1 out there (the CDF still 0.05-0.9 where the
    # observation sits), so the tails enter the score
    assert crps_from_quantiles(
        np.array([[0.0, 2.0, 4.0]]), np.array([10.0]), quantiles=[0.05, 0.5, 0.9]
    ) == pytest.approx(7.19166667, abs=1e-6)
    assert crps_from_quantiles(
        np.array([[5.0, 8.0, 10.0]]), np.array([1.0]), quantiles=[0.05, 0.5, 0.9]
    ) == pytest.approx(5.83416667, abs=1e-6)


def test_a_repeated_value_is_a_vertical_piece_not_extra_score():
    # two quantiles on one value: the CDF jumps there (a vertical piece of
    # measure zero for the integral) -- the score is the one of the
    # (2 -> 6) segment only
    assert crps_from_quantiles(
        np.array([[2.0, 2.0, 6.0]]), np.array([4.0]), quantiles=[0.25, 0.5, 0.75]
    ) == pytest.approx(0.83333333, abs=1e-6)
    # and the point forecast (all values one): exactly |obs - value|
    assert crps_from_quantiles(
        np.array([[4.0, 4.0, 4.0]]), np.array([6.0]), quantiles=[0.1, 0.5, 0.9]
    ) == pytest.approx(2.0, abs=1e-9)


def test_the_observation_on_a_quantile_break_point():
    assert crps_from_quantiles(
        np.array([[1.0, 3.0, 5.0]]), np.array([3.0]), quantiles=[0.1, 0.5, 0.9]
    ) == pytest.approx(0.41333333, abs=1e-6)
    # and on the smallest / largest value: the flat tails carry the rest
    assert crps_from_quantiles(
        np.array([[1.0, 3.0, 5.0]]), np.array([1.0]), quantiles=[0.1, 0.5, 0.9]
    ) == pytest.approx(1.21333333, abs=1e-6)
    assert crps_from_quantiles(
        np.array([[1.0, 3.0, 5.0]]), np.array([5.0]), quantiles=[0.1, 0.5, 0.9]
    ) == pytest.approx(1.21333333, abs=1e-6)


def test_crps_from_quantiles_matches_the_reference_on_random_tables():
    rng = np.random.default_rng(20261022)
    for _ in range(12):
        k = int(rng.integers(1, 9))
        n = int(rng.integers(1, 7))
        q = rng.normal(20, 5, size=(n, k))
        if k > 1 and rng.random() < 0.4:  # some ties
            q[rng.integers(0, n), rng.integers(0, k)] = q[rng.integers(0, n), rng.integers(0, k)]
        tau = rng.uniform(0.02, 0.98, size=k)
        y = rng.uniform(-10, 45, size=n)
        got = crps_from_quantiles(q, y, quantiles=list(tau))
        ref = float(np.mean([_reference(q[i], tau, y[i]) for i in range(n)]))
        assert got == pytest.approx(ref, rel=1e-5), (k, n, q, tau, y, got, ref)


def test_crps_from_quantiles_reads_the_qnnn_columns_directly():
    # the normalise-return_quantiles shape: the levels are the column names
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020", periods=3),
            "q025": [1.0, 1.0, 1.0],
            "q500": [2.0, 2.0, 2.0],
            "q975": [5.0, 5.0, 5.0],
        }
    )
    obs = pd.Series([1.0, 4.0, 5.0])
    got = crps_from_quantiles(df[["q025", "q500", "q975"]], obs)
    want = crps_from_quantiles(
        df[["q025", "q500", "q975"]].values, obs.values, quantiles=[0.025, 0.5, 0.975]
    )
    assert got == want
    # the unlabelled columns are not levels: the naming is what makes it a
    # quantile table, and a frame without one is refused
    with pytest.raises(DataError):
        crps_from_quantiles(pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]}), [1.0, 2.0])


def test_crps_from_quantiles_scores_the_finite_rows_only():
    df = pd.DataFrame(
        {"q025": [1.0, 1.0, 1.0], "q500": [2.0, 2.0, np.nan], "q975": [5.0, 5.0, 5.0]}
    )
    obs = pd.Series([1.0, 4.0, 5.0])
    got = crps_from_quantiles(df, obs)
    want = crps_from_quantiles(df.drop(index=2), obs.drop(index=2))
    assert got == pytest.approx(want)


def test_crps_from_quantiles_means_over_its_rows():
    # two identical rows: the mean of identical scores is that score
    one = crps_from_quantiles(
        np.array([[0.0, 1.0, 10.0]]), np.array([3.0]), quantiles=[0.05, 0.5, 0.9]
    )
    two = crps_from_quantiles(
        np.array([[0.0, 1.0, 10.0], [0.0, 1.0, 10.0]]),
        np.array([3.0, 3.0]),
        quantiles=[0.05, 0.5, 0.9],
    )
    assert two == pytest.approx(one)


# ------------------------------------------------------- the sample route


def test_crps_from_samples_scores_each_sample_by_its_rank():
    # by hand: two samples per row. Row 1: samples (1, 5), observed 3 -- the
    # CDF through (1, 1/3) and (5, 2/3); row 2: (10, 12), observed 11
    got = crps_from_samples(np.array([[1.0, 5.0], [10.0, 12.0]]), [3.0, 11.0])
    assert got == pytest.approx(0.52777778, abs=1e-6)
    # the sort is the ranking: unsorted input scores identically
    got_unsorted = crps_from_samples(np.array([[5.0, 1.0], [12.0, 10.0]]), [3.0, 11.0])
    assert got_unsorted == pytest.approx(got)


def test_crps_from_samples_matches_the_quantile_route_on_the_same_levels():
    # both routes are the same integral: the sorted samples at their rank
    # levels and the quantile table of the same values score identically
    rng = np.random.default_rng(11)
    S = rng.normal(20, 3, size=(5, 7))
    o = rng.normal(20, 3, size=5)
    via_samples = crps_from_samples(S, o)
    levels = [i / 8 for i in range(1, 8)]
    via_quantiles = crps_from_quantiles(np.sort(S, axis=1), o, quantiles=levels)
    assert via_samples == pytest.approx(via_quantiles, rel=1e-9)


def test_crps_from_samples_reads_the_wide_table_directly():
    # the normalise(aggregate=False) shape: date, observed, one column a seed
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "observed": [3.0, 11.0],
            0: [1.0, 10.0],
            1: [5.0, 12.0],
        }
    )
    assert crps_from_samples(df[[0, 1]], df["observed"]) == pytest.approx(0.52777778, abs=1e-6)
    with pytest.raises(DataError):
        crps_from_samples(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))  # one


def test_the_normal_spread_scores_below_the_point_forecast():
    # the plan's convention: a spread score and the point-forecast reference
    # are on one scale -- an honest large spread, scored at its own mean,
    # reads below the point forecast at the same mean, and a doubled
    # spread reads far worse than the doubled point error it adds
    rng = np.random.default_rng(7)
    s = 4.0
    obs = np.full(200, 20.0)
    honest = rng.normal(20, s, size=(200, 200))
    spread = crps_from_samples(honest, obs)
    # the point forecast at the mean on a record of the same size
    draws = rng.normal(20, s, size=200)
    point = crps_observed(np.full_like(draws, 20.0), draws)
    assert spread < point
    # and an over-wide spread is penalised: its CRPS exceeds the honest
    over = rng.normal(20, 2 * s, size=(200, 200))
    assert crps_from_samples(over, obs) > spread


# ------------------------------------------------------ the point forecast


def test_crps_observed_is_its_mae_times_the_normal_theory_factor():
    pred = np.array([1.0, 3.0, 5.0])
    obs = np.array([2.0, 1.0, 6.0])
    mae = np.mean(np.abs(obs - pred))
    assert crps_observed(pred, obs) == pytest.approx(np.sqrt(np.pi / 2.0) * mae)
    assert crps_observed(pd.Series(pred), pd.Series(obs)) == crps_observed(pred, obs)


def test_crps_observed_drops_the_non_finite_pairs():
    pred = np.array([1.0, np.nan, 3.0])
    obs = np.array([1.0, 1.0, 0.0])
    # finite rows: |1-1| = 0 and |0-3| = 3, so MAE 1.5, times the factor
    assert crps_observed(pred, obs) == pytest.approx(np.sqrt(np.pi / 2.0) * 1.5)


# ------------------------------------------------------------- refusals


def test_the_wrong_shapes_are_refused():
    with pytest.raises(DataError):
        crps_from_quantiles(np.array([[1.0, 2.0]]), np.array([1.0, 2.0]))  # lengths
    with pytest.raises(DataError):
        crps_from_quantiles(np.array([[1.0, 2.0]]), np.array([1.0]))  # no levels
    with pytest.raises(DataError):
        crps_from_quantiles(np.array([[1.0]]), np.array([1.0]))  # one level
    with pytest.raises(DataError):
        crps_from_samples(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(DataError):
        crps_observed(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(DataError):
        crps_from_quantiles(pd.DataFrame({"q025": [np.nan], "q500": [np.nan]}), [np.nan])


def test_every_score_is_non_negative_and_a_float():
    rng = np.random.default_rng(1)
    for _ in range(25):
        k = int(rng.integers(2, 12))
        n = int(rng.integers(1, 9))
        s = rng.normal(0, rng.uniform(0.01, 9), size=(n, k))
        o = rng.normal(0, rng.uniform(0.01, 9), size=n)
        assert isinstance(crps_from_samples(s, o), float)
        assert crps_from_samples(s, o) >= 0.0
        assert crps_observed(np.full_like(o, 1.0), o) >= 0.0
