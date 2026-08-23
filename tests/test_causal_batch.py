"""Tests for :func:`normet.causal.batch.scm_all`.

``scm_all`` fans :func:`normet.causal.run_scm.run_scm` out across every unit in
a panel, treating each one in turn -- the shape a placebo-in-space study needs.
Its defining behaviour is that one unit's failure must not take the batch down
with it: a donor pool that empties out, a unit too sparse to fit, a backend
that rejects one specification. Those are logged and skipped, and only a batch
where *nothing* succeeded is an error.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from normet.causal.batch import scm_all

CUTOFF = "2023-02-01"


UNITS = ("T", "D1", "D2", "D3", "D4")


def _panel(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """A dense panel: one treated-looking unit and four donors.

    The donors share a common factor so the ridge fit has something to work
    with. Four rather than three because every unit takes its turn as treated
    and is excluded from its own pool -- scm() needs at least three donors left
    over, and two is not enough to estimate weights from.
    """
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 1.0, n)
    rows = []
    for i, unit in enumerate(UNITS):
        values = 30.0 + 2.0 * common + rng.normal(0.0, 0.3, n) + i
        for date, value in zip(dates, values, strict=True):
            rows.append({"date": date, "site": unit, "no2": float(value)})
    return pd.DataFrame(rows)


def test_every_unit_gets_its_turn_as_treated():
    out = scm_all(
        _panel(),
        date_col="date",
        outcome_col="no2",
        unit_col="site",
        donors=None,
        cutoff_date=CUTOFF,
        n_cores=1,
    )

    assert set(out["site"]) == set(UNITS)
    assert {"date", "observed", "synthetic", "effect", "site"} <= set(out.columns)
    # One block per unit, concatenated long rather than joined wide.
    assert out.groupby("site").size().nunique() == 1
    assert np.isfinite(out["synthetic"]).all()


def test_effect_is_observed_minus_synthetic():
    """Pins the column contract the batch frame inherits from run_scm."""
    out = scm_all(
        _panel(),
        date_col="date",
        outcome_col="no2",
        unit_col="site",
        donors=None,
        cutoff_date=CUTOFF,
        n_cores=1,
    )

    np.testing.assert_allclose(out["effect"], out["observed"] - out["synthetic"], atol=1e-8)


def test_one_unit_failing_does_not_take_the_batch_down(caplog):
    """A unit that only came online after the cutoff has no pre-period to fit.

    It is held out of the donor pool so it breaks nobody else's fit, and its
    own failure must cost the batch that one unit and nothing more.
    """
    panel = _panel()
    late = panel[(panel["site"] == "D3") & (panel["date"] >= CUTOFF)].copy()
    late["site"] = "LATE"
    df = pd.concat([panel, late], ignore_index=True)

    with caplog.at_level(logging.WARNING):
        out = scm_all(
            df,
            date_col="date",
            outcome_col="no2",
            unit_col="site",
            donors=list(UNITS[1:]),
            cutoff_date=CUTOFF,
            n_cores=1,
        )

    assert "LATE" not in set(out["site"])
    assert set(out["site"]) == set(UNITS)
    assert "failed for unit LATE" in caplog.text


def test_a_batch_where_nothing_succeeded_is_an_error():
    """An empty frame would read downstream as 'no effect anywhere'."""
    with pytest.raises(RuntimeError, match="All synthetic-control runs failed"):
        scm_all(
            _panel(),
            date_col="date",
            outcome_col="no2",
            unit_col="site",
            donors=None,
            cutoff_date=CUTOFF,
            scm_backend="not-a-backend",
            n_cores=1,
        )


def test_worker_count_is_clamped_to_at_least_one():
    """``n_cores=0`` would otherwise reach joblib as a request for no workers."""
    out = scm_all(
        _panel(),
        date_col="date",
        outcome_col="no2",
        unit_col="site",
        donors=None,
        cutoff_date=CUTOFF,
        n_cores=0,
    )

    assert set(out["site"]) == set(UNITS)
