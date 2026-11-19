"""FLAML backend budget defaults: trial-bounded by default, clock on request.

The search bound decides whether model selection is reproducible, so these
tests pin which bound reaches ``AutoML.fit`` -- no FLAML import is needed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from normet.backends import flaml_backend


class _FakeAutoML:
    """Records ``fit``'s kwargs instead of searching."""

    def __init__(self) -> None:
        self.best_estimator = "lgbm"
        self.best_config: dict = {}
        self.best_loss = 0.0
        self.fit_kwargs: dict = {}

    def fit(self, X_train=None, y_train=None, seed=None, **kwargs):
        self.fit_kwargs = {"seed": seed, **kwargs}


@pytest.fixture
def fake_automl(monkeypatch):
    captured: dict = {}

    def factory():
        captured["automl"] = _FakeAutoML()
        return lambda: captured["automl"]

    monkeypatch.setattr(flaml_backend, "_import_flaml_automl", factory)
    return captured


def _df() -> pd.DataFrame:
    return pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 0.0, 1.0, 0.0]})


def test_default_search_is_trial_bounded(fake_automl):
    """The default must not depend on how loaded the node is.

    A wall-clock default returned a different model on a busy node with no
    sign that the search had been cut short; a fixed trial count with a
    fixed seed is reproducible.
    """
    flaml_backend.train_flaml(_df(), target="value", covariates=["x"])
    kw = fake_automl["automl"].fit_kwargs
    assert kw["max_iter"] == 30
    assert kw["time_budget"] == -1


def test_explicit_time_budget_drops_the_trial_cap(fake_automl):
    """A caller asking for 5 s must not silently be capped at 30 trials."""
    flaml_backend.train_flaml(
        _df(), target="value", covariates=["x"], model_config={"time_budget": 5}
    )
    kw = fake_automl["automl"].fit_kwargs
    assert kw["time_budget"] == 5
    assert "max_iter" not in kw


def test_explicit_max_iter_keeps_the_clock_disabled(fake_automl):
    flaml_backend.train_flaml(_df(), target="value", covariates=["x"], model_config={"max_iter": 7})
    kw = fake_automl["automl"].fit_kwargs
    assert kw["max_iter"] == 7
    assert kw["time_budget"] == -1


def test_both_bounds_pass_through(fake_automl):
    flaml_backend.train_flaml(
        _df(),
        target="value",
        covariates=["x"],
        model_config={"time_budget": 5, "max_iter": 7},
    )
    kw = fake_automl["automl"].fit_kwargs
    assert kw["max_iter"] == 7
    assert kw["time_budget"] == 5
