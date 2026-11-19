"""The SCM Studio's fit checks: the pre-trend test, the conformal interval, the
leave-one-donor-out weight drift and the event-study figure, the Abadie RMSPE-ratio
line on the Placebo Space tab, and the window end to end offscreen (the runner,
the Fit Checks tab, the light)."""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from normet.gui import scm_window as W  # noqa: E402


def _synth(n: int = 90, start: str = "2024-01-01", freq: str = "D") -> dict:
    """A finished fit as the window keeps it: a date-indexed synthetic table."""
    idx = pd.date_range(start, periods=n, freq=freq)
    rng = np.random.default_rng(0)
    observed = rng.normal(0, 0.3, n)
    synthetic = observed + rng.normal(0, 0.1, n)
    return {
        "synthetic": pd.DataFrame(
            {"observed": observed, "synthetic": synthetic, "effect": observed - synthetic},
            index=idx,
        ),
        "weights": None,
    }


# -- the verdict banners, pure ------------------------------------------------------


def test_pre_trend_banner_keeps_the_library_traffic_light():
    ok, text = W.pre_trend_banner("ok: no significant pre-trend (p=0.4, drift=0.9)")
    assert ok == "ok" and text.startswith("ok:")
    assert W.pre_trend_banner("caution: borderline pre-trend (p=0.04, drift=1.1)")[0] == "warn"
    assert W.pre_trend_banner("fail: significant pre-trend (p=0.001, drift=1.4)")[0] == "error"


def test_conformal_banner_reads_the_p_value():
    out = {"att": -1.0, "low": -1.6, "high": -0.4, "p_value": 0.02, "n_post": 30, "n_perm": 1000}
    level, text = W.conformal_banner(out)
    assert level == "ok" and "significant" in text and "95 % conformal interval" in text
    out["p_value"] = 0.4
    assert W.conformal_banner(out)[0] == "warn"


def _loo_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dropped_donor": ["a", "b", "c"],
            "mean_abs_drift": [0.01, 0.02, 0.09],
            "max_abs_drift": [0.02, 0.03, 0.18],
            "effect_shift": [0.001, -0.002, 0.03],
        }
    )


def test_loo_banner_reports_the_worst_donor():
    level, text = W.loo_banner(_loo_table())
    assert level == "warn" and "c" in text
    small = _loo_table().assign(mean_abs_drift=0.01, max_abs_drift=0.02)
    assert W.loo_banner(small)[0] == "ok"
    empty = pd.DataFrame(
        columns=["dropped_donor", "mean_abs_drift", "max_abs_drift", "effect_shift"]
    )
    level, text = W.loo_banner(empty)
    assert level == "warn" and "three donors" in text


def test_pre_trend_of_reads_only_the_fitted_synthetic():
    pre = W._pre_trend_of(_synth(), "2024-04-01")
    assert pre is not None and pre.n_periods > 0
    bad = {"synthetic": pd.DataFrame({"observed": [1.0], "synthetic": [1.0]})}
    assert W._pre_trend_of(bad, "2024-04-01") is None


# -- the window, end to end, offscreen ---------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _wait(qapp, until, timeout=240.0) -> None:
    t0 = time.time()
    while not until():
        qapp.processEvents()
        if time.time() - t0 > timeout:
            raise TimeoutError("the window did not reach what the test waits for")
        time.sleep(0.01)


def _click(qapp, win, btn) -> None:
    """Click a run button once the runner is idle again: while a task is still
    settling, its buttons are disabled and a click on one would be a no-op."""
    _wait(qapp, lambda: not win.runner.busy and btn.isEnabled())
    btn.click()


@pytest.fixture
def win(qapp):
    from normet.gui.scm_window import SCMWindow

    w = SCMWindow()
    yield w
    w.close()


def _tab_words(tab) -> str:
    from PySide6.QtWidgets import QLabel, QTableWidget

    parts = [w.text() for w in tab.findChildren(QLabel)]
    for t in tab.findChildren(QTableWidget):
        for c in range(t.columnCount()):
            head = t.horizontalHeaderItem(c)
            if head is not None:
                parts.append(head.text())
        for r in range(t.rowCount()):
            for c in range(t.columnCount()):
                item = t.item(r, c)
                if item is not None:
                    parts.append(item.text())
    return "\n".join(parts)


def test_the_fit_checks_run_on_the_example(win, qapp):
    win.load_example()
    assert win.df is not None
    # Gated until there is a fit: pre-trend and conformal read self.result;
    # the leave-one-donor-out refits the classic SCM itself.
    assert not win.pretrend_btn.isEnabled()
    assert not win.conf_btn.isEnabled()
    assert win.loo_btn.isEnabled()

    win.fit_btn.click()
    # The fit sets self.result, then the runner's finish signal re-syncs the
    # buttons: wait for both, not just the result.
    _wait(qapp, lambda: win.result is not None and win.pretrend_btn.isEnabled())
    assert win.conf_btn.isEnabled()
    assert "Pre-trend slope (per step)" in _tab_words(win.tab_fit)

    _click(qapp, win, win.pretrend_btn)
    _wait(qapp, lambda: "Pre-period slope (per step)" in _tab_words(win.tab_checks))
    assert win.tabs.currentIndex() == win.tabs.indexOf(win.tab_checks)

    win.conf_perm.setValue(100)
    _click(qapp, win, win.conf_btn)
    _wait(qapp, lambda: "95 % conformal interval" in _tab_words(win.tab_checks))

    _click(qapp, win, win.loo_btn)
    _wait(qapp, lambda: "Donor left out" in _tab_words(win.tab_checks), timeout=600)


def test_the_placebo_space_tab_carries_the_rmspe_ratio(win, qapp):
    win.load_example()
    win.placebo_space_btn.click()
    _wait(qapp, lambda: "RMSPE-ratio test" in _tab_words(win.tab_pspace), timeout=600)
