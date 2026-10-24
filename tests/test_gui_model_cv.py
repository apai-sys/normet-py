"""The Model tab's walk-forward CV view: the fold table and the verdict that
reads the folds against the random split (helpers pure), the CV run wired to
the library, and the tab's view selector between the split and the folds
(offscreen; the training itself is the library's own tests)."""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from normet.gui._run_train import cv_figure, cv_table, cv_verdict  # noqa: E402


def _folds(median: float = 0.73, holdout_dates: bool = True) -> pd.DataFrame:
    """A five-fold result as :func:`normet.cv_score` returns it: metadata
    first, then the statistics modStats is asked for."""
    rng = np.random.default_rng(7)
    k = 5
    r2 = median + rng.normal(0.0, 0.02, k)
    starts = pd.date_range("2024-06-01", periods=k, freq="7D")
    return pd.DataFrame(
        {
            "fold": list(range(k)),
            "train_start": starts - pd.Timedelta(days=60),
            "train_end": starts - pd.Timedelta(days=1),
            "test_start": starts if holdout_dates else starts.astype(str),
            "test_end": starts + pd.Timedelta(days=5),
            "n_train": [500 + 40 * i for i in range(k)],
            "n_test": [40] * k,
            "RMSE": [8.0, 8.2, 7.8, 8.1, 7.9],
            "MB": [-0.4, 0.2, -0.1, 0.3, -0.2],
            "R2": list(r2),
        }
    )


def test_cv_table_names_and_formats_the_folds():
    header, rows = cv_table(_folds())
    assert header == ["Fold", "Test starts", "Test ends", "Rows", "R2", "RMSE", "MB"]
    assert len(rows) == 5
    assert rows[0][0] == "0" and rows[0][3] == "40"
    assert rows[0][1].startswith("2024-06-01")
    assert float(rows[0][4]) == pytest.approx(_folds()["R2"].iloc[0], abs=5e-4)


def test_cv_table_shows_r_strings_and_missing_columns():
    folds = _folds(holdout_dates=False).drop(columns=["MB"])
    _header, rows = cv_table(folds)
    assert rows[0][1] == "2024-06-01"  # R returns dates as strings
    assert rows[0][6] == "—"  # a score the backend did not fill


def test_cv_verdict_catches_a_flattering_split():
    verdict, lines = cv_verdict(_folds(median=0.73), holdout_r2=0.88)
    assert verdict[0] == "warn" and "flatters" in verdict[1]
    assert len(lines) == 1 and "5 folds" in lines[0]


def test_cv_verdict_agrees_with_the_split_and_scores_the_folds():
    verdict, _ = cv_verdict(_folds(median=0.75), holdout_r2=0.79)
    assert verdict[0] == "ok" and "holds" in verdict[1]
    verdict, _ = cv_verdict(_folds(median=0.60), holdout_r2=0.62)
    assert verdict[0] == "warn" and "more features" in verdict[1]
    verdict, _ = cv_verdict(_folds(median=0.30), holdout_r2=0.31)
    assert verdict[0] == "error"
    verdict, _ = cv_verdict(_folds(median=0.75), holdout_r2=None)
    assert verdict[0] == "ok"  # no split to compare, the folds stand alone


def test_cv_verdict_without_a_scorable_fold():
    verdict, lines = cv_verdict(pd.DataFrame({"fold": [0]}), holdout_r2=0.9)
    assert verdict[0] == "warn" and "no scorable folds" in verdict[1]
    assert lines == []


def test_cv_figure_draws_bars_and_reference_lines():
    fig = cv_figure(_folds(), holdout_r2=0.88)
    ax = fig.axes[0]
    assert len(ax.containers) == 1 and len(ax.containers[0].patches) == 5
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert "CV median" in labels and "Random split" in labels
    assert "folds" in ax.get_title()


# -- the window, offscreen ----------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _wait(qapp, runner, timeout=60):
    t0 = time.time()
    while runner.busy and time.time() - t0 < timeout:
        qapp.processEvents()
        time.sleep(0.01)
    for _ in range(20):
        qapp.processEvents()


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


def _model_win(monkeypatch):
    """A window with data, a model and the Model tab rendered — everything
    except training, which the library's own tests cover."""
    import normet
    from normet.gui.main_window import MainWindow

    rng = np.random.default_rng(1)
    n = 480
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    value = rng.normal(30.0, 5.0, n)
    df_prep = pd.DataFrame(
        {"date": idx, "value": value, "set": ["training"] * (n // 2) + ["testing"] * (n - n // 2)}
    )
    monkeypatch.setattr(normet, "ml_predict", lambda model, d: d["value"] + 0.7)
    monkeypatch.setattr(
        normet, "modStats", lambda df_prep, model: pd.DataFrame({"set": ["testing"], "R2": [0.8]})
    )
    w = MainWindow()
    w.load_example()
    w.df_prep = df_prep
    w.model = object()
    w.trained_features = ["t2m"]
    w._render_model_tab()
    return w


def test_the_model_tab_holds_one_view_after_training(qapp, monkeypatch):
    from PySide6.QtWidgets import QComboBox

    w = _model_win(monkeypatch)
    try:
        assert "CRPS reference" in _tab_words(w.tab_model)
        assert [c for c in w.tab_model.findChildren(QComboBox) if c.count() > 1] == []
    finally:
        w.close()


def test_cv_done_adds_a_second_view_and_the_tab_switches(qapp, monkeypatch):
    from PySide6.QtWidgets import QComboBox

    w = _model_win(monkeypatch)
    try:
        w._cv_done(_folds(median=0.73))
        words = _tab_words(w.tab_model)
        assert "Fold" in words and "Test starts" in words
        assert "flatters" in words  # the stub's holdout R² is ~0.98 against folds at 0.73
        combos = [c for c in w.tab_model.findChildren(QComboBox) if c.count() > 1]
        assert len(combos) == 1
        assert set(combos[0].itemText(i) for i in range(combos[0].count())) == {
            "Holdout",
            "Walk-forward CV",
        }
        combos[0].setCurrentText("Holdout")
        for _ in range(10):
            qapp.processEvents()
        assert "CRPS reference" in _tab_words(w.tab_model)
        assert "Fold" not in _tab_words(w.tab_model)
    finally:
        w.close()


def test_run_cv_scores_folds_and_shows_them(qapp, monkeypatch):
    import normet
    from normet.gui.main_window import MainWindow

    captured: dict = {}

    def fake_cv_score(df, target, **kw):
        captured["args"] = (len(df), target, kw.get("n_splits"), kw.get("covariates"))
        return _folds(median=0.73)

    monkeypatch.setattr(normet, "cv_score", fake_cv_score)
    w = MainWindow()
    try:
        w.load_example()
        w.df_prep = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=300, freq="h")})
        w.df_prep["value"] = np.random.default_rng(2).normal(30.0, 5.0, 300)
        w.df_prep["set"] = "training"
        w.model = object()
        w.trained_features = ["t2m"]
        w.cv_folds_spin.setValue(5)
        w._run_cv()
        _wait(qapp, w.runner)
        assert captured["args"][1] == "value" and captured["args"][2] == 5
        assert "t2m" in captured["args"][3]
        assert "Fold" in _tab_words(w.tab_model)
        assert "flatters" in _tab_words(w.tab_model) or "holds" in _tab_words(w.tab_model)
    finally:
        w.close()


def test_run_cv_needs_a_model_and_chronos_has_its_own_check(qapp, monkeypatch):
    w = _model_win(monkeypatch)
    try:
        w.cv_button.setEnabled(True)  # the test asks the handler, not the button
        w.model = None
        w._run_cv()  # no model: nothing runs, nothing raises
        if hasattr(w, "_is_chronos"):  # the R window has neither chronos nor this branch
            w.backend_combo.setCurrentText("chronos-2")
            w.model = object()
            w._sync_enabled()
            assert not w.cv_button.isEnabled()
            assert "chronos-2" in w.cv_button.toolTip()
    finally:
        w.close()


def test_cv_done_with_no_folds_leaves_the_view_alone(qapp, monkeypatch):
    w = _model_win(monkeypatch)
    try:
        w._cv_done(pd.DataFrame())
        assert "CRPS reference" in _tab_words(w.tab_model)  # still the holdout
    finally:
        w.close()
