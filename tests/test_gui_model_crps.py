"""The Model tab's probabilistic scores: the point forecast's CRPS reference
line -- the helper pure, and the Model-tab renderer with the model stood in
for. The Chronos-2 quantile backtest that rides the same scale is tested in
``test_gui_chronos_crps.py`` (Chronos-2 is not part of the R window)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from normet.gui._run_train import crps_reference  # noqa: E402


def test_crps_reference_is_the_normal_scaled_mae():
    rng = np.random.default_rng(3)
    obs = rng.normal(10.0, 2.0, 400)
    pred = obs + rng.normal(0.0, 0.5, 400)
    ref = crps_reference(obs, pred)
    mae = float(np.mean(np.abs(pred - obs)))
    assert ref is not None and ref == pytest.approx(np.sqrt(np.pi / 2.0) * mae, rel=1e-9)


def test_crps_reference_survives_an_empty_holdout():
    assert crps_reference(np.array([]), np.array([])) is None


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


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


def test_the_model_tab_shows_the_point_reference(qapp, monkeypatch):
    import normet
    from normet.gui.main_window import MainWindow

    rng = np.random.default_rng(1)
    n = 240
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    value = rng.normal(30.0, 5.0, n)
    df_prep = pd.DataFrame(
        {"date": idx, "value": value, "set": ["training"] * (n // 2) + ["testing"] * (n - n // 2)}
    )
    monkeypatch.setattr(normet, "ml_predict", lambda model, d: d["value"] + 0.7)
    monkeypatch.setattr(
        normet,
        "modStats",
        lambda df_prep, model: pd.DataFrame({"set": ["testing"], "R2": [0.8]}),
    )
    w = MainWindow()
    try:
        w.df_prep = df_prep
        w.model = object()
        w.trained_features = ["t2m"]
        w._render_model_tab()
        assert "CRPS reference" in _tab_words(w.tab_model)
    finally:
        w.close()
