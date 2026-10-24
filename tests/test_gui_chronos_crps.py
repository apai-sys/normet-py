"""The Chronos-2 Model-tab quantile backtest: CRPS of the band against the
median's point reference, pinball per level -- helpers pure and the tab wired,
with the estimator stood in for. Chronos-2 is not part of the R window, so the
R vendor skips this file."""

from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from normet.gui._run_train import quantile_backtest  # noqa: E402


def _record(n: int = 300, seed: int = 5) -> pd.DataFrame:
    """An hourly record the stub estimator can forecast: a smooth level with
    noise the band is sized against."""
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-06-01", periods=n, freq="h")
    value = 20.0 + 3.0 * np.sin(np.arange(n) / 12.0) + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"date": idx, "value": value})


class _QuantileEstimator:
    """Stands in for Chronos2Estimator: a persistence median with a band of a
    fixed width around it -- near-calibrated against the record's step scale,
    so the CRPS beats the median's point reference."""

    model_name = "amazon/chronos-2"
    device = "cpu"
    context_length = 48
    prediction_length = 24

    def predict_quantiles(self, df, target, anchor, horizon=None, quantiles=None, **kw):
        past = df[df.index < anchor][target]
        level = float(past.iloc[-1])
        idx = df.loc[df.index >= anchor, target].index[: (horizon or self.prediction_length)]
        return pd.DataFrame(
            {f"q{q}": level + width for q, width in zip(quantiles, (-1.5, 0.0, 1.5), strict=True)},
            index=idx,
        )


def test_quantile_backtest_scores_the_last_horizon():
    rows, line = quantile_backtest(_QuantileEstimator(), _record())
    assert rows is not None and line is not None
    labels = [r[0] for r in rows]
    assert labels == [
        "Quantile backtest — CRPS",
        "Quantile backtest — point reference",
        "Pinball loss, q0.1",
        "Pinball loss, q0.5",
        "Pinball loss, q0.9",
    ]
    crps, ref = float(rows[0][1]), float(rows[1][1])
    assert 0.0 < crps < ref  # the band earns its keep on this record
    assert "earns its keep" in line


def test_quantile_backtest_is_none_when_the_record_is_too_short():
    est = _QuantileEstimator()
    assert quantile_backtest(est, _record(n=est.context_length + est.prediction_length)) is None


def test_quantile_backtest_swallows_a_failing_forecast():
    class Broken(_QuantileEstimator):
        def predict_quantiles(self, *a, **kw):
            raise RuntimeError("no weights")

    assert quantile_backtest(Broken(), _record()) is None


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


def test_the_chronos_model_tab_shows_the_backtest(qapp):
    from normet.gui.main_window import MainWindow

    w = MainWindow()
    try:
        w.load_example()
        w.backend_combo.setCurrentText("chronos-2")
        w.df_prep = w.df_raw.rename(columns={"PM2.5": "value"})
        w.model = _QuantileEstimator()
        w.trained_features = ["t2m", "blh", "u10", "v10"]
        w._render_chronos_model_tab()
        words = _tab_words(w.tab_model)
        assert "Quantile backtest — CRPS" in words
        assert "Pinball loss, q0.5" in words
        assert "earns its keep" in words
    finally:
        w.close()
