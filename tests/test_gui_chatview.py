"""The Assistant's conversation (normet.gui.chatview), offscreen."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def fresh_zoom():
    from PySide6.QtCore import QSettings

    from normet.gui.chatview import ZOOM_KEY

    prefs = QSettings("apai-sys", "normet-gui")
    prefs.remove(ZOOM_KEY)
    yield
    prefs.remove(ZOOM_KEY)


def test_a_dark_palette_keeps_the_bubble_readable(qapp):
    from PySide6.QtGui import QColor, QPalette

    from normet.gui.chatview import Look, contrast

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Base, QColor("#1e1e1e"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#f2f2f7"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#d0d0d0"))  # macOS's, light
    k = Look(palette=pal)
    assert k.dark and contrast(k.text, k.bubble) > 7  # mixed from the palette, not AlternateBase


def test_markdown_reads_at_the_texts_size(qapp):
    from normet.gui.chatview import Look, markdown_html

    k = Look()
    out = markdown_html(
        "# Title\n\nSome `code` and\n\n```\nx = 1\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n", k
    )
    assert "<h1" not in out and "Title" in out  # a heading an ordinary paragraph
    assert k.code.name() in out  # code shaded
    assert k.border.name() in out  # a table ruled


def test_the_conversation(qapp, fresh_zoom):
    from PySide6.QtWidgets import QPushButton

    from normet.gui.chatview import ChatView

    v = ChatView()
    v.resize(400, 600)
    v.you("Which predictors?")
    v.reply("Use **ws** and")
    v.reply(" **wd**.", more=True)
    item = v.tool("inspect_data", {"target": "NO2"})
    assert item.dot.property("tone") == "faint"  # running
    v.tool_done(item, "5,793 rows", False)
    assert item.dot.property("tone") == "ok" and item.chevron.text() == "›"
    item.toggle()
    assert item.opened and "5,793 rows" in item.detail.text()
    bad = v.tool("train", {})
    v.tool_done(bad, "No data", True)
    assert bad.dot.property("tone") == "error" and not bad.detail.isHidden()
    v.note("A note")
    btn = v.card("Proposal", "Target **NO2**", lambda: "Applied")
    assert isinstance(btn, QPushButton) and btn.text() == "Apply"
    btn.click()
    assert btn.text() == "Applied" and not btn.isEnabled()
    reply = [w for w in v.items() if type(w).__name__ == "_Reply"][0]
    assert reply.md == "Use **ws** and **wd**."

    size = v.look.size
    v.zoom_by(1)
    assert v.look.size == size + 1
    w = ChatView()
    assert w.zoom == 1.0  # kept
    v.zoom_by(0)
    assert v.look.size == size
    v.clear()
    assert v.items() == []
