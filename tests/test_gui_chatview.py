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


def test_the_conversation_as_text_and_its_menu(qapp, fresh_zoom):
    from PySide6.QtWidgets import QApplication

    from normet.gui.chatview import ChatView

    v = ChatView()
    v.resize(400, 600)
    v.you("Which predictors?\nAnd why?")
    v.reply("Use **ws**.")
    bad = v.tool("train", {})
    v.tool_done(bad, "No data", True)
    v.note("A note")
    btn = v.card("Proposed set-up", "Target **NO2**", lambda: "Applied", action="Apply and train")
    assert btn.text() == "Apply and train"
    assert v.toPlainText() == (
        "› Which predictors?\n  And why?\n\n"
        "Assistant: Use **ws**.\n\n"
        "● train()\n  ⎿ No data\n\n"
        "A note\n\n"
        "[Proposed set-up]\nTarget **NO2**\n[Apply and train]"
    )
    reply = [w for w in v.items() if type(w).__name__ == "_Reply"][0]
    menu = v.menu_for(reply.label)
    names = [a.text() for a in menu.actions() if a.text()]
    assert names[:3] == ["Copy text", "Copy message", "Copy conversation"]
    assert "Find…" in names and "Larger text" in names
    dict((a.text(), a) for a in menu.actions())["Copy text"].trigger()
    assert QApplication.clipboard().text() == "Use **ws**."
    dict((a.text(), a) for a in v.menu_for(None).actions())["Copy conversation"].trigger()
    assert QApplication.clipboard().text() == v.toPlainText()
    reply.label.setSelection(0, 3)
    assert [a.text() for a in v.menu_for(reply.label).actions()][0] == "Copy"


def test_find_steps_through_the_conversation(qapp, fresh_zoom):
    from normet.gui.chatview import ChatView

    v = ChatView()
    v.resize(400, 300)
    v.show()
    v.you("Is wind in it?")
    v.reply("Wind speed, and **wind** direction.")
    v.open_find()
    assert v.find_bar.isVisible()
    v.find_bar.edit.setText("wind")
    assert len(v._hits) == 3 and v.find_bar.count.text() == "1 of 3"
    v.find_step(1)
    assert v.find_bar.count.text() == "2 of 3" and v._shown.hasSelectedText()
    v.find_step(-2)
    assert v.find_bar.count.text() == "3 of 3"  # round from the start to the end
    v.find_bar.edit.setText("rain")
    assert v.find_bar.count.text() == "No matches"
    v.close_find()
    assert not v.find_bar.isVisible() and v._hits == []
    v.close()


def test_the_view_keeps_the_end_until_scrolled_up(qapp, fresh_zoom):
    from normet.gui.chatview import ChatView

    v = ChatView()
    v.resize(300, 200)
    v.show()
    for i in range(30):
        v.you(f"Line {i} of a long conversation.")

    def settle():
        for _ in range(10):
            qapp.sendPostedEvents()
            qapp.processEvents()

    settle()
    bar = v.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == bar.maximum() and v.to_end.isHidden()
    bar.setValue(0)  # the analyst scrolls up to read
    v.reply("Something new.")
    settle()
    assert bar.value() == 0 and not v.to_end.isHidden()
    v.to_end.click()
    settle()
    assert bar.value() == bar.maximum() and v.to_end.isHidden()
    v.close()


def test_the_view_follows_the_system_appearance(qapp, fresh_zoom):
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QApplication

    from normet.gui.chatview import ChatView

    v = ChatView()
    old = QApplication.palette()
    try:
        dark = QPalette(old)
        dark.setColor(QPalette.ColorRole.Base, QColor("#1e1e1e"))
        dark.setColor(QPalette.ColorRole.Text, QColor("#f2f2f7"))
        QApplication.setPalette(dark)
        v.follow_palette()
        assert v.look.dark
    finally:
        QApplication.setPalette(old)
        v.follow_palette()
    assert v.look.dark == (old.color(QPalette.ColorRole.Base).lightnessF() < 0.5)


def test_tex_in_dollars_reads_as_text(qapp):
    from normet.gui.chatview import Look, _math, markdown_html

    assert _math(r"Q/Q_{\text{exp}} \to 1") == "Q/Q<sub>exp</sub> → 1"
    assert _math(r"x^2 \geq \alpha") == "x<sup>2</sup> ≥ α"
    out = markdown_html("Costs $5 and $10, then $R^2 \\approx 0.9$; `a $b$ c` stays.", Look())
    assert "$5 and $10" in out and "R<sup>2</sup> ≈ 0.9" in out and "NORMETMATH" not in out
    assert "$b$" in out  # a dollar in code is left alone


def test_a_long_word_is_soft_broken_and_copies_whole(qapp, fresh_zoom):
    from PySide6.QtWidgets import QApplication

    from normet.gui.chatview import ZWSP, ChatView, _runs, _soften

    word = "x" * 100
    assert _runs("short words only") == "short words only"
    assert _runs(word).replace(ZWSP, "") == word and ZWSP in _runs(word)
    assert "&amp;" in _soften(f"<p>{word} &amp; more</p>", True)  # an entity kept whole
    v = ChatView()
    v.resize(300, 300)
    v.show()
    v.reply(f"see /data/{word}/end here")
    reply = [w for w in v.items() if type(w).__name__ == "_Reply"][0]
    assert ZWSP in reply.label.text()
    v.open_find()
    v.find_bar.edit.setText("/data/" + "x" * 30)  # a needle spanning a soft break
    assert len(v._hits) == 1
    lbl, at, n = v._hits[0]
    lbl.setSelection(at, n)
    menu = v.menu_for(lbl)
    dict((a.text(), a) for a in menu.actions())["Copy"].trigger()
    assert QApplication.clipboard().text() == "/data/" + "x" * 30
    assert v.toPlainText().endswith(f"see /data/{word}/end here")  # the transcript is whole
    v.close()


def test_a_table_scrolls_sideways_on_its_own(qapp, fresh_zoom):
    from normet.gui.chatview import ChatView, _Wide

    v = ChatView()
    v.resize(260, 400)
    v.show()
    row = " | ".join(f"column{i}" for i in range(12))
    v.reply(f"Before\n\n| {row} |\n|{'---|' * 12}\n| {row} |\n\nAfter")
    reply = [w for w in v.items() if type(w).__name__ == "_Reply"][0]
    for _ in range(10):
        qapp.sendPostedEvents()
        qapp.processEvents()
    wide = reply.findChildren(_Wide)
    assert len(wide) == 1 and wide[0].label.width() > 260  # wider than the panel
    assert wide[0].horizontalScrollBar().maximum() > 0
    assert len(reply._widgets) == 3  # text, table, text
    # streaming more words leaves the blocks alone
    before = list(reply._widgets)
    v.reply("!", more=True)
    assert reply._widgets == before
    v.close()


def test_notes_cards_and_the_working_line(qapp, fresh_zoom):
    from normet.gui.chatview import ChatView, Working

    v = ChatView()
    v.resize(400, 500)
    v.note("fine")
    v.note("It broke", error=True)
    names = [w.objectName() for w in v.items()]
    assert names == ["", "trouble"]
    assert v.toPlainText() == "fine\n\nIt broke"
    btn = v.card("Proposed set-up", "Target", lambda: "Done", kind="setup")
    card = v.items()[-1]
    assert card.tag.text() == "Set-up" and card.tag.isVisibleTo(card)
    assert not card.done
    v.card_done(btn)
    assert card.done and btn.text() == "Applied" and not btn.isEnabled()
    plain = v.card("Run in the window", "x", lambda: "Ran", "Ran", "Run")
    assert plain.text() == "Ran" and not plain.isEnabled()
    assert v.items()[-1].tag.isHidden()

    w = v.working
    assert isinstance(w, Working) and w.isHidden()
    w.say("Reading the data")
    assert w.timer.isActive() and "Reading the data" in w.label.text()
    first = w.spin.text()
    for _ in range(3):
        w._tick()
    assert w.spin.text() != first or len(w.FRAMES) == 1
    v.clear()
    assert not w.timer.isActive()


def test_a_call_line_is_one_painted_line(qapp, fresh_zoom):
    from normet.gui.chatview import ChatView, _CallLine

    v = ChatView()
    v.resize(300, 300)
    item = v.tool("inspect_data", {"target": "NO2"})
    line = item.line
    assert isinstance(line, _CallLine)
    assert line.name == "inspect_data" and line.rest == "(target='NO2')"
    v.show()
    line.grab()  # paints without error
    v.zoom_by(1)
    v.zoom_by(0)
    v.close()
