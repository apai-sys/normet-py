"""The Assistant's conversation, as the panel sets it out.

The analyst's messages sit in a rounded bubble on the right, as wide as their words; the
replies read as their Markdown does, in the window's own font and size -- headings bold at
the text's size, not a browser's sizes, lines a little apart, code shaded and wrapped, a
table ruled; each tool call is one quiet line: a dot, grey while it runs, green when it
answered, red when it was refused (the refusal under it), what it returned a click away;
a proposal is a card with an Apply button.

The colours are mixed from the application's palette -- the text's into the ground's --
and made again when it changes, light or dark: the dark palette's AlternateBase, which the
analyst's messages were shaded with, is a light grey under white text. The text size steps
with Cmd/Ctrl + and - (0 back to the system's; the wheel with the key held), and is kept.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QEvent, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

#: Where the conversation's text size is kept (points from the system's size).
ZOOM_KEY = "assistant/zoom"


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    """``b`` mixed into ``a`` by the fraction ``t``."""
    return QColor.fromRgbF(
        a.redF() + (b.redF() - a.redF()) * t,
        a.greenF() + (b.greenF() - a.greenF()) * t,
        a.blueF() + (b.blueF() - a.blueF()) * t,
    )


def contrast(a: QColor, b: QColor) -> float:
    """WCAG's contrast ratio of two colours: 1 (none) to 21 (black on white)."""

    def lum(c: QColor) -> float:
        ch = [
            v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
            for v in (c.redF(), c.greenF(), c.blueF())
        ]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


class Look:
    """The conversation's colours and sizes, from the application's palette; the text
    ``zoom`` points from the system's size."""

    def __init__(self, zoom: float = 0.0, palette: QPalette | None = None) -> None:
        pal = palette or QApplication.palette()
        role = QPalette.ColorRole
        ground, ink = pal.color(role.Base), pal.color(role.Text)
        self.dark = dark = ground.lightnessF() < 0.5
        self.ground = ground
        self.text = _mix(ink, ground, 0.08)
        self.muted = _mix(ink, ground, 0.42)
        self.faint = _mix(ink, ground, 0.62)
        self.bubble = _mix(ground, ink, 0.12 if dark else 0.055)
        self.card = _mix(ground, ink, 0.035 if dark else 0.0)
        self.border = _mix(ground, ink, 0.20 if dark else 0.13)
        self.code = _mix(ground, ink, 0.09 if dark else 0.045)
        self.hover = _mix(ground, ink, 0.06 if dark else 0.035)
        self.accent = pal.color(role.Accent) if hasattr(role, "Accent") else pal.color(role.Link)
        self.on_accent = QColor("#ffffff" if self.accent.lightnessF() < 0.62 else "#000000")
        self.ok = QColor("#4cc38a" if dark else "#2f9e5a")
        self.err = QColor("#ff7b72" if dark else "#c62828")
        base = QApplication.font().pointSizeF()
        self.size = max(9.0, (base if base > 0 else 13.0) + zoom)
        self.small = max(8.0, self.size - 1.5)
        self.code_size = max(8.0, self.size - 2.0)  # a fixed-width face reads larger
        self.mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()
        #: What it was made from: a palette event with the same is no change.
        self.key = (ground.rgb(), ink.rgb(), self.size)

    def font(self, size: float | None = None, *, mono: bool = False, bold: bool = False) -> QFont:
        f = (
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
            if mono
            else QFont(QApplication.font())
        )
        f.setPointSizeF(size or self.size)
        if bold:
            f.setBold(True)
        return f


def sheet(k: Look) -> str:
    """The conversation's style sheet, from its Look."""
    c = QColor.name
    return f"""
QScrollArea#chatview {{ background: {c(k.ground)}; border: none; border-radius: 10px; }}
QScrollArea#chatview > QWidget#qt_scrollarea_viewport {{ background: transparent; }}
QWidget#chat {{ background: transparent; }}
QWidget#chat QLabel {{ color: {c(k.text)}; background: transparent; }}
QWidget#chat QLabel[tone="muted"] {{ color: {c(k.muted)}; }}
QWidget#chat QLabel[tone="faint"] {{ color: {c(k.faint)}; }}
QWidget#chat QLabel[tone="ok"] {{ color: {c(k.ok)}; }}
QWidget#chat QLabel[tone="error"] {{ color: {c(k.err)}; }}
QFrame#you {{ background: {c(k.bubble)}; border-radius: 12px; }}
QFrame#proposal {{ background: {c(k.card)}; border: 1px solid {c(k.border)};
                   border-radius: 10px; }}
QFrame#tool {{ border-radius: 6px; }}
QFrame#tool[open="true"] {{ background: {c(k.hover)}; }}
QFrame#tool[clickable="true"]:hover {{ background: {c(k.hover)}; }}
QPushButton#apply {{ background: {c(k.accent)}; color: {c(k.on_accent)}; border: none;
                     border-radius: 6px; padding: 4px 14px; font-weight: 600; }}
QPushButton#apply:disabled {{ background: {c(k.border)}; color: {c(k.muted)}; }}
"""


def composer_sheet(k: Look) -> str:
    """The box the analyst writes in, set to match: rounded, the buttons inside it."""
    c = QColor.name
    return f"""
QFrame#composer {{ background: {c(k.ground)}; border: 1px solid {c(k.border)};
                   border-radius: 10px; }}
QFrame#composer[focus="true"] {{ border: 1px solid {c(k.accent)}; }}
QPlainTextEdit#composer_input {{ background: transparent; border: none; color: {c(k.text)}; }}
QPushButton#composer_send {{ background: {c(k.accent)}; color: {c(k.on_accent)}; border: none;
                             border-radius: 6px; padding: 4px 14px; font-weight: 600; }}
QPushButton#composer_send:hover {{ background: {c(k.accent.lighter(112))}; }}
QPushButton#composer_send:disabled {{ background: {c(k.border)}; color: {c(k.muted)}; }}
QPushButton#composer_stop, QPushButton#composer_new {{
    background: transparent; color: {c(k.text)}; border: 1px solid {c(k.border)};
    border-radius: 6px; padding: 4px 12px; }}
QPushButton#composer_new:hover {{ background: {c(k.hover)}; }}
QPushButton#composer_stop:disabled, QPushButton#composer_new:disabled {{ color: {c(k.faint)}; }}
"""


def markdown_html(md: str, k: Look) -> str:
    """The model's Markdown as Qt's HTML, set for the panel: a heading at the body's own
    size (bold only), paragraphs and list items closer, lines a little apart, code shaded
    (a block's lines wrapped), a table ruled in the palette's line colour, its first row
    shaded."""
    from PySide6.QtGui import QTextDocument

    doc = QTextDocument()
    doc.setMarkdown(md)
    text = doc.toHtml()
    text = text[text.find("<body") :] if "<body" in text else text
    text = re.sub(
        r"font-size:(xx-large|x-large|large|medium|small|x-small);",
        f"font-size:{k.size:.1f}pt;",
        text,
    )
    # A heading an ordinary paragraph: Qt sizes <h1>-<h6> itself, whatever the span says.
    text = re.sub(r"<h[1-6](\s[^>]*)?>", r"<p\1>", text)
    text = re.sub(r"</h[1-6]>", "</p>", text)
    text = text.replace(
        "margin-top:12px; margin-bottom:12px;", "margin-top:3px; margin-bottom:6px;"
    )
    text = re.sub(r'<(p|li) style="', r'<\1 style=" line-height:130%;', text)
    code = k.code.name()

    def table(m: re.Match[str]) -> str:
        t = re.sub(
            r'<table border="1" style="([^"]*)" cellpadding="\d+"',
            lambda s: (
                f'<table border="1" style="{s.group(1)} border-color:{k.border.name()}; '
                'border-style:solid;" cellpadding="5"'
            ),
            m.group(0),
            count=1,
        )
        first = re.search(r"<tr>.*?</tr>", t, re.S)
        if first:
            t = (
                t[: first.start()]
                + first.group(0).replace("<td", f'<td bgcolor="{code}"')
                + t[first.end() :]
            )
        return t

    text = re.sub(r'<table border="1".*?</table>', table, text, flags=re.S)

    def block(m: re.Match[str]) -> str:
        lines = re.sub(
            r'<pre style="[^"]*"',
            '<pre style=" margin-top:0px; margin-bottom:0px; white-space:pre-wrap;"',
            m.group(0).strip(),
        )
        return (
            f'<table width="100%" cellspacing="0" cellpadding="7" bgcolor="{code}" '
            f'style=" margin-top:4px; margin-bottom:6px;"><tr><td>{lines}</td></tr></table>'
        )

    text = re.sub(r"(?:<pre[^>]*>.*?</pre>\s*)+", block, text, flags=re.S)
    return text.replace(
        f"font-family:'{k.mono}';",
        f"font-family:'{k.mono}'; font-size:{k.code_size:.1f}pt; background-color:{code};",
    )


def _tone(w: QWidget, tone: str, prop: str = "tone") -> None:
    w.setProperty(prop, tone)
    w.style().unpolish(w)
    w.style().polish(w)


def _label(text: str, k: Look, *, rich: bool, keep: bool, tone: str | None = None) -> QLabel:
    lbl = QLabel()
    # The analyst's and the model's own words (*keep*): left as written in any language
    # the window is in (.uilang). A note is the window's own.
    lbl.setProperty("normetKeepText", keep)
    lbl.setTextFormat(Qt.TextFormat.RichText if rich else Qt.TextFormat.PlainText)
    lbl.setWordWrap(True)
    policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    policy.setHeightForWidth(True)
    lbl.setSizePolicy(policy)
    lbl.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.LinksAccessibleByMouse
    )
    lbl.setOpenExternalLinks(True)
    if tone:
        lbl.setProperty("tone", tone)
    lbl.setFont(k.font())
    lbl.setText(text)
    return lbl


class _Reply(QWidget):
    """The model's words, as their Markdown reads; written again as they stream in."""

    def __init__(self, md: str, k: Look) -> None:
        super().__init__()
        self.md = md
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 0)
        self.label = _label("", k, rich=True, keep=True)
        lay.addWidget(self.label)
        self.restyle(k)

    def set_md(self, md: str, k: Look) -> None:
        self.md = md
        self.label.setText(markdown_html(md, k))

    def restyle(self, k: Look) -> None:
        self.label.setFont(k.font())
        self.label.setText(markdown_html(self.md, k))


class _You(QFrame):
    """The analyst's message: a bubble on the right, as wide as its words."""

    def __init__(self, text: str, k: Look, room: int) -> None:
        super().__init__()
        self.setObjectName("you")
        self.text = text
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        self.label = _label(text, k, rich=False, keep=True)
        lay.addWidget(self.label)
        self.fit(k, room)

    def fit(self, k: Look, room: int) -> None:
        self.label.setFont(k.font())
        most = max(200, int(room * 0.85))
        widest = max(
            self.label.fontMetrics().horizontalAdvance(line)
            for line in self.text.splitlines() or [""]
        )
        self.setFixedWidth(min(most, widest + 24))


class _Tool(QFrame):
    """One tool call: a dot -- grey while it runs, green when it answered, red when it was
    refused -- and the call on one line; a refusal under it; what it returned shown when
    the line is clicked."""

    def __init__(self, call: str, k: Look) -> None:
        super().__init__()
        self.setObjectName("tool")
        self.call, self.result, self.error, self.opened = call, "", False, False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(2)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(7)
        self.dot = QLabel("●")
        self.dot.setProperty("tone", "faint")
        self.line = _label(html.escape(call), k, rich=True, keep=True, tone="muted")
        self.chevron = QLabel("")
        self.chevron.setProperty("tone", "faint")
        row.addWidget(self.dot)
        row.addWidget(self.line, 1)
        row.addWidget(self.chevron)
        lay.addLayout(row)
        self.detail = _label("", k, rich=False, keep=True, tone="muted")
        self.detail.setContentsMargins(17, 0, 0, 2)
        self.detail.hide()
        lay.addWidget(self.detail)
        self.setToolTip(call)
        self.restyle(k)

    def restyle(self, k: Look) -> None:
        self.dot.setFont(k.font(k.small - 1))
        self.line.setFont(k.font(k.small))
        self.chevron.setFont(k.font(k.small))
        self.detail.setFont(k.font(k.small) if self.error else k.font(k.code_size, mono=True))

    def finish(self, text: str, error: bool) -> None:
        self.result, self.error = text, error
        _tone(self.dot, "error" if error else "ok")
        if error:
            _tone(self.detail, "error")
            self.detail.setText(text[:1200])
            self.detail.show()
        elif text.strip():
            self.chevron.setText("›")
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            _tone(self, "true", "clickable")

    def mousePressEvent(self, e: Any) -> None:  # noqa: N802
        if self.error or not self.result.strip():
            return super().mousePressEvent(e)
        self.toggle()

    def toggle(self) -> None:
        self.opened = not self.opened
        self.detail.setText(self.result[:4000] + ("…" if len(self.result) > 4000 else ""))
        self.detail.setVisible(self.opened)
        self.chevron.setText("⌄" if self.opened else "›")
        _tone(self, "true" if self.opened else "false", "open")


class ChatView(QScrollArea):
    """The conversation: the analyst's messages on the right, the replies as their
    Markdown reads, each tool call one quiet line, a proposal a card with Apply."""

    #: The Look, after the palette or the text size changed: the box written in follows.
    restyled = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("chatview")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        host.setObjectName("chat")
        self._lay = QVBoxLayout(host)
        self._lay.setContentsMargins(8, 8, 8, 8)
        self._lay.setSpacing(8)
        self._lay.addStretch(1)
        self.setWidget(host)
        self._reply: _Reply | None = None
        try:
            self.zoom = float(QSettings("apai-sys", "normet-gui").value(ZOOM_KEY, 0.0))
        except (TypeError, ValueError):
            self.zoom = 0.0
        self.look = Look(self.zoom)
        self.setStyleSheet(sheet(self.look))
        self.verticalScrollBar().rangeChanged.connect(
            lambda _lo, hi: self.verticalScrollBar().setValue(hi)
        )
        for keys, step in (
            (QKeySequence.StandardKey.ZoomIn, 1),
            ("Ctrl+=", 1),
            (QKeySequence.StandardKey.ZoomOut, -1),
            ("Ctrl+0", 0),
        ):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda s=step: self.zoom_by(s))

    # -- what is in it -----------------------------------------------------------------
    def _add(self, w: QWidget, align: Qt.AlignmentFlag | None = None) -> None:
        if align is None:
            self._lay.insertWidget(self._lay.count() - 1, w)
        else:
            self._lay.insertWidget(self._lay.count() - 1, w, 0, align)

    def items(self) -> list[QWidget]:
        out = []
        for i in range(self._lay.count()):
            w = self._lay.itemAt(i).widget()
            if w is not None:
                out.append(w)
        return out

    def clear(self) -> None:
        while self._lay.count() > 1:
            w = self._lay.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        self._reply = None

    def you(self, text: str) -> None:
        self._add(_You(text, self.look, self.viewport().width()), Qt.AlignmentFlag.AlignRight)
        self._reply = None

    def reply(self, text: str, more: bool = False) -> None:
        """The model's words: a new reply, or (*more*) added to the one being written."""
        if not text.strip():
            return
        if more and self._reply is not None:
            self._reply.set_md(self._reply.md + text, self.look)
            return
        self._reply = _Reply(text, self.look)
        self._add(self._reply)

    def tool(self, name: str, arguments: dict[str, Any]) -> _Tool:
        args = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        item = _Tool(f"{name}({args[:120]}{'…' if len(args) > 120 else ''})", self.look)
        self._add(item)
        self._reply = None
        return item

    @staticmethod
    def tool_done(item: _Tool, output: str, error: bool) -> None:
        item.finish(output, error)

    def note(self, text: str, error: bool = False) -> None:
        lbl = _label(text, self.look, rich=False, keep=False, tone="error" if error else "muted")
        lbl.setFont(self.look.font(self.look.small))
        self._add(lbl)
        self._reply = None

    def card(
        self, title: str, body: str, apply: Callable[[], str], applied: str = ""
    ) -> QPushButton:
        """A proposal with its Apply button; ``applied``, what applying it said where it
        was applied already (the Assistant's full automatic), on the button instead."""
        frame = QFrame()
        frame.setObjectName("proposal")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(12, 10, 12, 10)
        # The title is the window's own words (translated); the body is the model's.
        head = _label(html.escape(title), self.look, rich=True, keep=False)
        head.setFont(self.look.font(bold=True))
        lay.addWidget(head)
        body_item = _Reply(body, self.look)
        lay.addWidget(body_item)
        btn = QPushButton("Apply")
        btn.setObjectName("apply")

        def clicked() -> None:
            try:
                said = apply()
            except Exception as exc:  # noqa: BLE001 -- shown on the card
                said = f"Could not apply: {exc}"
            btn.setText(said)
            btn.setEnabled(False)

        btn.clicked.connect(clicked)
        if applied:
            btn.setText(applied)
            btn.setEnabled(False)
        lay.addWidget(btn, 0, Qt.AlignmentFlag.AlignLeft)
        self._add(frame)
        self._reply = None
        return btn

    # -- how it looks ------------------------------------------------------------------
    def zoom_by(self, step: int) -> None:
        """The text a point larger (1), smaller (-1) or the system's size (0); kept."""
        z = 0.0 if step == 0 else max(-4.0, min(8.0, self.zoom + step))
        if z == self.zoom:
            return
        self.zoom = z
        QSettings("apai-sys", "normet-gui").setValue(ZOOM_KEY, z)
        self.restyle()

    def restyle(self) -> None:
        """Colours and sizes made again: the palette changed (light, dark) or the zoom."""
        self.look = k = Look(self.zoom)
        self.setStyleSheet(sheet(k))
        room = self.viewport().width()
        for w in self.items():
            if isinstance(w, _You):
                w.fit(k, room)
            elif isinstance(w, (_Reply, _Tool)):
                w.restyle(k)
            elif isinstance(w, QFrame) and w.objectName() == "proposal":
                for r in w.findChildren(_Reply):
                    r.restyle(k)
            elif isinstance(w, QLabel):
                w.setFont(k.font(k.small))
        self.restyled.emit(k)

    def changeEvent(self, e: Any) -> None:  # noqa: N802
        super().changeEvent(e)
        # The application's palette changed (light, dark). A style sheet set here posts
        # a palette event too: the same colours are no change, or it would never end.
        if e.type() == QEvent.Type.PaletteChange and Look(self.zoom).key != self.look.key:
            self.restyle()

    def wheelEvent(self, e: Any) -> None:  # noqa: N802
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_by(1 if e.angleDelta().y() > 0 else -1)
            return
        super().wheelEvent(e)
