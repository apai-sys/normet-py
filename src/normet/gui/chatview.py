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
The view follows the system's appearance as it changes (``follow_palette``: its own style
sheet sets its palette, so the panel holding it passes the change on).

A view at the end keeps the end as the conversation grows; scrolled up, it stays where the
analyst is reading, and a button at its corner goes back to the end. Cmd/Ctrl+F finds text
in the conversation, Enter and Shift+Enter stepping through, Esc letting it go. The menu
(a right click) copies what is selected, the message as it was written (Copy text), as the
transcript keeps it (Copy message), or the conversation entire (Copy conversation).

``toPlainText()`` is the conversation as text, as Copy conversation copies it: '› ' before
the analyst's messages, 'Assistant:' before each reply, '● ' before each call and '⎿ '
before a refusal.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QEvent, QSettings, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QKeySequence,
    QPalette,
    QShortcut,
    QTextDocument,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
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
QToolButton#to_end {{ background: {c(k.card if k.dark else k.ground)}; color: {c(k.text)};
                      border: 1px solid {c(k.border)}; border-radius: 14px; }}
QToolButton#to_end:hover {{ background: {c(k.hover)}; }}
QFrame#find {{ background: {c(k.ground)}; border: 1px solid {c(k.border)};
               border-radius: 8px; }}
QFrame#find QLineEdit {{ background: transparent; border: none; color: {c(k.text)}; }}
QFrame#find QLabel {{ color: {c(k.muted)}; background: transparent; }}
QFrame#find QToolButton {{ background: transparent; border: none; color: {c(k.text)};
                           padding: 0px 4px; }}
QFrame#find QToolButton:hover {{ background: {c(k.hover)}; border-radius: 4px; }}
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

    def bare(self) -> str:
        return self.md

    def plain(self) -> str:
        return "Assistant: " + self.md


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

    def bare(self) -> str:
        return self.text

    def plain(self) -> str:
        return "› " + self.text.replace("\n", "\n  ")


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
        if self.error or not self.result.strip() or e.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(e)
        self.toggle()

    def plain(self) -> str:
        return "● " + self.call + ("\n  ⎿ " + self.result.strip()[:1200] if self.error else "")

    def toggle(self) -> None:
        self.opened = not self.opened
        self.detail.setText(self.result[:4000] + ("…" if len(self.result) > 4000 else ""))
        self.detail.setVisible(self.opened)
        self.chevron.setText("⌄" if self.opened else "›")
        _tone(self, "true" if self.opened else "false", "open")


class _Card(QFrame):
    """A proposal: its title, the model's reasons and the button that carries it out."""

    def __init__(
        self,
        title: str,
        body: str,
        apply: Callable[[], str],
        applied: str,
        action: str,
        k: Look,
    ) -> None:
        super().__init__()
        self.setObjectName("proposal")
        self.title = title
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        # The title is the window's own words (translated); the body is the model's.
        head = _label(html.escape(title), k, rich=True, keep=False)
        head.setFont(k.font(bold=True))
        lay.addWidget(head)
        self.body = _Reply(body, k)
        lay.addWidget(self.body)
        self.button = btn = QPushButton(action)
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

    def restyle(self, k: Look) -> None:
        self.body.restyle(k)

    def bare(self) -> str:
        return self.body.md

    def plain(self) -> str:
        return f"[{self.title}]\n{self.body.md}\n[{self.button.text()}]"


def _label_text(label: QLabel) -> str:
    """A label's text as it reads: its rich text's characters, without the markup."""
    if label.textFormat() != Qt.TextFormat.RichText:
        return label.text()
    doc = QTextDocument()
    doc.setHtml(label.text())
    return doc.toPlainText()


def _put_away(w: QWidget | None, host: QWidget) -> bool:
    """Whether *w*, or what holds it below *host*, was hidden (a call's result folded
    away) -- not merely not shown yet: a label just added is shown a moment later."""
    while w is not None and w is not host:
        if w.testAttribute(Qt.WidgetAttribute.WA_WState_ExplicitShowHide) and w.isHidden():
            return True
        w = w.parentWidget()
    return False


class _FindBar(QFrame):
    """Cmd/Ctrl+F: the text to find in the conversation, the hits stepped through."""

    def __init__(self, view: ChatView) -> None:
        super().__init__(view)
        self.setObjectName("find")
        self.view = view
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 2, 4, 2)
        lay.setSpacing(2)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Find in the conversation")
        self.edit.textChanged.connect(lambda _t: view.find_again(first=True))
        self.edit.installEventFilter(self)
        self.count = QLabel("")
        buttons = []
        for text, tip, step in (("↑", "Previous (Shift+Enter)", -1), ("↓", "Next (Enter)", 1)):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, s=step: view.find_step(s))
            buttons.append(b)
        close = QToolButton()
        close.setText("×")
        close.setToolTip("Close (Esc)")
        close.clicked.connect(view.close_find)
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.count)
        for b in (*buttons, close):
            lay.addWidget(b)
        self.hide()

    def eventFilter(self, obj: Any, e: QEvent) -> bool:  # noqa: N802
        if obj is self.edit and e.type() == QEvent.Type.KeyPress:
            key = e.key()  # type: ignore[attr-defined]
            if key == Qt.Key.Key_Escape:
                self.view.close_find()
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                shift = e.modifiers() & Qt.KeyboardModifier.ShiftModifier  # type: ignore[attr-defined]
                self.view.find_step(-1 if shift else 1)
                return True
        return super().eventFilter(obj, e)


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
        #: Whether the view keeps the end as the conversation grows: it does until the
        #: analyst scrolls up to read, and again once they are back at the end.
        self._follow = True
        bar = self.verticalScrollBar()
        bar.rangeChanged.connect(self._on_range)
        bar.valueChanged.connect(self._on_value)
        self.to_end = QToolButton(self)
        self.to_end.setObjectName("to_end")
        self.to_end.setText("↓")
        self.to_end.setToolTip("Back to the end of the conversation")
        self.to_end.setFixedSize(28, 28)
        self.to_end.setCursor(Qt.CursorShape.PointingHandCursor)
        self.to_end.clicked.connect(self.to_bottom)
        self.to_end.hide()
        self.find_bar = _FindBar(self)
        self._hits: list[tuple[QLabel, int, int]] = []
        self._hit = -1
        self._shown: QLabel | None = None
        find = QShortcut(QKeySequence(QKeySequence.StandardKey.Find), self)
        find.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        find.activated.connect(self.open_find)
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
        # Each label's menu is the conversation's (the copies, find), not a label's own.
        for lbl in [w] if isinstance(w, QLabel) else w.findChildren(QLabel):
            lbl.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            lbl.customContextMenuRequested.connect(
                lambda pos, lbl=lbl: self.menu_for(lbl).exec(lbl.mapToGlobal(pos))
            )

    def items(self) -> list[QWidget]:
        out = []
        for i in range(self._lay.count()):
            w = self._lay.itemAt(i).widget()
            if w is not None:
                out.append(w)
        return out

    def clear(self) -> None:
        self._deselect()
        self._hits, self._hit = [], -1
        while self._lay.count() > 1:
            w = self._lay.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        self._reply = None
        self._follow = True

    def you(self, text: str) -> None:
        self._follow = True  # the analyst's own message: the view goes to the end
        self._add(_You(text, self.look, self.viewport().width()), Qt.AlignmentFlag.AlignRight)
        self._reply = None

    @property
    def writing(self) -> bool:
        """Whether a reply is being written (words added go to it)."""
        return self._reply is not None

    def reply_live(self, md: str) -> None:
        """The reply being written, as it stands so far (a new one if none is)."""
        if self._reply is None:
            self.reply(md)
        else:
            self._reply.set_md(md, self.look)

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
        self,
        title: str,
        body: str,
        apply: Callable[[], str],
        applied: str = "",
        action: str = "Apply",
    ) -> QPushButton:
        """A proposal with its button (``action``, what clicking it does); ``applied``,
        what applying it said where it was applied already (the Assistant's full
        automatic), on the button instead."""
        card = _Card(title, body, apply, applied, action, self.look)
        self._add(card)
        self._reply = None
        return card.button

    # -- as text ---------------------------------------------------------------------
    def _entry(self, w: QWidget | None) -> QWidget | None:
        """The entry of the conversation *w* is in (a message, a call, a card, a note)."""
        host = self.widget()
        while w is not None and w.parentWidget() is not host:
            w = w.parentWidget()
        return w

    @staticmethod
    def _plain(w: QWidget) -> str:
        if hasattr(w, "plain"):
            return str(w.plain())  # type: ignore[attr-defined]
        return w.text() if isinstance(w, QLabel) else ""

    def toPlainText(self) -> str:  # noqa: N802 -- Qt's name for a widget's text
        """The conversation as text, as Copy conversation copies it."""
        return "\n\n".join(s for s in (self._plain(w) for w in self.items()) if s.strip())

    def menu_for(self, label: QLabel | None = None) -> QMenu:
        """The menu for a label of the conversation (or none of them): copy what is
        selected, the message as it was written, the message as the transcript keeps it,
        the conversation entire; then find, and the text's size."""
        menu = QMenu(self)
        copy = QApplication.clipboard().setText
        if label is not None and label.hasSelectedText():
            selected = label.selectedText().replace("\u2029", "\n").replace("\u2028", "\n")
            menu.addAction("Copy", lambda: copy(selected))
        entry = self._entry(label)
        if entry is not None:
            if hasattr(entry, "bare"):
                menu.addAction("Copy text", lambda: copy(entry.bare()))  # type: ignore[attr-defined]
            menu.addAction("Copy message", lambda: copy(self._plain(entry)))
        whole = menu.addAction("Copy conversation", lambda: copy(self.toPlainText()))
        whole.setEnabled(bool(self.items()))
        menu.addSeparator()
        find = menu.addAction("Find…", self.open_find)
        find.setShortcut(QKeySequence(QKeySequence.StandardKey.Find))
        menu.addSeparator()
        menu.addAction("Larger text", lambda: self.zoom_by(1))
        menu.addAction("Smaller text", lambda: self.zoom_by(-1))
        menu.addAction("Actual size", lambda: self.zoom_by(0))
        return menu

    def contextMenuEvent(self, e: Any) -> None:  # noqa: N802
        self.menu_for(None).exec(e.globalPos())

    # -- the end -----------------------------------------------------------------------
    def _on_range(self, _lo: int, hi: int) -> None:
        if self._follow:
            self.verticalScrollBar().setValue(hi)
        self._place_floaters()

    def _on_value(self, value: int) -> None:
        bar = self.verticalScrollBar()
        self._follow = value >= bar.maximum() - 4
        self._place_floaters()

    def to_bottom(self) -> None:
        self._follow = True
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _place_floaters(self) -> None:
        """The button back to the end at the view's lower right, shown off the end; the
        find bar across its top."""
        vp = self.viewport().geometry()
        self.to_end.move(
            vp.right() - self.to_end.width() - 10, vp.bottom() - self.to_end.height() - 8
        )
        self.to_end.setVisible(not self._follow)
        self.to_end.raise_()
        width = min(vp.width() - 16, 360)
        self.find_bar.setGeometry(vp.right() - width - 8, vp.top() + 6, width, 30)
        self.find_bar.raise_()

    def resizeEvent(self, e: Any) -> None:  # noqa: N802
        super().resizeEvent(e)
        self._place_floaters()

    # -- find -------------------------------------------------------------------------
    def open_find(self) -> None:
        self._place_floaters()
        self.find_bar.show()
        self.find_bar.edit.setFocus()
        self.find_bar.edit.selectAll()
        self.find_again(first=True)

    def close_find(self) -> None:
        self.find_bar.hide()
        self._deselect()
        self._hits, self._hit = [], -1

    def _find_hits(self, needle: str) -> list[tuple[QLabel, int, int]]:
        """Where *needle* is (any case) in the labels shown: (label, start, length)."""
        hits: list[tuple[QLabel, int, int]] = []
        if not needle:
            return hits
        low = needle.lower()
        host = self.widget()
        for entry in self.items():
            labels = [entry] if isinstance(entry, QLabel) else entry.findChildren(QLabel)
            for lbl in labels:
                if _put_away(lbl, host):
                    continue
                text = _label_text(lbl).lower()
                at = text.find(low)
                while at >= 0:
                    hits.append((lbl, at, len(needle)))
                    at = text.find(low, at + len(needle))
        return hits

    def find_again(self, *, first: bool = False) -> None:
        """The hits for the find bar's text, made again; the first one shown."""
        self._deselect()
        self._hits = self._find_hits(self.find_bar.edit.text())
        self._hit = 0 if self._hits else -1
        self._show_hit()

    def find_step(self, step: int) -> None:
        """The next (1) or the previous (-1) hit shown, round from the end to the start."""
        if not self._hits:
            return
        self._hit = (self._hit + step) % len(self._hits)
        self._show_hit()

    def _deselect(self) -> None:
        if self._shown is not None:
            try:
                self._shown.setSelection(0, 0)
            except RuntimeError:  # deleted with a cleared conversation
                pass
            self._shown = None

    def _show_hit(self) -> None:
        needle = self.find_bar.edit.text()
        if self._hit < 0:
            self.find_bar.count.setText("No matches" if needle else "")
            return
        self._deselect()
        lbl, at, n = self._hits[self._hit]
        lbl.setSelection(at, n)
        self._shown = lbl
        self._follow = False
        self.ensureWidgetVisible(lbl, 0, 40)
        self.find_bar.count.setText(f"{self._hit + 1} of {len(self._hits)}")

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
            elif isinstance(w, (_Reply, _Tool, _Card)):
                w.restyle(k)
            elif isinstance(w, QLabel):
                w.setFont(k.font(k.small))
        self.restyled.emit(k)

    def follow_palette(self) -> None:
        """Made again in the application's palette if it has changed (light, dark). The
        view's own style sheet sets its palette, which then no longer follows the
        application's, so the panel holding it calls this too when the system's
        appearance changes. A style sheet set here posts a palette event as well: the
        same colours are no change, or it would never end."""
        if Look(self.zoom).key != self.look.key:
            self.restyle()

    def changeEvent(self, e: Any) -> None:  # noqa: N802
        super().changeEvent(e)
        if e.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self.follow_palette()

    def wheelEvent(self, e: Any) -> None:  # noqa: N802
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_by(1 if e.angleDelta().y() > 0 else -1)
            return
        super().wheelEvent(e)
