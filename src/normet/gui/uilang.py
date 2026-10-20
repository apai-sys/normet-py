"""The window in the analyst's language: English, or Chinese -- the system's by default,
View -> Language to choose (kept for the next session).

Ported from cume (cume_qt.uilang). The window is written in English and translated where
its texts are set -- a label's, a button's, a menu's and its actions', a tab's, a tooltip,
a box's items, a group's title, a dialog's -- through a catalogue engine (._i18n) and the
window's catalogue (.ui_zh). What the window's code reads back -- label.text(), combo.currentText(),
action.text(), tabText() -- is the English it set: its logic and its tests read the same
in either language, and only the screen changes. A text no template matches stays
English. Qt's own texts (a dialog's standard buttons, a text field's menu) come from
Qt's translations (qtbase_zh_CN.qm).

How: the setters of Qt's text-bearing classes are wrapped, so a text set is translated as
it is set; a text set in a constructor, or by Qt itself (a form's row label, a message
box's), is translated as its widget is first polished (an event filter, installed while
the language is not English). Each translated text keeps its English beside it -- a
dynamic property per text (`_EN` + the text's key), a combo item's in an item role -- and
the wrapped getters hand that back while the screen shows the translation. Switching
language walks every widget and action and writes each text again from its English.

A widget whose property "normetKeepText" is true is left as written: the Assistant's
conversation (the model's own words), logs, code.

    install(app)           wrap the setters, set the language the preferences or the
                           system give (call once, before the window is built)
    set_language(code)     "en" or "zh" now; every window's texts written again
    choice() / set_choice  the analyst's choice: "system", "en" or "zh"
    mac_app_language()     macOS's own menu items and panels kept to it (from the next start)
    language()             the language in use
    tr(text)               one text in the language in use (for text drawn by hand)
    translate_cells(t)     a table's cells translated (the window's words, not data)

NORMET_UI_AUDIT=<path>: every text met that the catalogue cannot translate is written to
<path> at exit, one per line -- the list a catalogue is filled from.
"""

import atexit
import os
import re
from collections.abc import Callable
from functools import cache, lru_cache
from typing import Any

from PySide6.QtCore import QEvent, QLibraryInfo, QLocale, QObject, Qt, QTranslator
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QSpinBox,
    QStatusBar,
    QTabBar,
    QTableWidget,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QTreeWidget,
    QWidget,
)


def _settings():
    from PySide6.QtCore import QSettings

    return QSettings("apai-sys", "normet-gui")  # where View → Appearance keeps its choice


def pref(key: str, default: Any = None) -> Any:
    return _settings().value(key, default)


def set_pref(key: str, value: Any) -> None:
    _settings().setValue(key, value)


#: The languages the window is written in, by code, each named in itself.
LANGUAGES: dict[str, str] = {"en": "English", "zh": "中文"}
#: The choices View -> Language offers: the system's language, or one of LANGUAGES.
CHOICES = ("system",) + tuple(LANGUAGES)
PREF = "ui_language"

_EN = "normetEn_"  # the dynamic property a text's English is kept in
_ITEM_ROLE = Qt.ItemDataRole.UserRole + 0x2E1  # a combo item's English
_KEEP = "normetKeepText"
_ELIDE = "normetElide"  # a text cut short with … to this many pixels
_CELLS = "normetCells"  # a table whose cells are the window's words (translate_cells)
#: Longer than this is a document, not a window's text: left as written (a pattern
#: search over a long text costs more than it can find).
_LONGEST = 6000

_state = {"lang": "en", "installed": False, "filter": None, "translator": None}
_listeners: list[Callable[[str], None]] = []
_missed: set = set()
_orig: dict[str, Callable] = {}


# -- the language ---------------------------------------------------------------------------
def system_language() -> str:
    """The system's language, as the window can be written in it: Chinese where the
    first of the system's interface languages is Chinese, English otherwise."""
    langs = QLocale.system().uiLanguages() or [QLocale.system().name()]
    return "zh" if str(langs[0]).lower().startswith("zh") else "en"


def choice() -> str:
    """The analyst's choice, kept between sessions: "system" (the default), "en", "zh"."""
    c = pref(PREF, "system")
    return c if c in CHOICES else "system"


def resolve(c: str | None = None) -> str:
    """The language a choice gives; NORMET_UI_LANGUAGE ("en", "zh") over the kept choice
    -- a test's, a screenshot's."""
    forced = os.environ.get("NORMET_UI_LANGUAGE")
    if c is None and forced in LANGUAGES:
        return forced
    c = choice() if c is None else c
    return system_language() if c == "system" else c


def language() -> str:
    return _state["lang"]


def on_change(f: Callable[[str], None]):
    """Call f(code) after each switch of language -- what is drawn by hand (a figure)
    draws itself again."""
    _listeners.append(f)


# -- translation ----------------------------------------------------------------------------
@cache
def _catalogue(lang: str):
    from ._i18n import build_catalogue

    if lang == "zh":
        from . import ui_zh

        return build_catalogue(ui_zh.TEXTS, ui_zh.INNER, ui_zh.DATE_PARTS, sentences=False)
    return build_catalogue({})


_LETTERS = re.compile(r"[A-Za-z]{2}")
_LETTER = re.compile(r"[A-Za-z]")  # a number with its unit ("14 d") has one
_TAG = re.compile(r"<[A-Za-z/][^>]*>")
_ENTITY = re.compile(r"&(?:#x?[0-9A-Fa-f]+|[a-z]+);")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_SPACES = re.compile(r"^(\s*)(.*?)(\s*)$", re.S)


@lru_cache(maxsize=1 << 16)
def _translate(text: str, lang: str) -> str | None:
    """`text` in `lang`, or None: whole; else, as HTML, each text between its tags; else
    in the parts the window joins it from (`_in_parts`), with a note in brackets apart
    (`_with_note`), or sentence by sentence (`_sentences`). Its leading and trailing
    space kept."""
    from ._i18n import translate

    cat = _catalogue(lang)
    lead, core, trail = _SPACES.match(text).groups()
    # A text the catalogue has whole, however short: a unit a spin box writes after its
    # number (" d", " h", " s") has one letter, and the search below wants two.
    exact = cat[0].get(" ".join(core.split()))
    if exact is not None:
        return lead + exact + trail
    if not core or not _LETTER.search(core):
        return None
    # The text whole first: a tooltip that writes <factor> or <min> is not HTML.
    got = translate(core, lang, cat)
    if got is not None:
        return lead + got + trail
    if _TAG.search(core) or _ENTITY.search(core):
        out = _html(core, lang)
        return None if out is None else lead + out + trail
    if got is None:
        got = _in_parts(core, lang, cat)
    if got is None:
        got = _with_note(core, lang, cat)
    if got is None:
        got = _sentences(core, lang, cat)
    return None if got is None else lead + got + trail


def _html(page: str, lang: str) -> str | None:
    """Rich text with each run of text between its tags translated as a text of its own
    (`_translate`, every fallback with it); code (<pre>, <code>) as it is written."""
    import html as _h

    from ._i18n import NOT_PROSE, tag_name
    from ._i18n import TAG_SPLIT as PIECES

    pieces, skip, changed = PIECES.split(page), 0, False
    for i, piece in enumerate(pieces):
        if piece.startswith("<"):
            if tag_name(piece) in NOT_PROSE:
                skip += -1 if piece.startswith("</") else 1
            continue
        if skip > 0 or not _LETTERS.search(piece):
            continue
        plain = _h.unescape(piece)
        got = _translate(plain, lang)
        if got is None:
            if _AUDIT and not _CJK.search(plain):
                _missed.add(" ".join(plain.split()))
            continue
        pieces[i] = _h.escape(got, quote=False)
        changed = True
    return "".join(pieces) if changed else None


#: Where a text the window builds from parts joins them: its lines, then " | " and " · "
#: (a status line's facts), " → " (a chain of steps) and " — " (a title's parts) -- each tried only
#: where the whole text matched no template. A part no template has stays as it is.
_JOINS = ("\n", " | ", " · ", " → ", " — ")


_NOTE = re.compile(r"^(.*?\S)\s*(\([^()]*\))$", re.S)


def _with_note(text: str, lang: str, cat) -> str | None:
    """A part with a note in brackets at its end -- 'missing: 0 (Strong S/N ≥ 1, ...)' --
    its text and its note each translated on their own."""
    from ._i18n import translate

    m = _NOTE.match(text)
    if not m:
        return None
    head, note = m.groups()
    th = translate(head, lang, cat) or _in_parts(head, lang, cat, (" — ",))
    tn = translate(note, lang, cat)
    if th is None and tn is None:
        return None
    return (th or head) + (tn or " " + note)


# A sentence's end -- a full stop after a word, a number or a bracket, a capital after it
# -- but not an abbreviation's (St. Louis, et al. 2015, e.g. X, vs. Y, Fig. 2).
_SENTENCE = re.compile(
    r"(?<=[a-z0-9)%\]])(?<!\bSt)(?<!\bal)(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)"
    r"(?<!\bFig)(?<!\bNo)\. (?=[A-Z(])"
)


def _sentences(text: str, lang: str, cat) -> str | None:
    """A text of several sentences the window strings together, each translated on its
    own -- with its full stop -- and joined as the language joins them (a Chinese one
    ends in its own stop, so no space between)."""
    from ._i18n import translate

    parts = _SENTENCE.split(text)
    if len(parts) < 2:
        return None
    done, any_done = [], False
    for i, part in enumerate(parts):
        whole = part if i == len(parts) - 1 else part + "."
        t = translate(whole, lang, cat) or _in_parts(whole, lang, cat, (" · ",))
        if t is None and _AUDIT and _LETTERS.search(whole):
            _missed.add(" ".join(whole.split()))
        any_done |= t is not None
        done.append(t if t is not None else whole)
    if not any_done:
        return None
    out = done[0]
    for d in done[1:]:
        out += d if out.endswith(("。", "；", "）")) else " " + d
    return out


def _in_parts(text: str, lang: str, cat, joins=_JOINS) -> str | None:
    from ._i18n import translate

    for k, join in enumerate(joins):
        if join not in text:
            continue
        parts = text.split(join)
        done = []
        for part in parts:
            t = translate(part, lang, cat) if _LETTERS.search(part) else part
            if t is None and _LETTERS.search(part):
                t = _in_parts(part, lang, cat, joins[k + 1 :])
            if t is None and _LETTERS.search(part):
                t = _with_note(part, lang, cat)
            if t is None and _AUDIT and _LETTERS.search(part) and join != " — ":
                _missed.add(" ".join(part.split()))
            done.append(part if t is None else t)
        if done != parts:
            return join.join(done)
        return None
    return None


def tr(text, lang: str | None = None):
    """`text` in the language in use (or `lang`); as it is where the catalogue has none."""
    lang = lang or _state["lang"]
    if lang == "en" or not isinstance(text, str) or not text or len(text) > _LONGEST:
        return text
    got = _translate(text, lang)
    if got is None:
        if (
            _AUDIT
            and _LETTERS.search(text)
            and not _TAG.search(text)
            and not _CJK.search(text)
            and not any(j in text for j in _JOINS[:3])
        ):
            _missed.add(" ".join(text.split()))
        return text
    return got


_AUDIT = os.environ.get("NORMET_UI_AUDIT")
if _AUDIT:

    def _write_audit():
        try:
            old = set()
            if os.path.exists(_AUDIT):
                with open(_AUDIT, encoding="utf-8") as fh:
                    old = {ln.rstrip("\n") for ln in fh if ln.strip()}
            with open(_AUDIT, "w", encoding="utf-8") as fh:
                fh.write("\n".join(sorted(old | _missed)) + "\n")
        except OSError:
            pass

    atexit.register(_write_audit)


# -- a text's English kept beside it --------------------------------------------------------
def _keep(obj) -> bool:
    try:
        return bool(obj.property(_KEEP))
    except RuntimeError:  # deleted underneath
        return True


def _english(obj, key: str, shown):
    """The English a text on screen was set as: its kept English where the screen still
    shows the translation that was made of it, the screen's own text otherwise."""
    if not isinstance(shown, str):
        return shown
    kept = obj.property(_EN + key)
    if isinstance(kept, (list, tuple)) and len(kept) == 2 and kept[1] == shown:
        return kept[0]
    return shown


def _note(obj, key: str, en: str, shown: str):
    if en != shown:
        obj.setProperty(_EN + key, [en, shown])
    elif obj.property(_EN + key) is not None:
        obj.setProperty(_EN + key, None)


def _write(obj, key: str, get: Callable, put: Callable):
    """One text of `obj` written again in the language in use, from its English."""
    shown = get(obj)
    if not isinstance(shown, str):
        return
    en = _english(obj, key, shown)
    new = _elided(obj, tr(en)) if key == "text" else tr(en)
    if new != shown:
        put(obj, new)
    _note(obj, key, en, new)


def _elided(w, text: str) -> str:
    width = w.property(_ELIDE) if isinstance(w, QWidget) else None
    if not width:
        return text
    return w.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, int(width))


def set_elided(w, text: str, width: int):
    """`w`'s text (a label's, a button's) on one line, cut short with … to `width` pixels
    -- translated first, then cut, and cut again in the new language on a switch. The
    English, whole, is what w.text() reads back."""
    w.setProperty(_ELIDE, int(width))
    shown = _elided(w, tr(text))
    put = _orig.get("QLabel.setText" if isinstance(w, QLabel) else "QAbstractButton.setText")
    (put or type(w).setText)(w, shown)
    _note(w, "text", text, shown)


def translate_cells(table):
    """A table whose cells are the window's own words (a check, its criterion), not the
    data's: its cells translated now and written again on each switch of language. Call
    it once the cells are set; item.text() reads back the English."""
    table.setProperty(_CELLS, True)
    _cells(table)


def _cells(w):
    t0 = _orig.get("QTableWidgetItem.text")
    if t0 is None:  # not installed: the window is in English
        return
    for r in range(w.rowCount()):
        for c in range(w.columnCount()):
            it = w.item(r, c)
            if it is None:
                continue
            en = it.data(_ITEM_ROLE)
            en = en if isinstance(en, str) else t0(it)
            it.setData(_ITEM_ROLE, en)
            new = tr(en)
            if new != t0(it):
                it.setText(new)


# -- the setters and getters wrapped ---------------------------------------------------------
def _wrap_text(cls, setter: str, getter: str | None, key: str):
    """cls.setter(text, ...) translated, its English kept under `key`; cls.getter() the
    English back."""
    s0 = getattr(cls, setter)
    _orig[f"{cls.__name__}.{setter}"] = s0

    def put(self, text, *a, **kw):
        if isinstance(text, str) and not _keep(self):
            shown = tr(text)
            _note(self, key, text, shown)
            return s0(self, shown, *a, **kw)
        return s0(self, text, *a, **kw)

    put.__name__ = setter
    setattr(cls, setter, put)
    if getter:
        g0 = getattr(cls, getter)
        _orig[f"{cls.__name__}.{getter}"] = g0

        def get(self, *a, **kw):
            return _english(self, key, g0(self, *a, **kw))

        get.__name__ = getter
        setattr(cls, getter, get)


def _wrap_combo():
    c = QComboBox
    add0, _addi0, ins0, _insi0 = c.addItem, c.addItems, c.insertItem, c.insertItems
    set0, txt0, cur0, find0 = c.setItemText, c.itemText, c.currentText, c.findText
    setcur0 = c.setCurrentText
    _orig.update({"QComboBox.itemText": txt0, "QComboBox.setItemText": set0})

    def _mark(self, i, en):
        shown = txt0(self, i)
        self.setItemData(i, en if en != shown else None, _ITEM_ROLE)

    def _str_at(a):
        # addItem(text, data) / addItem(icon, text, data): the text's position.
        return (
            0
            if a and isinstance(a[0], str)
            else 1
            if len(a) > 1 and isinstance(a[1], str)
            else None
        )

    def addItem(self, *a, **kw):
        at = _str_at(a)
        if at is None or _keep(self):
            return add0(self, *a, **kw)
        a = list(a)
        en = a[at]
        a[at] = tr(en)
        add0(self, *a, **kw)
        _mark(self, self.count() - 1, en)

    def addItems(self, texts):
        for t in texts:
            addItem(self, t)

    def insertItem(self, index, *a, **kw):
        at = _str_at(a)
        if at is None or _keep(self):
            return ins0(self, index, *a, **kw)
        a = list(a)
        en = a[at]
        a[at] = tr(en)
        n = self.count()
        ins0(self, index, *a, **kw)
        _mark(self, min(index, n) if index >= 0 else n, en)

    def insertItems(self, index, texts):
        for k, t in enumerate(texts):
            insertItem(self, index + k if index >= 0 else -1, t)

    def setItemText(self, i, text):
        if _keep(self):
            return set0(self, i, text)
        set0(self, i, tr(text))
        _mark(self, i, text)

    def itemText(self, i):
        en = self.itemData(i, _ITEM_ROLE)
        return en if isinstance(en, str) else txt0(self, i)

    def currentText(self):
        i = self.currentIndex()
        if i >= 0 and cur0(self) == txt0(self, i):
            return itemText(self, i)
        return cur0(self)

    def findText(self, text, flags=Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchCaseSensitive):
        for i in range(self.count()):
            if self.itemData(i, _ITEM_ROLE) == text:
                return i
        return find0(self, text, flags)

    def setCurrentText(self, text):
        i = findText(self, text)
        if i >= 0 and isinstance(self.itemData(i, _ITEM_ROLE), str):
            self.setCurrentIndex(i)
        else:
            setcur0(self, text)

    for f in (
        addItem,
        addItems,
        insertItem,
        insertItems,
        setItemText,
        itemText,
        currentText,
        findText,
        setCurrentText,
    ):
        setattr(c, f.__name__, f)
    # A combo's text signals carry the text its own code reads -- the English -- and not
    # the screen's: a slot given the text (currentTextChanged(text)) acts on what the
    # getters say. Each is the signal itself underneath, its argument read again.
    for name, get in (
        ("currentTextChanged", lambda box, t: box.currentText()),
        ("textActivated", _item_english),
        ("textHighlighted", _item_english),
    ):
        setattr(c, name, _EnglishSignal(c.__dict__[name], get))


def _item_english(box, shown):
    for i in range(box.count()):
        if _orig["QComboBox.itemText"](box, i) == shown:
            en = box.itemData(i, _ITEM_ROLE)
            return en if isinstance(en, str) else shown
    return shown


def _takes_argument(slot) -> bool:
    import inspect

    try:
        params = inspect.signature(slot).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        p.kind in (p.VAR_POSITIONAL,) or (p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
        for p in params
    )


class _EnglishSignal:
    """A combo's text signal, read through: connect(slot) has the slot called with the
    English of the text the signal carries; emit and disconnect as the signal's own."""

    def __init__(self, signal, english):
        self.signal, self.english = signal, english

    def __get__(self, box, owner=None):
        if box is None:
            return self.signal
        return _BoundEnglish(box, self.signal.__get__(box, owner), self.english)

    def __set__(self, box, value):
        # A data descriptor: PySide keeps the signal instance it makes on the object's
        # __dict__ the first time it is read, and an entry there would otherwise outrank
        # a descriptor with no __set__ -- the second read would reach the bare signal.
        pass


class _BoundEnglish:
    def __init__(self, box, raw, english):
        self.box, self.raw, self.english = box, raw, english

    def connect(self, slot, *a, **kw):
        box, english = self.box, self.english
        if not callable(slot) or hasattr(slot, "emit"):
            return self.raw.connect(slot, *a, **kw)  # a signal, chained as it is
        # As PySide calls a slot: with the text where it takes an argument, else without.
        takes = _takes_argument(slot)

        def relay(text, _slot=slot):
            if not takes:
                return _slot()
            # Read in English too: a switch back to English writes the items again
            # one by one, and an item not yet written still shows its translation.
            return _slot(english(box, text))

        relays = box.__dict__.setdefault("_normet_relays", {})
        relays[(id(self.raw), slot)] = relay
        return self.raw.connect(relay, *a, **kw)

    def disconnect(self, slot=None):
        if slot is None:
            return self.raw.disconnect()
        relay = self.box.__dict__.get("_normet_relays", {}).pop((id(self.raw), slot), slot)
        return self.raw.disconnect(relay)

    def emit(self, *a):
        return self.raw.emit(*a)


def _wrap_tabs(cls):
    """Tabs: their texts kept by what they show (a tab moves; its text goes with it)."""
    add0, ins0, set0, get0 = cls.addTab, cls.insertTab, cls.setTabText, cls.tabText
    _orig.update({f"{cls.__name__}.setTabText": set0, f"{cls.__name__}.tabText": get0})

    def _kept(self) -> dict[str, str]:
        d = self.property(_EN + "tabs")
        return dict(d) if isinstance(d, dict) else {}

    def _remember(self, en, shown):
        if en != shown:
            d = _kept(self)
            d[shown] = en
            self.setProperty(_EN + "tabs", d)

    def _str(a, at):
        return at < len(a) and isinstance(a[at], str)

    def addTab(self, *a):
        # (widget, text) or (widget, icon, text) -- QTabBar: (text) or (icon, text).
        at = next((i for i in range(len(a)) if isinstance(a[i], str)), None)
        if at is None or _keep(self):
            return add0(self, *a)
        a = list(a)
        en = a[at]
        a[at] = tr(en)
        _remember(self, en, a[at])
        return add0(self, *a)

    def insertTab(self, index, *a):
        at = next((i for i in range(len(a)) if isinstance(a[i], str)), None)
        if at is None or _keep(self):
            return ins0(self, index, *a)
        a = list(a)
        en = a[at]
        a[at] = tr(en)
        _remember(self, en, a[at])
        return ins0(self, index, *a)

    def setTabText(self, i, text):
        if isinstance(text, str) and not _keep(self):
            shown = tr(text)
            _remember(self, text, shown)
            return set0(self, i, shown)
        return set0(self, i, text)

    def tabText(self, i):
        shown = get0(self, i)
        return _kept(self).get(shown, shown)

    for f in (addTab, insertTab, setTabText, tabText):
        setattr(cls, f.__name__, f)


def _wrap_items():
    """A header item's text read back as the English it was set as (its item role); a
    cell holds the data's own text and none is translated."""
    from PySide6.QtWidgets import QTableWidgetItem

    t0 = QTableWidgetItem.text
    _orig["QTableWidgetItem.text"] = t0

    def text(self):
        en = self.data(_ITEM_ROLE)
        return en if isinstance(en, str) else t0(self)

    QTableWidgetItem.text = text


def _wrap_headers():
    """A table's and a tree's header labels: translated as set; read back through the
    header item, whose English is kept in its item role."""
    t0, v0, h0 = (
        QTableWidget.setHorizontalHeaderLabels,
        QTableWidget.setVerticalHeaderLabels,
        QTreeWidget.setHeaderLabels,
    )

    def _labels(f, item_of):
        def put(self, labels):
            labels = list(labels)
            if _keep(self) or _state["lang"] == "en":
                return f(self, labels)
            f(self, [tr(x) for x in labels])
            for i, en in enumerate(labels):
                it = item_of(self, i)
                if it is not None:
                    it.setData(_ITEM_ROLE, en)

        put.__name__ = f.__name__
        return put

    QTableWidget.setHorizontalHeaderLabels = _labels(t0, lambda s, i: s.horizontalHeaderItem(i))
    QTableWidget.setVerticalHeaderLabels = _labels(v0, lambda s, i: s.verticalHeaderItem(i))

    def tree_put(self, labels):
        labels = list(labels)
        if _keep(self) or _state["lang"] == "en":
            return h0(self, labels)
        h0(self, [tr(x) for x in labels])
        it = self.headerItem()
        for i, en in enumerate(labels):
            it.setData(i, _ITEM_ROLE, en)

    tree_put.__name__ = "setHeaderLabels"
    QTreeWidget.setHeaderLabels = tree_put


def _wrap_menu_title():
    """A menu's title is its menu action's text: one text, kept on the action, whichever
    of the two it is set or read through."""
    t0, g0 = QMenu.setTitle, QMenu.title
    _orig.update({"QMenu.setTitle": t0, "QMenu.title": g0})

    def setTitle(self, text):
        if isinstance(text, str) and not _keep(self):
            shown = tr(text)
            _note(self.menuAction(), "text", text, shown)
            return t0(self, shown)
        return t0(self, text)

    def title(self):
        return _english(self.menuAction(), "text", g0(self))

    QMenu.setTitle, QMenu.title = setTitle, title


def _wrap_menus():
    """A menu's actions and submenus made from text: translated as made (the menu's
    title, the action's text); an action added made already, as it is added."""
    for cls in (QMenu, QMenuBar, QToolBar):
        a0 = cls.addAction

        def addAction(self, *a, _a0=a0, **kw):
            act = _a0(self, *a, **kw)
            if isinstance(act, QAction) and not _keep(self):
                _action(act)
            return act

        cls.addAction = addAction
    for cls in (QMenu, QMenuBar):
        m0 = cls.addMenu

        def addMenu(self, *a, _m0=m0):
            menu = _m0(self, *a)
            if isinstance(menu, QMenu):
                _action(menu.menuAction())
            return menu

        cls.addMenu = addMenu
    s0 = QMenu.addSection

    def addSection(self, *a):
        a = list(a)
        at = next((i for i in range(len(a)) if isinstance(a[i], str)), None)
        if at is not None and not _keep(self):
            a[at] = tr(a[at])
        return s0(self, *a)

    QMenu.addSection = addSection


def _wrap_dialog_statics():
    """Message boxes and file dialogs opened by their static functions: the title and the
    text translated as they are asked for (the box's own children are polished too, but
    a native file dialog's caption is not a widget of ours)."""
    from PySide6.QtWidgets import QFileDialog, QInputDialog

    for name in ("information", "warning", "critical", "question", "about"):
        f0 = getattr(QMessageBox, name)

        def box(*a, _f0=f0, **kw):
            a = [tr(x) if isinstance(x, str) else x for x in a]
            kw = {
                k: (tr(v) if isinstance(v, str) and k in ("title", "text") else v)
                for k, v in kw.items()
            }
            return _f0(*a, **kw)

        setattr(QMessageBox, name, staticmethod(box))
    for name in ("getOpenFileName", "getOpenFileNames", "getSaveFileName", "getExistingDirectory"):
        f0 = getattr(QFileDialog, name)

        def ask(parent=None, caption="", *a, _f0=f0, **kw):
            if "caption" in kw:
                kw["caption"] = tr(kw["caption"])
                return _f0(parent, *a, **kw)
            return _f0(parent, tr(caption), *a, **kw)

        setattr(QFileDialog, name, staticmethod(ask))
    for name in ("getText", "getInt", "getDouble", "getItem", "getMultiLineText"):
        f0 = getattr(QInputDialog, name)

        def ask_value(parent, title, label, *a, _f0=f0, **kw):
            return _f0(parent, tr(title), tr(label), *a, **kw)

        setattr(QInputDialog, name, staticmethod(ask_value))


def _wrap_all():
    _wrap_text(QLabel, "setText", "text", "text")
    _wrap_text(QAbstractButton, "setText", "text", "text")
    _wrap_text(QMessageBox, "setText", "text", "text")
    _wrap_text(QMessageBox, "setInformativeText", "informativeText", "info")
    _wrap_text(QAction, "setText", "text", "text")
    _wrap_text(QAction, "setToolTip", "toolTip", "tip")
    _wrap_text(QAction, "setStatusTip", "statusTip", "status")
    _wrap_text(QAction, "setIconText", "iconText", "icon")
    _wrap_text(QWidget, "setToolTip", "toolTip", "tip")
    _wrap_text(QWidget, "setStatusTip", "statusTip", "status")
    _wrap_text(QWidget, "setWindowTitle", "windowTitle", "title")
    _wrap_text(QGroupBox, "setTitle", "title", "group")
    _wrap_menu_title()
    _wrap_text(QLineEdit, "setPlaceholderText", "placeholderText", "placeholder")
    _wrap_text(QTextEdit, "setPlaceholderText", "placeholderText", "placeholder")
    _wrap_text(QPlainTextEdit, "setPlaceholderText", "placeholderText", "placeholder")
    _wrap_text(QStatusBar, "showMessage", "currentMessage", "message")
    _wrap_text(QProgressDialog, "setLabelText", "labelText", "label")
    for cls in (QSpinBox, QDoubleSpinBox):
        _wrap_text(cls, "setSuffix", "suffix", "suffix")
        _wrap_text(cls, "setPrefix", "prefix", "prefix")
    _wrap_text(QAbstractSpinBox, "setSpecialValueText", "specialValueText", "special")
    _wrap_combo()
    _wrap_tabs(QTabWidget)
    _wrap_tabs(QTabBar)
    _wrap_headers()
    _wrap_items()
    _wrap_menus()
    _wrap_dialog_statics()
    b0 = QDialogButtonBox.addButton

    def addButton(self, *a):
        a = list(a)
        if a and isinstance(a[0], str) and not _keep(self):
            en = a[0]
            a[0] = tr(en)
            button = b0(self, *a)
            if button is not None:
                _note(button, "text", en, a[0])
            return button
        return b0(self, *a)

    QDialogButtonBox.addButton = addButton


# -- a widget's texts written again (first polish, a switch of language) --------------------
_TEXTS = [
    # (class, key, getter, setter): the texts a widget of the class holds on its own.
    (QLabel, "text", "QLabel.text", "QLabel.setText"),
    (QAbstractButton, "text", "QAbstractButton.text", "QAbstractButton.setText"),
    (QGroupBox, "group", "QGroupBox.title", "QGroupBox.setTitle"),
    (QLineEdit, "placeholder", "QLineEdit.placeholderText", "QLineEdit.setPlaceholderText"),
    (QTextEdit, "placeholder", "QTextEdit.placeholderText", "QTextEdit.setPlaceholderText"),
    (
        QPlainTextEdit,
        "placeholder",
        "QPlainTextEdit.placeholderText",
        "QPlainTextEdit.setPlaceholderText",
    ),
    (QWidget, "tip", "QWidget.toolTip", "QWidget.setToolTip"),
    (QWidget, "status", "QWidget.statusTip", "QWidget.setStatusTip"),
    (QWidget, "title", "QWidget.windowTitle", "QWidget.setWindowTitle"),
    (QSpinBox, "suffix", "QSpinBox.suffix", "QSpinBox.setSuffix"),
    (QSpinBox, "prefix", "QSpinBox.prefix", "QSpinBox.setPrefix"),
    (QDoubleSpinBox, "suffix", "QDoubleSpinBox.suffix", "QDoubleSpinBox.setSuffix"),
    (QDoubleSpinBox, "prefix", "QDoubleSpinBox.prefix", "QDoubleSpinBox.setPrefix"),
    (
        QAbstractSpinBox,
        "special",
        "QAbstractSpinBox.specialValueText",
        "QAbstractSpinBox.setSpecialValueText",
    ),
]


def _action(act: QAction):
    for key, g, s in (
        ("text", "QAction.text", "QAction.setText"),
        ("tip", "QAction.toolTip", "QAction.setToolTip"),
        ("status", "QAction.statusTip", "QAction.setStatusTip"),
    ):
        if key == "tip" and _orig[g](act) == _orig["QAction.text"](act).replace("&", ""):
            continue  # Qt's default tooltip: the text, translated with it
        _write(act, key, _orig[g], _orig[s])
    # Not act.menu(): on PySide6 it takes the submenu's ownership, and the submenu is
    # deleted with the wrapper (measured, the command palette's). A submenu's title is
    # its own widget's, written as the QMenu is.


def _widget(w: QWidget):
    """Every text `w` holds on its own, written in the language in use from its English."""
    if _keep(w):
        return
    for cls, key, g, s in _TEXTS:
        if isinstance(w, cls):
            if key == "title" and not (w.isWindow() or isinstance(w, QDockWidget)):
                continue
            _write(w, key, _orig[g], _orig[s])
    if isinstance(w, QComboBox) and not w.isEditable():
        for i in range(w.count()):
            shown = _orig["QComboBox.itemText"](w, i)
            en = w.itemData(i, _ITEM_ROLE)
            en = en if isinstance(en, str) else shown
            new = tr(en)
            # The English first: the current item's new text sends currentTextChanged,
            # and its slot is handed the English the item keeps (a time series' "Daily",
            # not 日均 -- read as no resampling rule at all).
            w.setItemData(i, en, _ITEM_ROLE)
            if new != shown:
                _orig["QComboBox.setItemText"](w, i, new)
            if en == new:
                w.setItemData(i, None, _ITEM_ROLE)
    if isinstance(w, QTabBar):
        get0, set0 = _orig["QTabBar.tabText"], _orig["QTabBar.setTabText"]
        kept = w.property(_EN + "tabs")
        kept = dict(kept) if isinstance(kept, dict) else {}
        owner = w.parent() if isinstance(w.parent(), QTabWidget) else None
        if owner is not None:
            more = owner.property(_EN + "tabs")
            kept.update(dict(more) if isinstance(more, dict) else {})
        fresh = {}
        for i in range(w.count()):
            shown = get0(w, i)
            en = kept.get(shown, shown)
            new = tr(en)
            if new != shown:
                set0(w, i, new)
            if new != en:
                fresh[new] = en
        w.setProperty(_EN + "tabs", fresh or None)
        if owner is not None:
            owner.setProperty(_EN + "tabs", fresh or None)
    if isinstance(w, (QTableWidget, QTreeWidget)):
        _headers(w)
    if isinstance(w, QTableWidget) and w.property(_CELLS):
        _cells(w)
    if isinstance(w, QMenu):
        _action(w.menuAction())
    for act in w.actions():
        _action(act)


def _headers(w):
    if isinstance(w, QTableWidget):
        items = [w.horizontalHeaderItem(i) for i in range(w.columnCount())] + [
            w.verticalHeaderItem(i) for i in range(w.rowCount())
        ]
        for it in items:
            if it is None:
                continue
            shown = _orig["QTableWidgetItem.text"](it)
            en = it.data(_ITEM_ROLE)
            en = en if isinstance(en, str) else shown
            new = tr(en)
            if new != shown:
                it.setText(new)
            it.setData(_ITEM_ROLE, en if en != new else None)
    else:
        it = w.headerItem()
        for i in range(w.columnCount()):
            shown = it.text(i)
            en = it.data(i, _ITEM_ROLE)
            en = en if isinstance(en, str) else shown
            new = tr(en)
            if new != shown:
                it.setText(i, new)
            it.setData(i, _ITEM_ROLE, en if en != new else None)


class _Polish(QObject):
    """A widget's texts translated as it is first polished -- those its constructor set,
    or Qt's own code (a form's row label, a message box's) -- and a menu's actions each
    time it opens (an action made and added after the menu was polished)."""

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == QEvent.Type.Polish and isinstance(obj, QWidget):
            _widget(obj)
        elif t == QEvent.Type.Show and isinstance(obj, QMenu):
            for act in obj.actions():
                _action(act)
        return False


# -- figures ------------------------------------------------------------------------------
def _words(s: str, lang: str) -> str:
    """A tick label in `lang` where it is words (a month, a day); a number as it is."""
    if "$" in s or not re.search(r"[A-Za-z]{2}", s):
        return s
    return tr(s, lang)


@cache
def translating_formatter():
    """A tick formatter that writes another's labels in a language: tick labels are
    written by the axes' formatters each time the figure is drawn, so the formatter is
    what is translated, not the labels."""
    from matplotlib.ticker import Formatter

    class Translating(Formatter):
        def __init__(self, inner, lang):
            self.inner, self.lang = inner, lang

        def __call__(self, x, pos=None):
            return _words(self.inner(x, pos), self.lang)

        def format_ticks(self, values):
            return [_words(s, self.lang) for s in self.inner.format_ticks(values)]

        def get_offset(self):
            return _words(self.inner.get_offset(), self.lang)

        def set_axis(self, axis):
            super().set_axis(axis)
            self.inner.set_axis(axis)

        def set_locs(self, locs):
            self.inner.set_locs(locs)

        def format_data_short(self, value):
            return self.inner.format_data_short(value)

    return Translating


def figure_texts(fig, lang: str | None = None):
    """A window figure's texts in the language in use (or `lang`: the report's, for a
    figure for print), from their English -- kept on each
    text (`_normet_en`), so a switch back writes the English again -- with the language's
    characters drawn from _i18n.figure_font, and its tick labels where they are words
    (months, days). Called by the window's canvas as it draws (FullWidthCanvas.draw). With
    no font for the language on the machine a figure stays English, rather than boxes."""
    lang = lang or _state["lang"]
    if lang == "en" and getattr(fig, "_normet_lang", "en") == "en":
        return
    from matplotlib.text import Text

    from ._i18n import figure_font

    font = figure_font(lang) if lang != "en" else None
    if lang != "en" and font is None:
        lang = "en"
    translating = translating_formatter()
    for t in fig.findobj(Text):
        shown = t.get_text()
        kept = getattr(t, "_normet_en", None)
        en = kept[0] if kept and kept[1] == shown else shown
        # A formula alone ($\mathregular{NO_{2}}$) has no words to translate; one in a
        # title is carried through the translation as a name is.
        words = re.sub(r"\$[^$]*\$", "", en or "")
        if not en or not en.strip() or not re.search(r"[A-Za-z]{2}", words):
            continue
        new = tr(en, lang) if lang != "en" else en
        if new != en and "$" in new:
            # Mathtext sets the whole text in its own face, which has no Chinese: the
            # formula is written out in sub- and superscripts instead (NO₂).
            from normet.style import plain

            new = plain(new)
        if new != shown:
            t.set_text(new)
        t._normet_en = (en, new) if new != en else None
        if font and new != en:
            families = list(t.get_fontfamily())
            if font not in families:
                t.set_fontfamily(families + [font])
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for get, put in (
                (axis.get_major_formatter, axis.set_major_formatter),
                (axis.get_minor_formatter, axis.set_minor_formatter),
            ):
                f = get()
                if lang == "en" and isinstance(f, translating):
                    put(f.inner)
                elif lang != "en" and not isinstance(f, translating):
                    put(translating(f, lang))
            if font:
                labels = axis.get_ticklabels(which="both")
                families = list(labels[0].get_fontfamily()) if labels else ["sans-serif"]
                if font not in families:
                    axis.set_tick_params(which="both", labelfontfamily=families + [font])
                # The axis' offset (a date axis' "2020-Jan") is a text of its own.
                off = list(axis.offsetText.get_fontfamily())
                if font not in off:
                    axis.offsetText.set_fontfamily(off + [font])
    fig._normet_lang = lang


# -- the switch -----------------------------------------------------------------------------
def _qt_translator(lang: str):
    app = QApplication.instance()
    old = _state["translator"]
    if old is not None:
        app.removeTranslator(old)
        _state["translator"] = None
    if lang == "en":
        return
    t = QTranslator(app)
    import PySide6

    # Qt's own path, then the package's: a frozen app carries them under PySide6/Qt.
    for path in (
        QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath),
        os.path.join(os.path.dirname(PySide6.__file__), "Qt", "translations"),
    ):
        if t.load(QLocale(QLocale.Language.Chinese, QLocale.Country.China), "qtbase", "_", path):
            app.installTranslator(t)
            _state["translator"] = t
            return


def _filter(on: bool):
    app = QApplication.instance()
    if on and _state["filter"] is None:
        _state["filter"] = _Polish(app)
        app.installEventFilter(_state["filter"])
    elif not on and _state["filter"] is not None:
        app.removeEventFilter(_state["filter"])
        _state["filter"] = None


def set_language(code: str):
    """The window's language now: every widget's and action's text written again from its
    English, Qt's own from its translations; then each `on_change` listener told."""
    if code not in LANGUAGES:
        raise ValueError(f"language must be one of {', '.join(LANGUAGES)}, not {code!r}")
    _state["lang"] = code
    _qt_translator(code)
    _filter(code != "en")
    for w in QApplication.allWidgets():
        try:
            _widget(w)
        except RuntimeError:  # deleted while the walk went on
            continue
    for f in list(_listeners):
        f(code)


def set_choice(c: str):
    """The analyst's choice (View -> Language), kept, and the language it gives set."""
    if c not in CHOICES:
        raise ValueError(f"choice must be one of {', '.join(CHOICES)}, not {c!r}")
    set_pref(PREF, c)
    set_language(resolve(c))
    mac_app_language(c)


#: What macOS's own texts are written in for each choice: its menu items (Edit's AutoFill,
#: Start Dictation, Emoji & Symbols; View's Enter Full Screen) and the open and save panels.
_MAC_LANGUAGES = {"zh": ("zh-Hans", "en"), "en": ("en", "zh-Hans")}


def mac_app_language(c: str | None = None) -> bool:
    """The packaged Mac app's own language (its AppleLanguages, as System Settings ->
    Language & Region -> Applications sets it) kept to the analyst's choice -- "system"
    removes it. macOS reads it as the app starts: its texts follow from the next start.
    Not from source (the setting would be Python's) or where NORMET_UI_LANGUAGE forces a
    language (a test's, a screenshot's). True where it was written."""
    import sys

    if (
        sys.platform != "darwin"
        or not getattr(sys, "frozen", False)
        or os.environ.get("NORMET_UI_LANGUAGE") in LANGUAGES
    ):
        return False
    c = choice() if c is None else c
    try:
        import ctypes
        import ctypes.util

        cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFArrayCreate.restype = ctypes.c_void_p
        cf.CFArrayCreate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_long,
            ctypes.c_void_p,
        ]
        cf.CFPreferencesSetAppValue.argtypes = [ctypes.c_void_p] * 3
        cf.CFPreferencesAppSynchronize.argtypes = [ctypes.c_void_p]

        def cfstr(t):
            return cf.CFStringCreateWithCString(None, t.encode(), 0x08000100)  # UTF-8

        app = ctypes.c_void_p.in_dll(cf, "kCFPreferencesCurrentApplication")
        langs = _MAC_LANGUAGES.get(c)
        value = None
        if langs:
            items = (ctypes.c_void_p * len(langs))(*[cfstr(x) for x in langs])
            value = cf.CFArrayCreate(
                None,
                items,
                len(langs),
                ctypes.addressof(ctypes.c_char.in_dll(cf, "kCFTypeArrayCallBacks")),
            )
        cf.CFPreferencesSetAppValue(cfstr("AppleLanguages"), value, app)
        cf.CFPreferencesAppSynchronize(app)
        return True
    except (OSError, AttributeError, ValueError):
        return False


def install(app: QApplication | None = None):
    """Wrap the setters (once) and set the language the preferences give -- the system's
    by default. Before the window is built, so its texts are translated as they are set."""
    if not _state["installed"]:
        _wrap_all()
        _state["installed"] = True
    set_language(resolve())
    mac_app_language()  # a choice made before this setting was kept
