"""View → Assistant: a language model working with the analyst in the window.

The model reads what the window holds
-- the table, the model trained, the normalised series, the decomposition --
through the readings the MCP server gives (:mod:`normet.agent.readings`), and
runs the window's steps the way their buttons do, so what it does is on screen
as the analyst's own runs are. Where it helps most is what to model and how to
read the result: it RECOMMENDS a target and predictors as a card with an Apply
button, and nothing is applied until the analyst clicks it -- the click ticks them
in the panel and starts training, and the next message tells the model so.

When a turn ends, the box written in offers the message the analyst is most likely
to send next: the model's own (its ``<!-- next: ... -->`` line, kept off the panel),
else one read off the window as it stands (``suggested_message``); Tab in the empty
box fills it in.

Models are named as :mod:`normet.agent.llm` names them
(``"anthropic:claude-sonnet-5-5"``, ``"deepseek:deepseek-chat"``,
``"ollama:qwen3:8b"``), or are a command-line agent signed in to the user's
own plan (:mod:`.assistant_cli`). A key is read from the macOS keychain
(service "normet", account the provider: Key… stores it there), then from
the provider's environment variable; it is never written anywhere else.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QKeyEvent, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from normet.agent import llm, readings
from normet.agent.mcp_http import ToolServer

from . import assistant_cli
from .chatview import ChatView, Look, composer_sheet
from .theme import theme_manager

if TYPE_CHECKING:
    from .main_window import MainWindow

SYSTEM = """\
You are an assistant to an analyst using normet, which removes the effect of weather
from air-quality time series (meteorological normalisation, or de-weathering): a
machine-learning model of the pollutant is trained on meteorology and time, then the
meteorology is resampled so that what remains is the change weather does not explain.
You work in normet's window, with the table the analyst has open; your tools read what
the window holds and run its steps the way its buttons do.

Work in the window's order, as the analyst asks: look at the data (inspect_data, with
the target), agree the target and predictors (propose_setup), train (train), read the
model (model_fit), normalise (normalise), read it (normalised), and decompose
(decompose, then decomposition).

Where you help most is what to model and how to read the result. RECOMMEND, with your
reasons and how sure you are; the analyst decides.
- Put a target and predictors to the analyst with propose_setup: a card whose Apply
  button ticks them in the panel and starts training. Predictors are meteorological
  columns (wind, temperature, humidity, pressure, boundary layer height, radiation,
  cloud, rain); normet adds the time features itself.
  Say what you left out and why. After proposing, stop and let the analyst answer: their
  Apply trains the model, and their next message says so -- do not train it again.
- Train, normalise or decompose only when the analyst has asked or agreed: each takes
  time (training can take minutes), so say what you start before starting it.
- Judge a model on the 'testing' rows (R2, RMSE, FAC2) and say plainly where it is weak.
- Describe the normalised series by the means the reading gives; do not claim a trend or
  a cause the numbers do not show.
Quote normet's numbers as the tools give them; do not compute or invent others.
Each message starts with the window as it stands ([Window now: ...]).
Answer in the language the analyst writes in. The panel shows Markdown; keep it short.
End every reply with one line of its own, <!-- next: ... -->, holding the message the
analyst is most likely to send next: in their language, as they would type it, following
from what you just said -- an offer you made, its acceptance; a question you asked, its
likely answer. The panel puts it in their box as a suggestion, not in your reply.
"""

#: What SYSTEM says of a proposal waiting for the analyst, and what full automatic says
#: instead (system_prompt).
_ASKED = """\
  Say what you left out and why. After proposing, stop and let the analyst answer: their
  Apply trains the model, and their next message says so -- do not train it again.
- Train, normalise or decompose only when the analyst has asked or agreed: each takes
  time (training can take minutes), so say what you start before starting it.
"""
_AUTO = """\
  Say what you left out and why. Full automatic is on: the set-up is applied the moment
  you propose it, and no one is asked.
- Work the chain to its end without waiting for a click: look at the data, propose the
  set-up, train, normalise and decompose, reading each result as it comes (a step returns
  its reading once it has ended). Say what ran and where it stands.
"""

#: The models offered: API models by provider:model, then the command-line agents.
API_MODELS = (
    "anthropic:claude-sonnet-5-5",
    "anthropic:claude-opus-5-5",
    "anthropic:claude-haiku-5-5",
    "deepseek:deepseek-chat",
)

#: Full automatic (the panel's Auto, kept in the settings): what the Assistant proposes is
#: applied at once, and a turn is worked to the chain's end.
AUTO_KEY = "assistant/auto"
#: The model calls one round of tool use may make before the turn is looked at again.
MAX_TURNS = 25
#: Full automatic: the times in a row the model is told the chain is unfinished without
#: the window moving, before its turn is let end; and the nudges a turn takes at most.
AUTO_STALLS = 2
AUTO_NUDGES = 8
#: Full automatic: a failed or empty call to the model tried again this many times in a
#: turn (after the provider's own retries, normet.agent.llm._post), this long apart, s.
AUTO_RETRIES = 2
AUTO_RETRY_WAIT = 20.0
#: The chain full automatic works, in the window's order (chain_left).
CHAIN = ("data", "setup", "train", "normalise", "decompose")
#: What each step left takes, as the nudge says it.
CHAIN_TEXT = {
    "wait": "the step under way: wait_for_run",
    "data": "the data looked at: inspect_data, with the target",
    "setup": "the target and predictors: propose_setup (applied at once)",
    "train": "the model: train, and say how well it fits the testing rows",
    "normalise": "the normalisation: normalise",
    "decompose": "the decomposition: decompose (method='meteorology' on chronos-2, where "
    "the emission one is not available)",
}
#: How the panel names each step left, in its note.
CHAIN_WORDS = {
    "wait": "the run under way",
    "data": "a look at the data",
    "setup": "the set-up",
    "train": "training",
    "normalise": "the normalisation",
    "decompose": "the decomposition",
}


def _settings() -> QSettings:
    return QSettings("apai-sys", "normet-gui")


def auto_on() -> bool:
    """Full automatic, as the panel's Auto last set it."""
    return bool(_settings().value(AUTO_KEY, False, type=bool))


def system_prompt() -> str:
    """SYSTEM, with full automatic on what it says of proposals waiting for a click
    replaced by what happens instead."""
    return SYSTEM.replace(_ASKED, _AUTO) if auto_on() else SYSTEM


# -- keys -------------------------------------------------------------------------------
_SESSION_KEYS: dict[str, str] = {}


def keychain_get(provider: str) -> str | None:
    """The key the macOS keychain holds for *provider* (service "normet"), or None."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", "normet", "-a", provider, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def keychain_store(provider: str, key: str) -> bool:
    """Keep *key* in the macOS keychain for *provider*; False where there is none."""
    if sys.platform != "darwin":
        return False
    try:
        out = subprocess.run(
            ["security", "add-generic-password", "-U", "-s", "normet", "-a", provider, "-w", key],
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0


def find_key(provider: str) -> str | None:
    """The provider's key: typed in this session, the keychain, or its environment variable."""
    import os

    env = llm.PROVIDERS.get(provider, ("", "", None))[2]
    return (
        _SESSION_KEYS.get(provider)
        or keychain_get(provider)
        or (os.environ.get(env) if env else None)
    )


# -- the window's tools -------------------------------------------------------------------
class _Bridge(QObject):
    """Carries a call from the model's thread to the window's, where widgets are."""

    ask = Signal(object)

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self.ask.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def _run(self, job: Callable[[], None]) -> None:
        job()


def _checked(widget: Any) -> list[str]:
    return [
        widget.item(i).text()
        for i in range(widget.count())
        if widget.item(i).checkState() == Qt.CheckState.Checked
    ]


def _set_checked(widget: Any, names: list[str]) -> list[str]:
    """Tick *names* in a check list and untick the rest; the names it does not hold."""
    have = {widget.item(i).text() for i in range(widget.count())}
    for i in range(widget.count()):
        item = widget.item(i)
        item.setCheckState(
            Qt.CheckState.Checked if item.text() in names else Qt.CheckState.Unchecked
        )
    return [n for n in names if n not in have]


class WindowTools:
    """The tools the model is given: readings of the window, and steps run in it."""

    def __init__(self, win: MainWindow, dock: AssistantDock) -> None:
        self.win, self.dock = win, dock
        self.stop = threading.Event()
        #: The tools called without error (full automatic's chain_left reads from it).
        self.called: set[str] = set()

    def _gui(self, fn: Callable[[], Any]) -> Any:
        """*fn* run on the window's thread, its answer (or its exception) brought back."""
        if QThread.currentThread() is self.win.thread():
            return fn()
        box: dict[str, Any] = {}
        done = threading.Event()

        def job() -> None:
            try:
                box["out"] = fn()
            except Exception as exc:  # noqa: BLE001 -- told to the model
                box["exc"] = exc
            done.set()

        self.dock.bridge.ask.emit(job)
        done.wait()
        if "exc" in box:
            raise box["exc"]
        return box.get("out")

    def _step(
        self, start: Callable[[], None], marker: Callable[[], Any], describe: Callable[[], str]
    ) -> str:
        """Run a step of the window's (``start``, on its thread) to its end, as its
        button does, and say what it made (``describe``); ``marker`` is what the
        step replaces when it succeeds."""
        ended = threading.Event()
        box: dict[str, Any] = {}

        def begin() -> None:
            runner = self.win.runner
            if runner.busy:
                box["error"] = "the window is running something: wait for it to end"
                ended.set()
                return
            box["before"] = marker()
            start()
            if not runner.busy:
                box["error"] = "the window did not start it (it says why on screen)"
                ended.set()
                return

            def finished() -> None:
                runner.finished.disconnect(finished)
                # Let the step's own result slot run first: it is queued as well.
                QTimer.singleShot(0, ended.set)

            runner.finished.connect(finished)

        self.dock.bridge.ask.emit(begin)
        while not ended.wait(0.25):
            if self.stop.is_set():
                return "stopped by the analyst; the window's step carries on and its result comes on screen"
        if "error" in box:
            return str(box["error"])

        def after() -> str:
            if marker() is box.get("before"):
                return "the step did not finish: the window says why on screen (and in its log)"
            return describe()

        return str(self._gui(after))

    # readings
    def window(self) -> str:
        """The window as it stands: the table open, the target and predictors ticked,
        the model and the results made."""
        return str(self._gui(lambda: state_text(self.win)))

    def inspect_data(self, target: str | None = None) -> str:
        """The table open in the window: its size and span, the time step and gaps,
        and each numeric column's role, missing values, range and -- with target, the
        window's target where left out -- its correlation with the target."""

        def read() -> str:
            if self.win.df_raw is None:
                raise ValueError("no table is open in the window")
            t = target or self.win.target_combo.currentText() or None
            return readings.data_reading(self.win.df_raw, t, name="The table open")

        return str(self._gui(read))

    def model_fit(self) -> str:
        """How well the trained model predicts: normet.modStats on the rows held out
        ('testing'), the training rows and all, and its feature importance."""

        def read() -> str:
            if self.win.model is None or self.win.df_prep is None:
                raise ValueError("no model is trained in the window yet")
            if self.win._is_chronos():
                return chronos_reading(self.win)
            return readings.model_reading(self.win.df_prep, self.win.model)

        return str(self._gui(read))

    def normalised(self) -> str:
        """The normalised series on screen against the observed one: means overall
        and by year or month, and the meteorological part (observed − normalised)."""

        def read() -> str:
            result = self.win.results.get("normalise")
            if result is None:
                raise ValueError("nothing is normalised in the window yet")
            return readings.normalised_reading(result)

        return str(self._gui(read))

    def decomposition(self) -> str:
        """The decomposition on screen: each component's mean, spread and range."""

        def read() -> str:
            result = self.win.results.get("decompose")
            if result is None:
                raise ValueError("nothing is decomposed in the window yet")
            return readings.decomposition_reading(result)

        return str(self._gui(read))

    def wait_for_run(self, max_wait: float = 600.0) -> str:
        """Wait for the step the window is running -- one the analyst started, or one
        before -- to end (at most max_wait seconds), then say how the window stands."""
        t0 = time.monotonic()
        while self._gui(lambda: self.win.runner.busy):
            if time.monotonic() - t0 > float(max_wait):
                return (
                    f"still running after {max_wait:.0f} s ({self.win.runner.task_name}); "
                    "call wait_for_run again"
                )
            if self.stop.wait(0.5):
                return "stopped by the analyst; the window's step carries on"
        return "nothing is running. The window: " + self.window()

    def log(self, lines: int = 40) -> str:
        """The last lines of the window's log: what it ran, warnings and errors."""
        text = str(self._gui(lambda: self.win.log_view.toPlainText()))
        return "\n".join(text.splitlines()[-max(1, int(lines)) :]) or "(the log is empty)"

    # proposals
    def propose_setup(
        self,
        target: str,
        features: list[str],
        reasons: str,
        confidence: float,
        resample_vars: list[str] | None = None,
    ) -> str:
        """Recommend the target and the predictors (features) to train on, and the
        variables to resample when normalising (resample_vars: the meteorological
        ones, as a rule): a card with your reasons, how sure you are (confidence, 0
        to 1) and an Apply button, which ticks them in the window's panel."""

        def check() -> None:
            df = self.win.df_raw
            if df is None:
                raise ValueError("no table is open in the window")
            missing = [c for c in [target, *features] if c not in df.columns]
            if missing:
                raise ValueError(f"the table has no column {', '.join(missing)}")
            if target in features:
                raise ValueError("the target cannot also be a predictor")

        self._gui(check)
        resample = list(resample_vars or [])
        body = (
            f"**Target:** {target}\n\n**Predictors:** {', '.join(features)}"
            + (f"\n\n**Resample when normalising:** {', '.join(resample)}" if resample else "")
            + f"\n\n{reasons}\n\n*Confidence {max(0.0, min(1.0, float(confidence))):.0%}*"
        )

        def apply() -> str:
            self.win.target_combo.setCurrentText(target)
            unknown = _set_checked(self.win.feature_list, features)
            self.dock.pending_resample = resample
            return "Applied" + (f" ({', '.join(unknown)} not in the list)" if unknown else "")

        def apply_and_train() -> str:
            # The analyst's click: the set-up ticked and the model trained at once, as
            # the card's button says; the next message tells the model.
            said = apply()
            self.dock.train_applied(f"target {target}; predictors {', '.join(features)}")
            return said

        if auto_on():
            said = str(self._gui(apply))
            self.dock.post_card.emit("Proposed set-up", body, apply, said, "Apply")
            return (
                f"{said} at once (full automatic is on): the panel has the target and "
                "predictors ticked. Go on with the chain."
            )
        self.dock.post_card.emit("Proposed set-up", body, apply_and_train, "", "Apply and train")
        return (
            "Shown to the analyst as a card; its Apply ticks the set-up and starts training. "
            "Stop here and let them answer."
        )

    # steps
    def train(self) -> str:
        """Train the model with the window's panel as it stands (target, predictors,
        backend, budget) -- once the analyst has agreed the set-up. Returns the fit."""
        return self._step(self.win._run_train, lambda: self.win.model, self._fit_now)

    def _fit_now(self) -> str:
        if self.win._is_chronos():
            return chronos_reading(self.win)
        return readings.model_reading(self.win.df_prep, self.win.model)

    def normalise(self, resample_vars: list[str] | None = None) -> str:
        """Normalise with the trained model, resampling resample_vars (the set-up the
        analyst applied, or the panel's ticks, where left out). Returns the reading."""

        def start() -> None:
            names = resample_vars or self.dock.pending_resample
            if names:
                _set_checked(self.win.norm_vars, list(names))
            self.win._run_normalise()

        return self._step(
            start,
            lambda: self.win.results.get("normalise"),
            lambda: readings.normalised_reading(self.win.results["normalise"]),
        )

    def decompose(self, method: str = "emission") -> str:
        """Decompose with the trained model: method 'emission' splits the normalised
        series into time scales; 'meteorology' splits observed − normalised by
        meteorological variable. Returns the reading."""
        if method not in ("emission", "meteorology"):
            raise ValueError("method is 'emission' or 'meteorology'")
        if method == "emission" and self._gui(self.win._is_chronos):
            raise ValueError(
                "the emission decomposition is not available on chronos-2: it reads the "
                "calendar off the target's own history, so the time-scale components would "
                "come back near zero. Use method='meteorology', or train with flaml or lightgbm."
            )

        def start() -> None:
            self.win.decom_method.setCurrentText(method)
            self.win._run_decompose()

        return self._step(
            start,
            lambda: self.win.results.get("decompose"),
            lambda: readings.decomposition_reading(self.win.results["decompose"]),
        )

    def tools(self) -> list[llm.Tool]:
        return [
            llm.as_tool(f)
            for f in (
                self.window,
                self.inspect_data,
                self.model_fit,
                self.normalised,
                self.decomposition,
                self.wait_for_run,
                self.log,
                self.propose_setup,
                self.train,
                self.normalise,
                self.decompose,
            )
        ]


def chronos_reading(win: MainWindow) -> str:
    """Chronos-2's quality check as text: how far its forecast moves when the
    future meteorology is shuffled -- the zero-shot stand-in for a model's fit."""
    est = win.model
    head = (
        f"The model is Chronos-2 ({getattr(est, 'model_name', '?')}, on {getattr(est, 'device', '?')}), "
        "zero-shot: nothing is fitted, so there is no held-out fit or feature importance. "
        "Its check is covariate sensitivity -- how far the median forecast moves when the "
        "future meteorology is shuffled. Near zero, the model is reading the series' own "
        "history, not the weather, and its normalised series means little."
    )
    if win.chronos_check is None:
        return head + "\nThe check could not be run on this record (see the Model tab)."
    shift, (level, said) = win.chronos_check
    rows = [
        ["mean absolute median shift", shift.get("mean_abs_shift")],
        ["max absolute median shift", shift.get("max_abs_shift")],
        ["shift as % of the prediction", shift.get("pct_of_prediction")],
    ]
    return "\n".join(
        [
            head,
            "",
            readings.table(["", "value"], rows),
            "",
            f"The window's verdict ({level}): {said}",
        ]
    )


#: The box written in before its first turn, and again on New chat (a turn's suggestion
#: is set over it).
INPUT_PLACEHOLDER = "Ask about the data, or what to model… (Enter sends)"

_NEXT = re.compile(r"\s*<!--\s*next:\s*(.*?)\s*-->\s*", re.S | re.I)


def split_next(text: str) -> tuple[str, str | None]:
    """A reply and the next message its model suggests (its ``<!-- next: ... -->`` line,
    the system prompt's), apart: the suggestion goes to the box written in, the reply to
    the panel without it. ``(text, None)`` where the reply gives none."""
    found = list(_NEXT.finditer(text or ""))
    if not found:
        return text, None
    said = " ".join(found[-1].group(1).split()) or None
    return _NEXT.sub("\n", text).strip(), said


def without_next(live: str) -> str:
    """A reply as it is still being written, the suggestion's line kept off the panel --
    whole or begun."""
    text, _ = split_next(live)
    at = text.rfind("<!--")
    return text[:at].rstrip() if at >= 0 and "-->" not in text[at:] else text


def suggested_message(win: MainWindow, called: set[str]) -> str:
    """What the box written in offers once a turn has ended, where the model suggested
    nothing: the window's next step, read off the window as it stands (``called``, the
    tools the model has called)."""
    if win.df_raw is None:
        return "What can you do in this window?"
    if win.runner.busy:
        return "Wait for the step under way, then tell me how it went."
    if win.model is None:
        if "propose_setup" in called:
            return "Train the model with the set-up in the panel."
        return "Look at the data and propose a target and predictors."
    if "normalise" not in win.results:
        return "How well does the model fit? If it is good enough, normalise."
    if "decompose" not in win.results:
        return "What does the normalised series show? Then decompose it."
    return "Summarise what the analysis shows."


def unstamp(text: str) -> str:
    """A reply with the ``[Window now: …]`` line it was sent with taken off its head:
    an agent that repeats its prompt's first line would show it as its own."""
    # Its bracket is not always kept: a model was seen to echo "Window now: ...]" bare.
    return re.sub(r"^\s*\[?Window now:[^\]\n]*\]\s*", "", text)


def state_text(win: MainWindow) -> str:
    """The window as it stands, in a line: told the model with each message."""
    if win.df_raw is None:
        return "no table open"
    parts = [f"table: {win.file_label.text().splitlines()[0]} ({len(win.df_raw):,} rows)"]
    target = win.target_combo.currentText()
    if target:
        parts.append(f"target: {target}")
    feats = [f for f in _checked(win.feature_list) if f != target]
    parts.append(f"predictors ticked: {', '.join(feats) or 'none'}")
    parts.append(f"backend: {win.backend_combo.currentText()}")
    if win.model is not None:
        parts.append(f"model trained on: {', '.join(win.trained_features)}")
    made = [k for k in ("normalise", "decompose", "rolling", "pdp") if k in win.results]
    if made:
        parts.append(f"results: {', '.join(made)}")
    if win.runner.busy:
        parts.append(f"running: {win.runner.task_name}")
    if auto_on():
        parts.append("the Assistant's full automatic is on: what it proposes is applied at once")
    return "; ".join(parts)


def chain_left(win: MainWindow, called: set[str]) -> list[str]:
    """The steps of full automatic's chain (CHAIN) not yet done, read off the window and
    the tools the model has called (``called``); "wait" first where a step is under way,
    and [] where no table is open -- there is nothing to start from."""
    if win.df_raw is None:
        return []
    trained = win.model is not None
    done = {
        "data": trained or "inspect_data" in called,
        "setup": trained or "propose_setup" in called,
        "train": trained,
        "normalise": "normalise" in win.results,
        "decompose": "decompose" in win.results,
    }
    left = [s for s in CHAIN if not done[s]]
    if win.runner.busy:
        left.insert(0, "wait")
    return left


def nudge_text(left: list[str], state: str) -> str:
    """What full automatic tells a model that ended its turn with the chain unfinished."""
    steps = "; ".join(CHAIN_TEXT[s] for s in left)
    return (
        "[Full automatic: the analysis is not finished, and no one will answer -- go on. "
        f"Still to do, in order: {steps}. Window now: {state}. Carry on with the first. "
        "Where a step cannot be done, say why in a line and go to the next; end once every "
        "step is done or cannot be.]"
    )


# -- the conversation ---------------------------------------------------------------------
_MUTED = "color: palette(placeholder-text);"


class _Composer(QPlainTextEdit):
    """The box the analyst writes in: Enter (or Cmd/Ctrl+Enter) sends, Shift+Enter starts
    a new line, and Tab in the empty box fills in the message suggested (``suggestion``,
    shown as its placeholder), as Claude Code's box does."""

    send = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.suggestion = ""

    def keyPressEvent(self, e: QKeyEvent) -> None:  # noqa: N802
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
            e.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.send.emit()
            return
        if e.key() == Qt.Key.Key_Tab and self.suggestion and not self.toPlainText().strip():
            self.setPlainText(self.suggestion)
            self.moveCursor(QTextCursor.MoveOperation.End)
            return
        super().keyPressEvent(e)


class _ChatThread(QThread):
    """One message of the analyst's, worked to the model's reply."""

    event = Signal(object)
    failed = Signal(str)

    def __init__(
        self, model: llm.ChatModel, messages: list[dict[str, Any]], tools: WindowTools
    ) -> None:
        super().__init__()
        self.model, self.messages, self.box = model, messages, tools

    def _on_event(self, e: dict[str, Any]) -> None:
        if e["type"] == "tool" and not e["error"]:
            self.box.called.add(e["name"])
        self.event.emit(e)

    def _note(self, text: str) -> None:
        self.event.emit({"type": "note", "text": text})

    def _left(self) -> list[str]:
        """Full automatic's steps not yet done (chain_left), read on the window's thread."""
        return list(self.box._gui(lambda: chain_left(self.box.win, self.box.called)))

    def _again(self, failures: int, wait: float) -> bool:
        """Full automatic: whether a failed or empty call to the model is made again --
        after ``wait`` seconds, which the analyst's Stop cuts short."""
        if not self.auto or failures > AUTO_RETRIES:
            return False
        self.box.stop.wait(wait)
        return not self.box.stop.is_set()

    def run(self) -> None:
        """The analyst's message worked to the model's reply. With full automatic on (no
        one to answer): a turn the model ends with the chain
        unfinished (chain_left) is told what is left and goes on, until the window stops
        moving (AUTO_STALLS) or AUTO_NUDGES nudges; a failed call to the model, or an
        empty reply, is made again (AUTO_RETRIES)."""
        self.auto = auto_on()
        failures = nudges = stalls = 0
        nudged: list[str] | None = None
        while True:
            try:
                turn = llm.converse(
                    self.model,
                    self.box.tools(),
                    system_prompt(),
                    self.messages,
                    max_turns=MAX_TURNS,
                    on_event=self._on_event,
                    stop=self.box.stop,
                )
                self.event.emit({"type": "usage", "usage": dict(turn.usage)})
            except Exception as exc:  # noqa: BLE001 -- shown in the panel
                failures += 1
                if self._again(failures, AUTO_RETRY_WAIT):
                    self._note(
                        f"The call to the model failed ({type(exc).__name__}: {exc}); full "
                        "automatic tries again."
                    )
                    continue
                # Every provider wants the turns to alternate: the model has the last word.
                if self.messages and self.messages[-1]["role"] != "assistant":
                    self.messages.append({"role": "assistant", "text": "(failed)", "calls": []})
                self.failed.emit(f"{type(exc).__name__}: {exc}")
                return
            if not self.auto or turn.stopped == "stopped":
                return
            last = self.messages[-1] if self.messages else {}
            if turn.stopped == "answered" and not (last.get("text") or "").strip():
                failures += 1
                if self._again(failures, 0.0):
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "[Your reply was empty. Full automatic: go on with "
                            "the analysis, a step at a time.]",
                        }
                    )
                    continue
                return
            left = self._left() if nudges < AUTO_NUDGES else []
            if not left:
                return
            # Told again only while the window moves: the same steps left at two nudges
            # running is a model that cannot do them.
            stalls = stalls + 1 if left == nudged else 0
            if stalls >= AUTO_STALLS:
                return
            nudges += 1
            nudged = left
            state = str(self.box._gui(lambda: state_text(self.box.win)))
            self.messages.append({"role": "user", "content": nudge_text(left, state)})
            self._note(
                "Full automatic: not finished, so it goes on -- left: "
                + ", ".join(CHAIN_WORDS[s] for s in left)
                + "."
            )


class AssistantDock(QDockWidget):
    """The Assistant panel: a model, a conversation, proposals the analyst applies."""

    post_card = Signal(str, str, object, str, str)
    _tool_started = Signal(str, object)
    _tool_ended = Signal(str, str, bool)

    def __init__(self, win: MainWindow) -> None:
        super().__init__("Assistant", win)
        self.setObjectName("assistantDock")
        self.win = win
        self.bridge = _Bridge(self)
        self.tools = WindowTools(win, self)
        self.pending_resample: list[str] = []
        self.messages: list[dict[str, Any]] = []
        #: What the analyst did in the window that the model is told with their next
        #: message (a set-up applied and trained), in brackets ahead of it.
        self.notes: list[str] = []
        #: The model's own suggestion of the next message, this turn's (split_next).
        self._model_next: str | None = None
        #: The tokens the conversation has taken (an API model's).
        self.usage = {"input": 0, "output": 0}
        #: A command-line agent's reply as it is written, whole (its suggestion split off
        #: at the turn's end).
        self._cli_text = ""
        self._thread: _ChatThread | None = None
        self._cli: assistant_cli.CliChat | None = None
        self._server: ToolServer | None = None
        self._open_tools: list[QLabel] = []
        self._retried = False
        #: The agents' entries in the model list: label -> (agent, model).
        self._agents = {label: (agent, model) for label, agent, model in assistant_cli.choices()}

        body = QWidget()
        v = QVBoxLayout(body)
        top = QHBoxLayout()
        self.model_box = QComboBox()
        self.model_box.setEditable(True)
        self.model_box.addItems([*API_MODELS, *self._agents])
        self.model_box.setToolTip(
            "An API model as provider:model (anthropic, openai, deepseek, gemini, ollama, "
            "lmstudio), or a command-line agent signed in to your own plan: Claude Code, "
            "Codex, Antigravity agy or GitHub Copilot."
        )
        self.model_box.setCurrentText(str(_settings().value("assistant/model", API_MODELS[0])))
        self.model_box.currentTextChanged.connect(self._model_changed)
        self.effort_box = QComboBox()
        self.effort_box.setToolTip(
            "How hard the agent thinks: more effort is slower and spends more of your plan."
        )
        self.effort_box.currentTextChanged.connect(self._effort_changed)
        self.key_btn = QPushButton("Key…")
        self.key_btn.setToolTip(
            "Store this provider's API key in the macOS keychain (or for this session)."
        )
        self.key_btn.clicked.connect(self.ask_key)
        # Full automatic: the button's own part is the switch; its menu says what it does.
        self.auto_btn = QToolButton()
        self.auto_btn.setText("Auto")
        self.auto_btn.setCheckable(True)
        self.auto_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.auto_btn.setToolTip(
            "Full automatic: the set-up the Assistant proposes is applied at once, and it "
            "works the analysis through -- data, set-up, training, normalisation, "
            "decomposition -- without waiting for you."
        )
        auto_menu = QMenu(self.auto_btn)
        auto_menu.addAction("What full automatic does…", self._about_auto)
        self.auto_btn.setMenu(auto_menu)
        self.auto_btn.setChecked(auto_on())
        self.auto_btn.toggled.connect(self._auto_chosen)
        # The row is what decides how narrow the dock can go: by default the
        # combos are as wide as their longest entry and Fusion's buttons 80 px,
        # which held the dock at 510 px and the window at that much more.
        for box in (self.model_box, self.effort_box):
            box.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
        self.model_box.setMinimumContentsLength(10)
        self.effort_box.setMinimumContentsLength(4)
        self.key_btn.setMinimumWidth(44)
        top.addWidget(self.model_box, 1)
        top.addWidget(self.effort_box)
        top.addWidget(self.key_btn)
        top.addWidget(self.auto_btn)
        # One height across the row: the Auto button's popup indicator left it a few
        # pixels shorter than Key… and the boxes.
        row_h = max(
            w.sizeHint().height()
            for w in (self.model_box, self.effort_box, self.key_btn, self.auto_btn)
        )
        for w in (self.model_box, self.effort_box, self.key_btn, self.auto_btn):
            w.setFixedHeight(row_h)
        v.addLayout(top)
        self.view = ChatView()
        v.addWidget(self.view, 1)
        self.working = QLabel("Working…")
        self.working.setStyleSheet(_MUTED)
        self.working.hide()
        v.addWidget(self.working)
        # The box the analyst writes in: rounded, growing with what is written, its
        # buttons inside it.
        self.composer_box = QFrame()
        self.composer_box.setObjectName("composer")
        box = QVBoxLayout(self.composer_box)
        box.setContentsMargins(10, 6, 8, 6)
        box.setSpacing(4)
        self.composer = _Composer()
        self.composer.setObjectName("composer_input")
        self.composer.setFrameShape(QFrame.Shape.NoFrame)
        self.composer.setPlaceholderText(INPUT_PLACEHOLDER)
        self.composer.send.connect(self.send)
        self.composer.textChanged.connect(self._fit_input)
        self.composer.installEventFilter(self)
        box.addWidget(self.composer)
        row = QHBoxLayout()
        row.setSpacing(6)
        # What the turn is doing (a step running in the window), and at its end the tokens
        # the conversation has taken.
        self.status = QLabel("")
        self.status.setObjectName("composer_status")
        self.status.setStyleSheet(_MUTED)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self.status, 1)
        self.new_btn = QPushButton("New chat")
        self.new_btn.setObjectName("composer_new")
        self.new_btn.setToolTip("Start a new conversation.")
        self.new_btn.clicked.connect(self.new_conversation)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("composer_stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("composer_send")
        self.send_btn.clicked.connect(self.send)
        for btn in (self.new_btn, self.stop_btn, self.send_btn):
            row.addWidget(btn)
        box.addLayout(row)
        v.addWidget(self.composer_box)
        self.setWidget(body)
        self.setMinimumWidth(300)
        self.view.restyled.connect(self._style_composer)
        self._style_composer(self.view.look)
        # Cmd/Ctrl+F finds in the conversation from anywhere in the panel, the box
        # written in too.
        find = QShortcut(QKeySequence(QKeySequence.StandardKey.Find), body)
        find.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        find.activated.connect(self.view.open_find)
        # The system's appearance changing (light, dark) reaches the conversation through
        # the theme manager: its own style sheet keeps the palette's events from it.
        theme_manager().themeChanged.connect(self._follow_theme)
        # A step the analyst ran ending (a set-up applied and trained): the suggestion
        # follows the window.
        win.runner.finished.connect(self._step_ended)

        self.post_card.connect(self.view.card)
        self._tool_started.connect(self._on_tool_started)
        self._tool_ended.connect(self._on_tool_ended)
        self._sync_key_button()
        self._fill_efforts()
        self.view.note(
            "The Assistant reads what the window holds and runs its steps. It proposes a "
            "target and predictors as a card; nothing is applied until you click Apply."
        )

    # -- the box written in -----------------------------------------------------------
    def _style_composer(self, look: Look) -> None:
        self.composer_box.setStyleSheet(composer_sheet(look))
        self.composer.setFont(look.font())
        self._fit_input()

    def _fit_input(self) -> None:
        """The box as tall as what is written in it: two lines to eight."""
        lines = int(self.composer.document().documentLayout().documentSize().height())
        margin = int(self.composer.document().documentMargin())
        self.composer.setFixedHeight(
            max(2, min(8, lines)) * self.composer.fontMetrics().lineSpacing() + 2 * margin + 4
        )

    def _follow_theme(self, _dark: bool = False) -> None:
        self.view.follow_palette()

    def changeEvent(self, e: QEvent) -> None:  # noqa: N802
        super().changeEvent(e)
        if e.type() == QEvent.Type.PaletteChange and hasattr(self, "view"):
            self.view.follow_palette()

    # -- the suggestion ---------------------------------------------------------------
    def _suggest(self, text: str) -> None:
        """The message offered in the box written in: its placeholder, Tab fills it."""
        self.composer.suggestion = text
        self.composer.setPlaceholderText(text or INPUT_PLACEHOLDER)

    def _step_ended(self) -> None:
        if not self.busy() and self.messages_sent():
            self._say_tokens()
            self._suggest(suggested_message(self.win, self.tools.called))

    def _say_tokens(self) -> None:
        """The tokens the conversation has taken, in the box's status line (none where the
        model is a command-line agent, which does not say)."""
        used = self.usage["input"] or self.usage["output"]
        self.status.setText(
            f"tokens: {self.usage['input']:,} in, {self.usage['output']:,} out" if used else ""
        )

    def messages_sent(self) -> bool:
        """Whether the conversation has begun (a suggestion means something only then)."""
        return bool(self.messages) or self._cli is not None

    def train_applied(self, what: str) -> None:
        """The set-up the analyst applied on a card (*what*), trained at once; the model
        is told with their next message."""
        if self.win.runner.busy:
            self.view.note("A step is running: train once it has ended.")
            self.notes.append(
                f"[The analyst applied the proposed set-up ({what}); a step was running, "
                "so training did not start.]"
            )
            return
        self.win._run_train()
        if not self.win.runner.busy:
            self.view.note("Training did not start: the window says why.")
            self.notes.append(
                f"[The analyst applied the proposed set-up ({what}), but the window did "
                "not start training (it says why on screen).]"
            )
            return
        self.view.note("Training started in the window.")
        self.status.setText(f"{self.win.runner.task_name} running in the window…")
        self.notes.append(
            f"[The analyst applied the proposed set-up ({what}) and training started in "
            "the window: wait_for_run, then model_fit reads it. Do not train it again.]"
        )

    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:  # noqa: N802
        box = getattr(self, "composer", None)  # events come while the dock is built
        if (
            box is not None
            and obj is box
            and ev.type()
            in (
                QEvent.Type.FocusIn,
                QEvent.Type.FocusOut,
            )
        ):
            # The box's edge in the accent colour while it is written in.
            self.composer_box.setProperty("focus", ev.type() == QEvent.Type.FocusIn)
            self.composer_box.style().unpolish(self.composer_box)
            self.composer_box.style().polish(self.composer_box)
        return super().eventFilter(obj, ev)

    # -- the model -------------------------------------------------------------------
    def _is_agent(self) -> bool:
        return self.model_box.currentText() in self._agents

    def _fill_efforts(self) -> None:
        """The efforts the chosen agent's model is offered at, the last one chosen kept;
        hidden for an API model or a model offered at none."""
        agent = self._agents.get(self.model_box.currentText())
        offered = assistant_cli.efforts(*agent) if agent else ()
        self.effort_box.blockSignals(True)
        self.effort_box.clear()
        self.effort_box.addItems(list(offered))
        kept = str(_settings().value("assistant/effort", "medium"))
        self.effort_box.setCurrentText(
            kept if kept in offered else (offered[len(offered) // 2] if offered else "")
        )
        self.effort_box.blockSignals(False)
        self.effort_box.setVisible(bool(offered))

    def _effort_changed(self, effort: str) -> None:
        if effort:
            _settings().setValue("assistant/effort", effort)
        self._close_cli()

    def _sync_key_button(self) -> None:
        name = self.model_box.currentText()
        self.key_btn.setEnabled(not self._is_agent() and name.partition(":")[0] in llm.PROVIDERS)

    def _model_changed(self, name: str) -> None:
        """The analyst chose another model: remembered for the next session."""
        self._sync_key_button()
        self._fill_efforts()
        _settings().setValue("assistant/model", name)
        self._close_cli()

    def ask_key(self) -> None:
        provider = self.model_box.currentText().partition(":")[0]
        if provider not in llm.PROVIDERS:
            return
        key, ok = QInputDialog.getText(
            self, "API key", f"Key for {provider}:", QLineEdit.EchoMode.Password
        )
        if not ok or not key.strip():
            return
        _SESSION_KEYS[provider] = key.strip()
        where = "the macOS keychain" if keychain_store(provider, key.strip()) else "this session"
        self.view.note(f"The {provider} key is kept in {where}.")

    def _model(self) -> llm.Model:
        name = self.model_box.currentText().strip()
        provider = name.partition(":")[0]
        return llm.Model(name, api_key=find_key(provider) if provider in llm.PROVIDERS else None)

    # -- a message -------------------------------------------------------------------
    def busy(self) -> bool:
        if self._thread is not None and self._thread.isRunning():
            return True
        return self._cli is not None and self._cli.running()

    def send(self, text: str | None = None) -> None:
        text = (text if isinstance(text, str) else self.composer.toPlainText()).strip()
        if not text or self.busy():
            return
        self.composer.clear()
        self.view.you(text)
        told = "".join(n + "\n" for n in self.notes)
        self.notes.clear()
        stamped = f"[Window now: {state_text(self.win)}]\n{told}{text}"
        self.tools.stop.clear()
        self._model_next = None
        self.status.setText("Thinking…")
        if self._is_agent():
            self._send_cli(stamped)
            return
        try:
            model = self._model()
        except (ValueError, RuntimeError) as exc:
            self.view.note(str(exc), error=True)
            return
        self.messages.append({"role": "user", "content": stamped})
        self._thread = _ChatThread(model, self.messages, self.tools)
        self._thread.event.connect(self._on_event)
        self._thread.failed.connect(lambda msg: self.view.note(msg, error=True))
        self._thread.finished.connect(self._idle)
        self._working(True)
        self._thread.start()

    def _on_event(self, e: dict[str, Any]) -> None:
        if e["type"] == "note":
            self.view.note(e["text"])
        elif e["type"] == "usage":
            for k in ("input", "output"):
                self.usage[k] += int(e["usage"].get(k, 0) or 0)
        elif e["type"] == "model":
            text, said = split_next(unstamp(e.get("text") or ""))
            if said:
                self._model_next = said
            self.view.reply(text)
            for c in e.get("calls") or []:
                self._open_tools.append(self.view.tool(c["name"], c["arguments"]))
                self._say_running(c["name"])
        elif e["type"] == "tool":
            if self._open_tools:
                self.view.tool_done(self._open_tools.pop(0), e["output"], e["error"])
            self.status.setText("Thinking…")

    def _say_running(self, name: str) -> None:
        """What the turn is doing, in the box's status line: a step of the window's, or a
        reading."""
        steps = ("train", "normalise", "decompose", "wait_for_run")
        self.status.setText(f"{name} running in the window…" if name in steps else f"{name}…")

    def _on_tool_started(self, name: str, arguments: dict[str, Any]) -> None:
        self._open_tools.append(self.view.tool(name, arguments))
        self._say_running(name)

    def _on_tool_ended(self, name: str, output: str, error: bool) -> None:
        if self._open_tools:
            self.view.tool_done(self._open_tools.pop(0), output, error)
        self.status.setText("Thinking…")

    def _working(self, on: bool) -> None:
        self.working.setVisible(on)
        self.send_btn.setEnabled(not on)
        self.stop_btn.setEnabled(on)

    def _idle(self) -> None:
        self._working(False)
        self._open_tools.clear()
        self._say_tokens()
        # The next message offered: the model's, following the conversation (an offer it
        # made, the reply to it); else the window's next step. Tab in the empty box fills
        # it; the next turn's end sets a fresher one.
        self._suggest(self._model_next or suggested_message(self.win, self.tools.called))
        self._model_next = None

    def _auto_chosen(self, on: bool) -> None:
        _settings().setValue(AUTO_KEY, bool(on))
        self.view.note(
            "Full automatic on: what the Assistant proposes is applied at once, and it works "
            "the analysis through."
            if on
            else "Full automatic off: proposals wait for Apply again."
        )

    def _about_auto(self) -> None:
        QMessageBox.information(
            self,
            "Full automatic",
            "With Auto on, the Assistant works without waiting for you:\n\n"
            "• the target and predictors it proposes are applied at once;\n"
            "• it goes through the analysis in the window's order -- the data, the set-up, "
            "training, normalisation, decomposition -- reading each result;\n"
            "• if it stops with steps left, it is told what is left and goes on (up to 8 "
            "times; it stops when two reminders in a row change nothing);\n"
            "• a failed or empty answer from the model is asked again, twice at most.\n\n"
            "Stop ends it at any time. With an API model all of this applies; a command-line "
            "agent has the set-up applied at once and works the chain as its own loop "
            "allows.",
        )

    def stop(self) -> None:
        self.tools.stop.set()
        if self._cli is not None:
            self._cli.stop()

    def new_conversation(self) -> None:
        if self.busy():
            return
        self.messages.clear()
        self.notes.clear()
        self.pending_resample = []
        self.tools.called.clear()
        self.usage = {"input": 0, "output": 0}
        self._model_next = None
        self._close_cli()
        self.view.clear()
        self.status.setText("")
        self._suggest("")

    # -- a command-line agent --------------------------------------------------------
    def _send_cli(self, text: str) -> None:
        if self._cli is None:
            if self._server is None:
                self._server = ToolServer(
                    self.tools.tools,
                    instructions=system_prompt(),
                    on_call=lambda n, a: self._tool_started.emit(n, a),
                    on_result=lambda n, t, err: self._tool_ended.emit(n, t, err),
                )
            agent, model = self._agents[self.model_box.currentText()]
            effort = self.effort_box.currentText() if self.effort_box.count() else ""
            self._cli = assistant_cli.make_chat(agent, model, effort, self._server)
            if self._cli is None:
                command = assistant_cli.AGENTS[agent][0]
                self.view.note(f"'{command}' is not installed, or not on PATH.", error=True)
                return
            self._cli.text.connect(self._cli_text_in)
            self._cli.step.connect(lambda name: self.view.note(f"the agent's own step: {name}"))
            self._cli.failed.connect(lambda msg: self.view.note(msg, error=True))
            self._cli.ended.connect(self._cli_ended)
            self._retried = False
        self._working(True)
        try:
            self._cli.send(text, system_prompt())
        except (OSError, RuntimeError) as exc:
            self.view.note(str(exc), error=True)
            self._idle()

    def _cli_text_in(self, t: str) -> None:
        """The agent's words as they arrive: the reply written so far shown whole, its
        suggestion's line (whole or begun) kept off the panel."""
        if not self.view.writing:
            self._cli_text = ""
        self._cli_text += t
        self.view.reply_live(without_next(unstamp(self._cli_text)))

    def _cli_ended(self) -> None:
        """The agent's message ended: asked once more on its own where its stream broke
        mid-answer (agy), else the panel is idle."""
        cli = self._cli
        if (
            cli is not None
            and cli.interrupted
            and not self._retried
            and not self.tools.stop.is_set()
        ):
            self._retried = True
            self.view.note("The agent's stream was interrupted: asking it to continue.")
            cli.send(assistant_cli.STREAM_INTERRUPTED, system_prompt())
            return
        self._retried = False
        text, said = split_next(unstamp(self._cli_text))
        if said:
            self._model_next = said
            self.view.reply_live(text)
        self._cli_text = ""
        self._idle()

    def _close_cli(self) -> None:
        if self._cli is not None:
            self._cli.close()
            self._cli = None

    def shutdown(self) -> None:
        """The window is closing: stop what runs, close the agent and the server."""
        self.stop()
        self._close_cli()
        if self._server is not None:
            self._server.close()
            self._server = None
        if self._thread is not None:
            self._thread.wait(2000)
