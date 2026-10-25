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

from . import _code, assistant_cli
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
model (model_fit), normalise (normalise), read it (normalised), decompose
(decompose, then decomposition), and test the trend of a series (trend; its
segmented form, where the trend breaks, is trend_breaks).

Which tool does what the analyst asks:
- The window's own steps are tools, each run as its button runs it, with the settings
  the analyst named (and no others): train (backend, budget, split, seed), normalise,
  decompose, rolling (rolling normalisation), multiscale, pdp (partial dependence),
  trend (the trend analysis of a series: Mann-Kendall, Theil-Sen slope, block-bootstrap
  corrected), polar_plot (a variable by wind direction and speed), and the SCM
  Studio's synthetic control: scm_open, scm_fit, scm_placebo, scm_bands. results
  reads every result tab.
- Everything else normet's API does goes to the analyst with carry_out: code on the
  table open, CODE_CONVENTIONS api_help("") lists the API and api_help(name) shows a
  call; write a call from what api_help shows. Do not say normet cannot do something
  before api_help("") has listed what it does.

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
#: Rounds of tool calls one message may take before the model is stopped (full automatic:
#: counted from the last step done, a round that only waits not counted).
MAX_ROUNDS = 30
#: Tokens a reply may take. A model that thinks first spends them on its thinking too:
#: at 4096, a thinking model's answer to a long question came back empty (cume's measure).
MAX_TOKENS = 16384
#: The tool-result rounds the payload keeps whole: older ones fold to their head line --
#: the tokens a long session spends are mostly the full readings it would otherwise carry
#: for good; the panel's conversation is untouched, and a fresh reading one call away.
KEEP_ROUNDS = 2
FOLDED = " ... [the head of an earlier reading; call the same tool once more for the whole]"
#: Readings the payload keeps whole however old in full automatic -- the latest of each:
#: what the set-up, the fit and the reading of the results stand on, read early in a
#: chain and decided on late.
PINNED = ("inspect_data", "model_fit", "normalised", "decomposition", "results")
#: Told to a model whose call's arguments were not JSON: cut off at its output limit.
UNPARSED = (
    "Your call's arguments were not valid JSON -- most likely cut off at your output "
    "limit. Call the tool again with shorter text (reasons in a few sentences)."
)
#: What each reading points at next, said at its tail: the model spends a round finding
#: out only where it does not know.
NEXT_TEXT = {
    "inspect_data": "propose_setup -- the target and the meteorological predictors, with "
    "your reasons (what you left out and why)",
    "model_fit": "where the testing rows fit well (R2, FAC2), normalise; where they do "
    "not, say what to change -- predictors, the budget, the backend",
    "normalised": "decompose: 'emission' splits the normalised series by time scale, "
    "'meteorology' the weather's part by variable; rolling or multiscale show how the "
    "weather's part moves over the record",
    "decomposition": "results reads each tab; trend tests the series for a trend "
    "(Mann-Kendall, Theil-Sen, the block-bootstrap-corrected p where the record is "
    "autocorrelated); pdp shows how the model answers each variable; polar_plot where "
    "the target comes from by the wind",
    "rolling": "multiscale reads the bands between rolling widths (it needs the "
    "normalisation); results reads every tab",
    "pdp": "results reads every tab; polar_plot reads the target by wind direction",
}
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


def _next(key: str) -> str:
    return "\nThen: " + NEXT_TEXT[key] + "."


def system_prompt() -> str:
    """SYSTEM, with full automatic on what it says of proposals waiting for a click
    replaced by what happens instead, and carry_out's language (:mod:`._code`) said."""
    text = SYSTEM.replace(_ASKED, _AUTO) if auto_on() else SYSTEM
    return text.replace("CODE_CONVENTIONS", f"written in {_code.CONVENTIONS}")


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
        #: Each reading's stamp at its last read (_same_once): what it depends on.
        self._readings: dict[str, tuple] = {}

    def _sig(self, key: str) -> tuple:
        """What the reading *key* depends on, stamped at each read: the window's state and
        the very objects read (a step run again makes a new one)."""
        w = self.win
        own = {
            "inspect_data": (id(w.df_raw), w.target_combo.currentText()),
            "model_fit": (id(w.model),),
            "normalised": (id(w.results.get("normalise")),),
            "decomposition": (id(w.results.get("decompose")),),
        }.get(key, ())
        return (*state_parts(w), *own)

    def _same_once(self, key: str) -> str | None:
        """The short answer when the reading is as it was at its last one (None else),
        given once: the whole is in the rounds the model already has, and asking once more
        reads it out whole. A step under way is no replay: the reading moves."""
        w = self.win
        if w.runner.busy:
            return None
        sig = self._sig(key)
        if self._readings.get(key) != sig:
            self._readings[key] = sig
            return None
        self._readings.pop(key, None)  # the next asking comes whole again
        return (
            "Unchanged since you last read it. The whole reading is in your earlier rounds "
            "while they last: one more call and it comes whole."
        )

    def _reading(self, key: str, read: Callable[[], str]) -> str:
        """A reading of the window's, said short where unchanged since the last (once), its
        pointer to the next step at its tail."""

        def go() -> str:
            same = self._same_once(key)
            if same is not None:
                return same
            return read() + (_next(key) if key in NEXT_TEXT else "")

        return str(self._gui(go))

    #: Each tool, and what the window must hold for it to be offered: one with nothing to
    #: act on is not offered, so not called. Read again at every round of a message: a
    #: step a tool runs brings its own.
    OFFERED = (
        ("window", None),
        ("inspect_data", "data"),
        ("model_fit", "model"),
        ("normalised", "normalised"),
        ("decomposition", "decomposed"),
        ("wait_for_run", None),
        ("log", None),
        ("propose_setup", "data"),
        ("train", "data"),
        ("normalise", "model"),
        ("decompose", "model"),
        ("trend", "decomposed"),
        ("trend_breaks", "decomposed"),
        ("rolling", "model"),
        ("multiscale", "normalised"),
        ("pdp", "model"),
        ("polar_plot", "data"),
        ("results", None),
        ("scm_open", None),
        ("scm_fit", "panel"),
        ("scm_placebo", "scm"),
        ("scm_bands", "scm"),
        ("api_help", None),
        ("carry_out", None),
    )
    #: Why a tool is not offered, as a model calling it anyway is told.
    WANTING = {
        "data": "no table is open",
        "model": "no model is trained: train first",
        "normalised": "nothing is normalised: normalise first",
        "decomposed": "nothing is decomposed: decompose first",
        "panel": "the SCM Studio has no panel open: scm_open first",
        "scm": "no synthetic control is fitted: scm_fit first",
    }
    #: Where a request goes that a tool's own arguments do not cover.
    BEYOND = {
        "train": "a model with any other setting is normet's build_model in carry_out "
        "(api_help('build_model'))",
        "normalise": "a normalisation with any other setting is normet's normalise in "
        "carry_out (api_help('normalise'))",
        "decompose": "a decomposition with any other setting is normet's decompose in "
        "carry_out (api_help('decompose'))",
        "trend": "a trend of any other series is normet's trend in carry_out (api_help('trend'))",
        "trend_breaks": (
            "a segment split of any other series is normet's detect_trend_breaks in "
            "carry_out (api_help('detect_trend_breaks'))"
        ),
        "api_help": "name is one of normet's calls, as api_help('') lists them",
        "carry_out": "code is normet's calls on the table open; reasons are said to the analyst",
    }
    #: Arguments a model names otherwise, taken as the tool's own.
    SYNONYMS = {
        "carry_out": {"call": "code", "calls": "code", "script": "code", "reason": "reasons"},
        "api_help": {
            "topic": "name",
            "query": "name",
            "symbol": "name",
            "function": "name",
            "call": "name",
        },
        "normalise": {"variables": "resample_vars", "variables_resample": "resample_vars"},
        "pdp": {"var_list": "variables", "vars": "variables"},
        "polar_plot": {"var": "variable", "pollutant": "variable", "stat": "statistic"},
        "train": {"n_trials": "budget", "time_budget": "budget", "split": "split_method"},
        "propose_setup": {"predictors": "features", "covariates": "features", "reason": "reasons"},
    }
    #: What an agent writes about its own call into the call's arguments, spelt without
    #: case or underscores -- dropped, not refused.
    ENVELOPE = {"servername", "toolname", "toolaction", "toolsummary"}
    #: Each argument said in its schema, by (tool, argument) or by argument.
    ARG_TEXT: dict[Any, str] = {
        "reasons": "your reasons, in the analyst's words, shown on the card",
        ("propose_setup", "confidence"): "how sure you are, from 0 to 1 (0.7 = fairly sure)",
        ("propose_setup", "features"): "the predictors: the table's meteorological columns",
        ("api_help", "name"): "one of normet's calls, e.g. 'rolling'; '' lists them",
        ("carry_out", "code"): "normet's calls on the table open (df), as api_help shows "
        "them, printing what the analyst should read",
        ("train", "budget"): "the search budget: seconds or trials, as the panel says",
        ("trend", "series"): "'emission' (the decomposition's emission part), 'met' "
        "(its weather part), or 'observed' (the target as recorded)",
        ("trend", "seasonal"): "'auto' (the window's own choice), 'none' (the "
        "classical test), or '24' (hourly), '7' (daily), '12' (monthly) calendar "
        "periods",
        ("trend_breaks", "series"): "'long-term' (the separated long-term trend, "
        "the default -- the cycles already taken out), 'emission' (its emission "
        "part), 'met' (its weather part) or 'observed' (the target as recorded)",
        ("trend_breaks", "segments"): "2 to 4: the most segments the fit may split "
        "the record into (it stops earlier where a split earns no keep)",
        ("scm_fit", "cutoff"): "the intervention date, YYYY-MM-DD",
    }

    def _holds(self) -> set[str]:
        w = self.win
        have = set()
        if w.df_raw is not None:
            have.add("data")
        if w.model is not None:
            have.add("model")
        if "normalise" in w.results:
            have.add("normalised")
        if "decompose" in w.results:
            have.add("decomposed")
        scm = getattr(w, "_scm_window", None)
        if scm is not None and getattr(scm, "df", None) is not None:
            have.add("panel")
            if getattr(scm, "result", None) is not None:
                have.add("scm")
        return have

    def check_arguments(self, t: llm.Tool, args: dict[str, Any]) -> str | None:
        """What is wrong with a call's arguments against the tool's own, said so the model
        can write it again -- None where nothing is. A synonym of an argument (SYNONYMS)
        is renamed in *args* to the argument it stands for; the agent's own account of
        the call (ENVELOPE) is dropped from *args*."""
        import difflib

        props = t.schema.get("properties", {})
        for k in [
            k for k in args if k not in props and k.lower().replace("_", "") in self.ENVELOPE
        ]:
            del args[k]
        for other, own in self.SYNONYMS.get(t.name, {}).items():
            if other in args and own not in args:
                args[own] = args.pop(other)
        need = t.schema.get("required", [])
        unknown = [k for k in args if k not in props]
        missing = [k for k in need if k not in args]
        if not unknown and not missing:
            return None
        said = []
        for k in unknown:
            close = difflib.get_close_matches(k, list(props), n=1)
            said.append(
                f"{k!r} is not one of its arguments"
                + (f" (did you mean {close[0]!r}?)" if close else "")
            )
        if missing:
            said.append("it needs " + ", ".join(missing))
        takes = ", ".join(p + ("" if p in need else " (optional)") for p in props) or "nothing"
        return (
            f"{t.name} takes {takes}: "
            + "; ".join(said)
            + "."
            + (f" {self.BEYOND[t.name]}." if t.name in self.BEYOND else "")
        )

    def not_offered(self, name: str) -> str:
        """Why a tool was not found, as the model calling it is told."""
        need = dict(self.OFFERED).get(name, "")
        if name in dict(self.OFFERED) and need not in self._gui(self._holds):
            return f"{name} is not offered now: {self.WANTING.get(need, need)}"
        return f"no tool {name!r}"

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
        self,
        start: Callable[[], None],
        marker: Callable[[], Any],
        describe: Callable[[], str],
        window: Any = None,
    ) -> str:
        """Run a step of the window's (``start``, on its thread) to its end, as its
        button does, and say what it made (``describe``); ``marker`` is what the
        step replaces when it succeeds. ``window``: the window whose runner runs it
        (a studio's), the main window's where left out."""
        ended = threading.Event()
        box: dict[str, Any] = {}

        def begin() -> None:
            runner = (window or self.win).runner
            if runner.busy:
                box["error"] = "the window is running something: wait for it to end"
                ended.set()
                return
            box["before"] = marker()
            try:
                start()
            except Exception as exc:  # noqa: BLE001 -- raised again in the tool's thread
                # A setting the panel refuses: said to the model, and the tool's thread
                # not left waiting for a step that never started.
                box["exc"] = exc
                ended.set()
                return
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
        if "exc" in box:
            raise box["exc"]
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

        return self._reading("inspect_data", read)

    def model_fit(self) -> str:
        """How well the trained model predicts: normet.modStats on the rows held out
        ('testing'), the training rows and all, and its feature importance."""

        def read() -> str:
            if self.win.model is None or self.win.df_prep is None:
                raise ValueError("no model is trained in the window yet")
            if self.win._is_chronos():
                return chronos_reading(self.win)
            return readings.model_reading(self.win.df_prep, self.win.model)

        return self._reading("model_fit", read)

    def normalised(self) -> str:
        """The normalised series on screen against the observed one: means overall
        and by year or month, and the meteorological part (observed − normalised)."""

        def read() -> str:
            result = self.win.results.get("normalise")
            if result is None:
                raise ValueError("nothing is normalised in the window yet")
            return readings.normalised_reading(result)

        return self._reading("normalised", read)

    def decomposition(self) -> str:
        """The decomposition on screen: each component's mean, spread and range."""

        def read() -> str:
            result = self.win.results.get("decompose")
            if result is None:
                raise ValueError("nothing is decomposed in the window yet")
            return readings.decomposition_reading(result)

        return self._reading("decomposition", read)

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
    def train(
        self,
        backend: str | None = None,
        budget: int | None = None,
        split_method: str | None = None,
        train_fraction: float | None = None,
        seed: int | None = None,
    ) -> str:
        """Train the model with the window's panel (target, predictors) -- once the
        analyst has agreed the set-up. Each setting given is set in the panel first, as
        the analyst would: backend (one the Backend list offers), budget (its search
        budget: seconds or trials, as the panel says), split_method (random, ts,
        month_ts, season_ts), train_fraction (0.1 to 0.95), seed. Returns the fit."""

        def start() -> None:
            w = self.win
            if backend is not None:
                _choose(w.backend_combo, backend, "backend")
            if budget is not None:
                w.budget_spin.setValue(int(budget))
            if split_method is not None:
                _choose(w.split_combo, split_method, "split_method")
            if train_fraction is not None:
                w.train_fraction_spin.setValue(float(train_fraction))
            if seed is not None:
                w.seed_spin.setValue(int(seed))
            w._run_train()

        return self._step(start, lambda: self.win.model, self._fit_now)

    def _fit_now(self) -> str:
        if self.win._is_chronos():
            return chronos_reading(self.win)
        return readings.model_reading(self.win.df_prep, self.win.model)

    def normalise(
        self,
        resample_vars: list[str] | None = None,
        n_samples: int | None = None,
        quantiles: bool | None = None,
    ) -> str:
        """Normalise with the trained model, resampling resample_vars (the set-up the
        analyst applied, or the panel's ticks, where left out), with n_samples
        Monte-Carlo draws and (quantiles) the 5-95 % band, where given. Returns the
        reading."""

        def start() -> None:
            names = resample_vars or self.dock.pending_resample
            if names:
                _set_checked(self.win.norm_vars, list(names))
            if n_samples is not None:
                self.win.norm_samples.setValue(int(n_samples))
            if quantiles is not None:
                self.win.norm_quantiles.setChecked(bool(quantiles))
            self.win._run_normalise()

        return self._step(
            start,
            lambda: self.win.results.get("normalise"),
            lambda: readings.normalised_reading(self.win.results["normalise"]),
        )

    def decompose(self, method: str = "emission", n_samples: int | None = None) -> str:
        """Decompose with the trained model: method 'emission' splits the normalised
        series into time scales; 'meteorology' splits observed − normalised by
        meteorological variable; n_samples, its draws, where given. Returns the
        reading."""
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
            if n_samples is not None:
                self.win.decom_samples.setValue(int(n_samples))
            self.win._run_decompose()

        return self._step(
            start,
            lambda: self.win.results.get("decompose"),
            lambda: readings.decomposition_reading(self.win.results["decompose"]),
        )

    def trend(self, series: str = "emission", seasonal: str = "auto") -> str:
        """The window's trend analysis (Mann-Kendall, Theil-Sen slope with its interval,
        and the block-bootstrap-corrected p where the record is autocorrelated) on
        series 'emission' (the decomposition's emission part), 'met' (its weather
        part) or 'observed' (the target as recorded); seasonal is 'auto', 'none' or
        the calendar period in steps of the record ('24', '7', '12'). Returns the
        verdict -- which p stands for the trend."""
        if series not in ("emission", "met", "observed"):
            raise ValueError("series is 'emission', 'met' or 'observed'")
        if seasonal not in ("auto", "none", "24", "7", "12"):
            raise ValueError("seasonal is 'auto', 'none', '24', '7' or '12'")
        if series == "emission" and self._gui(self.win._is_chronos):
            raise ValueError(
                "the emission part does not exist on chronos-2: its decomposition "
                "cannot make it. Take series='observed' instead, or 'met' once the "
                "meteorological decomposition has run."
            )

        def start() -> None:
            # The single segment: this tool is the one-slope test (its segments go
            # to trend_breaks), so a Segments row left higher is brought back.
            self.win.tre_segments.setValue(1)
            self.win.trend_series.setCurrentText(
                {
                    "emission": "emission part (emi_total)",
                    "met": "meteorology part (met_total)",
                    "observed": "observed target",
                }[series]
            )
            self.win.tre_seasonal.setCurrentText(seasonal)
            self.win._run_trend()

        return self._step(
            start,
            lambda: self.win.results.get("trend"),
            lambda: readings.trend_reading(
                self.win._trend_raw, name=self.win._trend_name or "series"
            ),
        )

    def trend_breaks(
        self, series: str = "long-term", segments: int = 2, n_bootstrap: int | None = None
    ) -> str:
        """The window's segmented trend analysis (normet.detect_trend_breaks): where the
        trend breaks, if it does, each segment with its own Theil-Sen slope and
        interval, and the block-bootstrap p of the search. series is 'long-term'
        (the decomposition's separated long-term component -- the cleaner input,
        the default), 'emission' (its emission part), 'met' (its weather part) or
        'observed' (the target as recorded); segments is 2 to 4; n_bootstrap the
        replicates of the search under the null. Returns the verdict -- where the
        break is, and which p stands for it."""
        if series not in ("long-term", "emission", "met", "observed"):
            raise ValueError("series is 'long-term' (the default), 'emission', 'met' or 'observed'")
        if not 2 <= int(segments) <= 4:
            raise ValueError("segments is 2 to 4")
        if series in ("emission", "long-term") and self._gui(self.win._is_chronos):
            raise ValueError(
                "the emission part and its separated long-term trend do not exist "
                "on chronos-2: its decomposition cannot make them. Take "
                "series='observed' instead, or 'met' once the meteorological "
                "decomposition has run."
            )

        def start() -> None:
            # Segments first, then the series: raising the Segments row may move a
            # default emission-part choice to the long-term one, and the series
            # named here must be the one that stands.
            self.win.tre_segments.setValue(int(segments))
            self.win.trend_series.setCurrentText(
                {
                    "long-term": "long-term trend (separated)",
                    "emission": "emission part (emi_total)",
                    "met": "meteorology part (met_total)",
                    "observed": "observed target",
                }[series]
            )
            if n_bootstrap is not None:
                self.win.tre_n_bootstrap.setValue(int(n_bootstrap))
            self.win._run_trend()

        return self._step(
            start,
            lambda: self.win.results.get("trend"),
            lambda: readings.trend_breaks_reading(
                self.win._trend_raw, name=self.win._trend_name or "series"
            ),
        )

    def rolling(
        self,
        window_days: int | None = None,
        step_days: int | None = None,
        n_samples: int | None = None,
    ) -> str:
        """Rolling normalisation with the trained model: the record normalised window by
        window (window_days long, a new one every step_days, n_samples draws each, where
        given), resampling the Normalise ticks. Returns the reading."""

        def start() -> None:
            w = self.win
            if window_days is not None:
                w.roll_window.setValue(int(window_days))
            if step_days is not None:
                w.roll_step.setValue(int(step_days))
            if n_samples is not None:
                w.roll_samples.setValue(int(n_samples))
            w._run_rolling()

        return self._step(
            start,
            lambda: self.win.results.get("rolling"),
            lambda: readings.rolling_reading(self.win.results["rolling"]) + _next("rolling"),
        )

    def multiscale(
        self,
        fast_days: int | None = None,
        meso_days: int | None = None,
        slow_days: int | None = None,
    ) -> str:
        """Multi-scale decomposition (needs the normalisation): rolling normalisations at
        three window widths (fast, meso, slow, in days, where given) and the bands
        between them. Returns what the Multi-scale tab says."""

        def start() -> None:
            w = self.win
            for spin, days in (
                (w.ms_fast, fast_days),
                (w.ms_meso, meso_days),
                (w.ms_slow, slow_days),
            ):
                if days is not None:
                    spin.setValue(int(days))
            w._run_multiscale()

        return self._step(
            start,
            lambda: getattr(self.win, "_multiscale_raw", None),
            lambda: _tab_says(self.win.tab_multiscale),
        )

    def pdp(self, variables: list[str] | None = None, grid_points: int | None = None) -> str:
        """Partial dependence of the trained model on variables (the PDP list's ticks
        where left out), at grid_points values each. Returns the reading."""

        def start() -> None:
            w = self.win
            if variables:
                unknown = _set_checked(w.pdp_vars, list(variables))
                if unknown:
                    raise ValueError(f"the model has no variable {', '.join(unknown)}")
            if grid_points is not None:
                w.pdp_grid.setValue(int(grid_points))
            w._run_pdp()

        return self._step(
            start,
            lambda: self.win.results.get("pdp"),
            lambda: readings.pdp_reading(self.win.results["pdp"]) + _next("pdp"),
        )

    def polar_plot(self, variable: str | None = None, statistic: str = "mean") -> str:
        """The polar plot (Analysis > Polar Plot) of variable (the target where left out)
        by wind direction and speed, statistic mean, median, p95 or max: opened for the
        analyst, and read here as the variable's mean by wind direction (8 sectors) and
        by wind speed."""
        from .polar_window import STATISTICS, wind_columns

        stats = [s for _label, s in STATISTICS]
        if statistic not in stats:
            raise ValueError(f"statistic is one of {', '.join(stats)}")

        def show() -> str:
            w = self.win
            if w.df_raw is None:
                raise ValueError("no table is open in the window")
            frame = w.df_raw.set_index("date") if "date" in w.df_raw.columns else w.df_raw
            frame, ws, wd = wind_columns(frame)
            if ws is None or wd is None:
                raise ValueError(
                    "the table has no wind speed and direction (ws, wd, or u10 and v10)"
                )
            var = variable or w.target_combo.currentText()
            if var not in frame.columns:
                raise ValueError(f"the table has no column {var}")
            w.open_polar_plot()
            pw = getattr(w, "_polar_window", None)
            if pw is not None:
                _choose(pw.var_combo, var, "variable")
                pw.stat_combo.setCurrentIndex(stats.index(statistic))
            return _wind_reading(frame, var, ws, wd)

        return str(self._gui(show))

    def results(self) -> str:
        """What every result tab of the window says now (its verdict and notes), the
        steps not yet run left out."""
        return str(self._gui(lambda: _all_tabs_say(self.win)))

    # the SCM Studio
    def _scm(self) -> Any:
        w = getattr(self.win, "_scm_window", None)
        if w is None or w.df is None:
            raise ValueError("the SCM Studio has no panel open: scm_open first")
        return w

    def scm_open(self, example: str = "") -> str:
        """Open the SCM Studio (synthetic control): with example 'ulez' or 'synthetic',
        that example panel; else the main window's table, where it is a panel (a unit
        column repeated over dates). Returns the design it shows."""

        def opened() -> str:
            self.win.open_scm_studio()
            w = self.win._scm_window
            if example:
                w.load_example(example)
            elif w.df is None and getattr(w, "_main_df", None) is not None:
                if w._panel_candidate(w._main_df) is None:
                    raise ValueError(
                        "the table open is not a panel: no unit column repeats over dates"
                    )
                w._use_main_data()
            if w.df is None:
                raise ValueError("no panel: give example='ulez', or open a panel in the SCM Studio")
            return _scm_design(w)

        return str(self._gui(opened))

    def scm_fit(
        self,
        treated: str | None = None,
        cutoff: str | None = None,
        backend: str | None = None,
        donors: list[str] | None = None,
    ) -> str:
        """Fit the synthetic control in the SCM Studio: the treated unit, the cutoff date
        (YYYY-MM-DD), the backend (one its Backend list offers) and the donors ticked,
        where given. Returns its fit and what the tab says."""

        def start() -> None:
            w = self._scm()
            if treated is not None:
                _choose(w.treated_combo, treated, "treated")
            if cutoff is not None:
                from PySide6.QtCore import QDate

                d = QDate.fromString(str(cutoff)[:10], "yyyy-MM-dd")
                if not d.isValid():
                    raise ValueError("cutoff is a date, YYYY-MM-DD")
                w.cutoff_edit.setDate(d)
            if backend is not None:
                _choose(w.backend_combo, backend, "backend")
            if donors:
                unknown = _set_checked(w.donor_list, list(donors))
                if unknown:
                    raise ValueError(f"the panel has no unit {', '.join(unknown)}")
            w._run_fit()

        w = self._gui(self._scm)
        return self._step(
            start, lambda: getattr(w, "result", None), lambda: _scm_fit_reading(w), window=w
        )

    def scm_placebo(self, kind: str = "space") -> str:
        """Placebo tests of the fit in the SCM Studio: kind 'space' (every donor as if
        treated) or 'time' (the cutoff moved back). Returns what the tab says."""
        if kind not in ("space", "time"):
            raise ValueError("kind is 'space' or 'time'")
        w = self._gui(self._scm)
        run = w._run_placebo_space if kind == "space" else w._run_placebo_time
        tab = w.tab_pspace if kind == "space" else w.tab_ptime
        return self._step(run, lambda: _tab_says(tab), lambda: _tab_says(tab), window=w)

    def scm_bands(self, method: str | None = None) -> str:
        """Uncertainty bands around the effect in the SCM Studio, by method (one its
        Uncertainty list offers, as jackknife), where given. Returns what the tab says."""
        w = self._gui(self._scm)

        def start() -> None:
            if method is not None:
                _choose(w.unc_method, method, "method")
            w._run_uncertainty()

        return self._step(
            start, lambda: _tab_says(w.tab_unc), lambda: _tab_says(w.tab_unc), window=w
        )

    # the rest of normet's API
    def api_help(self, name: str = "") -> str:
        """normet's API, for what carry_out runs: name="" lists its calls, a line each;
        a name shows the call in full (its arguments and what it does)."""
        own = {n for n, _need in self.OFFERED}
        if name.strip() in own:
            return (
                f"{name.strip()} is not a call for carry_out: it is a tool of yours -- call it "
                "as your tool list describes it."
            )
        return _code.api_help(name)

    def carry_out(self, code: str, reasons: str) -> str:
        """What the window's own steps do not do, from the rest of normet's API --
        written as code (the system prompt says in what language, and what is bound:
        the table open as df), run on the window's table. It is shown to the analyst as a
        card with your reasons and runs when they click Run; full automatic runs it at
        once. Returns what it printed and the tables it made (or that it waits for the
        analyst)."""
        if not code.strip():
            raise ValueError("no code")
        body = f"```\n{code.strip()}\n```\n\n{reasons}"

        def inputs() -> tuple[Any, Any, Any]:
            return self.win.df_raw, self.win.df_prep, self.win.model

        if auto_on():
            self.dock.post_card.emit("Run in the window", body, lambda: "Ran", "Ran", "Run")
            df, df_prep, model = self._gui(inputs)
            said = _code.run_code(code, df, df_prep, model)
            return "Ran at once (full automatic is on).\n\n" + said

        def run() -> str:
            self.dock.run_code(code)
            return "Running…"

        self.dock.post_card.emit("Run in the window", body, run, "", "Run")
        return (
            "Shown to the analyst as a card; Run carries it out on the table, and their "
            "next message says what it gave. Stop here and let them answer."
        )

    def tools(self) -> list[llm.Tool]:
        """The tools the window offers now (OFFERED), each argument described."""
        have = self._gui(self._holds)
        offered = {n for n, need in self.OFFERED if need is None or need in have}
        out = [t for t in self.all_tools() if t.name in offered]
        for t in out:
            for arg, schema in t.schema.get("properties", {}).items():
                text = self.ARG_TEXT.get((t.name, arg)) or self.ARG_TEXT.get(arg)
                if text:
                    schema["description"] = text
        return out

    def all_tools(self) -> list[llm.Tool]:
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
                self.trend,
                self.trend_breaks,
                self.rolling,
                self.multiscale,
                self.pdp,
                self.polar_plot,
                self.results,
                self.scm_open,
                self.scm_fit,
                self.scm_placebo,
                self.scm_bands,
                self.api_help,
                self.carry_out,
            )
        ]


def _choose(combo: Any, text: str, what: str) -> None:
    """*text* chosen in a list box, or a ValueError naming what it offers."""
    offered = [combo.itemText(i) for i in range(combo.count())]
    if text not in offered:
        raise ValueError(f"{what} is one of {', '.join(offered)}, not {text!r}")
    combo.setCurrentText(text)


def _tab_says(tab: Any) -> str:
    """What a result tab shows in words: its verdict, notes and table, as text."""
    from PySide6.QtWidgets import QLabel, QTableWidget

    lines = [
        lbl.text().strip()
        for lbl in tab.findChildren(QLabel)
        if lbl.isVisibleTo(tab) and lbl.text().strip() and "<" not in lbl.text()
    ]
    for tw in tab.findChildren(QTableWidget):
        if not tw.isVisibleTo(tab) or not tw.rowCount():
            continue
        header = [
            (tw.horizontalHeaderItem(c).text() if tw.horizontalHeaderItem(c) else "")
            for c in range(tw.columnCount())
        ]
        rows = [
            [(tw.item(r, c).text() if tw.item(r, c) else "") for c in range(tw.columnCount())]
            for r in range(min(tw.rowCount(), 30))
        ]
        lines += ["", readings.table(header, rows)]
    return "\n".join(lines) or "The tab shows nothing in words yet."


def _all_tabs_say(win: MainWindow) -> str:
    out = []
    for i in range(win.tabs.count()):
        tab = win.tabs.widget(i)
        said = _tab_says(tab)
        if said and not said.startswith("The tab shows nothing"):
            out.append(f"## {win.tabs.tabText(i)}\n{said}")
    return "\n\n".join(out) or "No result tab says anything yet: nothing has run."


def _wind_reading(frame: Any, var: str, ws: str, wd: str) -> str:
    """*var*'s mean by wind direction (8 sectors) and by wind speed (quartiles)."""
    import pandas as pd

    d = frame[[var, ws, wd]].apply(pd.to_numeric, errors="coerce").dropna()
    if d.empty:
        raise ValueError(f"no rows have {var}, {ws} and {wd} all")
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    sector = pd.Categorical(
        [names[int(((x % 360) + 22.5) // 45) % 8] for x in d[wd]], categories=names
    )
    by_dir = d.groupby(sector, observed=False)[var].agg(["mean", "count"])
    speed = pd.qcut(d[ws], 4, duplicates="drop")
    by_speed = d.groupby(speed, observed=True)[var].mean()
    share = 100.0 * by_dir["count"] / by_dir["count"].sum()
    return "\n".join(
        [
            f"{var} by wind direction ({len(d):,} rows with wind; the polar plot is open):",
            "",
            readings.table(
                ["from", f"mean {var}", "% of hours"],
                [[k, r["mean"], s] for (k, r), s in zip(by_dir.iterrows(), share, strict=True)],
            ),
            "",
            f"{var} by wind speed ({ws}, quartiles):",
            "",
            readings.table(
                [f"{ws} range", f"mean {var}"],
                [[f"{iv.left:.2f}–{iv.right:.2f}", v] for iv, v in by_speed.items()],
            ),
        ]
    )


def _scm_design(w: Any) -> str:
    """The SCM Studio's design as it stands: the panel, the treated unit, the cutoff."""
    n_units = w.df[w.unit_combo.currentText()].nunique() if w.unit_combo.currentText() else 0
    donors = _checked(w.donor_list)
    return (
        f"The SCM Studio has a panel of {len(w.df):,} rows, {n_units} units; outcome "
        f"{w.outcome_combo.currentText()}, unit column {w.unit_combo.currentText()}, date column "
        f"{w.date_combo.currentText()}. Treated unit {w.treated_combo.currentText()}, cutoff "
        f"{w.cutoff_edit.date().toString('yyyy-MM-dd')}, {len(donors)} donors ticked, backend "
        f"{w.backend_combo.currentText()} (offered: "
        + ", ".join(w.backend_combo.itemText(i) for i in range(w.backend_combo.count()))
        + ")."
    )


def _scm_fit_reading(w: Any) -> str:
    d = w.diagnostics or {}
    keys = ("pre_r2", "pre_rmse", "att", "effective_n_donors", "n_donors", "top_donors")
    rows = [[k, d.get(k)] for k in keys if k in d]
    head = readings.table(["diagnostic", "value"], rows) if rows else ""
    return "\n".join(s for s in (head, _tab_says(w.tab_fit)) if s)


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
    return "; ".join(state_parts(win))


def state_brief(parts: list[str]) -> str:
    """The line sent when the window is as it was at the message before: its table alone,
    said as unchanged -- repeating the whole state would spend its tokens telling the
    model what it knows."""
    return f"as before -- {parts[0]}; unchanged since the last message"


def state_parts(win: MainWindow) -> list[str]:
    """What the window holds as a message is sent, in parts (state_text joins them;
    state_brief says a message's state apart from the one before)."""
    if win.df_raw is None:
        return ["no table open"]
    parts = [f"table: {win.file_label.text().splitlines()[0]} ({len(win.df_raw):,} rows)"]
    left = 0 if win.df_loaded is None else len(win.df_loaded) - len(win.df_raw)
    if left:
        parts[0] += f", {left:,} rows left out of every step (Data tab, Exclude)"
    target = win.target_combo.currentText()
    if target:
        parts.append(f"target: {target}")
    feats = [f for f in _checked(win.feature_list) if f != target]
    parts.append(f"predictors ticked: {', '.join(feats) or 'none'}")
    parts.append(f"backend: {win.backend_combo.currentText()}")
    if win.model is not None:
        parts.append(f"model trained on: {', '.join(win.trained_features)}")
    made = [k for k in ("normalise", "decompose", "rolling", "pdp", "trend") if k in win.results]
    if made:
        from . import uilang
        from ._shell import report_language

        parts.append(f"results: {', '.join(made)}")
        # The analyst picks it under File > Export HTML Report.
        parts.append(f"the report in {uilang.LANGUAGES[report_language()]}")
    if win.runner.busy:
        parts.append(f"running: {win.runner.task_name}")
    scm = getattr(win, "_scm_window", None)
    if scm is not None and getattr(scm, "df", None) is not None:
        parts.append(
            "SCM Studio: a panel open"
            + (", a fit made" if getattr(scm, "result", None) is not None else "")
        )
    if auto_on():
        parts.append("the Assistant's full automatic is on: what it proposes is applied at once")
    return parts


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

    def _mend(self) -> None:
        """A reply with neither text nor a tool call is refused when sent back ("content
        or tool_calls must be set"): kept in words instead."""
        for m in self.messages:
            if (
                m["role"] == "assistant"
                and not m.get("calls")
                and not (m.get("text") or "").strip()
            ):
                m["text"] = "(empty reply)"

    def _close(self, text: str) -> None:
        # Every provider wants the turns to alternate: the model has the last word.
        if self.messages and self.messages[-1]["role"] != "assistant":
            self.messages.append({"role": "assistant", "text": text, "calls": []})

    def _payload(self) -> list[dict[str, Any]]:
        """``self.messages`` as the next call gets it: rounds older than the last
        KEEP_ROUNDS have their tool results folded to the head line (FOLDED) and their
        reasoning left out. A long session otherwise carries every full reading to every
        round -- the largest thing it spends; the panel's conversation is not touched,
        and a fresh reading is one call of the same tool away. Full automatic keeps the
        latest of each PINNED reading whole."""
        kept: int | None = None
        rounds = 0
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i]["role"] == "assistant":
                rounds += 1
                if rounds > KEEP_ROUNDS:
                    kept = i
                    break
        if kept is None:
            return self.messages
        pinned = self._pinned() if self.auto else set()
        out: list[dict[str, Any]] = []
        for pos, m in enumerate(self.messages):
            if pos >= kept:
                out.append(m)
            elif m["role"] == "assistant":
                out.append({k: v for k, v in m.items() if k != "reasoning"})
            elif m["role"] == "tool":
                results = []
                for r in m.get("results", []):
                    r = dict(r)
                    text = r.get("text") or ""
                    if len(text) > len(FOLDED) + 60 and r.get("id") not in pinned:
                        r["text"] = text[:160].rstrip() + FOLDED
                    results.append(r)
                out.append({"role": "tool", "results": results})
            else:
                out.append(dict(m))
        return out

    def _pinned(self) -> set[str]:
        """The ids of the results kept whole however old (full automatic): the latest of
        each PINNED reading that did not fail."""
        name_of: dict[str, str] = {}
        for m in self.messages:
            if m["role"] == "assistant":
                for c in m.get("calls") or []:
                    name_of[c.id] = c.name
        latest: dict[str, str] = {}
        for m in self.messages:
            if m["role"] == "tool":
                for r in m.get("results", []):
                    name = name_of.get(r.get("id", ""))
                    if name in PINNED and not r.get("error"):
                        latest[name] = r["id"]
        return set(latest.values())

    def _call(self, c: llm.Call, by_name: dict[str, llm.Tool]) -> tuple[str, bool]:
        """A call of the model's carried out: its arguments checked first (renamed where a
        synonym, refused where wrong, said so the model can write it again)."""
        t = by_name.get(c.name)
        if t is None:
            check = getattr(self.box, "not_offered", None)
            return (check(c.name) if check else f"no tool {c.name!r}"), True
        if "_unparsed" in c.arguments:
            return UNPARSED, True
        check_arguments = getattr(self.box, "check_arguments", None)
        problem = check_arguments(t, c.arguments) if check_arguments else None
        if problem:
            return problem, True
        try:
            text = str(t.fn(**c.arguments))
        except Exception as exc:  # noqa: BLE001 -- told to the model
            return f"{type(exc).__name__}: {exc}", True
        self.box.called.add(c.name)
        return text, False

    def run(self) -> None:
        """The rounds of one message: the model called, and its tool calls carried out,
        until it answers without one. With full automatic on (no one to answer): a turn
        the model ends with the chain unfinished (chain_left) is told what is left and
        goes on, until the window stops moving (AUTO_STALLS) or AUTO_NUDGES nudges; the
        MAX_ROUNDS budget runs from the last step done, and a round that only waits for a
        step is not counted; the latest PINNED readings are kept whole (_payload); a
        failed call to the model, or an empty reply, is made again (AUTO_RETRIES)."""
        self.auto = auto_on()
        rounds = failures = nudges = stalls = 0
        free_waits = 10
        best = len([s for s in self._left() if s != "wait"]) if self.auto else 0
        nudged: list[str] | None = None
        try:
            while rounds < MAX_ROUNDS:
                if self.box.stop.is_set():
                    self._close("(stopped by the analyst)")
                    self._note("Stopped.")
                    return
                self._mend()
                # The tools the window offers now: a step run a round ago brings its own.
                tools = self.box.tools()
                by_name = {t.name: t for t in tools}
                try:
                    reply = self.model.chat(system_prompt(), self._payload(), tools)
                except Exception as exc:  # noqa: BLE001 -- shown in the panel
                    failures += 1
                    if self._again(failures, AUTO_RETRY_WAIT):
                        self._note(
                            f"The call to the model failed ({type(exc).__name__}: {exc}); full "
                            "automatic tries again."
                        )
                        continue
                    raise
                rounds += 1
                self.event.emit({"type": "usage", "usage": dict(reply.usage)})
                if not reply.calls and not (reply.text or "").strip():
                    raw = reply.raw if isinstance(reply.raw, dict) else {}
                    why = (
                        ((raw.get("choices") or [{}])[0]).get("finish_reason")
                        or raw.get("stop_reason")
                        or "none given"
                    )
                    self.messages.append(
                        {"role": "assistant", "text": "(empty reply)", "calls": []}
                    )
                    failures += 1
                    if self._again(failures, 0.0):
                        self.messages.append(
                            {
                                "role": "user",
                                "content": f"[Your reply was empty (finish reason: {why}). Full "
                                "automatic: go on with the analysis, a step at a time.]",
                            }
                        )
                        continue
                    self.failed.emit(
                        f"The model's reply was empty (finish reason: {why})"
                        + (
                            "; it reached its output limit while thinking. Ask again, perhaps "
                            "for less at a time."
                            if why in ("length", "max_tokens")
                            else ". Ask again."
                        )
                    )
                    return
                self.messages.append(
                    {
                        "role": "assistant",
                        "text": reply.text,
                        "calls": reply.calls,
                        "reasoning": reply.reasoning,
                    }
                )
                self.event.emit(
                    {
                        "type": "model",
                        "text": reply.text,
                        "calls": [{"name": c.name, "arguments": c.arguments} for c in reply.calls],
                    }
                )
                if not reply.calls:
                    left = self._left() if self.auto and nudges < AUTO_NUDGES else []
                    # Told again only while the window moves: the same steps left at two
                    # nudges running is a model that cannot do them.
                    stalls = stalls + 1 if left and left == nudged else 0
                    if not left or stalls >= AUTO_STALLS:
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
                    continue
                results = []
                for c in reply.calls:
                    if self.box.stop.is_set():
                        text, error = "stopped by the analyst before this call ran", True
                    else:
                        text, error = self._call(c, by_name)
                    self.event.emit(
                        {
                            "type": "tool",
                            "name": c.name,
                            "arguments": c.arguments,
                            "output": text,
                            "error": error,
                        }
                    )
                    results.append({"id": c.id, "text": text, "error": error})
                self.messages.append({"role": "tool", "results": results})
                if self.auto:
                    if free_waits and all(c.name == "wait_for_run" for c in reply.calls):
                        free_waits -= 1
                        rounds -= 1  # waiting is not working
                    now = len([s for s in self._left() if s != "wait"])
                    if now < best:
                        best, rounds = now, 0  # a step done: the budget anew
            self._close("(stopped: too many rounds of tool calls)")
            self.failed.emit(f"Stopped after {MAX_ROUNDS} rounds of tool calls.")
        except Exception as exc:  # noqa: BLE001 -- the provider's word, shown in the panel
            self._close("(no reply: the call failed)")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


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
        #: The window's state as the last message told it (state_brief).
        self._state_parts: list[str] | None = None
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

    def run_code(self, code: str) -> None:
        """The code of a card the analyst ran (carry_out), on the window's table, off the
        interface thread; what it gave shown and told to the model with the next
        message."""
        w = self.win
        item = self.view.tool("carry_out", {"lines": len(code.strip().splitlines())})

        def done(said: str) -> None:
            failed = "\nError:\n" in f"\n{said}"
            self.view.tool_done(item, said, failed)
            self.notes.append("[The analyst ran your code. What it gave:\n" + said + "]")
            if not self.busy() and self.messages_sent():
                self._suggest(suggested_message(w, self.tools.called))

        def failed(message: str) -> None:
            self.view.tool_done(item, message, True)
            self.notes.append(f"[The analyst ran your code; it failed: {message}]")

        if not w.runner.submit(
            "carry out",
            _code.run_code,
            done,
            failed,
            code,
            w.df_raw,
            w.df_prep,
            w.model,
        ):
            self.view.tool_done(item, "A step is running: run it again once it has ended.", True)

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
        return llm.Model(
            name,
            api_key=find_key(provider) if provider in llm.PROVIDERS else None,
            max_tokens=MAX_TOKENS,
        )

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
        # The whole state where it moved since the last message; a brief line where not.
        parts = state_parts(self.win)
        state = state_brief(parts) if parts == self._state_parts else "; ".join(parts)
        self._state_parts = parts
        stamped = f"[Window now: {state}]\n{told}{text}"
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
        self._state_parts = None
        self.tools._readings.clear()
        self._close_cli()
        self.view.clear()
        self.status.setText("")
        self._suggest("")

    # -- a command-line agent --------------------------------------------------------
    def _send_cli(self, text: str) -> None:
        if self._cli is None:
            if self._server is None:
                # Every tool: an agent lists them once, as its session starts, and a tool
                # the window cannot run yet says why when it is called.
                self._server = ToolServer(
                    self.tools.all_tools,
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
