"""File -> Save Results / Load Results / Reopen Last Session.

A session -- the table, the panel's settings, the model trained, every step's result and
the command log -- written to one file (``.normet``, joblib) and read back into the
window as it was: each result is handed to the step's own handler, so its tab, verdict
and figure are drawn again exactly as the run drew them. After each step the session is
also kept, off the GUI thread, as ``~/.normet/last_session.normet``, the safety net for a
window closed or crashed before its results were saved.

A Chronos-2 checkpoint is not written: it is the pretrained model, half a gigabyte, and
is loaded again from its cache with *Load Chronos-2*. A file of this kind is a pickle:
open only one you or the window wrote.
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
from typing import Any

import pandas as pd
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox

from . import _chronos, _settings

log = logging.getLogger(__name__)

#: The format's version, written into every file.
VERSION = 1
SUFFIX = ".normet"
#: The steps whose result is a table handed back to its handler, in the order they ran.
_STEPS = ("normalise", "decompose", "rolling", "pdp")


def last_session_path() -> str:
    """Where the window keeps the session after each step."""
    return os.path.join(_settings.config_dir(), "last_session" + SUFFIX)


class SavedResultsMixin(QMainWindow):
    """Saving and restoring a whole session. Mixed into the main window."""

    # ------------------------------------------------------------- writing
    def session_bundle(self) -> dict[str, Any]:
        """What a saved session holds."""
        chronos = self._is_chronos()
        return {
            "version": VERSION,
            "saved": datetime.datetime.now().isoformat(timespec="seconds"),
            "label": self.file_label.text().split("\n")[0] if self.df_raw is not None else "",
            "df_raw": self.df_raw,
            "config": self.get_config(),
            "df_prep": self.df_prep,
            "model": None if chronos else self.model,
            "chronos": chronos and self.model is not None,
            "trained_features": list(self.trained_features),
            "results": {k: v for k, v in self.results.items() if k in _STEPS},
            "multiscale": getattr(self, "_multiscale_raw", None),
            "commands": list(self.command_log.entries),
            "ran": dict(self._ran),  # what the Methods paragraph is written from
        }

    def save_results(self) -> None:
        """File -> Save Results: the session to a file of the analyst's choosing."""
        if self.df_raw is None:
            self.notify("Nothing to save yet: open a table and run a step first.", key="nothing")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save results", "normet_session" + SUFFIX, f"normet session (*{SUFFIX})"
        )
        if not path:
            return
        if not path.endswith(SUFFIX):
            path += SUFFIX
        try:
            import joblib

            joblib.dump(self.session_bundle(), path, compress=3)
        except Exception as exc:  # noqa: BLE001 - say why
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved {os.path.basename(path)}")
        log.info("Session saved to %s", path)

    def autosave_session(self) -> None:
        """Keep the session as the last one, off the GUI thread (a step has finished)."""
        if self.df_raw is None or getattr(self, "_restoring", False):
            return
        bundle = self.session_bundle()
        path = last_session_path()

        def write() -> None:
            try:
                import joblib

                tmp = path + ".part"
                joblib.dump(bundle, tmp, compress=3)
                os.replace(tmp, path)  # a crash mid-write leaves the previous one whole
            except Exception:  # noqa: BLE001 - a safety net that failed is logged, not raised
                log.debug("Autosave failed", exc_info=True)

        threading.Thread(target=write, name="normet-autosave", daemon=True).start()

    # ------------------------------------------------------------- reading
    def load_results(self, path: str | None = None) -> None:
        """File -> Load Results: a saved session into the window."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load results", "", f"normet session (*{SUFFIX});;All files (*)"
            )
        if not path:
            return
        try:
            import joblib

            bundle = joblib.load(path)
        except Exception as exc:  # noqa: BLE001 - say why
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        if not isinstance(bundle, dict) or "df_raw" not in bundle:
            QMessageBox.critical(self, "Load failed", "Not a normet session file.")
            return
        self.restore_session(bundle)
        self.statusBar().showMessage(f"Loaded {os.path.basename(path)}")

    def reopen_last_session(self) -> None:
        """File -> Reopen Last Session: the session kept after the last step."""
        path = last_session_path()
        if not os.path.exists(path):
            self.notify(
                "No session has been kept yet: one is after each step that finishes.",
                key="no-session",
            )
            return
        self.load_results(path)

    def load_session_script(self, path: str | None = None, *, ask: bool = True) -> None:
        r"""File -> Load Session Script: a script the command log wrote, run again and
        its results brought into the window (.\_script). It is Python, run as written:
        the window asks first, unless told not to (``ask``)."""
        from . import _script

        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load session script", "", "Python (*.py);;All files (*)"
            )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            calls = _script.read_calls(text)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            QMessageBox.critical(self, "Not a session script", str(exc))
            return
        if "build_model" not in calls and "load_estimator" not in calls:
            QMessageBox.critical(
                self,
                "Not a session script",
                "It trains no model: a session script is one the command log wrote "
                "(File -> Save Session as Script).",
            )
            return
        if ask:
            answer = QMessageBox.question(
                self,
                "Run this script?",
                f"{os.path.basename(path)} is Python and runs as it is written, as running "
                "it yourself would. Run only a script you or the window wrote.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        # What the script's calls set, for when it has run. Its result comes back through
        # a method of the window: Qt queues that onto the GUI thread (a plain function
        # would run on the worker's, and draw figures there).
        self._pending_script = (calls, os.path.basename(path))
        self.runner.submit(
            "session script",
            _script.run_script,
            self._script_result,
            self._show_error,
            text,
            os.path.dirname(os.path.abspath(path)),
        )

    def _script_result(self, made: dict[str, Any]) -> None:
        calls, name = self._pending_script
        self._script_done(made, calls, name)

    def _script_done(self, made: dict[str, Any], calls: dict[str, Any], name: str) -> None:
        """What a session script made, brought into the window as a saved session is."""
        from . import _script

        df = made.get("df")
        if not isinstance(df, pd.DataFrame):
            self.notify(f"{name} ran but made no table called df.", kind="warn")
            return
        config = _script.config_from_calls(calls)
        trained = list(config.get("features") or [])
        results = {
            step: made[var]
            for var, step in _script.RESULTS.items()
            if isinstance(made.get(var), pd.DataFrame)
        }
        multiscale = made.get("multiscale")
        bundle = {
            "saved": name,
            "label": name,
            "df_raw": df,
            "config": config,
            "df_prep": made.get("df_prep"),
            "model": made.get("model"),
            "chronos": False,
            "trained_features": trained,
            "results": results,
            "multiscale": multiscale if isinstance(multiscale, dict) else None,
            "commands": [],
            "ran": calls,
        }
        self.restore_session(bundle)

    def restore_session(self, bundle: dict[str, Any]) -> None:
        """The window as a saved session left it: its table, settings, model and results."""
        df = bundle["df_raw"]
        if df is None:
            return
        self._restoring = True
        try:
            self._data_path = None  # not a file the time zone can read again
            self._show_dates(None)
            self._ingest(df, bundle.get("label") or "restored session")
            config = bundle.get("config") or {}
            self.apply_config(config)
            model = bundle.get("model")
            if model is not None and bundle.get("df_prep") is not None:
                self._pending_features = list(bundle.get("trained_features") or [])
                self._train_done((bundle["df_prep"], model))
                self.apply_config(config)  # training resets Step 2's ticks to Met only
            results = bundle.get("results") or {}
            if model is not None:
                for step in _STEPS:
                    res = results.get(step)
                    if res is None:
                        continue
                    if step == "decompose":
                        self.decom_method.setCurrentText(
                            "meteorology" if "met_total" in res.columns else "emission"
                        )
                    getattr(self, f"_{step}_done")(res)
                if bundle.get("multiscale") is not None:
                    self._multiscale_done(bundle["multiscale"])
            self.command_log.entries = list(bundle.get("commands") or [])
            self._ran = dict(bundle.get("ran") or {})
            self._refresh_command_log()
        finally:
            self._restoring = False
        self._sync_enabled()
        when = bundle.get("saved", "")
        if bundle.get("chronos"):
            self.backend_combo.setCurrentText(_chronos.BACKEND)
            self.notify(
                f"Session of {when} restored without its Chronos-2 model, which is not "
                "saved: Load Chronos-2 again to go on.",
                key="restored",
                kind="warn",
                actions=(("Load Chronos-2", self._run_train),),
            )
        else:
            steps = [s for s in ("train", *_STEPS, "multiscale") if s in self.results]
            self.notify(
                f"Session of {when} restored: "
                + (", ".join(steps) if steps else "the table and its settings")
                + ".",
                key="restored",
            )
        if isinstance(df, pd.DataFrame):
            log.info("Session restored: %d rows, %d results", len(df), len(self.results))
