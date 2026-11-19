"""Shared fixtures for the normet test suite."""

from __future__ import annotations

import importlib.util
import logging
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

_MACOS = sys.platform == "darwin"


def _kill_cfprefsd() -> None:
    """Stop macOS's preferences daemon and wait for it to actually go.

    It keeps the settings in memory and flushes them to the file *as it exits*: with
    only a fixed sleep, a copy made underneath it can be overwritten by the cached
    values a moment later (which is how tests' scraps survive a restore). Waiting for
    ``pgrep`` to come back empty closes that race. It restarts by itself when next used.
    Only ever called on macOS.
    """
    try:
        subprocess.run(["killall", "cfprefsd"], capture_output=True, check=False)
    except OSError:
        return
    # Only this user's daemon: killall (run without sudo here) cannot stop the copies
    # other accounts keep, and a bare `pgrep cfprefsd` would wait for those forever.
    for _ in range(100):  # up to 10 s; the flush is a few hundred milliseconds
        try:
            gone = subprocess.run(
                ["pgrep", "-u", str(os.getuid()), "cfprefsd"], capture_output=True
            )
            if gone.returncode != 0:
                return
        except OSError:
            return
        time.sleep(0.1)


def _plist_dict(path: str) -> dict | None:
    """The plist as a dict, or None if there is no file (the file is read directly:
    ``defaults``/QSettings go through the daemon)."""
    try:
        with open(path, "rb") as f:
            return dict(plistlib.load(f))
    except (FileNotFoundError, ValueError, OSError):
        return None


def _normet_plist() -> str:
    return os.path.join(
        os.path.expanduser("~"), "Library", "Preferences", "com.apai-sys.normet-gui.plist"
    )


def _restore_normet_plist(config) -> None:
    """Put the user's real normet settings back (macOS only; a no-op elsewhere): the file
    as it was, or gone again if the run is what made it.

    cfprefsd can still have a cached copy that outlives one kill, so the file is read
    back after the second kill and, if the daemon overwrote it again, the sequence is
    retried. This is what keeps a test run from leaving its geometry, panel page or
    result tab in the settings of the window the user works in.
    """
    plist = getattr(config, "_normet_plist_path", None)
    if not plist:
        return
    backup = getattr(config, "_normet_plist_backup", None)
    want = None
    if backup and os.path.exists(backup):
        with open(backup, "rb") as f:
            want = dict(plistlib.load(f))
    for _ in range(5):
        _kill_cfprefsd()
        if want is not None and backup:
            shutil.copy(str(backup), plist)
        elif os.path.exists(plist):
            os.remove(plist)
        _kill_cfprefsd()
        if _plist_dict(plist) == want:
            return


def pytest_configure(config):
    """Snapshot the user's real normet settings before any test touches them.

    The GUI persists window geometry, the panel page and the result tab through
    QSettings, which on macOS writes straight to the user's plist regardless of
    $HOME or QSettings.setPath. Every test that builds a MainWindow would
    otherwise leave its scraps there. The snapshot is restored in
    pytest_sessionfinish, after the windows are closed. Other platforms keep their
    settings elsewhere (an ini file, the registry) and are left alone.

    The file's path is worked out here, once: some tests move ``$HOME`` to a folder of
    their own, and a run interrupted inside one would otherwise look for the file there.
    """
    if not _MACOS:
        return
    _kill_cfprefsd()
    plist = config._normet_plist_path = _normet_plist()
    if os.path.exists(plist):
        config._normet_plist_backup = os.path.join(
            tempfile.mkdtemp(prefix="normet_plist_bak_"), "backup.plist"
        )
        shutil.copy(plist, config._normet_plist_backup)


@pytest.fixture(autouse=True)
def _config_dir_of_its_own(request, monkeypatch, tmp_path_factory):
    """No test writes into the user's ``~/.normet``: the recent files, the run history and
    the last session (which the end of every step rewrites) are the user's own, and a test
    that runs a step would append to the first and overwrite the last. Each test gets a
    folder of its own for them. The tests that make a home of their own (``home``) keep
    it: there ``config_dir`` follows ``$HOME``."""
    if "home" in request.fixturenames:
        return
    try:
        from normet.gui import _settings
    except Exception:  # noqa: BLE001 - not the window's tests
        return
    config = str(tmp_path_factory.mktemp("config"))
    monkeypatch.setattr(_settings, "config_dir", lambda: config)


@pytest.fixture(autouse=True)
def _close_figures():
    """Release every matplotlib figure a test leaves behind.

    pyplot keeps each figure it creates alive until something closes it, and
    normet's plotting helpers hand back an Axes or a Figure for the caller to
    own. Across a full run that accumulated past matplotlib's 20-figure alarm
    and reported it as a RuntimeWarning against whichever test happened to
    cross the threshold -- a number that moves whenever tests are reordered.
    """
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture(autouse=True)
def _restore_normet_logger():
    """Put the shared "normet" logger back the way each test found it.

    ``enable_default_logging`` -- reached directly from test_logging.py and
    through the CLI in test_cli.py -- sets ``propagate = False`` and a level on
    that logger. Left in place, every later test that reads a warning through
    ``caplog`` (whose handler sits on the root logger) saw nothing and failed,
    but only when run after those modules, never on its own.
    """
    logger = logging.getLogger("normet")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def _close_windows() -> None:
    if "PySide6.QtWidgets" not in sys.modules:
        return
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return
    for w in QApplication.topLevelWidgets():
        w.close()
        w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QCoreApplication.processEvents()


def pytest_sessionfinish(session, exitstatus):
    """Delete the windows the GUI tests left open while the QApplication still runs.

    Left to the interpreter's exit, Python's wrappers and Qt's parents freed the same
    windows in no fixed order: on Linux CI the run passed every test and then aborted
    ("shared QObject was deleted directly", "corrupted double-linked list"). Closed
    and deleted here, Qt frees each child before its parent.

    Then, on macOS, the user's real normet settings go back: the windows' closeEvents
    write them, so it has to be after the windows are closed -- and whether or not
    there was a QApplication to close them in.
    """
    try:
        _close_windows()
    finally:
        _restore_normet_plist(session.config)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20240101)


@pytest.fixture(scope="session")
def synthetic_aq(rng) -> pd.DataFrame:
    """Hourly synthetic air-quality panel with a clear weather signal."""
    n = 24 * 60  # 60 days hourly
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    t2m = 10 + 8 * np.sin(2 * np.pi * np.arange(n) / (24 * 30)) + rng.normal(0, 1, n)
    blh = 800 + 400 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 50, n)
    u10 = rng.normal(0, 3, n)
    v10 = rng.normal(0, 3, n)
    # Concentration: increases when blh is low and temperature deviates from mean.
    pm = 20 - 0.01 * blh + 0.3 * (t2m - t2m.mean()) ** 2 + 0.5 * u10 + rng.normal(0, 2, n)
    pm = np.clip(pm, 0, None)
    return pd.DataFrame(
        {"date": dates, "PM2.5": pm, "t2m": t2m, "blh": blh, "u10": u10, "v10": v10}
    )


@pytest.fixture(scope="session")
def scm_panel(rng) -> pd.DataFrame:
    """
    Toy panel: one treated unit + 6 donors. Treated unit equals 0.5*D1 + 0.5*D2
    in the pre-period, then gets a shock of +10 after cutoff.

    Each donor carries a *distinct* deterministic signal (a different sinusoid
    frequency and phase), so the donors are not mutually collinear and the true
    0.5/0.5 weight identity on D1+D2 is actually recoverable. (If every donor
    shared the same signal — as in a naive construction — the synthetic-control
    weights would be unidentifiable: many donor combinations fit equally well,
    so only the *effect* would be recoverable, not the weights. Note the
    augmented SCM balances ridge *residuals*, so the donor signals must also be
    cleanly predictable — a large independent random-walk component would
    dominate the residuals and again wash out weight identity.)
    """
    n_days = 200
    t = np.arange(n_days)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    cutoff = pd.Timestamp("2023-05-01")

    # Distinct deterministic signal per donor: different period + phase so no
    # donor is a linear combination of the others.
    periods = {"D1": 30, "D2": 17, "D3": 45, "D4": 11, "D5": 60, "D6": 23}
    donors_data = {}
    for j, name in enumerate(["D1", "D2", "D3", "D4", "D5", "D6"]):
        donors_data[name] = 5 + j + 3 * np.sin(2 * np.pi * t / periods[name] + j)

    treated = 0.5 * donors_data["D1"] + 0.5 * donors_data["D2"] + rng.normal(0, 0.05, n_days)
    treated = treated + np.where(dates >= cutoff, 10.0, 0.0)

    rows = []
    for d, val in zip(dates, treated, strict=False):
        rows.append({"date": d, "ID": "T", "value": val})
    for name, arr in donors_data.items():
        for d, val in zip(dates, arr, strict=False):
            rows.append({"date": d, "ID": name, "value": val})
    return pd.DataFrame(rows)


def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


needs_flaml = pytest.mark.skipif(not _has("flaml"), reason="flaml not installed")
needs_lgb = pytest.mark.skipif(not _has("lightgbm"), reason="lightgbm not installed")


@pytest.fixture(scope="session")
def chronos_device() -> str:
    """Where the foundation tests put the checkpoint.

    The CPU by default, so the suite runs anywhere and gives the same numbers on
    every machine. ``NORMET_TEST_DEVICE=cuda`` re-runs the same tests on a GPU,
    which is worth doing after touching the shared forward pass: batching sends
    several inputs through one call, and a device can disagree with the CPU
    about a batched op in a way no CPU-only run would ever show.
    """
    return os.environ.get("NORMET_TEST_DEVICE", "cpu")


@pytest.fixture(scope="session")
def chronos2_pipeline(chronos_device):
    """Load ``amazon/chronos-2`` once for the whole session.

    Both foundation test modules hit the real checkpoint; re-reading ~500 MB of
    weights per module dominates runtime on a shared filesystem. The pipeline is
    stateless across calls, so sharing it is safe.
    """
    from chronos import Chronos2Pipeline

    from normet.foundation.estimator import DEFAULT_MODEL

    return Chronos2Pipeline.from_pretrained(DEFAULT_MODEL, device_map=chronos_device)
