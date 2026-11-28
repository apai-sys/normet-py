"""Jobs: a ``normet`` command run in a child process, in a folder of its own.

A job's folder holds what it is and how
it went, so a server started again finds its jobs where they were::

    job.json      what was asked: the kind, the command's arguments, the command line
    started.json  the child's process id and when it started (written by the child)
    stdout.txt    what the command printed
    stderr.txt    its log lines and errors
    exit.json     how it ended: done, failed or cancelled, and the exit code
    cancel        there when a cancel was asked
    *.csv, run.*  the command's files

The command is the job's record: ``command`` in ``job.json`` runs again, as it
is, from the job's folder, without the server or an assistant. Jobs run one at
a time, in the order asked.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

#: How long a job asked to stop has to stop before it is killed.
CANCEL_GRACE = 10.0

#: How long an un-PID'd ``launched`` marker is treated as starting before
#: assuming the launch never completed.
LAUNCH_GRACE = 15.0

STATES = ("queued", "running", "done", "failed", "cancelled")


def default_home() -> str:
    """Where jobs are kept: ``$NORMET_MCP_HOME``, or ``~/.normet/mcp``."""
    return os.path.abspath(
        os.path.expanduser(os.environ.get("NORMET_MCP_HOME") or os.path.join("~", ".normet", "mcp"))
    )


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: str, value: dict[str, Any]) -> None:
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=1)
    os.replace(tmp, path)


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":  # pragma: no cover - Windows
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A child of this process that has ended but not been waited for is a zombie: gone.
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


def _folder_pid_alive(folder: str) -> bool:
    """True when ``started.json`` or ``launched`` records a live process id."""
    for name in ("started.json", "launched"):
        meta = _read_json(os.path.join(folder, name))
        if meta and _alive(meta.get("pid")):
            return True
    return False


def _child_env() -> dict[str, str]:
    """The server's environment for a job, which runs in its own folder: the normet
    this server runs is the one the job imports, and a relative PYTHONPATH is made
    absolute."""
    import normet

    here = os.path.dirname(os.path.dirname(os.path.abspath(normet.__file__)))
    paths = [here] + [
        os.path.abspath(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p
    ]
    return dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=os.pathsep.join(dict.fromkeys(paths)))


def command_line(argv: list[str]) -> str:
    """The job as a command a shell runs."""
    return "normet " + " ".join(shlex.quote(a) for a in argv)


def _folder_mtime(folder: str) -> float:
    """When the job was asked for, from its ``job.json`` (the folder's own time if
    that is gone)."""
    for path in (os.path.join(folder, "job.json"), folder):
        try:
            return os.path.getmtime(path)
        except OSError:
            continue
    return 0.0


def _dir_size(folder: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class JobStore:
    """The jobs kept in ``home/jobs``, run one at a time."""

    def __init__(self, home: str | None = None, python: str | None = None) -> None:
        self.home = os.path.abspath(home or default_home())
        self.root = os.path.join(self.home, "jobs")
        os.makedirs(self.root, exist_ok=True)
        self.python = python or sys.executable
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._lock = threading.RLock()

    # -- asking -----------------------------------------------------------------
    def submit(self, kind: str, argv: list[str], label: str = "") -> dict[str, Any]:
        """Queue ``normet argv`` as a job; it starts once those before it have ended."""
        # Ids sort as the jobs were asked -- the queue's order -- to the microsecond.
        now = time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        job_id = f"{stamp}.{int(now * 1e6) % 1_000_000:06d}-{uuid.uuid4().hex[:4]}"
        folder = os.path.join(self.root, job_id)
        os.makedirs(folder)
        argv = [str(a) for a in argv]
        _write_json(
            os.path.join(folder, "job.json"),
            {
                "id": job_id,
                "kind": kind,
                "label": label,
                "argv": argv,
                "command": command_line(argv),
                "folder": folder,
                "submitted": now,
            },
        )
        self.pump()
        return self.status(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Stop a job: one waiting never starts; one running is asked to stop, and
        killed if it has not within :data:`CANCEL_GRACE` seconds."""
        folder = self._folder(job_id)
        with self._lock:
            open(os.path.join(folder, "cancel"), "w").close()
            started = _read_json(os.path.join(folder, "started.json")) or {}
            launched = _read_json(os.path.join(folder, "launched")) or {}
            proc = self._procs.get(job_id)
            pid = proc.pid if proc is not None else (started.get("pid") or launched.get("pid"))

            def running() -> bool:
                # This server's own child is asked through its Popen, which reaps it;
                # one an earlier server started, by its process id.
                return proc.poll() is None if proc is not None else _alive(pid)

            if pid and running():
                self._signal(pid, proc, signal.SIGTERM)
                deadline = time.time() + CANCEL_GRACE
                while time.time() < deadline and running():
                    time.sleep(0.1)
                if running():
                    self._signal(pid, proc, getattr(signal, "SIGKILL", signal.SIGTERM))
                if proc is not None:
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:  # pragma: no cover
                        pass
        self.pump()
        return self.status(job_id)

    @staticmethod
    def _signal(pid: int, proc: subprocess.Popen[bytes] | None, sig: int) -> None:
        try:
            if os.name == "nt":  # pragma: no cover - Windows
                if proc is not None:
                    proc.terminate()
                else:
                    os.kill(pid, sig)
            else:
                os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    # -- reading ----------------------------------------------------------------
    def ids(self) -> list[str]:
        """Every job, oldest first."""
        return sorted(
            d
            for d in os.listdir(self.root)
            if os.path.isfile(os.path.join(self.root, d, "job.json"))
        )

    def _folder(self, job_id: str) -> str:
        job_id = str(job_id).strip()
        folder = os.path.join(self.root, job_id)
        if os.path.dirname(os.path.abspath(folder)) != self.root or not os.path.isfile(
            os.path.join(folder, "job.json")
        ):
            raise KeyError(f"no job {job_id!r}")
        return folder

    def state(self, job_id: str) -> str:
        folder = self._folder(job_id)
        ended = _read_json(os.path.join(folder, "exit.json"))
        if ended:
            return str(ended["state"])
        proc = self._procs.get(job_id)
        if proc is not None:
            if proc.poll() is None:
                return "running"
        else:
            started = _read_json(os.path.join(folder, "started.json"))
            if _folder_pid_alive(folder):
                return "running"
            if not started:
                launched_path = os.path.join(folder, "launched")
                if not os.path.exists(launched_path):
                    return (
                        "cancelled" if os.path.exists(os.path.join(folder, "cancel")) else "queued"
                    )
                launched = _read_json(launched_path)
                if not (launched and launched.get("pid") is not None):
                    # A freshly spawned child may not have written started.json yet (or
                    # `launched` may not yet carry its pid); allow a short startup grace
                    # window before classifying an orphaned marker as failed.
                    try:
                        age = time.time() - os.path.getmtime(launched_path)
                    except OSError:
                        age = 0.0
                    if age < LAUNCH_GRACE:
                        return "running"
        # Re-check exit.json in case the child finished and exited while we checked.
        ended = _read_json(os.path.join(folder, "exit.json"))
        if ended:
            return str(ended["state"])
        # Ended without saying how: killed, or the machine stopped.
        return "cancelled" if os.path.exists(os.path.join(folder, "cancel")) else "failed"

    def status(self, job_id: str) -> dict[str, Any]:
        """What a job is, how far it is, and -- once it has ended -- what it left."""
        folder = self._folder(job_id)
        job = _read_json(os.path.join(folder, "job.json")) or {}
        state = self.state(job_id)
        started = _read_json(os.path.join(folder, "started.json")) or {}
        ended = _read_json(os.path.join(folder, "exit.json")) or {}
        t0 = started.get("started")
        t1 = ended.get("finished") or (time.time() if state == "running" else None)
        out = dict(
            job,
            state=state,
            exit_code=ended.get("code"),
            elapsed_s=round(t1 - t0, 1) if t0 and t1 else None,
        )
        hidden = {"job.json", "started.json", "exit.json", "cancel", "launched"}
        files = sorted(f for f in os.listdir(folder) if not f.endswith(".part") and f not in hidden)
        out["files"] = [os.path.join(folder, f) for f in files]
        if state == "queued":
            ahead = [j for j in self.ids() if j < job_id and self.state(j) in ("queued", "running")]
            out["ahead"] = len(ahead)
        return out

    def text(self, job_id: str, name: str = "stdout.txt", tail: int | None = None) -> str:
        """A job's printed output, or its last *tail* characters."""
        try:
            with open(
                os.path.join(self._folder(job_id), name), encoding="utf-8", errors="replace"
            ) as fh:
                text = fh.read()
        except OSError:
            return ""
        return text[-tail:] if tail and len(text) > tail else text

    # -- pruning ----------------------------------------------------------------
    def prune(
        self,
        *,
        max_age_days: float | None = None,
        states: list[str] | tuple[str, ...] | None = None,
        keep_latest: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Remove finished job folders by age, state and retention.

        Active jobs (``queued``, ``running``) are never removed. With no
        argument only ``failed`` and ``cancelled`` jobs go; asking by age or by
        ``keep_latest`` also allows finished ones. The age is read from the
        job's ``job.json``, written when it was asked for, so a long job is
        pruned by when it was submitted, not by when it ended.
        """
        if states is not None:
            target_states = {str(s) for s in states} - {"queued", "running"}
        elif max_age_days is not None or keep_latest is not None:
            target_states = {"done", "failed", "cancelled"}
        else:
            target_states = {"failed", "cancelled"}

        cutoff = None if max_age_days is None else time.time() - float(max_age_days) * 86400.0
        keep_n = max(0, int(keep_latest)) if keep_latest is not None else 0

        removed: list[str] = []
        kept: list[str] = []
        reclaimed = 0
        with self._lock:
            ids = self.ids()
            protected = set(ids[-keep_n:]) if keep_n else set()
            for job_id in ids:
                state = self.state(job_id)
                folder = self._folder(job_id)
                proc = self._procs.get(job_id)
                still_alive = (proc is not None and proc.poll() is None) or (
                    not _read_json(os.path.join(folder, "exit.json")) and _folder_pid_alive(folder)
                )
                if (
                    state in ("queued", "running")
                    or still_alive
                    or job_id in protected
                    or state not in target_states
                ):
                    kept.append(job_id)
                    continue
                if cutoff is not None and _folder_mtime(folder) > cutoff:
                    kept.append(job_id)
                    continue
                reclaimed += _dir_size(folder)
                if not dry_run:
                    shutil.rmtree(folder, ignore_errors=True)
                removed.append(job_id)
        return {
            "removed": removed,
            "kept": kept,
            "reclaimed_bytes": int(reclaimed),
            "dry_run": bool(dry_run),
        }

    # -- running ----------------------------------------------------------------
    def pump(self) -> None:
        """Start the next waiting job if none is running; reap those that ended."""
        with self._lock:
            for job_id, proc in list(self._procs.items()):
                if proc.poll() is not None:
                    del self._procs[job_id]
            ids = self.ids()
            if any(self.state(j) == "running" for j in ids):
                return
            for job_id in ids:
                if self.state(job_id) == "queued":
                    self._launch(job_id)
                    return

    def _launch(self, job_id: str) -> None:
        folder = self._folder(job_id)
        launched_path = os.path.join(folder, "launched")
        _write_json(launched_path, {"launched": time.time()})
        kw: dict[str, Any] = {}
        if os.name == "nt":  # pragma: no cover - Windows
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            kw["start_new_session"] = True
        with (
            open(os.path.join(folder, "stdout.txt"), "wb") as out,
            open(os.path.join(folder, "stderr.txt"), "wb") as err,
        ):
            try:
                proc = subprocess.Popen(
                    [self.python, "-m", "normet.agent.runner", folder],
                    cwd=folder,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    env=_child_env(),
                    **kw,
                )
            except OSError:
                _write_json(
                    os.path.join(folder, "exit.json"),
                    {"state": "failed", "code": 1, "finished": time.time()},
                )
                raise
            self._procs[job_id] = proc
            _write_json(launched_path, {"pid": proc.pid, "launched": time.time()})

    def wait(self, job_id: str, timeout: float | None = None, poll: float = 0.2) -> dict[str, Any]:
        """Wait for a job to end (for tests and scripts; a server polls :meth:`status`)."""
        deadline = None if timeout is None else time.time() + timeout
        while self.state(job_id) in ("queued", "running"):
            self.pump()
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(f"job {job_id} still {self.state(job_id)}")
            time.sleep(poll)
        self.pump()
        return self.status(job_id)

    def start_pumping(self, every: float = 2.0) -> threading.Thread:
        """A thread that starts each waiting job in turn while the server runs."""

        def loop() -> None:
            while True:
                try:
                    self.pump()
                except Exception:  # pragma: no cover
                    pass
                time.sleep(every)

        thread = threading.Thread(target=loop, name="normet-jobs", daemon=True)
        thread.start()
        return thread
