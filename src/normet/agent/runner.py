"""A job's child process: ``python -m normet.agent.runner FOLDER`` runs the
``normet`` command of ``FOLDER/job.json`` there and writes how it ended
(:mod:`normet.agent.jobs`)."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
import traceback
from types import FrameType
from typing import Any


class Cancelled(BaseException):
    """Asked to stop (SIGTERM). A BaseException, so no ``except Exception`` in the
    run swallows it."""


def _write(path: str, value: dict[str, Any]) -> None:
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=1)
    os.replace(tmp, path)


def main(folder: str) -> int:
    with open(os.path.join(folder, "job.json"), encoding="utf-8") as fh:
        job = json.load(fh)
    _write(os.path.join(folder, "started.json"), {"pid": os.getpid(), "started": time.time()})

    def stop(signum: int, frame: FrameType | None) -> None:
        raise Cancelled()

    signal.signal(signal.SIGTERM, stop)
    os.chdir(folder)
    state, code = "done", 0
    try:
        import click

        from normet.cli import _build_cli

        try:
            _build_cli().main(args=job["argv"], prog_name="normet", standalone_mode=False)
        except click.ClickException as exc:  # a bad option: said as the command line says it
            exc.show()
            state, code = "failed", exc.exit_code
    except Cancelled:
        state, code = "cancelled", -int(signal.SIGTERM)
        print("cancelled", file=sys.stderr)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        state = "done" if code == 0 else "failed"
    except BaseException:  # noqa: BLE001 - its record is the point
        traceback.print_exc()
        state, code = "failed", 1
    sys.stdout.flush()
    sys.stderr.flush()
    _write(
        os.path.join(folder, "exit.json"), {"state": state, "code": code, "finished": time.time()}
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
