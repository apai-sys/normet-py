"""Per-user GUI state under ``~/.normet``: recent files and run history."""

from __future__ import annotations

import json
import os
from typing import Any


def config_dir() -> str:
    d = os.path.join(os.path.expanduser("~"), ".normet")
    os.makedirs(d, exist_ok=True)
    return d


def _recent_path() -> str:
    return os.path.join(config_dir(), "gui_recent_files.json")


def read_recent() -> list[str]:
    try:
        with open(_recent_path()) as f:
            return [p for p in json.load(f) if os.path.exists(p)]
    except Exception:
        return []


def add_recent(path: str, keep: int = 8) -> None:
    path = os.path.abspath(path)
    recent = [p for p in read_recent() if p != path]
    recent.insert(0, path)
    try:
        with open(_recent_path(), "w") as f:
            json.dump(recent[:keep], f)
    except Exception:
        pass


def clear_recent() -> None:
    try:
        os.remove(_recent_path())
    except OSError:
        pass


def _history_path() -> str:
    return os.path.join(config_dir(), "gui_run_history.jsonl")


def append_history(record: dict[str, Any]) -> None:
    try:
        with open(_history_path(), "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def read_history() -> list[dict[str, Any]]:
    try:
        with open(_history_path()) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []
    except Exception:
        return []


def clear_history() -> None:
    try:
        open(_history_path(), "w").close()
    except OSError:
        pass
