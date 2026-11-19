"""CLI agents: usage per turn (``turn_done``) and the discovery of the agents' commands."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from normet.gui import assistant_cli as ac  # noqa: E402


class _Server:
    url = "http://127.0.0.1:1/mcp"
    token = "t"
    tools: tuple[str, ...] = ()


def _usage(cls, events):
    chat = cls("/bin/true", "m", "", _Server())
    got = []
    chat.turn_done.connect(got.append)
    try:
        for e in events:
            chat.parse(e)
    finally:
        chat.close()
    return got


def test_claude_usage_sums_cache_tokens():
    ev = {
        "type": "result",
        "subtype": "success",
        "usage": {
            "input_tokens": 3,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 100,
            "output_tokens": 7,
        },
    }
    assert _usage(ac.ClaudeChat, [ev]) == [{"input": 113, "output": 7}]


def test_claude_error_still_reports_usage():
    chat = ac.ClaudeChat("/bin/true", "m", "", _Server())
    failed, done = [], []
    chat.failed.connect(failed.append)
    chat.turn_done.connect(done.append)
    chat.parse({"type": "result", "is_error": True, "result": "boom"})
    chat.close()
    assert failed == ["claude: boom"] and done == [{"input": 0, "output": 0}]


def test_codex_usage_on_turn_completed():
    ev = {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}}
    assert _usage(ac.CodexChat, [ev]) == [{"input": 5, "output": 2}]


def test_agy_usage_in_result():
    ev = {
        "event": "result",
        "result": {"status": "SUCCESS", "usage": {"input_tokens": 9, "output_tokens": 4}},
    }
    assert _usage(ac.AgyChat, [ev]) == [{"input": 9, "output": 4}]


def test_copilot_reports_zeros():
    assert _usage(ac.CopilotChat, [{"type": "result"}]) == [{"input": 0, "output": 0}]


def test_no_usage_without_a_result():
    assert _usage(ac.CodexChat, [{"type": "thread.started", "thread_id": "x"}]) == []


def test_find_cli_searches_version_manager_folders(tmp_path, monkeypatch):
    for v in ("v9.0.0", "v22.11.0"):
        d = tmp_path / ".nvm" / "versions" / "node" / v / "bin"
        d.mkdir(parents=True)
        f = d / "fakeagent"
        f.write_text("#!/bin/sh\n")
        f.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(ac, "_FOUND", {})
    monkeypatch.setattr(ac, "LOGIN_PATH", None)
    found = ac.find_cli("fakeagent")
    assert found is not None and "v22.11.0" in found


def test_find_cli_falls_back_to_login_path(tmp_path, monkeypatch):
    f = tmp_path / "otheragent"
    f.write_text("#!/bin/sh\n")
    f.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(ac, "_FOUND", {})
    monkeypatch.setattr(ac, "LOGIN_PATH", None)
    assert ac.find_cli("otheragent") is None
    monkeypatch.setattr(ac, "LOGIN_PATH", str(tmp_path))
    assert ac.find_cli("otheragent") == str(f)


@pytest.mark.skipif(os.name == "nt", reason="needs a POSIX shell")
def test_read_login_path_reads_the_shells_path(tmp_path, monkeypatch):
    sh = tmp_path / "fakeshell"
    sh.write_text('#!/bin/sh\necho noise\nprintf "\\n__NORMET_PATH__/a:/b\\n"\n')
    sh.chmod(0o755)
    monkeypatch.setenv("SHELL", str(sh))
    assert ac.read_login_path(timeout=10) == "/a:/b"


def test_login_path_in_background_runs_once(monkeypatch):
    import threading

    monkeypatch.setattr(ac, "_login_asked", False)
    monkeypatch.setattr(ac, "LOGIN_PATH", None)
    monkeypatch.setattr(ac, "read_login_path", lambda: "/x:/y")
    seen = threading.Event()
    ac.login_path_in_background(seen.set)
    assert seen.wait(5) and ac.LOGIN_PATH == "/x:/y"
    ac.login_path_in_background(lambda: pytest.fail("asked twice"))


def test_stream_interrupted_wording():
    assert ac.stream_interrupted(ac.STREAM_INTERRUPTED)
    assert not ac.stream_interrupted("rate limited")
