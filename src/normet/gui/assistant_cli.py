"""The Assistant through a command-line agent signed in to the user's own plan,
which no API key gives: Claude Code (a Claude plan) or Codex (ChatGPT).

Adapted from the cume app's. The agent brings its own loop and model; what it
is given are the window's tools, the same the API models get
(:class:`normet.gui.assistant.WindowTools`), served by the window over MCP's
HTTP transport on 127.0.0.1 with a token drawn per window
(:class:`normet.agent.mcp_http.ToolServer`). How each agent is told of it:

- ``claude``: ``--mcp-config`` with ``--strict-mcp-config`` for the one call --
  no other server, nothing written to its configuration; ``--tools ""`` leaves
  it no tool of its own (no shell, no files) and ``--allowedTools`` lets it call
  the window's unasked.
- ``codex``: ``-c mcp_servers.normet-window.url=...`` with
  ``--ignore-user-config`` for the one call (the token in an environment
  variable, not on the command line); its shell is held to a read-only sandbox
  in an empty folder.

Each message is one process; the conversation is carried from one to the next
by its id (``claude --resume``, ``codex exec resume``). Stop ends the process;
the conversation is kept for the next message.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from normet.agent.mcp_http import SERVER_NAME, ToolServer

#: The agents offered, by the name the model list shows: (command, model, effort).
AGENTS: dict[str, tuple[str, str, str]] = {
    "Claude Code · Sonnet (your Claude plan)": ("claude", "sonnet", ""),
    "Claude Code · Opus (your Claude plan)": ("claude", "opus", ""),
    "Codex (your ChatGPT plan)": ("codex", "", ""),
}

#: Where installers put an agent when a GUI app's PATH does not reach it.
_EXTRA_DIRS = ("~/.local/bin", "/opt/homebrew/bin", "/usr/local/bin", "~/.npm-global/bin", "~/bin")

#: Told to Codex with the first message, which has no system prompt of its own.
NOTE = "(Above: your instructions. Below: the user's first message.)"


def find_cli(command: str) -> str | None:
    """The agent's executable: on PATH, else where installers put it."""
    found = shutil.which(command)
    if found:
        return found
    extra = os.pathsep.join(os.path.expanduser(d) for d in _EXTRA_DIRS)
    return shutil.which(command, path=extra)


def _env_with(command: str, env: dict[str, str]) -> dict[str, str]:
    """*env* with the command's folder first on PATH: an npm-installed agent is a
    Node script that finds ``node`` beside it."""
    folder = os.path.dirname(command)
    return dict(env, PATH=folder + os.pathsep + env.get("PATH", ""))


class _Reader(QThread):
    """Reads an agent's events, a JSON object per line, and hands each to its chat."""

    line = Signal(object)

    def __init__(self, proc: subprocess.Popen[str], chat: CliChat) -> None:
        super().__init__()
        self.proc, self.chat = proc, chat
        self.raw: list[str] = []

    def run(self) -> None:
        assert self.proc.stdout is not None
        for text in self.proc.stdout:
            text = text.strip()
            if not text:
                continue
            try:
                self.line.emit(json.loads(text))
            except ValueError:  # not an event: an error the agent printed
                self.raw.append(text)
        self.proc.wait()


class CliChat(QObject):
    """The conversation with an agent, a process per message.

    Signals: ``text`` (the reply's text as it comes), ``step`` (a tool of the
    agent's own, by name), ``failed(str)`` and ``ended()`` (the message's
    process is gone).
    """

    text = Signal(str)
    step = Signal(str)
    failed = Signal(str)
    ended = Signal()

    def __init__(self, command: str, model: str, effort: str, server: ToolServer) -> None:
        super().__init__()
        self.command, self.model, self.effort, self.server = command, model, effort, server
        self.proc: subprocess.Popen[str] | None = None
        self.conversation: str | None = None
        self.workspace = tempfile.mkdtemp(prefix="normet-agent-")  # empty: nothing to edit
        self._reader: _Reader | None = None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # what each agent is given, and how its events read
    def argv(self, system: str) -> list[str]:
        raise NotImplementedError

    def message(self, text: str, system: str) -> str:
        raise NotImplementedError

    def parse(self, e: dict[str, Any]) -> None:
        raise NotImplementedError

    def env(self) -> dict[str, str]:
        return dict(os.environ)

    def send(self, text: str, system: str) -> None:
        """One message: a process that answers it and ends."""
        if self.running():
            raise RuntimeError("the agent is still answering the last message")
        msg = self.message(text, system)
        self.proc = subprocess.Popen(
            self.argv(system),
            cwd=self.workspace,
            env=_env_with(self.command, self.env()),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader = _Reader(self.proc, self)
        self._reader.line.connect(self._event)
        self._reader.finished.connect(self._done)
        self._reader.start()
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(msg + "\n")
            self.proc.stdin.close()
        except (OSError, ValueError) as exc:
            self.failed.emit(f"{os.path.basename(self.command)}: {exc}")

    def _event(self, e: Any) -> None:
        if isinstance(e, dict):
            self.parse(e)

    def _done(self) -> None:
        code = self.proc.returncode if self.proc is not None else 0
        raw = self._reader.raw if self._reader is not None else []
        if code not in (0, None) and code > 0 and raw:
            self.failed.emit(f"{os.path.basename(self.command)}: " + " ".join(raw[-3:])[:500])
        self.ended.emit()

    def stop(self) -> None:
        """End the message's process; the conversation is kept for the next."""
        if self.proc is not None and self.running():
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def close(self) -> None:
        self.stop()
        shutil.rmtree(self.workspace, ignore_errors=True)


class ClaudeChat(CliChat):
    """Claude Code: the window's server for this call alone, no tool of its own,
    the system prompt as its own, the session resumed by id."""

    def argv(self, system: str) -> list[str]:
        cfg = {
            "mcpServers": {
                SERVER_NAME: {
                    "type": "http",
                    "url": self.server.url,
                    "headers": {"Authorization": "Bearer " + self.server.token},
                }
            }
        }
        cmd = [
            self.command,
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--mcp-config",
            json.dumps(cfg),
            "--strict-mcp-config",
            "--tools",
            "",
            "--allowedTools",
            "mcp__" + SERVER_NAME,
            "--setting-sources",
            "",
            "--system-prompt",
            system,
        ]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        if self.conversation:
            cmd += ["--resume", self.conversation]
        return cmd

    def env(self) -> dict[str, str]:
        # Started from inside a Claude Code session, it would be taken for that
        # session's own child.
        return {
            k: v for k, v in os.environ.items() if not k.startswith(("CLAUDECODE", "CLAUDE_CODE_"))
        }

    def message(self, text: str, system: str) -> str:
        return json.dumps({"type": "user", "message": {"role": "user", "content": text}})

    def parse(self, e: dict[str, Any]) -> None:
        kind = e.get("type")
        self.conversation = e.get("session_id") or self.conversation
        if kind == "assistant":
            for block in (e.get("message") or {}).get("content") or []:
                if block.get("type") == "text" and block.get("text"):
                    self.text.emit(block["text"])
                elif block.get("type") == "tool_use" and not str(block.get("name", "")).startswith(
                    "mcp__" + SERVER_NAME
                ):
                    self.step.emit(str(block.get("name")))
        elif kind == "result" and (e.get("is_error") or e.get("subtype") not in (None, "success")):
            self.failed.emit(f"claude: {e.get('result') or e.get('subtype')}")


class CodexChat(CliChat):
    """Codex: the window's server for this call alone (the user's config.toml not
    read), the system prompt in the first message, the thread resumed by id."""

    TOKEN_ENV = "NORMET_WINDOW_TOKEN"

    def argv(self, system: str) -> list[str]:
        key = f"mcp_servers.{SERVER_NAME}"
        cmd = [
            self.command,
            "exec",
            "--json",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "-C",
            self.workspace,
            "-c",
            f'{key}.url="{self.server.url}"',
            "-c",
            f'{key}.bearer_token_env_var="{self.TOKEN_ENV}"',
            # Its own limit on a call is a minute; a training run can take longer.
            "-c",
            f"{key}.tool_timeout_sec=1800",
            # Unapproved, exec cancels every call ("user cancelled MCP tool call").
            "-c",
            f'{key}.default_tools_approval_mode="approve"',
        ]
        if self.model:
            cmd += ["-m", self.model]
        if self.effort:
            cmd += ["-c", f'model_reasoning_effort="{self.effort}"']
        if self.conversation:
            cmd += ["resume", self.conversation]
        return [*cmd, "-"]

    def env(self) -> dict[str, str]:
        return dict(os.environ, **{self.TOKEN_ENV: self.server.token})

    def message(self, text: str, system: str) -> str:
        if self.conversation is None:
            text = system + "\n" + NOTE + "\n" + text
        return text

    def parse(self, e: dict[str, Any]) -> None:
        kind = e.get("type", "")
        if kind == "thread.started":
            self.conversation = e.get("thread_id") or self.conversation
        elif kind in ("item.started", "item.completed"):
            item = e.get("item") or {}
            it = item.get("type")
            if kind == "item.completed" and it == "agent_message" and item.get("text"):
                self.text.emit(item["text"] + "\n")
            elif kind == "item.started" and it == "command_execution":
                self.step.emit("shell: " + str(item.get("command", ""))[:80])
            elif (
                kind == "item.started"
                and it == "mcp_tool_call"
                and item.get("server") != SERVER_NAME
            ):
                self.step.emit(f"{item.get('server')}/{item.get('tool')}")
        elif kind == "turn.failed":
            err = e.get("error") if isinstance(e.get("error"), dict) else e
            msg = str((err or {}).get("message") or kind)
            try:  # the provider's own error, as JSON in it
                msg = json.loads(msg)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                pass
            self.failed.emit(f"codex: {msg}")


def make_chat(agent: str, server: ToolServer) -> CliChat | None:
    """The conversation for an agent named in :data:`AGENTS`; None where its
    command is not installed."""
    command, model, effort = AGENTS[agent]
    path = find_cli(command)
    if path is None:
        return None
    cls = ClaudeChat if command == "claude" else CodexChat
    return cls(path, model, effort, server)
