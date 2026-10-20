"""The Assistant through a command-line agent signed in to the user's own plan,
which no API key gives: Claude Code (a Claude plan), Codex (ChatGPT), Google
Antigravity's agy (Google AI Pro) or GitHub Copilot's CLI (a Copilot plan).

The agent brings its own loop and model; what it
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
- ``copilot``: ``--additional-mcp-config`` for the one call, its built-in
  servers off, and ``--available-tools`` naming the window's tools alone (no
  shell, no files).
- ``agy``: its own configuration (``agy mcp add normet-window URL``), which is
  global: the entry is enabled while agy answers a message and disabled once it
  has, so an agy started for anything else meanwhile finds nothing there. Its
  own tools are denied by hooks in its (empty) workspace (:func:`agy_guard`).

Each message is one process; the conversation is carried from one to the next
by its id (``claude --resume``, ``codex exec resume``, ``agy --conversation``,
``copilot --session-id``). Stop ends the process; the conversation is kept.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from normet.agent.mcp_http import SERVER_NAME, ToolServer

#: The agents: name -> (command, how the model list names it).
AGENTS: dict[str, tuple[str, str]] = {
    "claude-code": ("claude", "Claude Code"),
    "codex": ("codex", "Codex"),
    "agy": ("agy", "Antigravity agy"),
    "copilot": ("copilot", "GitHub Copilot"),
}

CLAUDE_MODELS = ("sonnet", "opus", "haiku")
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
#: Codex: "default" is the model of the user's own ~/.codex/config.toml (a call
#: made with --ignore-user-config does not read it); the others its cache lists.
CODEX_EFFORTS = ("low", "medium", "high")
#: agy's models, each offered at these efforts (``agy models``, 2026-10).
AGY_EFFORTS: dict[str, tuple[str, ...]] = {
    "gemini-4-argon": ("low", "medium", "high"),
    "gemini-3.8-flash": ("low", "medium", "high"),
    "claude-sonnet-5-5": ("low", "medium", "high"),
    "claude-opus-5-5": ("low", "medium", "high"),
    "gpt-oss-120b": ("medium",),
}
COPILOT_MODELS = ("default", "auto")
COPILOT_EFFORTS = ("low", "medium", "high")

#: Where installers put an agent when a GUI app's PATH does not reach it.
_EXTRA_DIRS = ("~/.local/bin", "/opt/homebrew/bin", "/usr/local/bin", "~/.npm-global/bin", "~/bin")

#: Told with the first message to an agent that takes no system prompt of its own.
NOTE = (
    "Use only the tools of the normet-window server: they act on the analyst's open window. "
    "Do not use a terminal, files or a browser for this work."
)

#: What agy reports when its model stream broke (seen repeatedly):
#: the turn ends on this line. A message ending so is asked once more on its own.
STREAM_INTERRUPTED = "The stream was interrupted. Please continue the task you were working on."


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


# -- models and efforts -------------------------------------------------------------------
def _codex_cache() -> list[dict[str, Any]]:
    home = os.path.expanduser(os.environ.get("CODEX_HOME") or os.path.join("~", ".codex"))
    try:
        with open(os.path.join(home, "models_cache.json"), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    models = data.get("models") if isinstance(data, dict) else data
    return [m for m in models or [] if isinstance(m, dict) and m.get("slug")]


def models(agent: str) -> tuple[str, ...]:
    """The models an agent is offered with; "default" is the agent's own choice."""
    if agent == "claude-code":
        return CLAUDE_MODELS
    if agent == "codex":
        return ("default", *(m["slug"] for m in _codex_cache() if m.get("visibility") == "list"))
    if agent == "agy":
        return tuple(AGY_EFFORTS)
    return COPILOT_MODELS


def efforts(agent: str, model: str) -> tuple[str, ...]:
    """The efforts a model is offered at by its agent; () where none is chosen."""
    if agent == "claude-code":
        return CLAUDE_EFFORTS
    if agent == "codex":
        for m in _codex_cache():
            if m["slug"] == model:
                levels = tuple(
                    x["effort"]
                    for x in m.get("supported_reasoning_levels") or []
                    if isinstance(x, dict) and x.get("effort")
                )
                if levels:
                    return levels
        return CODEX_EFFORTS
    if agent == "agy":
        return AGY_EFFORTS.get(model, ())
    return () if model in COPILOT_MODELS else COPILOT_EFFORTS


def agy_model(name: str, effort: str) -> str:
    """agy's name for a model at an effort: ``gemini-4-argon-high``."""
    offered = AGY_EFFORTS.get(name)
    if not offered:
        return name
    return f"{name}-{effort if effort in offered else offered[len(offered) // 2]}"


def choices() -> list[tuple[str, str, str]]:
    """Every (label, agent, model) the panel's model list offers for the agents."""
    return [
        (f"{AGENTS[agent][1]} · {model}", agent, model)
        for agent in AGENTS
        for model in models(agent)
    ]


# -- the conversation ---------------------------------------------------------------------
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
    process is gone). ``interrupted`` is set when the agent's stream broke
    mid-answer, for the panel to ask once more.
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
        self.interrupted = False
        self._reader: _Reader | None = None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # what each agent is given, and how its events read
    def argv(self, system: str) -> list[str]:
        raise NotImplementedError

    def message(self, text: str, system: str) -> str | None:
        """What goes to the agent's input; None where the message is on its command line."""
        raise NotImplementedError

    def parse(self, e: dict[str, Any]) -> None:
        raise NotImplementedError

    def env(self) -> dict[str, str]:
        return dict(os.environ)

    def before(self) -> None:
        """Before the message's process starts."""

    def after(self) -> None:
        """Once it has ended."""

    def send(self, text: str, system: str) -> None:
        """One message: a process that answers it and ends."""
        if self.running():
            raise RuntimeError("the agent is still answering the last message")
        self.interrupted = False
        self.before()
        msg = self.message(text, system)  # before argv: a prompt may go in it
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
            if msg is not None:
                self.proc.stdin.write(msg + "\n")
            self.proc.stdin.close()
        except (OSError, ValueError) as exc:
            self.failed.emit(f"{os.path.basename(self.command)}: {exc}")

    def _event(self, e: Any) -> None:
        if isinstance(e, dict):
            self.parse(e)

    def _done(self) -> None:
        self.after()
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
        if self.model and self.model != "default":
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
            text = system + "\n" + NOTE + "\n\n" + text
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


class CopilotChat(CliChat):
    """GitHub Copilot's CLI: the window's server added for the one call, only its
    tools available, the prompt on the command line, the session named by an id
    of ours."""

    _prompt = ""

    def argv(self, system: str) -> list[str]:
        cfg = {
            "mcpServers": {
                SERVER_NAME: {
                    "type": "http",
                    "url": self.server.url,
                    "tools": ["*"],
                    "headers": {"Authorization": "Bearer " + self.server.token},
                }
            }
        }
        cmd = [
            self.command,
            "-p",
            self._prompt,
            "--output-format",
            "json",
            "--additional-mcp-config",
            json.dumps(cfg),
            "--disable-builtin-mcps",
            "--allow-tool",
            SERVER_NAME,
            "--available-tools",
            *[f"{SERVER_NAME}-{n}" for n in self.server.tools],
            "--no-ask-user",
            "--no-custom-instructions",
            "--no-color",
            "--session-id",
            str(self.conversation),
        ]
        if self.model and self.model != "default":
            cmd += ["--model", self.model]
        if self.effort and self.model not in COPILOT_MODELS:
            cmd += ["--reasoning-effort", self.effort]
        return cmd

    def message(self, text: str, system: str) -> None:
        if self.conversation is None:  # a new session, by an id of ours
            self.conversation = str(uuid.uuid4())
            text = system + "\n" + NOTE + "\n\n" + text
        self._prompt = text
        return None

    def parse(self, e: dict[str, Any]) -> None:
        kind, data = e.get("type", ""), e.get("data") or {}
        if kind == "assistant.message":
            if data.get("content"):
                self.text.emit(data["content"] + "\n")
        elif kind == "tool.execution_start":
            if data.get("mcpServerName") != SERVER_NAME:
                self.step.emit(str(data.get("toolName") or "tool"))
        elif kind.endswith(".error") or kind == "error":
            self.failed.emit(f"copilot: {data.get('message') or kind}")


# -- agy ------------------------------------------------------------------------------------
#: agy's own tools, by name, for Windows, where no shell reads its calls.
AGY_OWN_TOOLS = (
    "run_command|shell_exec|send_command_input|read_terminal|command_status|"
    "replace_file_content|multi_replace_file_content|write_to_file|file_change|propose_code|"
    "code_action|write_blob|delete_directory|move|edit_notebook|execute_notebook|"
    "read_notebook|grep_search|find_by_name|find|list_dir|list_directory|view_file_outline|"
    "view_code_item|view_content_chunk|search_web|read_url_content|open_browser_url|"
    "read_browser_page|browser_.*|execute_browser_javascript|capture_browser_.*|"
    "click_browser_pixel|list_browser_pages|invoke_subagent|define_subagent|"
    "manage_subagents|send_message|schedule|generate_image|git_commit|run_extension_code"
)
_DENY = (
    '{"decision":"deny","reason":"In the normet window the analysis is done with the tools '
    'of the window (normet-window) alone: no shell, files or web."}'
)


def agy_guard(workspace: str, home: str | None = None) -> str:
    """agy held to the window's tools, by hooks in its workspace
    (``.agents/hooks.json``, read for that workspace alone -- the user's own agy
    configuration is not touched). Asked about the window, an agy otherwise
    reaches for its shell, and a shell, files and the web reach the analyst's
    data, which only the window's tools are to read. Everything is denied but a
    call to the window's server, agy reading that server's tool schemas, and its
    task list; on Windows, agy's own tools by name. Returns the hooks file's path."""
    home = home or os.path.expanduser("~")
    folder = os.path.join(workspace, ".agents")
    os.makedirs(folder, exist_ok=True)
    rule: dict[str, Any]
    if os.name == "nt":  # pragma: no cover - Windows
        rule = {
            "matcher": f"^({AGY_OWN_TOOLS})$",
            "hooks": [{"type": "command", "timeout": 10, "command": f"echo {_DENY}"}],
        }
    else:
        schemas = os.path.join(home, ".gemini", "antigravity-cli", "mcp", SERVER_NAME) + "/"
        script = os.path.join(folder, "window_only.sh")
        with open(script, "w") as f:
            f.write(
                f"""#!/bin/sh
# normet: agy in the Assistant calls the window's tools and nothing else (assistant_cli.agy_guard).
allow='{{"decision":"allow"}}'
deny='{_DENY}'
in=$(cat)
case "$in" in
  *'"name":"call_mcp_tool"'*)
    case "$in" in *'"ServerName":"{SERVER_NAME}"'*) echo "$allow" ;; *) echo "$deny" ;; esac ;;
  *'"name":"view_file"'*)
    case "$in" in
      *..*) echo "$deny" ;;
      *'"AbsolutePath":"{schemas}'*) echo "$allow" ;;
      *) echo "$deny" ;;
    esac ;;
  *'"name":"manage_task"'*) echo "$allow" ;;
  *) echo "$deny" ;;
esac
"""
            )
        rule = {
            "matcher": "*",
            "hooks": [{"type": "command", "timeout": 10, "command": "sh window_only.sh"}],
        }
    path = os.path.join(folder, "hooks.json")
    with open(path, "w") as f:
        json.dump({"normet-window-only": {"PreToolUse": [rule]}}, f, indent=1)
    return path


def _agy(agy: str, *args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [agy, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_env_with(agy, dict(os.environ)),
    )


def register(agy: str, server: ToolServer) -> None:
    """The window's server in agy's configuration, enabled (add updates an entry there)."""
    p = _agy(
        agy,
        "mcp",
        "add",
        "--header",
        f"Authorization: Bearer {server.token}",
        SERVER_NAME,
        server.url,
    )
    if p.returncode != 0:
        raise RuntimeError(f"agy mcp add failed: {(p.stderr or p.stdout).strip()[:300]}")
    _agy(agy, "mcp", "enable", SERVER_NAME)


def forget(agy: str) -> None:
    """The window's entry taken out of agy's configuration, as it was before."""
    try:
        _agy(agy, "mcp", "remove", SERVER_NAME, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pass


def unregister(agy: str) -> None:
    """Disabled: an agy started for anything else does not look for the window."""
    try:
        _agy(agy, "mcp", "disable", SERVER_NAME, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pass


class AgyChat(CliChat):
    """agy: the model named with its effort, the system prompt in the first message
    (agy takes none of its own), the window's entry enabled for the message, and
    its own tools denied in its workspace (:func:`agy_guard`)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        agy_guard(self.workspace)
        self._said = ""

    def argv(self, system: str) -> list[str]:
        cmd = [
            self.command,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--model",
            agy_model(self.model, self.effort),
            "--sandbox",
        ]
        if self.conversation:
            cmd += ["--conversation", self.conversation]
        return [*cmd, "-p="]

    def message(self, text: str, system: str) -> str:
        if self.conversation is None:
            text = system + "\n" + NOTE + "\n\n" + text
        self._said = ""
        return json.dumps({"event": "user", "message": {"role": "user", "content": text}})

    def before(self) -> None:
        register(self.command, self.server)
        self._registered = True

    def close(self) -> None:
        super().close()
        if getattr(self, "_registered", False):
            forget(self.command)
            self._registered = False

    def after(self) -> None:
        unregister(self.command)
        self.interrupted = STREAM_INTERRUPTED in self._said[-len(STREAM_INTERRUPTED) - 20 :]

    def parse(self, e: dict[str, Any]) -> None:
        kind = e.get("event")
        if kind == "init":
            self.conversation = e.get("conversation_id") or self.conversation
        elif kind == "step_update":
            s = e.get("step_update") or {}
            self.conversation = s.get("conversation_id") or self.conversation
            st = s.get("step_type", "")
            if st == "agent_response" and s.get("text_delta"):
                self._said += s["text_delta"]
                self.text.emit(s["text_delta"])
            elif st == "tool" and s.get("state") == "ACTIVE":
                # Its reading of the tools' schemas, and its calls to them, are
                # shown as the server takes them.
                if s.get("tool_name") not in ("call_mcp_tool", "view_file"):
                    self.step.emit(s.get("tool_name") or "tool")
        elif kind == "result":
            r = e.get("result") or {}
            self.conversation = r.get("conversation_id") or self.conversation
            if r.get("status") not in (None, "SUCCESS"):
                self.failed.emit(f"agy: {r.get('error') or r.get('status')}")


CHATS: dict[str, type[CliChat]] = {
    "claude-code": ClaudeChat,
    "codex": CodexChat,
    "agy": AgyChat,
    "copilot": CopilotChat,
}


def make_chat(agent: str, model: str, effort: str, server: ToolServer) -> CliChat | None:
    """The conversation with *agent* (a key of :data:`AGENTS`) at *model* and
    *effort*; None where its command is not installed."""
    path = find_cli(AGENTS[agent][0])
    if path is None:
        return None
    return CHATS[agent](path, model, effort, server)
