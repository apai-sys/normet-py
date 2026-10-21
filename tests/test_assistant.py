"""The Assistant: the model loop, the window's tools over HTTP, the command-line
agents and the panel in the window (offscreen; the GUI parts skip without PySide6)."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from normet.agent import llm
from normet.agent.mcp_http import SERVER_NAME, ToolServer


# -- the loop -----------------------------------------------------------------------------
def _echo(
    word: str, times: int = 1, tags: list[str] | None = None, scale: float | None = None
) -> str:
    """Say a word."""
    return " ".join([word] * times)


def test_tool_schema_from_signature():
    t = llm.as_tool(_echo)
    assert t.name == "_echo" and t.description == "Say a word."
    props = t.schema["properties"]
    assert props["word"] == {"type": "string"}
    assert props["times"] == {"type": "integer"}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["scale"] == {"type": "number"}
    assert t.schema["required"] == ["word"]


def _script(*replies):
    """A FakeModel giving *replies* in turn."""
    queue = list(replies)

    def policy(messages, tools):
        return queue.pop(0)

    return llm.FakeModel(policy)


def test_converse_runs_tools_until_the_answer():
    def boom() -> str:
        raise ValueError("no table")

    tools = [llm.as_tool(_echo), llm.as_tool(boom)]
    model = _script(
        llm.Reply(
            "Looking.",
            [llm.Call("1", "_echo", {"word": "hi", "times": 2}), llm.Call("2", "boom", {})],
            {"input": 5, "output": 1},
        ),
        llm.Reply("", [llm.Call("3", "nope", {})], {}),
        llm.Reply("Done.", [], {"input": 2, "output": 3}),
    )
    messages = [{"role": "user", "content": "go"}]
    seen = []
    turn = llm.converse(model, tools, "sys", messages, on_event=seen.append)
    assert turn.stopped == "answered" and turn.usage == {"input": 7, "output": 4}
    outputs = [e for e in seen if e["type"] == "tool"]
    assert outputs[0]["output"] == "hi hi" and not outputs[0]["error"]
    assert outputs[1]["output"] == "ValueError: no table" and outputs[1]["error"]
    assert outputs[2]["error"] and "no tool 'nope'" in outputs[2]["output"]
    assert [m["role"] for m in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


def test_converse_stops_and_runs_out_of_turns():
    stop = threading.Event()
    stop.set()
    turn = llm.converse(_script(), [], "sys", [{"role": "user", "content": "go"}], stop=stop)
    assert turn.stopped == "stopped"
    loop = llm.FakeModel(lambda m, t: llm.Reply("", [llm.Call("x", "_echo", {"word": "a"})], {}))
    messages = [{"role": "user", "content": "go"}]
    turn = llm.converse(loop, [llm.as_tool(_echo)], "sys", messages, max_turns=2)
    assert turn.stopped == "turns" and messages[-1]["role"] == "assistant"


def test_model_names_and_keys(monkeypatch):
    with pytest.raises(ValueError, match="provider:model"):
        llm.Model("nobody:x")
    with pytest.raises(ValueError, match="provider:model"):
        llm.Model("anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm.Model("anthropic:claude-sonnet-5-5")
    assert llm.Model("ollama:qwen3:8b").model == "qwen3:8b"


def test_adapters_write_and_read_each_format(monkeypatch):
    sent = {}

    def fake_post(url, body, headers, **kw):
        sent["url"], sent["body"], sent["headers"] = url, body, headers
        if url.endswith("/v1/messages"):
            return {
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "t1", "name": "_echo", "input": {"word": "w"}},
                ],
                "usage": {"input_tokens": 3, "output_tokens": 4},
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "_echo", "arguments": '{"word": "w"}'},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }

    monkeypatch.setattr(llm, "_post", fake_post)
    history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "text": "", "calls": [llm.Call("a", "_echo", {"word": "x"})]},
        {"role": "tool", "results": [{"id": "a", "text": "x", "error": False}]},
    ]
    tools = [llm.as_tool(_echo)]

    reply = llm.Model("anthropic:claude-sonnet-5-5", api_key="k").chat("sys", history, tools)
    assert sent["headers"]["x-api-key"] == "k" and sent["body"]["system"] == "sys"
    assert sent["body"]["messages"][2]["content"][0]["type"] == "tool_result"
    assert reply.text == "ok" and reply.calls[0].arguments == {"word": "w"}
    assert reply.usage == {"input": 3, "output": 4}

    reply = llm.Model("ollama:qwen3:8b").chat("sys", history, tools)
    assert sent["url"].endswith("/chat/completions") and "Authorization" not in sent["headers"]
    assert sent["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["body"]["messages"][3] == {"role": "tool", "tool_call_id": "a", "content": "x"}
    assert reply.calls[0].name == "_echo" and reply.usage == {"input": 1, "output": 2}


# -- the window's tools over HTTP -------------------------------------------------------
def _rpc(server, body, token=True):
    req = urllib.request.Request(
        server.url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "Content-Type": "application/json",
            **({"Authorization": "Bearer " + server.token} if token else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, (json.loads(r.read() or b"null"))


def test_tool_server_speaks_mcp_over_http():
    calls = []
    server = ToolServer(
        [llm.as_tool(_echo)], instructions="be brief", on_call=lambda n, a: calls.append(n)
    )
    try:
        code, r = _rpc(server, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert code == 200 and r["result"]["serverInfo"]["name"] == SERVER_NAME
        assert r["result"]["instructions"] == "be brief"
        code, _ = _rpc(server, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert code == 202
        _, r = _rpc(server, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert [t["name"] for t in r["result"]["tools"]] == ["_echo"]
        _, r = _rpc(
            server,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "_echo", "arguments": {"word": "hi"}},
            },
        )
        assert r["result"] == {"content": [{"type": "text", "text": "hi"}], "isError": False}
        _, r = _rpc(
            server,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "_echo", "arguments": {}},
            },
        )
        assert r["result"]["isError"]
        _, r = _rpc(server, {"jsonrpc": "2.0", "id": 5, "method": "nope"})
        assert r["error"]["code"] == -32601
        assert calls == ["_echo", "_echo"]
        with pytest.raises(urllib.error.HTTPError) as err:
            _rpc(server, {"jsonrpc": "2.0", "id": 6, "method": "ping"}, token=False)
        assert err.value.code == 401
    finally:
        server.close()


# -- the command-line agents ------------------------------------------------------------
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _wait(qapp, until, timeout=60.0):
    t0 = time.time()
    while not until():
        qapp.processEvents()
        if time.time() - t0 > timeout:
            raise TimeoutError
        time.sleep(0.01)


def test_claude_and_codex_command_lines():
    from normet.gui.assistant_cli import ClaudeChat, CodexChat, find_cli

    server = ToolServer([])
    try:
        claude = ClaudeChat("/x/claude", "sonnet", "", server)
        argv = claude.argv("SYS")
        assert argv[argv.index("--tools") + 1] == "" and "--strict-mcp-config" in argv
        cfg = json.loads(argv[argv.index("--mcp-config") + 1])
        assert cfg["mcpServers"][SERVER_NAME]["url"] == server.url
        assert argv[argv.index("--allowedTools") + 1] == "mcp__" + SERVER_NAME
        claude.conversation = "s1"
        assert argv != claude.argv("SYS") and claude.argv("SYS")[-2:] == ["--resume", "s1"]
        assert json.loads(claude.message("hi", "SYS"))["message"]["content"] == "hi"
        got = []
        claude.text.connect(got.append)
        claude.parse(
            {
                "type": "assistant",
                "session_id": "s2",
                "message": {"content": [{"type": "text", "text": "yo"}]},
            }
        )
        assert got == ["yo"] and claude.conversation == "s2"

        codex = CodexChat("/x/codex", "", "", server)
        argv = codex.argv("SYS")
        assert argv[:2] == ["/x/codex", "exec"] and argv[-1] == "-" and "read-only" in argv
        assert server.token not in " ".join(argv)  # the token goes in the environment
        assert codex.env()[CodexChat.TOKEN_ENV] == server.token
        assert codex.message("hi", "SYS").startswith("SYS\n")
        codex.parse({"type": "thread.started", "thread_id": "t1"})
        assert codex.message("hi", "SYS") == "hi" and codex.argv("SYS")[-3:] == [
            "resume",
            "t1",
            "-",
        ]
        claude.close()
        codex.close()
    finally:
        server.close()
    assert find_cli("no-such-agent-here") is None


def test_an_agent_process_end_to_end(qapp, tmp_path):
    from normet.gui.assistant_cli import ClaudeChat

    script = tmp_path / "fake-claude"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "msg = json.loads(sys.stdin.readline())['message']['content']\n"
        "print(json.dumps({'type': 'system', 'session_id': 'abc'}))\n"
        "print(json.dumps({'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'you said ' + msg}]}}))\n"
        "print(json.dumps({'type': 'result', 'subtype': 'success'}))\n"
    )
    script.chmod(0o755)
    server = ToolServer([])
    chat = ClaudeChat(str(script), "", "", server)
    got, ended = [], []
    chat.text.connect(got.append)
    chat.ended.connect(lambda: ended.append(True))
    try:
        chat.send("hello", "SYS")
        _wait(qapp, lambda: ended)
        assert got == ["you said hello"] and chat.conversation == "abc"
    finally:
        chat.close()
        server.close()


# -- the panel in the window ------------------------------------------------------------
@pytest.fixture
def win(qapp, monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings

    from normet.gui import assistant
    from normet.gui.main_window import MainWindow

    ini = str(tmp_path / "normet-gui.ini")  # never the user's own settings
    monkeypatch.setattr(assistant, "_settings", lambda: QSettings(ini, QSettings.Format.IniFormat))
    w = MainWindow()
    yield w
    w.close()


def test_window_tools_read_and_propose(win, qapp):
    tools = win.assistant.tools
    assert "no table open" in tools.window()
    with pytest.raises(ValueError, match="no table"):
        tools.inspect_data()
    win.load_example()
    assert "target: PM2.5" in tools.window()
    assert "r with PM2.5" in tools.inspect_data()
    with pytest.raises(ValueError, match="no model"):
        tools.model_fit()
    with pytest.raises(ValueError, match="nothing is normalised"):
        tools.normalised()
    with pytest.raises(ValueError, match="no column"):
        tools.propose_setup("PM2.5", ["nope"], "r", 0.5)
    with pytest.raises(ValueError, match="cannot also"):
        tools.propose_setup("PM2.5", ["PM2.5", "t2m"], "r", 0.5)
    said = tools.propose_setup(
        "NO2", ["t2m", "blh"], "Wind and mixing.", 0.8, resample_vars=["t2m", "blh"]
    )
    assert "card" in said and "starts training" in said
    qapp.processEvents()
    from PySide6.QtWidgets import QPushButton

    trained = []
    win._run_train = lambda: trained.append(win.target_combo.currentText())
    apply = [
        b for b in win.assistant.view.findChildren(QPushButton) if b.text() == "Apply and train"
    ]
    assert len(apply) == 1
    apply[0].click()
    assert win.target_combo.currentText() == "NO2"
    assert win._selected_features() == ["t2m", "blh"]
    assert win.assistant.pending_resample == ["t2m", "blh"]
    assert apply[0].text() == "Applied" and not apply[0].isEnabled()
    # The click trains at once, with the set-up just ticked (here the window did not
    # start: the model is told so with the next message).
    assert trained == ["NO2"]
    assert "did not start training" in win.assistant.notes[-1]
    assert [t.name for t in tools.tools()][:3] == ["window", "inspect_data", "model_fit"]


def test_a_conversation_in_the_panel(win, qapp, monkeypatch):
    win.load_example()
    dock = win.assistant
    seen = []

    def policy(messages, tools):
        seen.append(messages[-1])
        if len(seen) == 1:
            return llm.Reply("Reading the table.", [llm.Call("1", "inspect_data", {})], {})
        return llm.Reply("It has **8,760 rows**.", [], {})

    monkeypatch.setattr(dock, "_model", lambda: llm.FakeModel(policy))
    dock.composer.setPlainText("What is in the table?")
    dock.send()
    _wait(qapp, lambda: not dock.busy() and not dock.working.isVisible())
    assert seen[0]["content"].startswith("[Window now: table: ")
    assert seen[1]["role"] == "tool" and "r with PM2.5" in seen[1]["results"][0]["text"]
    assert [m["role"] for m in dock.messages] == ["user", "assistant", "tool", "assistant"]
    dock.new_conversation()
    assert dock.messages == []


def test_train_and_normalise_through_the_tools(win, qapp):
    pytest.importorskip("lightgbm")
    win.load_example()
    win.df_raw = win.df_raw.iloc[: 24 * 45].reset_index(drop=True)
    win.backend_combo.setCurrentText("lightgbm")
    win.budget_spin.setValue(3)
    win.norm_samples.setValue(5)
    box = {}

    def work():
        box["fit"] = win.assistant.tools.train()
        box["norm"] = win.assistant.tools.normalise()

    t = threading.Thread(target=work)
    t.start()
    _wait(qapp, lambda: not t.is_alive(), timeout=300)
    assert "Model fit" in box["fit"], box["fit"]
    assert "Means by month" in box["norm"], box["norm"]
    assert "model trained on" in win.assistant.tools.window()


def test_unknown_agent_and_missing_key(win, qapp, monkeypatch):
    from normet.gui import assistant, assistant_cli

    dock = win.assistant
    monkeypatch.setattr(assistant, "find_key", lambda provider: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    dock.model_box.setCurrentText("anthropic:claude-sonnet-5-5")
    assert dock.key_btn.isEnabled()
    dock.send("hi")
    assert not dock.busy() and dock.messages == []
    assert dock.effort_box.isHidden()  # an API model has no effort to choose
    monkeypatch.setattr(assistant_cli, "find_cli", lambda c: None)
    dock.model_box.setCurrentText("Claude Code · sonnet")
    assert not dock.key_btn.isEnabled()
    assert not dock.effort_box.isHidden() and dock.effort_box.count() == 5
    dock.model_box.setCurrentText("GitHub Copilot · auto")
    assert dock.effort_box.isHidden()
    dock.model_box.setCurrentText("Antigravity agy · gpt-oss-120b")
    assert [dock.effort_box.itemText(i) for i in range(dock.effort_box.count())] == ["medium"]
    dock.send("hi")
    assert not dock.busy()


# -- agy and Copilot ----------------------------------------------------------------------
def test_agent_models_and_efforts(monkeypatch, tmp_path):
    from normet.gui import assistant_cli as cli

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "models_cache.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "gpt-x",
                        "visibility": "list",
                        "supported_reasoning_levels": [{"effort": "low"}, {"effort": "xhigh"}],
                    },
                    {"slug": "hidden", "visibility": "hide"},
                ]
            }
        )
    )
    assert cli.models("codex") == ("default", "gpt-x")
    assert cli.efforts("codex", "gpt-x") == ("low", "xhigh")
    assert cli.efforts("codex", "default") == cli.CODEX_EFFORTS
    assert cli.models("claude-code") == ("sonnet", "opus", "haiku")
    assert (
        cli.efforts("copilot", "auto") == ()
        and cli.efforts("copilot", "gpt-5.4") == cli.COPILOT_EFFORTS
    )
    assert cli.agy_model("gemini-4-argon", "high") == "gemini-4-argon-high"
    assert cli.agy_model("gpt-oss-120b", "low") == "gpt-oss-120b-medium"  # the one it offers
    assert cli.agy_model("someone-else", "high") == "someone-else"
    labels = [label for label, _, _ in cli.choices()]
    assert "Antigravity agy · gemini-4-argon" in labels and "Codex · gpt-x" in labels


def test_copilot_command_line_and_events():
    from normet.gui.assistant_cli import CopilotChat

    server = ToolServer([llm.as_tool(_echo)])
    try:
        chat = CopilotChat("/x/copilot", "gpt-5.4", "high", server)
        assert chat.message("hi", "SYS") is None and chat.conversation
        argv = chat.argv("SYS")
        assert argv[argv.index("-p") + 1].startswith("SYS\n") and argv[
            argv.index("-p") + 1
        ].endswith("hi")
        assert argv[argv.index("--available-tools") + 1] == f"{SERVER_NAME}-_echo"
        assert "--disable-builtin-mcps" in argv and argv[-2:] == ["--reasoning-effort", "high"]
        sid = chat.conversation
        chat.message("again", "SYS")
        assert chat.conversation == sid and chat.argv("SYS")[2] == "again"
        got, steps = [], []
        chat.text.connect(got.append)
        chat.step.connect(steps.append)
        chat.parse({"type": "assistant.message", "data": {"content": "ok"}})
        chat.parse(
            {
                "type": "tool.execution_start",
                "data": {"mcpServerName": SERVER_NAME, "toolName": "x"},
            }
        )
        chat.parse(
            {
                "type": "tool.execution_start",
                "data": {"mcpServerName": "other", "toolName": "shell"},
            }
        )
        assert got == ["ok\n"] and steps == ["shell"]
        chat.close()
    finally:
        server.close()


def test_agy_guard_allows_only_the_window(tmp_path):
    import subprocess

    from normet.gui.assistant_cli import agy_guard

    hooks = agy_guard(str(tmp_path / "ws"), home="/h")
    with open(hooks) as fh:
        rule = json.load(fh)["normet-window-only"]["PreToolUse"][0]
    assert rule["matcher"] == "*"

    def decide(event):
        out = subprocess.run(
            ["sh", "window_only.sh"],
            input=json.dumps(event, separators=(",", ":")),  # compact, as agy sends it
            capture_output=True,
            text=True,
            cwd=os.path.dirname(hooks),
        )
        return json.loads(out.stdout)["decision"]

    schemas = f"/h/.gemini/antigravity-cli/mcp/{SERVER_NAME}/tools.json"
    assert decide({"name": "call_mcp_tool", "args": {"ServerName": SERVER_NAME}}) == "allow"
    assert decide({"name": "call_mcp_tool", "args": {"ServerName": "elsewhere"}}) == "deny"
    assert decide({"name": "view_file", "args": {"AbsolutePath": schemas}}) == "allow"
    assert decide({"name": "view_file", "args": {"AbsolutePath": "/h/data.csv"}}) == "deny"
    assert decide({"name": "view_file", "args": {"AbsolutePath": schemas + "/../../x"}}) == "deny"
    assert decide({"name": "run_command", "args": {"CommandLine": "ls"}}) == "deny"
    assert decide({"name": "manage_task"}) == "allow"


def test_agy_registers_answers_and_is_asked_again_when_its_stream_breaks(qapp, tmp_path):
    from normet.gui.assistant_cli import STREAM_INTERRUPTED, AgyChat

    calls = tmp_path / "calls.txt"
    script = tmp_path / "agy"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"open({str(calls)!r}, 'a').write(' '.join(sys.argv[1:4]) + '\\n')\n"
        "if sys.argv[1] == 'mcp':\n    sys.exit(0)\n"
        "msg = json.loads(sys.stdin.readline())['message']['content']\n"
        "print(json.dumps({'event': 'init', 'conversation_id': 'c1'}))\n"
        f"text = 'done' if msg == {STREAM_INTERRUPTED!r} else 'partial ' + {STREAM_INTERRUPTED!r}\n"
        "print(json.dumps({'event': 'step_update', 'step_update': {'step_type': 'agent_response', 'text_delta': text}}))\n"
        "print(json.dumps({'event': 'result', 'result': {'status': 'SUCCESS'}}))\n"
    )
    script.chmod(0o755)
    server = ToolServer([])
    chat = AgyChat(str(script), "gemini-4-argon", "high", server)
    assert os.path.exists(os.path.join(chat.workspace, ".agents", "hooks.json"))
    argv = chat.argv("SYS")
    assert argv[argv.index("--model") + 1] == "gemini-4-argon-high" and "--sandbox" in argv
    got, ended = [], []
    chat.text.connect(got.append)
    chat.ended.connect(lambda: ended.append(True))
    try:
        chat.send("hello", "SYS")
        _wait(qapp, lambda: ended)
        assert chat.interrupted and chat.conversation == "c1"
        assert chat.argv("SYS")[-3:] == ["--conversation", "c1", "-p="]
        chat.send(STREAM_INTERRUPTED, "SYS")
        _wait(qapp, lambda: len(ended) == 2)
        assert not chat.interrupted and got[-1] == "done"
    finally:
        chat.close()
        server.close()
    lines = calls.read_text().splitlines()
    assert lines[0].startswith("mcp add --header") and lines[1] == f"mcp enable {SERVER_NAME}"
    assert lines[3] == f"mcp disable {SERVER_NAME}"  # after the message, off again
    assert lines[-1] == f"mcp remove {SERVER_NAME}"  # and gone once the chat closes


def test_the_window_stamp_is_not_shown_as_the_reply():
    from normet.gui.assistant import unstamp

    assert (
        unstamp("[Window now: table: x; target: y] The table has 8 rows.")
        == "The table has 8 rows."
    )
    assert unstamp("No stamp [Window now: here] stays.") == "No stamp [Window now: here] stays."
    # Echoed without its opening bracket, as a model was seen to do.
    assert unstamp("Window now: table: x; results: normalise]\n当前窗口") == "当前窗口"


# -- full automatic: the turn worked to the chain's end ---------------------------------
class _Box:
    """WindowTools as _ChatThread sees it: one tool, the window's thread being this one."""

    def __init__(self, tool):
        self.stop, self.called, self.win = threading.Event(), set(), None
        self._tool = tool

    def _gui(self, fn):
        return fn()

    def tools(self):
        return [self._tool]


def _scripted_chat(monkeypatch, replies, left, auto=True):
    """A _ChatThread on a model giving *replies* in turn (an Exception raised), the chain's
    steps left read from ``left()``; its events and failures kept."""
    from types import SimpleNamespace

    from normet.gui import assistant as A

    monkeypatch.setattr(A, "auto_on", lambda: auto)
    monkeypatch.setattr(A, "state_text", lambda win: "test")
    monkeypatch.setattr(A, "chain_left", lambda win, called: left())
    monkeypatch.setattr(A, "AUTO_RETRY_WAIT", 0.0)
    queue = list(replies)

    def policy(messages, tools):
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def check() -> str:
        """Read the checks."""
        return "the checks"

    c = A._ChatThread.__new__(A._ChatThread)
    c.model, c.box = llm.FakeModel(policy), _Box(llm.as_tool(check))
    c.messages = [{"role": "user", "content": "全自动分析这张表。"}]
    events, failed = [], []
    c.event, c.failed = SimpleNamespace(emit=events.append), SimpleNamespace(emit=failed.append)
    return c, events, failed


def _told(c):
    return [m["content"] for m in c.messages if m["role"] == "user"][1:]


def test_full_automatic_tells_a_model_that_stops_short_to_go_on(monkeypatch):
    from normet.gui import assistant as A

    state = {"left": ["train", "normalise"]}
    c, events, failed = _scripted_chat(
        monkeypatch,
        [
            llm.Reply("I will train next.", [], {}),  # stops short
            llm.Reply("", [llm.Call("1", "check", {})], {}),
            llm.Reply("Done.", [], {}),
        ],
        lambda: state["left"],
    )
    real = c.box._tool.fn
    c.box._tool = llm.Tool(
        c.box._tool.name,
        c.box._tool.description,
        c.box._tool.schema,
        lambda: (state.update(left=[]), real())[1],
    )
    c.run()
    told = _told(c)
    assert len(told) == 1 and "Still to do, in order" in told[0], told
    assert A.CHAIN_TEXT["train"] in told[0] and A.CHAIN_TEXT["normalise"] in told[0]
    assert "check" in c.box.called and not failed
    notes = [e["text"] for e in events if e["type"] == "note"]
    assert notes == [
        "Full automatic: not finished, so it goes on -- left: training, the normalisation."
    ]


def test_full_automatic_lets_a_turn_end_once_the_window_stops_moving(monkeypatch):
    from normet.gui import assistant as A

    c, events, failed = _scripted_chat(
        monkeypatch, [llm.Reply(f"cannot {i}", [], {}) for i in range(10)], lambda: ["train"]
    )
    c.run()
    assert len(_told(c)) == A.AUTO_STALLS and not failed


def test_full_automatic_calls_the_model_again_after_a_failure_and_an_empty_reply(monkeypatch):
    c, events, failed = _scripted_chat(
        monkeypatch,
        [RuntimeError("HTTP 503"), llm.Reply("", [], {}), llm.Reply("ok", [], {})],
        lambda: [],
    )
    c.run()
    assert [e["type"] for e in events if e["type"] == "note"] == ["note"] and not failed
    assert any("empty" in t for t in _told(c))
    assert c.messages[-1] == {"role": "assistant", "text": "ok", "calls": [], "reasoning": None}


def test_without_full_automatic_a_turn_ends_where_the_model_ends_it(monkeypatch):
    c, events, failed = _scripted_chat(
        monkeypatch, [RuntimeError("HTTP 503")], lambda: ["train"], auto=False
    )
    c.run()
    assert failed == ["RuntimeError: HTTP 503"] and c.messages[-1]["role"] == "assistant"
    c, events, failed = _scripted_chat(
        monkeypatch, [llm.Reply("stop here", [], {})], lambda: ["train"], auto=False
    )
    c.run()
    assert _told(c) == [] and not [e for e in events if e["type"] == "note"]


def test_the_chain_left_is_read_off_the_window(win, qapp):
    from normet.gui import assistant as A

    assert A.chain_left(win, set()) == []  # no table: nothing to start from
    win.load_example()
    assert A.chain_left(win, set()) == list(A.CHAIN)
    assert A.chain_left(win, {"inspect_data", "propose_setup"}) == [
        "train",
        "normalise",
        "decompose",
    ]


def test_full_automatic_applies_the_set_up_at_once(win, qapp, monkeypatch):
    from normet.gui import assistant as A

    win.load_example()
    assert "Full automatic is on" not in A.system_prompt()
    win.assistant.auto_btn.setChecked(True)
    assert A.auto_on() and "full automatic is on" in A.state_text(win)
    prompt = A.system_prompt()
    assert "Full automatic is on" in prompt and "stop and let the analyst answer" not in prompt
    said = win.assistant.tools.propose_setup("PM2.5", ["ws", "wd"], "wind only", 0.6)
    assert said.startswith("Applied at once")
    assert A._checked(win.feature_list) == ["ws", "wd"]
    win.assistant.auto_btn.setChecked(False)
    assert not A.auto_on()
    said = win.assistant.tools.propose_setup("PM2.5", ["t2m"], "warmth", 0.5)
    assert "Stop here" in said and A._checked(win.feature_list) == ["ws", "wd"]


def test_full_automatic_carries_a_lazy_model_through_the_window(win, qapp, monkeypatch):
    """End to end in the panel: a model that does one step a turn and stops is told
    what is left each time, and the window ends up trained, normalised and decomposed."""
    pytest.importorskip("lightgbm")
    from normet.gui import assistant as A

    win.load_example()
    win.df_raw = win.df_raw.iloc[: 24 * 45].reset_index(drop=True)
    win.backend_combo.setCurrentText("lightgbm")
    win.budget_spin.setValue(3)
    win.norm_samples.setValue(5)
    win.decom_samples.setValue(5)
    dock = win.assistant
    dock.auto_btn.setChecked(True)
    step = {
        "data": llm.Call("d", "inspect_data", {}),
        "setup": llm.Call(
            "s",
            "propose_setup",
            {
                "target": "PM2.5",
                "features": ["ws", "wd", "t2m", "blh"],
                "reasons": "met",
                "confidence": 0.7,
            },
        ),
        "train": llm.Call("t", "train", {}),
        "normalise": llm.Call("n", "normalise", {"resample_vars": ["ws", "wd", "t2m", "blh"]}),
        "decompose": llm.Call("c", "decompose", {"method": "emission"}),
    }
    done = []

    def lazy(messages, tools):
        last = messages[-1]
        if last["role"] == "tool":  # one step a turn, then it stops short
            return llm.Reply("Done that one.", [], {})
        text = last["content"]
        name = next((s for s in A.CHAIN if A.CHAIN_TEXT[s] in text), "data")
        done.append(name)
        return llm.Reply("", [step[name]], {})

    monkeypatch.setattr(dock, "_model", lambda: llm.FakeModel(lazy))
    dock.send("Analyse this table, all of it.")
    _wait(qapp, lambda: not dock.busy() and not dock.working.isVisible(), timeout=600)
    assert done == list(A.CHAIN), done
    assert win.model is not None and {"normalise", "decompose"} <= set(win.results)
    assert A.chain_left(win, dock.tools.called) == []
    nudges = [m for m in dock.messages if m["role"] == "user"][1:]
    assert len(nudges) == len(A.CHAIN) - 1


# -- the panel as cume's: the next message, Apply that trains, the status line -------------
def test_a_replys_next_message_is_split_off():
    from normet.gui.assistant import split_next, without_next

    text, said = split_next("Shall I train it?\n<!-- next: Yes, train it -->")
    assert text == "Shall I train it?" and said == "Yes, train it"
    assert split_next("No suggestion.") == ("No suggestion.", None)
    # The last of several, its spaces folded.
    assert split_next("a <!-- next: x --> b <!--next:  y\n z-->")[1] == "y z"
    # While it is being written, the line is kept off the panel whole or begun.
    assert without_next("Done.\n<!-- next: Yes") == "Done."
    assert without_next("Done.\n<!-- next: Yes -->") == "Done."
    assert without_next("Done.") == "Done."


def test_the_window_suggests_its_next_step(win, qapp):
    from normet.gui.assistant import suggested_message

    assert suggested_message(win, set()) == "What can you do in this window?"
    win.load_example()
    assert "propose a target" in suggested_message(win, set())
    assert suggested_message(win, {"propose_setup"}).startswith("Train the model")


def test_tab_fills_the_suggestion_and_enter_sends(win, qapp, monkeypatch):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    dock = win.assistant
    box = dock.composer
    monkeypatch.setattr(
        dock, "_model", lambda: llm.FakeModel(lambda m, t: llm.Reply("Hi.", [], {}))
    )

    def key(k, mods=Qt.KeyboardModifier.NoModifier):
        box.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, k, mods))

    dock._suggest("Yes, train it")
    assert box.placeholderText() == "Yes, train it"
    key(Qt.Key.Key_Tab)
    assert box.toPlainText() == "Yes, train it"
    key(Qt.Key.Key_Tab)  # the box is not empty: the suggestion is not put in again
    assert box.toPlainText().count("Yes, train it") == 1
    box.setPlainText("hello")
    key(Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)  # a new line, not sent
    assert box.toPlainText().strip() == "hello" and "\n" in box.toPlainText()
    key(Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)  # Cmd/Ctrl+Enter sends
    assert box.toPlainText() == "" and "› hello" in dock.view.toPlainText()
    _wait(qapp, lambda: not dock.busy() and box.suggestion != "Yes, train it")
    dock.new_conversation()
    assert box.suggestion == "" and box.placeholderText().startswith("Ask about the data")


def test_a_turn_leaves_its_suggestion_and_tokens(win, qapp, monkeypatch):
    win.load_example()
    dock = win.assistant

    def policy(messages, tools):
        return llm.Reply(
            "Train on wind and mixing?\n<!-- next: Yes, train it -->",
            [],
            {"input": 1200, "output": 34},
        )

    monkeypatch.setattr(dock, "_model", lambda: llm.FakeModel(policy))
    dock.send("What should I model?")
    _wait(qapp, lambda: not dock.busy() and dock.composer.suggestion)
    reply = [w for w in dock.view.items() if type(w).__name__ == "_Reply"][-1]
    assert reply.md == "Train on wind and mixing?"  # the suggestion is not in the reply
    assert dock.composer.suggestion == "Yes, train it"
    assert dock.status.text() == "tokens: 1,200 in, 34 out"


def test_apply_trains_at_once_and_the_model_is_told(win, qapp, monkeypatch):
    win.load_example()
    dock = win.assistant

    def quick():  # the window's training, as quick as a test wants it
        win.runner.submit(
            "train (test)",
            lambda: time.sleep(0.2),
            lambda _r: setattr(win, "model", "a model"),
            lambda _e: None,
        )

    monkeypatch.setattr(win, "_run_train", quick)
    dock.tools.propose_setup("PM2.5", ["ws", "wd"], "wind", 0.7)
    qapp.processEvents()
    from PySide6.QtWidgets import QPushButton

    btn = [b for b in dock.view.findChildren(QPushButton) if b.text() == "Apply and train"][0]
    btn.click()
    assert win.runner.busy and "running in the window" in dock.status.text()
    assert "training started" in dock.notes[-1]
    seen = []

    def policy(messages, tools):
        seen.append(messages[-1]["content"])
        return llm.Reply("Reading it.", [], {})

    monkeypatch.setattr(dock, "_model", lambda: llm.FakeModel(policy))
    dock.messages.append({"role": "assistant", "text": "(card)", "calls": []})
    _wait(qapp, lambda: not win.runner.busy)
    # The step ended with the conversation begun: the window's next step is offered.
    assert dock.composer.suggestion.startswith("How well does the model fit?")
    dock.composer.suggestion = ""
    dock.send("How did it go?")
    _wait(qapp, lambda: not dock.busy() and dock.composer.suggestion)
    told = seen[0].splitlines()
    assert told[0].startswith("[Window now:") and "training started" in told[1]
    assert told[-1] == "How did it go?" and dock.notes == []
