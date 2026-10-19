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
    assert "card" in said
    qapp.processEvents()
    from PySide6.QtWidgets import QPushButton

    apply = [b for b in win.assistant.view.findChildren(QPushButton) if b.text() == "Apply"]
    assert len(apply) == 1
    apply[0].click()
    assert win.target_combo.currentText() == "NO2"
    assert win._selected_features() == ["t2m", "blh"]
    assert win.assistant.pending_resample == ["t2m", "blh"]
    assert apply[0].text() == "Applied" and not apply[0].isEnabled()
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
    agent = next(iter(assistant_cli.AGENTS))
    monkeypatch.setattr(assistant_cli, "find_cli", lambda c: None)
    dock.model_box.setCurrentText(agent)
    assert not dock.key_btn.isEnabled()
    dock.send("hi")
    assert not dock.busy()
