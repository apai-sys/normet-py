"""A Model Context Protocol server over HTTP for a list of :class:`~normet.agent.llm.Tool`.

The GUI serves the window's tools with it to a
command-line agent (Claude Code, Codex) that brings its own model and loop:
on 127.0.0.1 only, with a token drawn per server that every request must
carry, JSON responses and no event stream. The standard library only, so a
packaged app needs neither the protocol's SDK nor a Python beside it.
"""

from __future__ import annotations

import json
import secrets
import threading
from collections.abc import Callable, Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .llm import Tool

PROTOCOL = "2025-06-18"
SERVER_NAME = "normet-window"


class ToolServer:
    """An MCP server over HTTP for *tools* -- a list, or a callable giving them,
    asked at each listing. ``on_call(name, arguments)`` and ``on_result(name,
    text, error)`` are told of each call, from the server's threads."""

    def __init__(
        self,
        tools: Iterable[Tool] | Callable[[], Iterable[Tool]],
        *,
        instructions: str = "",
        on_call: Callable[[str, dict[str, Any]], None] | None = None,
        on_result: Callable[[str, str, bool], None] | None = None,
    ) -> None:
        self._tools = tools
        self.instructions = instructions
        self.on_call, self.on_result = on_call, on_result
        self.token = secrets.token_urlsafe(24)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, code: int, body: Any = None) -> None:
                data = b"" if body is None else json.dumps(body).encode("utf-8")
                self.send_response(code)
                if body is not None:
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:  # noqa: N802
                if self.headers.get("Authorization") != "Bearer " + outer.token:
                    self._send(401, {"error": "unauthorized"})
                    return
                try:
                    size = int(self.headers.get("Content-Length") or 0)
                    msg = json.loads(self.rfile.read(size) or b"null")
                except ValueError:
                    self._send(
                        400,
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32700, "message": "parse error"},
                        },
                    )
                    return
                batch = isinstance(msg, list)
                out = [
                    r for r in (outer.handle(m) for m in (msg if batch else [msg])) if r is not None
                ]
                if not out:
                    self._send(202)
                else:
                    self._send(200, out if batch else out[0])

            def do_GET(self) -> None:  # noqa: N802 - no server-sent event stream
                self._send(405)

            def do_DELETE(self) -> None:  # noqa: N802
                self._send(200, {})

            def log_message(self, *args: Any) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/mcp"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def tools(self) -> dict[str, Tool]:
        tools = self._tools() if callable(self._tools) else self._tools
        return {t.name: t for t in tools}

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def handle(self, m: Any) -> dict[str, Any] | None:
        """The response owed to one JSON-RPC message, or None for a notification."""
        if not isinstance(m, dict) or "method" not in m or "id" not in m:
            return None
        rid, method, params = m["id"], m["method"], m.get("params") or {}
        try:
            result = self._method(method, params)
        except KeyError as exc:
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "error": {"code": -32601, "message": f"no method {exc}"},
            }
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def _method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            from normet import __version__

            return {
                "protocolVersion": params.get("protocolVersion") or PROTOCOL,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": __version__},
                "instructions": self.instructions,
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {
                "tools": [
                    {"name": t.name, "description": t.description, "inputSchema": t.schema}
                    for t in self.tools.values()
                ]
            }
        if method == "tools/call":
            return self._call(str(params.get("name")), params.get("arguments") or {})
        raise KeyError(method)

    def _call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tools = self.tools
        t = tools.get(name)
        if self.on_call:
            self.on_call(name, args)
        error = False
        if t is None:
            text, error = f"no tool {name!r}; the tools are {', '.join(tools)}", True
        else:
            try:
                text = str(t.fn(**args))
            except Exception as exc:  # noqa: BLE001 -- told to the model
                text, error = f"{type(exc).__name__}: {exc}", True
        if self.on_result:
            self.on_result(name, text, error)
        return {"content": [{"type": "text", "text": text}], "isError": error}
