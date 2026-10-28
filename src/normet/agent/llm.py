"""A language model calling normet's tools: the adapters, and the loop between them.

One loop, any model; the adapters speak the two
formats the providers share:

- ``anthropic`` -- Anthropic's Messages API (Claude);
- ``openai`` -- the OpenAI chat-completions API with function calling: OpenAI's
  own, and every server that speaks it -- DeepSeek, Gemini's compatible
  endpoint, a model on this machine (Ollama, LM Studio);
- :class:`FakeModel` -- a scripted policy, for tests.

A model is named ``"provider:model"``, e.g. ``"anthropic:claude-sonnet-5-5"`` or
``"ollama:qwen3:8b"``; :data:`PROVIDERS` gives each provider's format, address
and the environment variable its key is read from. Two more kinds of provider
are resolved by :func:`provider_spec`:

- a cluster whose models hpc-agent forwards to this machine --
  ``"bede:hf.co/unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL"``, or any unique part of
  the model's name: the address is the forward's port, read from hpc-agent's
  registry each time (:func:`hpc_forward`), so a restarted job still answers;
- a server the analyst names with :func:`register_provider`, any
  OpenAI-compatible address.

Nothing here writes a key down or prints one.

:func:`converse` carries a conversation on: it sends the messages, carries out
the tool calls the model asks for (an error is returned to the model in
normet's words) and stops when the model answers without one. Only the
standard library is used: no SDK to install.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import socket
import threading
import time
import typing
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import UnionType
from typing import Any, Protocol

#: provider -> (format, base address, environment variable of the key or None)
PROVIDERS: dict[str, tuple[str, str, str | None]] = {
    "anthropic": ("anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    "openai": ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "deepseek": ("openai", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "gemini": (
        "openai",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
    ),
    "ollama": ("openai", "http://localhost:11434/v1", None),
    "lmstudio": ("openai", "http://localhost:1234/v1", None),
}

# -- models forwarded by hpc-agent, and servers the analyst names ---------------------------
#: hpc-agent keeps one entry per model it forwards from a cluster job to this machine
#: (``cluster``, ``model``, ``local`` port, ...). The port is reassigned when a job is
#: restarted, so a model is looked up here each time it is used, never stored.
HPC_FORWARDS = "~/.config/hpc-agent/forwards.json"
#: Where hpc-agent copies each cluster's llama.cpp key (OpenCode's config).
OPENCODE_DIR = "~/.config/opencode"
#: The clusters hpc-agent knows; any other cluster found in the registry counts too.
HPC_CLUSTERS = ("csf3", "bede")
#: Servers named by the analyst: provider -> (format, base address, key variable or None).
#: Kept apart from :data:`PROVIDERS`, which stays the fixed set the library ships.
_CUSTOM: dict[str, tuple[str, str, str | None]] = {}
_PROVIDER_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def hpc_forwards(path: str | None = None, *, alive_only: bool = False) -> list[dict[str, Any]]:
    """The models hpc-agent forwards to this machine, from its registry.

    Each entry has at least ``cluster``, ``model`` and ``local`` (the port on 127.0.0.1).
    The registry is a plain file hpc-agent rewrites as jobs start and end, so a missing
    or unreadable one means "none", not an error. A tunnel that died with its ssh
    session can leave an entry behind: ``alive_only`` keeps those that still accept a
    connection. The path is ``path``, else ``$NORMET_HPC_FORWARDS``, else
    :data:`HPC_FORWARDS`.
    """
    p = Path(path or os.environ.get("NORMET_HPC_FORWARDS") or HPC_FORWARDS).expanduser()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[dict[str, Any]] = []
    for e in data if isinstance(data, list) else []:
        try:
            entry = {**e, "cluster": str(e["cluster"]), "model": str(e["model"])}
            entry["local"] = int(e["local"])
        except (KeyError, TypeError, ValueError):
            continue
        if not alive_only or _listening(entry["local"]):
            out.append(entry)
    return out


def hpc_clusters(path: str | None = None) -> tuple[str, ...]:
    """The clusters a model can be named after: hpc-agent's two, and any in the registry."""
    found = [f["cluster"] for f in hpc_forwards(path)]
    return tuple(dict.fromkeys([*HPC_CLUSTERS, *found]))


def hpc_models(path: str | None = None) -> list[str]:
    """``cluster:model`` for every forwarded model that answers now (see :func:`hpc_forwards`)."""
    return [f"{f['cluster']}:{f['model']}" for f in hpc_forwards(path, alive_only=True)]


def hpc_forward(cluster: str, model: str, path: str | None = None) -> dict[str, Any]:
    """The forward of ``model`` on ``cluster``: the full name, else a part of it that is unique.

    hpc-agent names a model as the server knows it
    (``hf.co/unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL``); ``bede:glm-5.3`` is enough here
    while only one model on Bede contains that. Raises ``RuntimeError`` when the cluster has
    no such forward (the message says what to start), ``ValueError`` when the part is
    ambiguous.
    """
    mine = [f for f in hpc_forwards(path) if f["cluster"] == cluster]
    if not mine:
        raise RuntimeError(
            f"{cluster}:{model}: hpc-agent forwards no model from {cluster} -- start one "
            f"(hpc-agent --cluster {cluster} start) and use Forward models in its app"
        )
    hits = [f for f in mine if f["model"] == model] or [
        f for f in mine if model.lower() in f["model"].lower()
    ]
    if not hits:
        raise RuntimeError(
            f"{cluster}:{model} is not forwarded; on {cluster}: "
            + ", ".join(f["model"] for f in mine)
        )
    if len({f["model"] for f in hits}) > 1:
        raise ValueError(
            f"{cluster}:{model} could be " + ", ".join(f["model"] for f in hits) + ": name one"
        )
    live = [f for f in hits if _listening(f["local"])]
    return (live or hits)[0]


def hpc_key(forward: dict[str, Any]) -> str | None:
    """The API key of a forward, as hpc-agent copied it into OpenCode's config.

    llama.cpp on the cluster refuses a request without its key, and hpc-agent keeps that key in
    the OpenCode provider whose address is this forward's port. Ports are reused, so a
    provider left over from an earlier job can share one (its key is another cluster's):
    those in ``disabled_providers`` are skipped, and of the rest the one that lists this
    model, then the one named for this cluster, comes first. It is read, never written or
    printed. ``None`` when OpenCode has no such provider; then the key is typed in the
    Assistant, or set as ``<CLUSTER>_API_KEY``.
    """
    base = Path(OPENCODE_DIR).expanduser()
    for name in ("opencode.jsonc", "opencode.json"):
        try:
            text = (base / name).read_text(encoding="utf-8")
            cfg = json.loads(re.sub(r"^\s*//.*$", "", text, flags=re.M))
        except (OSError, ValueError):
            continue
        off = set(cfg.get("disabled_providers") or [])
        found: list[tuple[int, str]] = []
        for table in ("providers", "provider"):
            for pid, prov in (cfg.get(table) or {}).items():
                if pid in off or not isinstance(prov, dict):
                    continue
                s = prov.get("settings") or prov.get("options") or {}
                m = re.match(r"^https?://(?:127\.0\.0\.1|localhost):(\d+)", str(s.get("baseURL")))
                if not (m and int(m.group(1)) == forward["local"] and s.get("apiKey")):
                    continue
                cluster = forward["cluster"]
                rank = (forward["model"] not in (prov.get("models") or {})) * 2 + (
                    not (pid == cluster or pid.startswith(cluster + "-"))
                )
                found.append((rank, str(s["apiKey"])))
        if found:
            return min(found, key=lambda f: f[0])[1]
    return None


def register_provider(
    name: str, base_url: str, *, key_env: str | None = None, fmt: str = "openai"
) -> None:
    """Name a server so that ``name:model`` reaches it.

    For an OpenAI-compatible server the library does not ship: a model on another port, a
    llama.cpp or vLLM behind a tunnel. ``key_env`` is the variable its key is read from
    (``None``: no key). A built-in provider or a cluster cannot be redefined. Held for
    this process; the caller keeps what must outlast it.
    """
    if not _PROVIDER_NAME.match(name):
        raise ValueError(f"{name!r}: a provider name is letters, digits, '_', '-' and '.'")
    if name in PROVIDERS or name in hpc_clusters():
        raise ValueError(f"{name!r} is a provider of its own; choose another name")
    if fmt not in ("openai", "anthropic"):
        raise ValueError("fmt is 'openai' or 'anthropic'")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError(f"{base_url!r}: an address begins with http:// or https://")
    _CUSTOM[name] = (fmt, base_url.rstrip("/"), key_env)


def forget_provider(name: str) -> None:
    """Drop a server named with :func:`register_provider` (a built-in one stays)."""
    _CUSTOM.pop(name, None)


def custom_providers() -> dict[str, tuple[str, str, str | None]]:
    """The servers named in this process, a copy."""
    return dict(_CUSTOM)


def provider_spec(provider: str) -> tuple[str, str, str | None] | None:
    """``(format, base address, key variable)`` of a provider, or ``None`` if it is unknown.

    A cluster's address is empty: it is the forward's, found when the model is used.
    """
    if provider in PROVIDERS:
        return PROVIDERS[provider]
    if provider in _CUSTOM:
        return _CUSTOM[provider]
    if provider in hpc_clusters():
        return ("openai", "", f"{re.sub(r'[^A-Za-z0-9]', '_', provider).upper()}_API_KEY")
    return None


def known_providers() -> list[str]:
    """Every name a model can start with: built in, named by the analyst, or a cluster."""
    return list(dict.fromkeys([*PROVIDERS, *_CUSTOM, *hpc_clusters()]))


# -- tools ----------------------------------------------------------------------------
@dataclass
class Tool:
    """A function the model may call: its name, description and JSON schema."""

    name: str
    description: str
    schema: dict[str, Any]
    fn: Callable[..., Any]


def _json_type(hint: Any) -> dict[str, Any]:
    origin = typing.get_origin(hint)
    args = [a for a in typing.get_args(hint) if a is not type(None)]
    if origin in (typing.Union, UnionType) and len(args) == 1:
        return _json_type(args[0])
    if origin is list:
        return {"type": "array", "items": _json_type(args[0]) if args else {}}
    return {
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        str: {"type": "string"},
    }.get(hint, {"type": "string"})


def as_tool(
    fn: Callable[..., Any], name: str | None = None, description: str | None = None
) -> Tool:
    """A :class:`Tool` from a function's signature and docstring."""
    hints = typing.get_type_hints(fn)
    props: dict[str, Any] = {}
    required: list[str] = []
    for p in inspect.signature(fn).parameters.values():
        if p.name in ("self", "args", "kwargs"):
            continue
        props[p.name] = _json_type(hints.get(p.name, str))
        if p.default is inspect.Parameter.empty:
            required.append(p.name)
    schema = {"type": "object", "properties": props, "required": required}
    return Tool(name or fn.__name__, inspect.cleandoc(description or fn.__doc__ or ""), schema, fn)


# -- messages -------------------------------------------------------------------------
@dataclass
class Call:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Reply:
    text: str
    calls: list[Call]
    usage: dict[str, int]
    raw: Any = None
    reasoning: str | None = None


class ChatModel(Protocol):
    name: str

    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[Tool]) -> Reply: ...


def _post(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: float = 600.0,
    retries: int = 4,
) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url, data=data, method="POST", headers={"Content-Type": "application/json", **headers}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out: dict[str, Any] = json.load(r)
                return out
        except urllib.error.HTTPError as e:
            text = e.read()[:600].decode("utf-8", "replace")
            # Too many requests, or the server's own trouble: wait and ask again.
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < retries:
                time.sleep(min(60.0, 2.0 * 2**attempt))
                continue
            raise RuntimeError(f"{url}: HTTP {e.code}: {text}") from None
        except urllib.error.URLError as e:
            if attempt < retries:
                time.sleep(min(60.0, 2.0 * 2**attempt))
                continue
            raise RuntimeError(f"{url}: {e.reason}") from None
    raise RuntimeError(f"{url}: no answer")  # pragma: no cover - the loop returns or raises


class Model:
    """A model by name: ``"provider:model"`` (:data:`PROVIDERS`)."""

    def __init__(
        self,
        name: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = 0.0,
    ) -> None:
        provider, _, model = name.partition(":")
        spec = provider_spec(provider)
        if spec is None or not model:
            raise ValueError(
                "name a model as provider:model, the provider one of "
                + ", ".join(known_providers())
            )
        self.format, url, key_env = spec
        self.key = (
            api_key if api_key is not None else (os.environ.get(key_env) if key_env else None)
        )
        if provider not in PROVIDERS and provider not in _CUSTOM and not base_url:
            # A model hpc-agent forwards: its port is the registry's now, its full name the
            # server's. The key may sit in OpenCode's config, where hpc-agent put it.
            fwd = hpc_forward(provider, model)
            model, url = fwd["model"], f"http://127.0.0.1:{fwd['local']}/v1"
            name = f"{provider}:{model}"
            if not self.key:
                self.key = hpc_key(fwd)
        elif key_env and not self.key:
            raise RuntimeError(f"{name}: no key -- set {key_env}, or store one in the Assistant")
        self.name, self.model = name, model
        self.base_url = (base_url or url).rstrip("/")
        self.max_tokens, self.temperature = max_tokens, temperature

    # The loop keeps messages in a neutral form; each adapter writes its own.
    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[Tool]) -> Reply:
        if self.format == "anthropic":
            return self._anthropic(system, messages, tools)
        return self._openai(system, messages, tools)

    def _anthropic(self, system: str, messages: list[dict[str, Any]], tools: list[Tool]) -> Reply:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                content: list[dict[str, Any]] = (
                    [{"type": "text", "text": m["text"]}] if m["text"] else []
                )
                content += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in m["calls"]
                ]
                out.append({"role": "assistant", "content": content})
            else:  # tool results, one user turn
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": r["id"],
                                "content": r["text"],
                                "is_error": r.get("error", False),
                            }
                            for r in m["results"]
                        ],
                    }
                )
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": out,
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.schema}
                for t in tools
            ],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        r = _post(
            self.base_url + "/v1/messages",
            body,
            {"x-api-key": self.key or "", "anthropic-version": "2023-06-01"},
        )
        text = "".join(b.get("text", "") for b in r["content"] if b["type"] == "text")
        calls = [
            Call(b["id"], b["name"], b.get("input") or {})
            for b in r["content"]
            if b["type"] == "tool_use"
        ]
        u = r.get("usage") or {}
        return Reply(
            text, calls, {"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0)}, r
        )

    def _openai(self, system: str, messages: list[dict[str, Any]], tools: list[Tool]) -> Reply:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m["text"] or None}
                if m["calls"]:
                    msg["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in m["calls"]
                    ]
                if m.get("reasoning"):  # DeepSeek's thinking, sent back
                    msg["reasoning_content"] = m["reasoning"]
                out.append(msg)
            else:
                for r in m["results"]:
                    out.append({"role": "tool", "tool_call_id": r["id"], "content": r["text"]})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": out,
            "max_tokens": self.max_tokens,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.schema,
                    },
                }
                for t in tools
            ],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {"Authorization": "Bearer " + self.key} if self.key else {}
        r = _post(self.base_url + "/chat/completions", body, headers)
        msg = r["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_unparsed": c["function"].get("arguments")}
            calls.append(Call(c["id"], c["function"]["name"], args))
        u = r.get("usage") or {}
        return Reply(
            msg.get("content") or "",
            calls,
            {"input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0)},
            r,
            reasoning=msg.get("reasoning_content"),
        )


class FakeModel:
    """A scripted policy, ``policy(messages, tools) -> Reply``: free and the same
    every time, for tests."""

    def __init__(
        self, policy: Callable[[list[dict[str, Any]], list[Tool]], Reply], name: str = "fake"
    ) -> None:
        self.policy, self.name = policy, name

    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[Tool]) -> Reply:
        return self.policy(messages, tools)


@dataclass
class Turn:
    """What one message of the user's came to: the events, the tokens spent and
    why the model stopped."""

    events: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})
    stopped: str = ""


def converse(
    model: ChatModel,
    tools: list[Tool],
    system: str,
    messages: list[dict[str, Any]],
    *,
    max_turns: int = 25,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    stop: threading.Event | None = None,
) -> Turn:
    """Carry the conversation in *messages* on until the model answers without a
    tool call, *stop* is set, or *max_turns* model calls have been made.

    *messages* is extended in place: the model's turns and the tools' results
    are appended to it, so the next message of the user's continues it. Each
    event -- ``{"type": "model", "text", "calls"}`` or ``{"type": "tool", "name",
    "arguments", "output", "error"}`` -- is passed to *on_event* as it happens.
    """
    turn = Turn()
    by_name = {t.name: t for t in tools}

    def event(e: dict[str, Any]) -> None:
        turn.events.append(e)
        if on_event:
            on_event(e)

    for _ in range(max_turns):
        if stop is not None and stop.is_set():
            turn.stopped = "stopped"
            return turn
        reply = model.chat(system, messages, tools)
        for k in ("input", "output"):
            turn.usage[k] += int(reply.usage.get(k, 0) or 0)
        event(
            {
                "type": "model",
                "text": reply.text,
                "calls": [{"name": c.name, "arguments": c.arguments} for c in reply.calls],
            }
        )
        messages.append(
            {
                "role": "assistant",
                "text": reply.text,
                "calls": reply.calls,
                "reasoning": reply.reasoning,
            }
        )
        if not reply.calls:
            turn.stopped = "answered"
            return turn
        results = []
        for c in reply.calls:
            t = by_name.get(c.name)
            error = False
            if stop is not None and stop.is_set():
                text, error = "stopped by the user before this call ran", True
            elif t is None:
                text, error = f"no tool {c.name!r}; the tools are {', '.join(by_name)}", True
            else:
                try:
                    text = str(t.fn(**c.arguments))
                except Exception as exc:  # noqa: BLE001 -- told to the model
                    text, error = f"{type(exc).__name__}: {exc}", True
            event(
                {
                    "type": "tool",
                    "name": c.name,
                    "arguments": c.arguments,
                    "output": text,
                    "error": error,
                }
            )
            results.append({"id": c.id, "text": text, "error": error})
        messages.append({"role": "tool", "results": results})
    turn.stopped = "turns"
    # Every provider wants the turns to alternate: the model has the last word.
    messages.append({"role": "assistant", "text": "(stopped: too many steps)", "calls": []})
    return turn
