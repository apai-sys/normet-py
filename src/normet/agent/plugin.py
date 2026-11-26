"""Package validation and multi-platform installation for the normet agent plugin.

Supports Antigravity, Claude Code, DeepSeek Harness (`dsh`), OpenCode, and
OpenAI Codex CLI (`codex`) without requiring any third-party runtime
dependencies beyond Python's standard library.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

try:  # Python 3.11+; tomli is the backport the project pins for 3.10
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]
from collections.abc import Sequence
from typing import Any

PLUGIN_DIR_NAME = "normet"
TARGETS = ("antigravity", "claude-code", "dsh", "opencode", "codex")
_NORMET_SKILLS = (
    "deweather-workflow",
    "deweather-model-choice",
    "scm-intervention",
    "transport-receptor",
)

#: Top-level keys permitted in Antigravity's plugin.json (unknown keys are dropped).
ANTIGRAVITY_ALLOWED_FIELDS = frozenset(
    {
        "name",
        "displayName",
        "description",
        "version",
        "logo",
        "suggestedPrompts",
        "author",
        "homepage",
        "repository",
        "disabled",
    }
)


def find_plugin_source(source: str | None = None) -> Path:
    """Locate the plugin bundle (`plugin/` in a git checkout or packaged `plugin_bundle/`)."""
    if source is not None:
        root = Path(source).expanduser().resolve()
    else:
        repo_root = Path(__file__).resolve().parents[3] / "plugin"
        pkg_root = Path(__file__).resolve().parent / "plugin_bundle"
        root = repo_root if (repo_root / "plugin.json").is_file() else pkg_root
    if not (root / "plugin.json").is_file():
        raise FileNotFoundError(
            f"Plugin bundle not found at {root}; pass --source to a valid plugin directory."
        )
    return root


def _parse_extra_env(items: Sequence[str] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE for --env, got {item!r}.")
        key, val = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty environment variable name in {item!r}.")
        env[key] = val
    return env


def resolve_server_spec(
    command: str | None = None,
    use_python: bool = False,
    use_uvx: bool = False,
    toolset: str = "all",
    roots: str | None = None,
    mcp_home: str | None = None,
    extra_env: Sequence[str] | None = None,
    url: str | None = None,
    auth_token: str | None = None,
) -> dict[str, Any]:
    """Resolve stdio command/args/env or remote HTTP url/headers for launching `normet-mcp`."""
    if toolset not in ("all", "core"):
        raise ValueError(f"Unknown toolset {toolset!r}; expected 'all' or 'core'.")

    if url:
        if use_uvx or use_python or command:
            raise ValueError("--url cannot be combined with --uvx, --python, or --command.")
        spec: dict[str, Any] = {
            "transport": "http",
            "url": url.strip(),
            "headers": {"Authorization": f"Bearer {auth_token}"} if auth_token else {},
        }
        return spec

    args: list[str] = []
    env: dict[str, str] = {}
    is_python_module = False

    if use_uvx:
        cmd = "uvx"
        args = ["--from", "normet[mcp]", "normet-mcp"]
    elif command:
        cmd = command
    elif use_python:
        cmd = sys.executable
        args = ["-m", "normet.mcp_server"]
        is_python_module = True
    else:
        exe_name = "normet-mcp.exe" if os.name == "nt" else "normet-mcp"
        sibling = Path(sys.executable).resolve().parent / exe_name
        if sibling.is_file() and os.access(sibling, os.X_OK):
            cmd = str(sibling)
        else:
            cmd = sys.executable
            args = ["-m", "normet.mcp_server"]
            is_python_module = True

    if toolset != "all":
        args.extend(["--toolset", toolset])

    if is_python_module:
        src_dir = Path(__file__).resolve().parents[2]
        if src_dir.name == "src" and (src_dir / "normet" / "mcp_server.py").is_file():
            env["PYTHONPATH"] = str(src_dir)

    if roots:
        env["NORMET_MCP_ROOTS"] = roots
    if mcp_home:
        env["NORMET_MCP_HOME"] = mcp_home
    env.update(_parse_extra_env(extra_env))

    return {"command": cmd, "args": args, "env": env}


def _server_entry(spec: dict[str, Any], include_type: bool = False) -> dict[str, Any]:
    if spec.get("transport") == "http":
        entry: dict[str, Any] = {}
        if include_type:
            entry["type"] = "http"
        entry["url"] = spec["url"]
        if spec.get("headers"):
            entry["headers"] = dict(spec["headers"])
        return entry
    entry = {}
    if include_type:
        entry["type"] = "stdio"
    entry["command"] = spec["command"]
    entry["args"] = list(spec.get("args") or [])
    if spec.get("env"):
        entry["env"] = dict(spec["env"])
    return entry


def render_antigravity_mcp(spec: dict[str, Any]) -> dict[str, Any]:
    """Render `mcp_config.json` payload for Antigravity."""
    return {"mcpServers": {"normet": _server_entry(spec, include_type=False)}}


def render_claude_mcp(spec: dict[str, Any]) -> dict[str, Any]:
    """Render `.mcp.json` payload for Claude Code (and `dsh-mcp-project`)."""
    return {"mcpServers": {"normet": _server_entry(spec, include_type=True)}}


def _dsh_normet_lines(spec: dict[str, Any]) -> list[str]:
    if spec.get("transport") == "http":
        lines = [
            '  - name: "normet"',
            '    transport: "http"',
            f"    url: {json.dumps(spec['url'])}",
        ]
        headers = spec.get("headers") or {}
        if headers:
            lines.append("    headers:")
            for key, value in headers.items():
                lines.append(f"      {key}: {json.dumps(str(value))}")
        return lines
    lines = [
        '  - name: "normet"',
        f"    command: {json.dumps(spec['command'])}",
        f"    args: {json.dumps(list(spec.get('args') or []))}",
    ]
    env = spec.get("env") or {}
    if env:
        lines.append("    env:")
        for key, value in env.items():
            lines.append(f"      {key}: {json.dumps(str(value))}")
    return lines


def render_dsh_mcp_yaml(spec: dict[str, Any]) -> str:
    """Render `.dsh/mcp.yml` payload for DeepSeek Harness."""
    return "\n".join(["mcpServers:"] + _dsh_normet_lines(spec)) + "\n"


def render_dsh_cordis_patch(spec: dict[str, Any]) -> str:
    """Render a `- insert:` entry for DeepSeek Harness `cordis.patch.yml`."""
    if spec.get("transport") == "http":
        lines = [
            "- insert:",
            "    - id: mcp-normet",
            '      name: "@deepseek-ai/dsh-mcp-client"',
            "      config:",
            "        serverName: normet",
            "        transport: http",
            f"        url: {json.dumps(spec['url'])}",
        ]
        headers = spec.get("headers") or {}
        if headers:
            lines.append("        headers:")
            for key, value in headers.items():
                lines.append(f"          {key}: {json.dumps(str(value))}")
        return "\n".join(lines) + "\n"
    lines = [
        "- insert:",
        "    - id: mcp-normet",
        '      name: "@deepseek-ai/dsh-mcp-client"',
        "      config:",
        "        serverName: normet",
        "        transport: stdio",
        f"        command: {json.dumps(spec['command'])}",
    ]
    args = list(spec.get("args") or [])
    if args:
        lines.append("        args:")
        for arg in args:
            lines.append(f"          - {json.dumps(str(arg))}")
    else:
        lines.append("        args: []")
    env = spec.get("env") or {}
    if env:
        lines.append("        env:")
        for key, value in env.items():
            lines.append(f"          {key}: {json.dumps(str(value))}")
    return "\n".join(lines) + "\n"


def _strip_normet_from_insert_block(entry_lines: list[str]) -> list[str]:
    """Remove `id: mcp-normet` from a top-level `- insert:` entry, keeping sibling items."""
    child_starts: list[tuple[int, int]] = []
    for i in range(1, len(entry_lines)):
        m = re.match(r"^(\s*)-\s+", entry_lines[i])
        if m:
            child_starts.append((i, len(m.group(1))))
    if not child_starts:
        return entry_lines
    base_indent = min(indent for _, indent in child_starts)
    top_children = [idx for idx, indent in child_starts if indent == base_indent]
    kept_children: list[str] = []
    for pos, start in enumerate(top_children):
        stop = top_children[pos + 1] if pos + 1 < len(top_children) else len(entry_lines)
        block = "\n".join(entry_lines[start:stop])
        if re.search(r'(?:^|\n)\s*(?:-\s*)?id:\s*["\']?mcp-normet["\']?\s*(?:#.*)?(?:\n|$)', block):
            continue
        kept_children.extend(entry_lines[start:stop])
    if not kept_children:
        return []
    return [entry_lines[0], *kept_children]


def _merge_dsh_cordis_patch(dest: Path, spec: dict[str, Any]) -> str:
    """Insert or replace `mcp-normet` in a DeepSeek Harness `cordis.patch.yml`."""
    patch_block = render_dsh_cordis_patch(spec)
    if not dest.is_file():
        return patch_block
    raw = dest.read_text(encoding="utf-8")
    if not raw.strip():
        return patch_block

    lines = raw.splitlines()
    top_starts = [i for i, ln in enumerate(lines) if ln.startswith("- ")]
    if not top_starts:
        return raw.rstrip() + "\n" + patch_block

    kept_lines: list[str] = list(lines[: top_starts[0]])
    for pos, start in enumerate(top_starts):
        stop = top_starts[pos + 1] if pos + 1 < len(top_starts) else len(lines)
        entry = lines[start:stop]
        if re.match(r"^-\s+insert:\s*(?:#.*)?$", entry[0]):
            entry = _strip_normet_from_insert_block(entry)
        kept_lines.extend(entry)

    prefix = "\n".join(kept_lines).rstrip()
    return (prefix + "\n" + patch_block) if prefix else patch_block


def _opencode_server_entry(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("transport") == "http":
        entry: dict[str, Any] = {
            "type": "remote",
            "url": spec["url"],
            "enabled": True,
        }
        if spec.get("headers"):
            entry["headers"] = dict(spec["headers"])
        return entry
    entry = {
        "type": "local",
        "command": [spec["command"], *list(spec.get("args") or [])],
        "enabled": True,
    }
    if spec.get("env"):
        entry["environment"] = dict(spec["env"])
    return entry


def render_opencode_mcp(spec: dict[str, Any]) -> dict[str, Any]:
    """Render `opencode.json` payload for OpenCode."""
    return {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {"normet": _opencode_server_entry(spec)},
    }


def _merge_opencode_json(
    dest: Path,
    spec: dict[str, Any],
    instruction_path: str | None = None,
) -> dict[str, Any]:
    if dest.is_file():
        data = json.loads(dest.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in {dest}")
    else:
        data = {}
    data.setdefault("$schema", "https://opencode.ai/config.json")
    mcp = data.setdefault("mcp", {})
    if not isinstance(mcp, dict):
        raise ValueError(f"Expected 'mcp' object in {dest}")
    mcp["normet"] = _opencode_server_entry(spec)
    if instruction_path:
        inst = data.setdefault("instructions", [])
        if isinstance(inst, list) and instruction_path not in inst:
            inst.append(instruction_path)
    return data


def _toml_key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key)


def render_codex_mcp_toml(spec: dict[str, Any]) -> str:
    """Render `[mcp_servers.normet]` TOML block for OpenAI Codex CLI."""
    if spec.get("transport") == "http":
        lines = [
            "[mcp_servers.normet]",
            "enabled = true",
            f"url = {json.dumps(spec['url'])}",
        ]
        headers = spec.get("headers") or {}
        if headers:
            pairs = ", ".join(
                f"{_toml_key(str(k))} = {json.dumps(str(v))}" for k, v in headers.items()
            )
            lines.append(f"http_headers = {{ {pairs} }}")
        return "\n".join(lines) + "\n"
    lines = [
        "[mcp_servers.normet]",
        "enabled = true",
        f"command = {json.dumps(spec['command'])}",
        f"args = {json.dumps(list(spec.get('args') or []))}",
    ]
    env = spec.get("env") or {}
    if env:
        pairs = ", ".join(f"{_toml_key(str(k))} = {json.dumps(str(v))}" for k, v in env.items())
        lines.append(f"env = {{ {pairs} }}")
    return "\n".join(lines) + "\n"


def _merge_codex_toml(dest: Path, spec: dict[str, Any]) -> str:
    """Insert or replace `[mcp_servers.normet]` in `config.toml`, preserving comments."""
    block = render_codex_mcp_toml(spec)
    if not dest.is_file():
        return block
    raw = dest.read_text(encoding="utf-8")
    if not raw.strip():
        return block

    kept: list[str] = []
    skipping = False
    for line in raw.splitlines():
        m = re.match(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$", line)
        if m:
            section = m.group(1).strip()
            skipping = section == "mcp_servers.normet" or section.startswith("mcp_servers.normet.")
        if not skipping:
            kept.append(line)

    prefix = "\n".join(kept).rstrip()
    merged = (prefix + "\n\n" + block) if prefix else block
    tomllib.loads(merged)
    return merged


def _merge_json_mcp(dest: Path, server_name: str, entry: dict[str, Any]) -> dict[str, Any]:
    if dest.is_file():
        data = json.loads(dest.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in {dest}")
    else:
        data = {}
    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError(f"Expected 'mcpServers' object in {dest}")
    servers[server_name] = entry
    return data


def _merge_dsh_mcp_yaml(dest: Path, spec: dict[str, Any]) -> str:
    """Insert or replace the `normet` entry under `mcpServers:` without PyYAML."""
    if not dest.is_file():
        return render_dsh_mcp_yaml(spec)
    raw = dest.read_text(encoding="utf-8")
    if not raw.strip():
        return render_dsh_mcp_yaml(spec)

    lines = raw.splitlines()
    normet_lines = _dsh_normet_lines(spec)

    top_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "mcpServers:"), None)
    if top_idx is None:
        return raw.rstrip("\n") + "\n\n" + render_dsh_mcp_yaml(spec)

    # Locate the span of the `mcpServers:` block up to the next top-level key.
    end_idx = len(lines)
    for i in range(top_idx + 1, len(lines)):
        ln = lines[i]
        if ln.strip() and not ln[0].isspace() and not ln.lstrip().startswith("#"):
            end_idx = i
            break

    # Find list items starting inside the `mcpServers:` section.
    item_starts: list[tuple[int, int]] = []
    for i in range(top_idx + 1, end_idx):
        m = re.match(r"^(\s*)-\s+", lines[i])
        if m:
            item_starts.append((i, len(m.group(1))))

    if item_starts:
        base_indent = min(indent for _, indent in item_starts)
        top_items = [idx for idx, indent in item_starts if indent == base_indent]
        for pos, start in enumerate(top_items):
            stop = top_items[pos + 1] if pos + 1 < len(top_items) else end_idx
            block = "\n".join(lines[start:stop])
            if re.search(r'(?:^|\n)\s*(?:-\s*)?name:\s*["\']?normet["\']?\s*(?:\n|$)', block):
                new_lines = lines[:start] + normet_lines + lines[stop:]
                return "\n".join(new_lines) + "\n"

    insert_at = end_idx
    while insert_at > top_idx + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    new_lines = lines[:insert_at] + normet_lines + lines[insert_at:]
    return "\n".join(new_lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _copy_dir(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_skills(skills_src: Path, skills_dst: Path, dry_run: bool) -> list[str]:
    names: list[str] = []
    for entry in sorted(skills_src.iterdir()):
        if entry.is_dir() and (entry / "SKILL.md").is_file():
            names.append(entry.name)
            _copy_dir(entry, skills_dst / entry.name, dry_run=dry_run)
    return names


def _copy_rule_file(source: Path, dest_file: Path, dry_run: bool) -> str | None:
    src = source / "rules" / "AGENTS.md"
    if not src.is_file():
        return None
    if not dry_run:
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_file)
    return str(dest_file)


def _remove_skills(skills_dst: Path, dry_run: bool) -> list[str]:
    removed: list[str] = []
    for name in _NORMET_SKILLS:
        target = skills_dst / name
        if target.is_dir():
            removed.append(name)
            if not dry_run:
                shutil.rmtree(target)
    return removed


def _remove_file(path: Path, dry_run: bool) -> bool:
    if not path.is_file():
        return False
    if not dry_run:
        path.unlink()
    return True


def _resolve_home(home: Path | None) -> Path:
    return Path(home).expanduser().resolve() if home is not None else Path.home().resolve()


def _resolve_project(project_dir: Path | None) -> Path:
    return (
        Path(project_dir).expanduser().resolve()
        if project_dir is not None
        else Path.cwd().resolve()
    )


def install_antigravity(
    source: Path,
    spec: dict[str, Any],
    scope: str = "user",
    project_dir: Path | None = None,
    home: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Install the Antigravity plugin bundle (never modifying `config.json`)."""
    if scope not in ("user", "project"):
        raise ValueError(f"Unknown scope {scope!r}; expected 'user' or 'project'.")
    if scope == "user":
        dest = _resolve_home(home) / ".gemini" / "config" / "plugins" / PLUGIN_DIR_NAME
    else:
        dest = _resolve_project(project_dir) / ".agents" / "plugins" / PLUGIN_DIR_NAME

    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "plugin.json", dest / "plugin.json")
        if (source / "README.md").is_file():
            shutil.copy2(source / "README.md", dest / "README.md")
    for sub in ("assets", "rules", "skills"):
        if (source / sub).is_dir():
            _copy_dir(source / sub, dest / sub, dry_run=dry_run)
    _write_json(dest / "mcp_config.json", render_antigravity_mcp(spec), dry_run=dry_run)

    return {
        "target": "antigravity",
        "scope": scope,
        "dest": str(dest),
        "mcp_config": str(dest / "mcp_config.json"),
        "rule": str(dest / "rules" / "AGENTS.md"),
        "dry_run": dry_run,
        "note": f"Manage in Antigravity Customizations UI (dir: {PLUGIN_DIR_NAME}).",
    }


def install_claude_code(
    source: Path,
    spec: dict[str, Any],
    scope: str = "user",
    project_dir: Path | None = None,
    home: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Install the Claude Code plugin bundle (user) or `.mcp.json` + skills + rules (project)."""
    if scope not in ("user", "project"):
        raise ValueError(f"Unknown scope {scope!r}; expected 'user' or 'project'.")

    if scope == "user":
        dest = _resolve_home(home) / ".normet" / "plugins" / "claude-code" / PLUGIN_DIR_NAME
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            if (source / "README.md").is_file():
                shutil.copy2(source / "README.md", dest / "README.md")
        _copy_dir(source / ".claude-plugin", dest / ".claude-plugin", dry_run=dry_run)
        _copy_dir(source / "skills", dest / "skills", dry_run=dry_run)
        rule_path = _copy_rule_file(source, dest / "rules" / "normet.md", dry_run=dry_run)
        _write_json(dest / ".mcp.json", render_claude_mcp(spec), dry_run=dry_run)
        return {
            "target": "claude-code",
            "scope": scope,
            "dest": str(dest),
            "mcp_config": str(dest / ".mcp.json"),
            "rule": rule_path,
            "dry_run": dry_run,
            "note": f"Load with: claude --plugin-dir {dest} (or claude plugin add {dest})",
        }

    proj = _resolve_project(project_dir)
    mcp_path = proj / ".mcp.json"
    skills_dst = proj / ".claude" / "skills"
    merged = _merge_json_mcp(mcp_path, "normet", _server_entry(spec, include_type=True))
    _write_json(mcp_path, merged, dry_run=dry_run)
    installed_skills = _copy_skills(source / "skills", skills_dst, dry_run=dry_run)
    rule_path = _copy_rule_file(source, proj / ".claude" / "rules" / "normet.md", dry_run=dry_run)
    return {
        "target": "claude-code",
        "scope": scope,
        "dest": str(proj),
        "mcp_config": str(mcp_path),
        "skills": installed_skills,
        "rule": rule_path,
        "dry_run": dry_run,
        "note": f"Merged 'normet' into {mcp_path} and installed skills to {skills_dst}.",
    }


def install_dsh(
    source: Path,
    spec: dict[str, Any],
    scope: str = "user",
    project_dir: Path | None = None,
    home: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Install skills, guardrails, and `.dsh/mcp.yml` for DeepSeek Harness (`dsh`)."""
    if scope not in ("user", "project"):
        raise ValueError(f"Unknown scope {scope!r}; expected 'user' or 'project'.")

    base = _resolve_home(home) if scope == "user" else _resolve_project(project_dir)
    dsh_dir = base / ".dsh"
    skills_dst = dsh_dir / "skills"
    mcp_path = dsh_dir / "mcp.yml"
    installed_skills = _copy_skills(source / "skills", skills_dst, dry_run=dry_run)
    rule_path = _copy_rule_file(source, dsh_dir / "rules" / "normet.md", dry_run=dry_run)
    merged_yaml = _merge_dsh_mcp_yaml(mcp_path, spec)
    _write_text(mcp_path, merged_yaml, dry_run=dry_run)

    cordis_patches: list[str] = []
    if scope == "user":
        profiles_dir = dsh_dir / "profiles"
        if profiles_dir.is_dir():
            for prof in sorted(profiles_dir.iterdir()):
                if not prof.is_dir():
                    continue
                patch_path = prof / "cordis.patch.yml"
                if patch_path.is_file() or (prof / "cordis.yml").is_file():
                    merged_patch = _merge_dsh_cordis_patch(patch_path, spec)
                    _write_text(patch_path, merged_patch, dry_run=dry_run)
                    cordis_patches.append(str(patch_path))

    return {
        "target": "dsh",
        "scope": scope,
        "dest": str(dsh_dir),
        "mcp_config": str(mcp_path),
        "cordis_patches": cordis_patches,
        "skills": installed_skills,
        "rule": rule_path,
        "dry_run": dry_run,
        "note": (
            f"Installed skills to {skills_dst}, merged 'normet' into {mcp_path}"
            + (f", and patched {', '.join(cordis_patches)}" if cordis_patches else "")
            + " (compatible with DeepSeek Harness Desktop, dsh-mcp-project, and .agents/skills)."
        ),
    }


def install_opencode(
    source: Path,
    spec: dict[str, Any],
    scope: str = "user",
    project_dir: Path | None = None,
    home: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Install `opencode.json` (`mcp.normet` + `instructions`), guardrails, and skills for OpenCode."""
    if scope not in ("user", "project"):
        raise ValueError(f"Unknown scope {scope!r}; expected 'user' or 'project'.")

    if scope == "user":
        base = _resolve_home(home) / ".config" / "opencode"
        cfg_path = base / "opencode.json"
        skills_dst = base / "skills"
        rule_dst = base / "rules" / "normet.md"
    else:
        proj = _resolve_project(project_dir)
        cfg_path = proj / "opencode.json"
        skills_dst = proj / ".opencode" / "skills"
        rule_dst = proj / ".opencode" / "rules" / "normet.md"

    rule_path = _copy_rule_file(source, rule_dst, dry_run=dry_run)
    merged = _merge_opencode_json(cfg_path, spec, instruction_path=rule_path)
    _write_json(cfg_path, merged, dry_run=dry_run)
    installed_skills = _copy_skills(source / "skills", skills_dst, dry_run=dry_run)
    return {
        "target": "opencode",
        "scope": scope,
        "dest": str(cfg_path.parent),
        "mcp_config": str(cfg_path),
        "skills": installed_skills,
        "rule": rule_path,
        "dry_run": dry_run,
        "note": (
            f"Merged 'normet' into {cfg_path} and installed skills to {skills_dst} "
            "(verify with: opencode mcp list)."
        ),
    }


def install_codex(
    source: Path,
    spec: dict[str, Any],
    scope: str = "user",
    project_dir: Path | None = None,
    home: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Install `[mcp_servers.normet]` into `.codex/config.toml`, guardrails, and skills for Codex."""
    if scope not in ("user", "project"):
        raise ValueError(f"Unknown scope {scope!r}; expected 'user' or 'project'.")

    base = _resolve_home(home) if scope == "user" else _resolve_project(project_dir)
    codex_dir = base / ".codex"
    cfg_path = codex_dir / "config.toml"
    skills_dst = codex_dir / "skills"

    merged_toml = _merge_codex_toml(cfg_path, spec)
    _write_text(cfg_path, merged_toml, dry_run=dry_run)
    installed_skills = _copy_skills(source / "skills", skills_dst, dry_run=dry_run)
    rule_path = _copy_rule_file(source, codex_dir / "rules" / "normet.md", dry_run=dry_run)
    return {
        "target": "codex",
        "scope": scope,
        "dest": str(codex_dir),
        "mcp_config": str(cfg_path),
        "skills": installed_skills,
        "rule": rule_path,
        "dry_run": dry_run,
        "note": (
            f"Merged [mcp_servers.normet] into {cfg_path} and installed skills to "
            f"{skills_dst} (verify with: codex mcp list; .agents/skills is also supported)."
        ),
    }


def _remove_json_mcp(dest: Path, server_name: str, dry_run: bool) -> bool:
    if not dest.is_file():
        return False
    data = json.loads(dest.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return False
    servers = data.get("mcpServers")
    if not isinstance(servers, dict) or server_name not in servers:
        return False
    del servers[server_name]
    _write_json(dest, data, dry_run=dry_run)
    return True


def _remove_dsh_mcp_yaml(dest: Path, dry_run: bool) -> bool:
    if not dest.is_file():
        return False
    raw = dest.read_text(encoding="utf-8")
    lines = raw.splitlines()
    top_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "mcpServers:"), None)
    if top_idx is None:
        return False
    end_idx = len(lines)
    for i in range(top_idx + 1, len(lines)):
        ln = lines[i]
        if ln.strip() and not ln[0].isspace() and not ln.lstrip().startswith("#"):
            end_idx = i
            break
    item_starts: list[tuple[int, int]] = []
    for i in range(top_idx + 1, end_idx):
        m = re.match(r"^(\s*)-\s+", lines[i])
        if m:
            item_starts.append((i, len(m.group(1))))
    if not item_starts:
        return False
    base_indent = min(indent for _, indent in item_starts)
    top_items = [idx for idx, indent in item_starts if indent == base_indent]
    for pos, start in enumerate(top_items):
        stop = top_items[pos + 1] if pos + 1 < len(top_items) else end_idx
        block = "\n".join(lines[start:stop])
        if re.search(r'(?:^|\n)\s*(?:-\s*)?name:\s*["\']?normet["\']?\s*(?:\n|$)', block):
            new_lines = lines[:start] + lines[stop:]
            text = ("\n".join(new_lines) + "\n") if new_lines else ""
            _write_text(dest, text, dry_run=dry_run)
            return True
    return False


def _remove_dsh_cordis_patch(dest: Path, dry_run: bool) -> bool:
    if not dest.is_file():
        return False
    raw = dest.read_text(encoding="utf-8")
    if "mcp-normet" not in raw:
        return False
    lines = raw.splitlines()
    top_starts = [i for i, ln in enumerate(lines) if ln.startswith("- ")]
    if not top_starts:
        return False
    kept_lines: list[str] = list(lines[: top_starts[0]])
    for pos, start in enumerate(top_starts):
        stop = top_starts[pos + 1] if pos + 1 < len(top_starts) else len(lines)
        entry = lines[start:stop]
        if re.match(r"^-\s+insert:\s*(?:#.*)?$", entry[0]):
            entry = _strip_normet_from_insert_block(entry)
        kept_lines.extend(entry)
    text = "\n".join(kept_lines).rstrip()
    _write_text(dest, (text + "\n") if text else "", dry_run=dry_run)
    return True


def _remove_opencode_json(dest: Path, rule_path: Path, dry_run: bool) -> bool:
    if not dest.is_file():
        return False
    data = json.loads(dest.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return False
    changed = False
    mcp = data.get("mcp")
    if isinstance(mcp, dict) and "normet" in mcp:
        del mcp["normet"]
        changed = True
    inst = data.get("instructions")
    if isinstance(inst, list) and str(rule_path) in inst:
        data["instructions"] = [x for x in inst if x != str(rule_path)]
        if not data["instructions"]:
            del data["instructions"]
        changed = True
    if changed:
        _write_json(dest, data, dry_run=dry_run)
    return changed


def _remove_codex_toml(dest: Path, dry_run: bool) -> bool:
    if not dest.is_file():
        return False
    raw = dest.read_text(encoding="utf-8")
    try:
        parsed = tomllib.loads(raw)
    except tomllib.TOMLDecodeError:
        parsed = {}
    if "normet" not in (parsed.get("mcp_servers") or {}):
        return False
    kept: list[str] = []
    skipping = False
    for line in raw.splitlines():
        m = re.match(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$", line)
        if m:
            section = m.group(1).strip()
            skipping = section == "mcp_servers.normet" or section.startswith("mcp_servers.normet.")
        if not skipping:
            kept.append(line)
    text = "\n".join(kept).strip()
    new_raw = (text + "\n") if text else ""
    if new_raw:
        tomllib.loads(new_raw)
    _write_text(dest, new_raw, dry_run=dry_run)
    return True


def uninstall_antigravity(
    scope: str = "user",
    project_dir: Path | None = None,
    home: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if scope not in ("user", "project"):
        raise ValueError(f"Unknown scope {scope!r}; expected 'user' or 'project'.")
    if scope == "user":
        dest = _resolve_home(home) / ".gemini" / "config" / "plugins" / PLUGIN_DIR_NAME
    else:
        dest = _resolve_project(project_dir) / ".agents" / "plugins" / PLUGIN_DIR_NAME
    existed = dest.is_dir()
    if existed and not dry_run:
        shutil.rmtree(dest)
    return {
        "target": "antigravity",
        "scope": scope,
        "dest": str(dest),
        "removed": existed,
        "dry_run": dry_run,
    }


def uninstall_claude_code(
    scope: str = "user",
    project_dir: Path | None = None,
    home: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if scope not in ("user", "project"):
        raise ValueError(f"Unknown scope {scope!r}; expected 'user' or 'project'.")
    if scope == "user":
        dest = _resolve_home(home) / ".normet" / "plugins" / "claude-code" / PLUGIN_DIR_NAME
        existed = dest.is_dir()
        if existed and not dry_run:
            shutil.rmtree(dest)
        return {
            "target": "claude-code",
            "scope": scope,
            "dest": str(dest),
            "removed": existed,
            "dry_run": dry_run,
        }
    proj = _resolve_project(project_dir)
    mcp_removed = _remove_json_mcp(proj / ".mcp.json", "normet", dry_run=dry_run)
    skills_removed = _remove_skills(proj / ".claude" / "skills", dry_run=dry_run)
    rule_removed = _remove_file(proj / ".claude" / "rules" / "normet.md", dry_run=dry_run)
    return {
        "target": "claude-code",
        "scope": scope,
        "dest": str(proj),
        "removed": bool(mcp_removed or skills_removed or rule_removed),
        "skills_removed": skills_removed,
        "dry_run": dry_run,
    }


def uninstall_dsh(
    scope: str = "user",
    project_dir: Path | None = None,
    home: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if scope not in ("user", "project"):
        raise ValueError(f"Unknown scope {scope!r}; expected 'user' or 'project'.")
    base = _resolve_home(home) if scope == "user" else _resolve_project(project_dir)
    dsh_dir = base / ".dsh"
    yaml_removed = _remove_dsh_mcp_yaml(dsh_dir / "mcp.yml", dry_run=dry_run)
    skills_removed = _remove_skills(dsh_dir / "skills", dry_run=dry_run)
    rule_removed = _remove_file(dsh_dir / "rules" / "normet.md", dry_run=dry_run)
    patches_cleaned: list[str] = []
    if scope == "user":
        profiles_dir = dsh_dir / "profiles"
        if profiles_dir.is_dir():
            for prof in sorted(profiles_dir.iterdir()):
                if prof.is_dir() and _remove_dsh_cordis_patch(
                    prof / "cordis.patch.yml", dry_run=dry_run
                ):
                    patches_cleaned.append(str(prof / "cordis.patch.yml"))
    return {
        "target": "dsh",
        "scope": scope,
        "dest": str(dsh_dir),
        "removed": bool(yaml_removed or skills_removed or rule_removed or patches_cleaned),
        "skills_removed": skills_removed,
        "cordis_patches": patches_cleaned,
        "dry_run": dry_run,
    }


def uninstall_opencode(
    scope: str = "user",
    project_dir: Path | None = None,
    home: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if scope not in ("user", "project"):
        raise ValueError(f"Unknown scope {scope!r}; expected 'user' or 'project'.")
    if scope == "user":
        base = _resolve_home(home) / ".config" / "opencode"
        cfg_path = base / "opencode.json"
        skills_dst = base / "skills"
        rule_dst = base / "rules" / "normet.md"
    else:
        proj = _resolve_project(project_dir)
        cfg_path = proj / "opencode.json"
        skills_dst = proj / ".opencode" / "skills"
        rule_dst = proj / ".opencode" / "rules" / "normet.md"
    cfg_removed = _remove_opencode_json(cfg_path, rule_dst, dry_run=dry_run)
    skills_removed = _remove_skills(skills_dst, dry_run=dry_run)
    rule_removed = _remove_file(rule_dst, dry_run=dry_run)
    return {
        "target": "opencode",
        "scope": scope,
        "dest": str(cfg_path.parent),
        "removed": bool(cfg_removed or skills_removed or rule_removed),
        "skills_removed": skills_removed,
        "dry_run": dry_run,
    }


def uninstall_codex(
    scope: str = "user",
    project_dir: Path | None = None,
    home: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if scope not in ("user", "project"):
        raise ValueError(f"Unknown scope {scope!r}; expected 'user' or 'project'.")
    base = _resolve_home(home) if scope == "user" else _resolve_project(project_dir)
    codex_dir = base / ".codex"
    cfg_removed = _remove_codex_toml(codex_dir / "config.toml", dry_run=dry_run)
    skills_removed = _remove_skills(codex_dir / "skills", dry_run=dry_run)
    rule_removed = _remove_file(codex_dir / "rules" / "normet.md", dry_run=dry_run)
    return {
        "target": "codex",
        "scope": scope,
        "dest": str(codex_dir),
        "removed": bool(cfg_removed or skills_removed or rule_removed),
        "skills_removed": skills_removed,
        "dry_run": dry_run,
    }


def _present_skills(skills_dir: Path) -> list[str]:
    return [s for s in _NORMET_SKILLS if (skills_dir / s / "SKILL.md").is_file()]


def status_plugin(
    scope: str = "user",
    project_dir: Path | None = None,
    home: Path | None = None,
) -> list[dict[str, Any]]:
    """Inspect installed normet plugin files and MCP endpoints across all 5 targets."""
    if scope not in ("user", "project"):
        raise ValueError(f"Unknown scope {scope!r}; expected 'user' or 'project'.")
    h = _resolve_home(home)
    p = _resolve_project(project_dir)
    out: list[dict[str, Any]] = []

    # 1. Antigravity
    ag_dir = (
        h / ".gemini" / "config" / "plugins" / PLUGIN_DIR_NAME
        if scope == "user"
        else p / ".agents" / "plugins" / PLUGIN_DIR_NAME
    )
    ag_cfg = ag_dir / "mcp_config.json"
    ag_srv: dict[str, Any] = {}
    if ag_cfg.is_file():
        try:
            ag_srv = (json.loads(ag_cfg.read_text(encoding="utf-8")).get("mcpServers") or {}).get(
                "normet"
            ) or {}
        except (OSError, ValueError):
            ag_srv = {}
    out.append(
        {
            "target": "antigravity",
            "scope": scope,
            "installed": bool(ag_srv) or (ag_dir / "plugin.json").is_file(),
            "mcp_configured": bool(ag_srv),
            "config_path": str(ag_cfg),
            "skills": _present_skills(ag_dir / "skills"),
            "rule_installed": (ag_dir / "rules" / "AGENTS.md").is_file(),
            "transport": "http" if "url" in ag_srv else ("stdio" if "command" in ag_srv else None),
            "command": ag_srv.get("command"),
            "url": ag_srv.get("url"),
            "env": ag_srv.get("env") or {},
        }
    )

    # 2. Claude Code
    cc_dest = h / ".normet" / "plugins" / "claude-code" / PLUGIN_DIR_NAME if scope == "user" else p
    cc_cfg = cc_dest / ".mcp.json"
    cc_skills = cc_dest / "skills" if scope == "user" else p / ".claude" / "skills"
    cc_rule = (
        cc_dest / "rules" / "normet.md"
        if scope == "user"
        else p / ".claude" / "rules" / "normet.md"
    )
    cc_srv: dict[str, Any] = {}
    if cc_cfg.is_file():
        try:
            cc_srv = (json.loads(cc_cfg.read_text(encoding="utf-8")).get("mcpServers") or {}).get(
                "normet"
            ) or {}
        except (OSError, ValueError):
            cc_srv = {}
    out.append(
        {
            "target": "claude-code",
            "scope": scope,
            "installed": bool(cc_srv) or bool(_present_skills(cc_skills)),
            "mcp_configured": bool(cc_srv),
            "config_path": str(cc_cfg),
            "skills": _present_skills(cc_skills),
            "rule_installed": cc_rule.is_file(),
            "transport": "http" if "url" in cc_srv else ("stdio" if "command" in cc_srv else None),
            "command": cc_srv.get("command"),
            "url": cc_srv.get("url"),
            "env": cc_srv.get("env") or {},
        }
    )

    # 3. DeepSeek Harness
    dsh_dir = (h if scope == "user" else p) / ".dsh"
    dsh_cfg = dsh_dir / "mcp.yml"
    dsh_has = False
    if dsh_cfg.is_file():
        dsh_text = dsh_cfg.read_text(encoding="utf-8", errors="replace")
        dsh_has = bool(re.search(r'name:\s*["\']?normet["\']?', dsh_text))
    cordis_ok = False
    if scope == "user" and (dsh_dir / "profiles").is_dir():
        for prof in (dsh_dir / "profiles").iterdir():
            cp = prof / "cordis.patch.yml"
            if cp.is_file() and "mcp-normet" in cp.read_text(encoding="utf-8", errors="replace"):
                cordis_ok = True
                break
    out.append(
        {
            "target": "dsh",
            "scope": scope,
            "installed": dsh_has or cordis_ok or bool(_present_skills(dsh_dir / "skills")),
            "mcp_configured": dsh_has or cordis_ok,
            "config_path": str(dsh_cfg),
            "skills": _present_skills(dsh_dir / "skills"),
            "rule_installed": (dsh_dir / "rules" / "normet.md").is_file(),
            "transport": None,
            "command": None,
            "url": None,
            "env": {},
        }
    )

    # 4. OpenCode
    oc_base = h / ".config" / "opencode" if scope == "user" else p
    oc_cfg = oc_base / "opencode.json"
    oc_skills = oc_base / "skills" if scope == "user" else p / ".opencode" / "skills"
    oc_rule = (
        oc_base / "rules" / "normet.md"
        if scope == "user"
        else p / ".opencode" / "rules" / "normet.md"
    )
    oc_srv: dict[str, Any] = {}
    if oc_cfg.is_file():
        try:
            oc_srv = (json.loads(oc_cfg.read_text(encoding="utf-8")).get("mcp") or {}).get(
                "normet"
            ) or {}
        except (OSError, ValueError):
            oc_srv = {}
    cmd_list = oc_srv.get("command") if isinstance(oc_srv.get("command"), list) else []
    out.append(
        {
            "target": "opencode",
            "scope": scope,
            "installed": bool(oc_srv) or bool(_present_skills(oc_skills)),
            "mcp_configured": bool(oc_srv),
            "config_path": str(oc_cfg),
            "skills": _present_skills(oc_skills),
            "rule_installed": oc_rule.is_file(),
            "transport": "http"
            if oc_srv.get("type") == "remote"
            else ("stdio" if oc_srv.get("type") == "local" else None),
            "command": cmd_list[0] if cmd_list else None,
            "url": oc_srv.get("url"),
            "env": oc_srv.get("environment") or {},
        }
    )

    # 5. Codex
    cx_dir = (h if scope == "user" else p) / ".codex"
    cx_cfg = cx_dir / "config.toml"
    cx_srv: dict[str, Any] = {}
    if cx_cfg.is_file():
        try:
            cx_srv = (
                tomllib.loads(cx_cfg.read_text(encoding="utf-8")).get("mcp_servers") or {}
            ).get("normet") or {}
        except (OSError, tomllib.TOMLDecodeError):
            cx_srv = {}
    out.append(
        {
            "target": "codex",
            "scope": scope,
            "installed": bool(cx_srv) or bool(_present_skills(cx_dir / "skills")),
            "mcp_configured": bool(cx_srv),
            "config_path": str(cx_cfg),
            "skills": _present_skills(cx_dir / "skills"),
            "rule_installed": (cx_dir / "rules" / "normet.md").is_file(),
            "transport": "http" if "url" in cx_srv else ("stdio" if "command" in cx_srv else None),
            "command": cx_srv.get("command"),
            "url": cx_srv.get("url"),
            "env": cx_srv.get("env") or {},
        }
    )
    return out


def _describe_exit(rc: int) -> str:
    if rc < 0:
        try:
            return f"signal {signal.Signals(-rc).name} (exit code {rc})"
        except ValueError:
            return f"signal {-rc} (exit code {rc})"
    return f"exit code {rc}"


def doctor_plugin(
    python_exe: str | None = None,
    source: str | None = None,
    scope: str = "user",
    home: Path | None = None,
    project_dir: Path | None = None,
    skip_torch_probe: bool = False,
) -> dict[str, Any]:
    """Run health checks across plugin bundle, Python/OpenMP safety, MCP SDK, roots, and installs."""
    exe = python_exe or sys.executable
    checks: list[dict[str, Any]] = []

    # 1. Plugin bundle validity
    bundle_errors = validate_plugin(source)
    checks.append(
        {
            "name": "bundle",
            "status": "error" if bundle_errors else "ok",
            "detail": "; ".join(bundle_errors)
            if bundle_errors
            else f"Valid ({find_plugin_source(source)})",
        }
    )

    # 2. Interpreter & conda / ~/.local user-site collision detection
    probe_env = dict(os.environ, NORMET_NO_CALIBRATION="1")
    src_dir = Path(__file__).resolve().parents[2]
    if src_dir.name == "src" and (src_dir / "normet" / "mcp_server.py").is_file():
        old_pp = probe_env.get("PYTHONPATH", "")
        probe_env["PYTHONPATH"] = os.pathsep.join([str(src_dir)] + ([old_pp] if old_pp else []))

    env_code = (
        "import json, os, site, sys; "
        "is_conda = os.path.isdir(os.path.join(sys.prefix, 'conda-meta')); "
        "user_site = site.getusersitepackages() if site.ENABLE_USER_SITE else ''; "
        "user_torch = bool(user_site and os.path.isdir(os.path.join(user_site, 'torch'))); "
        "print(json.dumps({'executable': sys.executable, 'version': sys.version.split()[0], "
        "'is_conda': is_conda, 'user_site': user_site, 'user_torch': user_torch}))"
    )
    try:
        res_env = subprocess.run(
            [exe, "-c", env_code],
            capture_output=True,
            text=True,
            timeout=15,
            env=probe_env,
        )
        if res_env.returncode == 0:
            info = json.loads(res_env.stdout.strip())
            if info.get("is_conda") and info.get("user_torch"):
                checks.append(
                    {
                        "name": "interpreter",
                        "status": "warn",
                        "detail": (
                            f"{info['executable']} (Python {info['version']}) is conda-based while "
                            f"{info['user_site']} contains torch; dual-libomp SIGSEGV risk — prefer an "
                            "isolated virtualenv or set PYTHONNOUSERSITE=1."
                        ),
                    }
                )
            else:
                checks.append(
                    {
                        "name": "interpreter",
                        "status": "ok",
                        "detail": f"{info['executable']} (Python {info['version']})",
                    }
                )
        else:
            checks.append(
                {
                    "name": "interpreter",
                    "status": "error",
                    "detail": f"Interpreter probe failed with {_describe_exit(res_env.returncode)}: {res_env.stderr.strip()}",
                }
            )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        checks.append({"name": "interpreter", "status": "error", "detail": str(exc)})

    # 3. MCP SDK check
    mcp_code = "import importlib.metadata as m, mcp; print(m.version('mcp'))"
    try:
        res_mcp = subprocess.run(
            [exe, "-c", mcp_code],
            capture_output=True,
            text=True,
            timeout=15,
            env=probe_env,
        )
        if res_mcp.returncode == 0:
            checks.append(
                {
                    "name": "mcp_sdk",
                    "status": "ok",
                    "detail": f"mcp {res_mcp.stdout.strip()}",
                }
            )
        else:
            checks.append(
                {
                    "name": "mcp_sdk",
                    "status": "error",
                    "detail": 'MCP SDK not importable; run `pip install "normet[mcp]"`.',
                }
            )
    except (OSError, subprocess.SubprocessError) as exc:
        checks.append({"name": "mcp_sdk", "status": "error", "detail": str(exc)})

    # 4. Torch + hardware safety probe
    if skip_torch_probe:
        checks.append(
            {"name": "torch_runtime", "status": "ok", "detail": "Skipped (--skip-torch-probe)."}
        )
    else:
        torch_code = (
            "import torch; "
            "_ = (torch.zeros((2, 2)) @ torch.ones((2, 2))).sum().item(); "
            "print(torch.__version__)"
        )
        try:
            res_torch = subprocess.run(
                [exe, "-c", torch_code],
                capture_output=True,
                text=True,
                timeout=25,
                env=probe_env,
            )
            if res_torch.returncode == 0:
                checks.append(
                    {
                        "name": "torch_runtime",
                        "status": "ok",
                        "detail": f"torch {res_torch.stdout.strip()} ok",
                    }
                )
            else:
                hint = (
                    " (likely dual-libomp or native crash; use an isolated virtualenv)"
                    if res_torch.returncode < 0
                    else ""
                )
                err_tail = (res_torch.stderr or res_torch.stdout or "").strip()[-240:]
                checks.append(
                    {
                        "name": "torch_runtime",
                        "status": "error",
                        "detail": f"Torch probe failed with {_describe_exit(res_torch.returncode)}{hint}: {err_tail}",
                    }
                )
        except (OSError, subprocess.SubprocessError) as exc:
            checks.append({"name": "torch_runtime", "status": "error", "detail": str(exc)})

    # 5. NORMET_MCP_ROOTS check
    roots_raw = os.environ.get("NORMET_MCP_ROOTS", "").strip()
    resolved_home = _resolve_home(home)
    if not roots_raw:
        checks.append(
            {
                "name": "roots",
                "status": "ok",
                "detail": f"NORMET_MCP_ROOTS unset (defaults to home directory {resolved_home}).",
            }
        )
    else:
        entries = [
            os.path.realpath(os.path.expanduser(r))
            for r in roots_raw.split(os.pathsep)
            if r.strip()
        ]
        missing = [r for r in entries if not os.path.isdir(r)]
        covers_home = any(
            str(resolved_home) == r or str(resolved_home).startswith(r.rstrip(os.sep) + os.sep)
            for r in entries
        )
        if missing:
            checks.append(
                {
                    "name": "roots",
                    "status": "warn",
                    "detail": f"NORMET_MCP_ROOTS contains non-existent directory: {', '.join(missing)}.",
                }
            )
        elif not covers_home:
            checks.append(
                {
                    "name": "roots",
                    "status": "warn",
                    "detail": (
                        f"NORMET_MCP_ROOTS ({roots_raw}) narrows access away from {resolved_home}; "
                        "datasets outside listed roots (e.g. Desktop/Downloads) will be blocked."
                    ),
                }
            )
        else:
            checks.append(
                {"name": "roots", "status": "ok", "detail": f"NORMET_MCP_ROOTS={roots_raw}"}
            )

    # 6. Platform installation scan
    installs = status_plugin(scope=scope, project_dir=project_dir, home=home)
    configured = [row["target"] for row in installs if row["mcp_configured"]]
    checks.append(
        {
            "name": "installations",
            "status": "ok" if configured else "warn",
            "detail": (
                f"Configured ({scope}): {', '.join(configured)}"
                if configured
                else f"No normet MCP configs found in {scope} scope yet (run `normet-plugin install`)."
            ),
        }
    )

    ok = not any(c["status"] == "error" for c in checks)
    return {
        "ok": ok,
        "python": exe,
        "checks": checks,
        "installations": installs,
    }


def _parse_frontmatter(text: str) -> dict[str, str] | None:
    """Extract `name` and `description` from a SKILL.md YAML frontmatter block."""
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        return None
    lines = text.splitlines()
    end = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end = idx
            break
    if end is None:
        return None
    fm: dict[str, str] = {}
    i = 1
    while i < end:
        ln = lines[i]
        m = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", ln)
        if not m:
            i += 1
            continue
        key, val = m.group(1), m.group(2).strip()
        if val in (">-", ">", "|", "|-"):
            block: list[str] = []
            i += 1
            while i < end and (lines[i].startswith("  ") or not lines[i].strip()):
                block.append(lines[i].strip())
                i += 1
            fm[key] = " ".join(part for part in block if part).strip()
            continue
        fm[key] = val.strip("\"'")
        i += 1
    return fm


def _known_mcp_tools(source: Path) -> set | None:
    """Return the set of `normet_*` tool names defined in `normet/app/mcp_server.py`, if found."""
    candidates = [
        source.parent / "src" / "normet" / "app" / "mcp_server.py",
        source.parent / "src" / "normet" / "mcp_server.py",
        Path(__file__).resolve().parents[1] / "app" / "mcp_server.py",
        Path(__file__).resolve().parents[1] / "mcp_server.py",
    ]
    for path in candidates:
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            names = set(re.findall(r"def (normet_[a-z0-9_]+)\s*\(", text))
            if names:
                return names
    return None


def validate_plugin(source: str | None = None) -> list[str]:
    """Validate the `plugin/` bundle across Antigravity, Claude Code, DSH, OpenCode, and Codex."""
    errors: list[str] = []
    try:
        root = find_plugin_source(source)
    except FileNotFoundError as exc:
        return [str(exc)]

    # 1. Antigravity plugin.json
    ag_path = root / "plugin.json"
    # `Any`, not `dict`: the JSON may be a list or a string, and the check below
    # is what rejects that -- annotating it as a dict made mypy call the branch
    # unreachable.
    ag_data: Any = {}
    try:
        ag_data = json.loads(ag_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"Invalid plugin.json: {exc}")
    else:
        if not isinstance(ag_data, dict):
            errors.append("plugin.json must be a JSON object.")
        else:
            extra = set(ag_data) - ANTIGRAVITY_ALLOWED_FIELDS
            if extra:
                errors.append(f"plugin.json has unsupported fields: {sorted(extra)}")
            name = ag_data.get("name", "")
            if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
                errors.append(f"plugin.json 'name' must be lowercase kebab-case: {name!r}")
            for req in ("description", "version"):
                if not str(ag_data.get(req, "")).strip():
                    errors.append(f"plugin.json missing non-empty '{req}'.")
            logo = ag_data.get("logo")
            if not isinstance(logo, str) or not (root / logo).is_file():
                errors.append(f"plugin.json logo does not exist: {logo!r}")
            prompts = ag_data.get("suggestedPrompts")
            if (
                not isinstance(prompts, list)
                or not (1 <= len(prompts) <= 3)
                or any(not isinstance(p, str) or not p.strip() for p in prompts)
            ):
                errors.append("plugin.json 'suggestedPrompts' must contain 1..3 non-empty strings.")

    # 2. Claude Code .claude-plugin/plugin.json
    cc_path = root / ".claude-plugin" / "plugin.json"
    if not cc_path.is_file():
        errors.append("Missing .claude-plugin/plugin.json.")
    else:
        try:
            cc_data = json.loads(cc_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"Invalid .claude-plugin/plugin.json: {exc}")
        else:
            for req in ("name", "version", "description"):
                if not str(cc_data.get(req, "")).strip():
                    errors.append(f".claude-plugin/plugin.json missing '{req}'.")
            if ag_data.get("version") and cc_data.get("version") != ag_data.get("version"):
                errors.append(
                    f"Version mismatch between plugin.json ({ag_data.get('version')}) "
                    f"and .claude-plugin/plugin.json ({cc_data.get('version')})."
                )

    # 3. MCP configs (Antigravity + Claude Code)
    for rel in ("mcp_config.json", ".mcp.json"):
        cfg_path = root / rel
        if not cfg_path.is_file():
            errors.append(f"Missing {rel}.")
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"Invalid {rel}: {exc}")
            continue
        normet_srv = (cfg.get("mcpServers") or {}).get("normet") if isinstance(cfg, dict) else None
        if not isinstance(normet_srv, dict) or not normet_srv.get("command"):
            errors.append(f"{rel} must define mcpServers.normet.command.")

    # 4. DeepSeek Harness .dsh/mcp.yml and .dsh/cordis.patch.yml
    dsh_path = root / ".dsh" / "mcp.yml"
    if not dsh_path.is_file():
        errors.append("Missing .dsh/mcp.yml.")
    else:
        dsh_text = dsh_path.read_text(encoding="utf-8")
        if "mcpServers:" not in dsh_text or not re.search(r'name:\s*["\']?normet["\']?', dsh_text):
            errors.append(".dsh/mcp.yml must declare mcpServers with name 'normet'.")

    cordis_tpl = root / ".dsh" / "cordis.patch.yml"
    if not cordis_tpl.is_file():
        errors.append("Missing .dsh/cordis.patch.yml.")
    else:
        cordis_text = cordis_tpl.read_text(encoding="utf-8")
        if "@deepseek-ai/dsh-mcp-client" not in cordis_text or "mcp-normet" not in cordis_text:
            errors.append(
                ".dsh/cordis.patch.yml must declare id 'mcp-normet' for '@deepseek-ai/dsh-mcp-client'."
            )

    # 5. OpenCode opencode.json
    oc_path = root / "opencode.json"
    if not oc_path.is_file():
        errors.append("Missing opencode.json.")
    else:
        try:
            oc_cfg = json.loads(oc_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"Invalid opencode.json: {exc}")
        else:
            oc_normet = (
                (oc_cfg.get("mcp") or {}).get("normet") if isinstance(oc_cfg, dict) else None
            )
            if (
                not isinstance(oc_normet, dict)
                or oc_normet.get("type") != "local"
                or not isinstance(oc_normet.get("command"), list)
                or not oc_normet["command"]
                or not all(isinstance(arg, str) and arg for arg in oc_normet["command"])
            ):
                errors.append(
                    "opencode.json must define mcp.normet with type='local' and non-empty command list."
                )

    # 6. Codex .codex/config.toml
    codex_path = root / ".codex" / "config.toml"
    if not codex_path.is_file():
        errors.append("Missing .codex/config.toml.")
    else:
        try:
            codex_cfg = tomllib.loads(codex_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"Invalid .codex/config.toml: {exc}")
        else:
            codex_normet = (codex_cfg.get("mcp_servers") or {}).get("normet")
            if (
                not isinstance(codex_normet, dict)
                or not isinstance(codex_normet.get("command"), str)
                or not codex_normet["command"].strip()
                or not isinstance(codex_normet.get("args"), list)
            ):
                errors.append(
                    ".codex/config.toml must define [mcp_servers.normet] with command and args list."
                )

    # 7. Antigravity rules/AGENTS.md (no YAML frontmatter)
    rules_path = root / "rules" / "AGENTS.md"
    if not rules_path.is_file():
        errors.append("Missing rules/AGENTS.md.")
    else:
        rules_text = rules_path.read_text(encoding="utf-8")
        if not rules_text.strip():
            errors.append("rules/AGENTS.md must not be empty.")
        if rules_text.lstrip().startswith("---"):
            errors.append("rules/AGENTS.md must be plain markdown without YAML frontmatter.")

    # 6. Shared skills
    skills_dir = root / "skills"
    skill_dirs = (
        [d for d in sorted(skills_dir.iterdir()) if d.is_dir()] if skills_dir.is_dir() else []
    )
    if not skill_dirs:
        errors.append("skills/ must contain at least one skill directory.")

    known_tools = _known_mcp_tools(root)
    for sdir in skill_dirs:
        if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", sdir.name):
            errors.append(
                f"Skill directory {sdir.name!r} must be strict kebab-case (^[a-z0-9]+(-[a-z0-9]+)*$)."
            )
        skill_md = sdir / "SKILL.md"
        if not skill_md.is_file():
            errors.append(f"Missing SKILL.md in {sdir}.")
            continue
        text = skill_md.read_text(encoding="utf-8")
        fm = _parse_frontmatter(text)
        if fm is None:
            errors.append(f"{skill_md} missing valid YAML frontmatter.")
            continue
        if fm.get("name") != sdir.name:
            errors.append(
                f"{skill_md} frontmatter name {fm.get('name')!r} != directory {sdir.name!r}."
            )
        if not fm.get("description", "").strip():
            errors.append(f"{skill_md} missing non-empty 'description'.")
        if known_tools is not None:
            # The prose names files as well as tools: `normet_results` is the folder a
            # finished job's bundle is written to beside the data, and a token ending in
            # `_` is a prefix wildcard (`normet_start_*`), not a tool name.
            not_tools = {"normet_results"}
            for md_file in sorted(sdir.rglob("*.md")):
                md_text = md_file.read_text(encoding="utf-8")
                for token in sorted(set(re.findall(r"\bnormet_[a-z0-9_]+\b", md_text))):
                    if (
                        token not in known_tools
                        and token not in not_tools
                        and not token.endswith("_")
                    ):
                        errors.append(f"{md_file} references unknown MCP tool {token!r}.")

    return errors


def _add_server_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--command", help="Explicit normet-mcp executable or command.")
    parser.add_argument(
        "--python",
        dest="use_python",
        action="store_true",
        help="Invoke the current Python interpreter with -m normet.mcp_server.",
    )
    parser.add_argument(
        "--uvx",
        dest="use_uvx",
        action="store_true",
        help="Use 'uvx --from normet[mcp] normet-mcp' for zero-config startup.",
    )
    parser.add_argument(
        "--toolset",
        choices=("all", "core"),
        default="all",
        help="Expose all 26 tools or only the 14 of the standard de-weathering workflow.",
    )
    parser.add_argument("--roots", help="Value for NORMET_MCP_ROOTS.")
    parser.add_argument("--mcp-home", help="Value for NORMET_MCP_HOME.")
    parser.add_argument(
        "--env",
        dest="extra_env",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Extra environment variable to pass to stdio normet-mcp (repeatable).",
    )
    parser.add_argument(
        "--url",
        help="Connect to a remote `normet-mcp --transport streamable-http` URL instead of stdio.",
    )
    parser.add_argument(
        "--auth-token",
        help="Optional Bearer token for --url remote MCP servers.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="normet-plugin",
        description=(
            "Validate, inspect, diagnose, install, and uninstall normet AI agent plugins "
            "(Antigravity, Claude Code, DSH, OpenCode, Codex)."
        ),
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_install = sub.add_parser("install", help="Install plugin files for target agent(s).")
    p_install.add_argument(
        "--target",
        choices=TARGETS + ("all",),
        default="all",
        help="Target agent platform (default: all).",
    )
    p_install.add_argument(
        "--scope",
        choices=("user", "project"),
        default="user",
        help="Install into user home or a project workspace (default: user).",
    )
    p_install.add_argument("--project-dir", help="Project workspace root for --scope project.")
    p_install.add_argument("--source", help="Path to plugin/ bundle directory.")
    p_install.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing files.",
    )
    _add_server_args(p_install)

    p_uninstall = sub.add_parser(
        "uninstall", help="Uninstall normet plugin entries from target agent(s)."
    )
    p_uninstall.add_argument("--target", choices=TARGETS + ("all",), default="all")
    p_uninstall.add_argument("--scope", choices=("user", "project"), default="user")
    p_uninstall.add_argument("--project-dir", help="Project workspace root for --scope project.")
    p_uninstall.add_argument("--dry-run", action="store_true")

    p_status = sub.add_parser("status", help="Show installed plugin status across all platforms.")
    p_status.add_argument("--scope", choices=("user", "project"), default="user")
    p_status.add_argument("--project-dir", help="Project workspace root for --scope project.")
    p_status.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON output.")

    p_doc = sub.add_parser(
        "doctor", help="Diagnose plugin bundle, Python/torch safety, MCP SDK, roots, and installs."
    )
    p_doc.add_argument("--python", dest="python_exe", help="Python interpreter to check.")
    p_doc.add_argument("--source", help="Path to plugin/ bundle directory.")
    p_doc.add_argument("--scope", choices=("user", "project"), default="user")
    p_doc.add_argument("--project-dir", help="Project workspace root for --scope project.")
    p_doc.add_argument(
        "--skip-torch-probe", action="store_true", help="Skip torch/hardware subprocess probe."
    )
    p_doc.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON output.")

    p_prune = sub.add_parser("prune", help="Clean up finished job directories in $NORMET_MCP_HOME.")
    p_prune.add_argument("--mcp-home", help="Override NORMET_MCP_HOME directory.")
    p_prune.add_argument(
        "--days", type=float, default=None, help="Remove finished jobs older than N days."
    )
    p_prune.add_argument(
        "--states",
        nargs="+",
        default=None,
        help="Job states to prune (e.g. failed cancelled done).",
    )
    p_prune.add_argument("--keep", type=int, default=None, help="Always keep the newest K jobs.")
    p_prune.add_argument("--dry-run", action="store_true")

    p_print = sub.add_parser("print-config", help="Print MCP config snippet for a target.")
    p_print.add_argument("--target", choices=TARGETS, required=True)
    _add_server_args(p_print)

    p_val = sub.add_parser("validate", help="Validate plugin manifests and skill files.")
    p_val.add_argument("--source", help="Path to plugin/ bundle directory.")

    args = parser.parse_args(argv)

    if args.subcommand == "validate":
        errors = validate_plugin(args.source)
        if errors:
            for err in errors:
                print(f"ERROR: {err}", file=sys.stderr)
            return 1
        print("Plugin bundle is valid.")
        return 0

    if args.subcommand == "status":
        project_path = Path(args.project_dir) if args.project_dir else None
        rows = status_plugin(scope=args.scope, project_dir=project_path)
        if args.as_json:
            print(json.dumps(rows, indent=2))
        else:
            for r in rows:
                mark = "installed" if r["installed"] else "not installed"
                ep = r.get("url") or r.get("command") or "-"
                print(
                    f"{r['target']:<12} {mark:<14} skills={len(r['skills'])} rule={r['rule_installed']} endpoint={ep}"
                )
        return 0

    if args.subcommand == "doctor":
        project_path = Path(args.project_dir) if args.project_dir else None
        report = doctor_plugin(
            python_exe=args.python_exe,
            source=args.source,
            scope=args.scope,
            project_dir=project_path,
            skip_torch_probe=args.skip_torch_probe,
        )
        if args.as_json:
            print(json.dumps(report, indent=2))
        else:
            for chk in report["checks"]:
                print(f"[{chk['status'].upper():<5}] {chk['name']}: {chk['detail']}")
        return 0 if report["ok"] else 1

    if args.subcommand == "prune":
        from .jobs import JobStore

        store = JobStore(home=args.mcp_home)
        res = store.prune(
            max_age_days=args.days,
            states=args.states,
            keep_latest=args.keep,
            dry_run=args.dry_run,
        )
        print(json.dumps(res, indent=2))
        return 0

    if args.subcommand == "uninstall":
        targets = TARGETS if args.target == "all" else (args.target,)
        project_path = Path(args.project_dir) if args.project_dir else None
        out_rows = []
        for target in targets:
            if target == "antigravity":
                out_rows.append(
                    uninstall_antigravity(
                        scope=args.scope, project_dir=project_path, dry_run=args.dry_run
                    )
                )
            elif target == "claude-code":
                out_rows.append(
                    uninstall_claude_code(
                        scope=args.scope, project_dir=project_path, dry_run=args.dry_run
                    )
                )
            elif target == "opencode":
                out_rows.append(
                    uninstall_opencode(
                        scope=args.scope, project_dir=project_path, dry_run=args.dry_run
                    )
                )
            elif target == "codex":
                out_rows.append(
                    uninstall_codex(
                        scope=args.scope, project_dir=project_path, dry_run=args.dry_run
                    )
                )
            else:
                out_rows.append(
                    uninstall_dsh(scope=args.scope, project_dir=project_path, dry_run=args.dry_run)
                )
        print(json.dumps(out_rows, indent=2))
        return 0

    spec = resolve_server_spec(
        command=args.command,
        use_python=args.use_python,
        use_uvx=args.use_uvx,
        toolset=args.toolset,
        roots=args.roots,
        mcp_home=args.mcp_home,
        extra_env=args.extra_env,
        url=args.url,
        auth_token=args.auth_token,
    )

    if args.subcommand == "print-config":
        if args.target == "antigravity":
            print(json.dumps(render_antigravity_mcp(spec), indent=2))
        elif args.target == "claude-code":
            print(json.dumps(render_claude_mcp(spec), indent=2))
        elif args.target == "opencode":
            print(json.dumps(render_opencode_mcp(spec), indent=2))
        elif args.target == "codex":
            sys.stdout.write(render_codex_mcp_toml(spec))
        else:
            sys.stdout.write(render_dsh_mcp_yaml(spec))
        return 0

    source = find_plugin_source(args.source)
    targets = TARGETS if args.target == "all" else (args.target,)
    project_path = Path(args.project_dir) if args.project_dir else None
    results = []
    for target in targets:
        if target == "antigravity":
            results.append(
                install_antigravity(
                    source,
                    spec,
                    scope=args.scope,
                    project_dir=project_path,
                    dry_run=args.dry_run,
                )
            )
        elif target == "claude-code":
            results.append(
                install_claude_code(
                    source,
                    spec,
                    scope=args.scope,
                    project_dir=project_path,
                    dry_run=args.dry_run,
                )
            )
        elif target == "opencode":
            results.append(
                install_opencode(
                    source,
                    spec,
                    scope=args.scope,
                    project_dir=project_path,
                    dry_run=args.dry_run,
                )
            )
        elif target == "codex":
            results.append(
                install_codex(
                    source,
                    spec,
                    scope=args.scope,
                    project_dir=project_path,
                    dry_run=args.dry_run,
                )
            )
        else:
            results.append(
                install_dsh(
                    source,
                    spec,
                    scope=args.scope,
                    project_dir=project_path,
                    dry_run=args.dry_run,
                )
            )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
