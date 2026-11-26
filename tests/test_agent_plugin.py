"""Tests for the multi-platform AI agent plugin bundle and installer."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from normet import __version__
from normet.agent.plugin import (
    find_plugin_source,
    install_antigravity,
    install_claude_code,
    install_codex,
    install_dsh,
    install_opencode,
    main,
    resolve_server_spec,
    validate_plugin,
)

try:  # Python 3.11+; the backport comes with the dev extra on 3.10
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS = ("deweather-workflow", "deweather-model-choice", "scm-intervention", "transport-receptor")


def test_plugin_bundle_is_valid():
    errors = validate_plugin()
    assert errors == [], errors

    source = find_plugin_source()
    ag_manifest = json.loads((source / "plugin.json").read_text(encoding="utf-8"))
    cc_manifest = json.loads(
        (source / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert ag_manifest["version"] == __version__
    assert cc_manifest["version"] == __version__
    assert pyproject["project"]["version"] == __version__
    assert pyproject["project"]["scripts"]["normet-plugin"] == "normet.agent.plugin:main"


def test_install_antigravity_user_and_project(tmp_path):
    source = find_plugin_source()
    spec = resolve_server_spec(command="/opt/bin/normet-mcp", roots=str(tmp_path))

    home = tmp_path / "home"
    user_res = install_antigravity(source, spec, scope="user", home=home)
    user_dest = Path(user_res["dest"])
    assert user_dest == home / ".gemini" / "config" / "plugins" / "normet"
    assert (user_dest / "plugin.json").is_file()
    assert (user_dest / "assets" / "logo.svg").is_file()
    assert (user_dest / "rules" / "AGENTS.md").is_file()
    for skill in SKILLS:
        assert (user_dest / "skills" / skill / "SKILL.md").is_file()

    cfg = json.loads((user_dest / "mcp_config.json").read_text(encoding="utf-8"))
    assert cfg["mcpServers"]["normet"]["command"] == "/opt/bin/normet-mcp"
    assert cfg["mcpServers"]["normet"]["env"]["NORMET_MCP_ROOTS"] == str(tmp_path)
    # Never touch Antigravity's global config.json
    assert not (home / ".gemini" / "config" / "config.json").exists()

    proj = tmp_path / "project"
    proj_res = install_antigravity(source, spec, scope="project", project_dir=proj)
    assert Path(proj_res["dest"]) == proj / ".agents" / "plugins" / "normet"
    assert (proj / ".agents" / "plugins" / "normet" / "mcp_config.json").is_file()


def test_install_claude_code_user_and_project_merges_mcp(tmp_path):
    source = find_plugin_source()
    spec = resolve_server_spec(command="normet-mcp")

    home = tmp_path / "home"
    user_res = install_claude_code(source, spec, scope="user", home=home)
    user_dest = Path(user_res["dest"])
    assert (user_dest / ".claude-plugin" / "plugin.json").is_file()
    assert (user_dest / "skills" / "deweather-workflow" / "SKILL.md").is_file()
    user_mcp = json.loads((user_dest / ".mcp.json").read_text(encoding="utf-8"))
    assert user_mcp["mcpServers"]["normet"]["type"] == "stdio"
    assert user_mcp["mcpServers"]["normet"]["command"] == "normet-mcp"

    proj = tmp_path / "workspace"
    proj.mkdir()
    existing = {"mcpServers": {"other": {"type": "stdio", "command": "other-mcp", "args": []}}}
    (proj / ".mcp.json").write_text(json.dumps(existing), encoding="utf-8")

    install_claude_code(source, spec, scope="project", project_dir=proj)
    merged = json.loads((proj / ".mcp.json").read_text(encoding="utf-8"))
    assert "other" in merged["mcpServers"]
    assert merged["mcpServers"]["normet"]["command"] == "normet-mcp"
    assert (proj / ".claude" / "skills" / "deweather-workflow" / "SKILL.md").is_file()
    assert (proj / ".claude" / "skills" / "deweather-model-choice" / "SKILL.md").is_file()


def test_install_dsh_user_and_project_merges_yaml(tmp_path):
    source = find_plugin_source()
    spec = resolve_server_spec(command="normet-mcp", roots="/data/normet")

    home = tmp_path / "home"
    install_dsh(source, spec, scope="user", home=home)
    assert (home / ".dsh" / "skills" / "deweather-workflow" / "SKILL.md").is_file()
    yml_text = (home / ".dsh" / "mcp.yml").read_text(encoding="utf-8")
    assert 'name: "normet"' in yml_text
    assert 'NORMET_MCP_ROOTS: "/data/normet"' in yml_text

    proj = tmp_path / "dsh_proj"
    (proj / ".dsh").mkdir(parents=True)
    (proj / ".dsh" / "mcp.yml").write_text(
        'mcpServers:\n  - name: "other"\n    command: "other-mcp"\n    args: []\n',
        encoding="utf-8",
    )
    install_dsh(source, spec, scope="project", project_dir=proj)
    install_dsh(source, spec, scope="project", project_dir=proj)  # idempotent re-run
    proj_yml = (proj / ".dsh" / "mcp.yml").read_text(encoding="utf-8")
    assert 'name: "other"' in proj_yml
    assert proj_yml.count('name: "normet"') == 1
    assert (proj / ".dsh" / "skills" / "transport-receptor" / "SKILL.md").is_file()


def test_install_dsh_patches_cordis_profile_idempotently(tmp_path):
    source = find_plugin_source()
    spec = resolve_server_spec(command="/opt/bin/normet-mcp", toolset="core", roots="/data/normet")

    home = tmp_path / "home"
    prof = home / ".dsh" / "profiles" / "desktop"
    prof.mkdir(parents=True)
    patch_path = prof / "cordis.patch.yml"
    original = (
        "# Existing profile patch\n"
        "- id: agent-default-model\n"
        '  name: "@deepseek-ai/dsh-agent-default-model"\n'
        "  config:\n"
        "    model: deepseek-flash\n"
        "- insert:\n"
        "    - id: mcp-other\n"
        '      name: "@deepseek-ai/dsh-mcp-client"\n'
        "      config:\n"
        "        serverName: other\n"
        "    - id: mcp-normet\n"
        '      name: "@deepseek-ai/dsh-mcp-client"\n'
        "      config:\n"
        "        serverName: normet\n"
        "        command: old-normet\n"
    )
    patch_path.write_text(original, encoding="utf-8")

    # dry_run must not modify cordis.patch.yml
    dry = install_dsh(source, spec, scope="user", home=home, dry_run=True)
    assert str(patch_path) in dry["cordis_patches"]
    assert patch_path.read_text(encoding="utf-8") == original

    # Two real runs should be idempotent and keep comments + sibling items
    install_dsh(source, spec, scope="user", home=home)
    res = install_dsh(source, spec, scope="user", home=home)
    assert str(patch_path) in res["cordis_patches"]

    updated = patch_path.read_text(encoding="utf-8")
    assert "# Existing profile patch" in updated
    assert "id: agent-default-model" in updated
    assert "id: mcp-other" in updated
    assert "old-normet" not in updated
    assert updated.count("id: mcp-normet") == 1
    assert 'command: "/opt/bin/normet-mcp"' in updated
    assert '"--toolset"' in updated
    assert 'NORMET_MCP_ROOTS: "/data/normet"' in updated


def test_install_opencode_user_and_project_merges_config(tmp_path):
    source = find_plugin_source()
    spec = resolve_server_spec(use_uvx=True, toolset="core", roots="/data/normet")

    home = tmp_path / "home"
    install_opencode(source, spec, scope="user", home=home)
    user_cfg = json.loads(
        (home / ".config" / "opencode" / "opencode.json").read_text(encoding="utf-8")
    )
    normet_srv = user_cfg["mcp"]["normet"]
    assert normet_srv["type"] == "local"
    assert normet_srv["command"] == [
        "uvx",
        "--from",
        "normet[mcp]",
        "normet-mcp",
        "--toolset",
        "core",
    ]
    assert normet_srv["enabled"] is True
    assert normet_srv["environment"]["NORMET_MCP_ROOTS"] == "/data/normet"
    assert (home / ".config" / "opencode" / "skills" / "deweather-workflow" / "SKILL.md").is_file()

    proj = tmp_path / "oc_proj"
    proj.mkdir()
    existing = {
        "model": "anthropic/claude-sonnet-4",
        "mcp": {"other": {"type": "local", "command": ["other-mcp"], "enabled": True}},
    }
    (proj / "opencode.json").write_text(json.dumps(existing), encoding="utf-8")
    install_opencode(source, spec, scope="project", project_dir=proj)
    merged = json.loads((proj / "opencode.json").read_text(encoding="utf-8"))
    assert merged["model"] == "anthropic/claude-sonnet-4"
    assert "other" in merged["mcp"]
    assert merged["mcp"]["normet"]["type"] == "local"
    assert (proj / ".opencode" / "skills" / "scm-intervention" / "SKILL.md").is_file()


def test_install_codex_user_and_project_merges_toml(tmp_path):
    source = find_plugin_source()
    spec = resolve_server_spec(use_uvx=True, toolset="core", roots="/data/normet")

    home = tmp_path / "home"
    install_codex(source, spec, scope="user", home=home)
    user_toml_path = home / ".codex" / "config.toml"
    user_cfg = tomllib.loads(user_toml_path.read_text(encoding="utf-8"))
    normet_srv = user_cfg["mcp_servers"]["normet"]
    assert normet_srv["enabled"] is True
    assert normet_srv["command"] == "uvx"
    assert normet_srv["args"] == ["--from", "normet[mcp]", "normet-mcp", "--toolset", "core"]
    assert normet_srv["env"]["NORMET_MCP_ROOTS"] == "/data/normet"
    assert (home / ".codex" / "skills" / "deweather-workflow" / "SKILL.md").is_file()

    proj = tmp_path / "codex_proj"
    (proj / ".codex").mkdir(parents=True)
    cfg_path = proj / ".codex" / "config.toml"
    cfg_path.write_text(
        '# Codex project config\nmodel = "o3"\n\n'
        '[mcp_servers.other]\ncommand = "other-mcp"\nargs = []\n\n'
        '[mcp_servers.normet]\ncommand = "old-normet"\nargs = []\n',
        encoding="utf-8",
    )
    install_codex(source, spec, scope="project", project_dir=proj)
    install_codex(source, spec, scope="project", project_dir=proj)  # idempotent re-run
    raw_proj = cfg_path.read_text(encoding="utf-8")
    assert "# Codex project config" in raw_proj
    assert raw_proj.count("[mcp_servers.normet]") == 1
    parsed_proj = tomllib.loads(raw_proj)
    assert parsed_proj["model"] == "o3"
    assert parsed_proj["mcp_servers"]["other"]["command"] == "other-mcp"
    assert parsed_proj["mcp_servers"]["normet"]["command"] == "uvx"
    assert (proj / ".codex" / "skills" / "transport-receptor" / "SKILL.md").is_file()


def test_dry_run_and_uvx_and_toolset_core(tmp_path):
    source = find_plugin_source()
    spec = resolve_server_spec(use_uvx=True, toolset="core", roots="/tmp/data")
    assert spec["command"] == "uvx"
    assert spec["args"] == ["--from", "normet[mcp]", "normet-mcp", "--toolset", "core"]
    assert spec["env"]["NORMET_MCP_ROOTS"] == "/tmp/data"

    home = tmp_path / "dry_home"
    install_antigravity(source, spec, scope="user", home=home, dry_run=True)
    install_claude_code(source, spec, scope="user", home=home, dry_run=True)
    install_dsh(source, spec, scope="user", home=home, dry_run=True)
    install_opencode(source, spec, scope="user", home=home, dry_run=True)
    install_codex(source, spec, scope="user", home=home, dry_run=True)
    assert not home.exists()


def test_cli_validate_and_print_config(capsys):
    assert main(["validate"]) == 0
    out = capsys.readouterr().out
    assert "valid" in out.lower()

    for target in ("antigravity", "claude-code", "dsh", "opencode", "codex"):
        assert main(["print-config", "--target", target, "--uvx", "--toolset", "core"]) == 0
        cfg_out = capsys.readouterr().out
        assert "uvx" in cfg_out
        assert "--toolset" in cfg_out


def test_mcp_server_exports_core_toolset_without_torch():
    code = (
        "import json, sys; import normet.mcp_server as m; "
        "print(json.dumps({'torch': 'torch' in sys.modules, "
        "'core': len(m.CORE_TOOLS), "
        "'names': sorted(m.CORE_TOOLS)}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(out.stdout)
    assert data["torch"] is False
    assert data["core"] == 14
    for name in ("normet_inspect_data", "normet_start_do_all", "normet_job_status"):
        assert name in data["names"]


def test_build_server_core_toolset_registers_only_core(monkeypatch):
    from normet.app import mcp_server as m

    class FakeServer:
        def __init__(self, *args, **kwargs):
            self.names = []

        def tool(self, name=None, description=None):
            def decorator(fn):
                self.names.append(name)
                return fn

            return decorator

        def resource(self, *args, **kwargs):
            return lambda fn: fn

        def prompt(self, *args, **kwargs):
            return lambda fn: fn

    monkeypatch.setattr(m, "_sdk", lambda: (FakeServer, object, RuntimeError))
    all_tools = m.build_server(object(), toolset="all")
    core_tools = m.build_server(object(), toolset="core")
    assert len(all_tools.names) == 26
    assert set(core_tools.names) == set(m.CORE_TOOLS)
    assert len(core_tools.names) == 14
    with pytest.raises(ValueError):
        m.build_server(object(), toolset="everything")


def test_plugin_bundle_parity_with_package_data():
    repo_bundle = REPO_ROOT / "plugin"
    pkg_bundle = REPO_ROOT / "src" / "normet" / "agent" / "plugin_bundle"
    assert pkg_bundle.is_dir()
    repo_files = sorted(p.relative_to(repo_bundle) for p in repo_bundle.rglob("*") if p.is_file())
    pkg_files = sorted(p.relative_to(pkg_bundle) for p in pkg_bundle.rglob("*") if p.is_file())
    assert repo_files == pkg_files
    for rel in repo_files:
        assert (repo_bundle / rel).read_bytes() == (pkg_bundle / rel).read_bytes(), str(rel)

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pkg_data = pyproject["tool"]["setuptools"]["package-data"]["normet.agent"]
    assert "plugin_bundle/*" in pkg_data
    assert "plugin_bundle/.*" in pkg_data
    assert "plugin_bundle/.*/*" in pkg_data
    assert any(dep.startswith("tomli") for dep in pyproject["project"]["dependencies"])


def test_remote_url_env_guardrails_status_uninstall_and_doctor(tmp_path, monkeypatch, capsys):
    from normet.agent.plugin import (
        doctor_plugin,
        render_antigravity_mcp,
        render_claude_mcp,
        render_codex_mcp_toml,
        render_dsh_cordis_patch,
        render_dsh_mcp_yaml,
        render_opencode_mcp,
        status_plugin,
        uninstall_antigravity,
        uninstall_claude_code,
        uninstall_codex,
        uninstall_dsh,
        uninstall_opencode,
    )

    spec_env = resolve_server_spec(
        command="normet-mcp", extra_env=["OMP_NUM_THREADS=4", "FOO=bar=baz"]
    )
    assert spec_env["env"]["OMP_NUM_THREADS"] == "4"
    assert spec_env["env"]["FOO"] == "bar=baz"

    http_spec = resolve_server_spec(url="http://127.0.0.1:8765/mcp", auth_token="tok123")
    assert (
        render_antigravity_mcp(http_spec)["mcpServers"]["normet"]["url"]
        == "http://127.0.0.1:8765/mcp"
    )
    assert render_claude_mcp(http_spec)["mcpServers"]["normet"]["type"] == "http"
    assert 'transport: "http"' in render_dsh_mcp_yaml(http_spec)
    assert "transport: http" in render_dsh_cordis_patch(http_spec)
    assert render_opencode_mcp(http_spec)["mcp"]["normet"]["type"] == "remote"
    codex_http = tomllib.loads(render_codex_mcp_toml(http_spec))
    assert codex_http["mcp_servers"]["normet"]["url"] == "http://127.0.0.1:8765/mcp"
    assert codex_http["mcp_servers"]["normet"]["http_headers"]["Authorization"] == "Bearer tok123"

    # Install across all 5 targets in user scope and verify guardrails, status, and clean uninstall
    source = find_plugin_source()
    home = tmp_path / "home"
    (home / ".dsh" / "profiles" / "desktop").mkdir(parents=True)
    (home / ".dsh" / "profiles" / "desktop" / "cordis.patch.yml").write_text(
        '- insert:\n    - id: mcp-other\n      name: "@deepseek-ai/dsh-mcp-client"\n',
        encoding="utf-8",
    )
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "config.toml").write_text(
        '[mcp_servers.other]\ncommand = "other-mcp"\nargs = []\n', encoding="utf-8"
    )
    (home / ".config" / "opencode").mkdir(parents=True)
    (home / ".config" / "opencode" / "opencode.json").write_text(
        json.dumps({"mcp": {"other": {"type": "local", "command": ["other"], "enabled": True}}}),
        encoding="utf-8",
    )

    install_antigravity(source, spec_env, scope="user", home=home)
    install_claude_code(source, spec_env, scope="user", home=home)
    install_dsh(source, spec_env, scope="user", home=home)
    install_opencode(source, spec_env, scope="user", home=home)
    install_codex(source, spec_env, scope="user", home=home)

    # Guardrails rules copied to cross-platform locations
    assert (
        home / ".normet" / "plugins" / "claude-code" / "normet" / "rules" / "normet.md"
    ).is_file()
    assert (home / ".dsh" / "rules" / "normet.md").is_file()
    oc_rule = home / ".config" / "opencode" / "rules" / "normet.md"
    assert oc_rule.is_file()
    oc_cfg = json.loads(
        (home / ".config" / "opencode" / "opencode.json").read_text(encoding="utf-8")
    )
    assert str(oc_rule) in oc_cfg["instructions"]
    assert (home / ".codex" / "rules" / "normet.md").is_file()

    st = status_plugin(scope="user", home=home)
    assert all(row["installed"] and row["rule_installed"] for row in st)

    # Uninstall cleanly preserves siblings
    assert uninstall_antigravity(scope="user", home=home)["removed"] is True
    assert uninstall_claude_code(scope="user", home=home)["removed"] is True
    assert uninstall_dsh(scope="user", home=home)["removed"] is True
    assert uninstall_opencode(scope="user", home=home)["removed"] is True
    assert uninstall_codex(scope="user", home=home)["removed"] is True

    st_after = status_plugin(scope="user", home=home)
    assert all(not row["installed"] for row in st_after)
    assert "mcp-other" in (home / ".dsh" / "profiles" / "desktop" / "cordis.patch.yml").read_text(
        encoding="utf-8"
    )
    assert (
        "other"
        in json.loads(
            (home / ".config" / "opencode" / "opencode.json").read_text(encoding="utf-8")
        )["mcp"]
    )
    assert (
        "other"
        in tomllib.loads((home / ".codex" / "config.toml").read_text(encoding="utf-8"))[
            "mcp_servers"
        ]
    )

    monkeypatch.setenv("NORMET_MCP_ROOTS", str(tmp_path / "only_sub"))
    (tmp_path / "only_sub").mkdir()
    doc = doctor_plugin(home=home, skip_torch_probe=True)
    roots_chk = next(c for c in doc["checks"] if c["name"] == "roots")
    assert roots_chk["status"] == "warn"
    assert main(["doctor", "--skip-torch-probe", "--json"]) == 0
    doc_out = json.loads(capsys.readouterr().out)
    assert doc_out["ok"] is True


def test_job_store_prune_and_cli(tmp_path, capsys):
    from normet.agent.jobs import JobStore

    store = JobStore(home=str(tmp_path / "mcp_home"))
    old = time.time() - 10 * 86400.0
    for job_id, state in (
        ("j1", "failed"),
        ("j2", "done"),
        ("j3", "done"),
        ("j4", "queued"),
    ):
        folder = Path(store.root) / job_id
        folder.mkdir(parents=True)
        (folder / "job.json").write_text(
            json.dumps(
                {"id": job_id, "kind": "run", "argv": [], "label": "", "folder": str(folder)}
            ),
            encoding="utf-8",
        )
        if state in ("done", "failed"):
            (folder / "exit.json").write_text(
                json.dumps({"state": state, "code": 0}), encoding="utf-8"
            )
            (folder / "stderr.txt").write_text(f"step line {job_id}\n", encoding="utf-8")
        if job_id in ("j1", "j2"):
            os.utime(folder / "job.json", (old, old))

    dry = store.prune(states=("failed",), dry_run=True)
    assert dry["removed"] == ["j1"] and (Path(store.root) / "j1").is_dir()

    pruned = store.prune(max_age_days=1.0, keep_latest=1)
    assert set(pruned["removed"]) == {"j1", "j2"}
    assert "j3" in pruned["kept"] and "j4" in pruned["kept"]
    assert not (Path(store.root) / "j1").exists()
    assert main(["prune", "--mcp-home", str(tmp_path / "mcp_home"), "--dry-run"]) == 0
    assert "removed" in capsys.readouterr().out
