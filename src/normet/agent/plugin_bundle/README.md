# normet Agent Plugin

Multi-platform AI agent plugin bundle for **normet** (meteorological
normalisation of air-quality time series and synthetic control for intervention
effects), targeting:

- **Antigravity** (`plugin.json`, `mcp_config.json`, `rules/AGENTS.md`, `skills/`, `assets/logo.svg`)
- **Claude Code** (`.claude-plugin/plugin.json`, `.mcp.json`, `skills/`)
- **DeepSeek Harness (`dsh`)** (`.dsh/mcp.yml`, `.mcp.json`, `skills/`)
- **OpenCode** (`opencode.json`, `skills/`)
- **OpenAI Codex CLI (`codex`)** (`.codex/config.toml`, `skills/`)

## Quick Install

Install `normet` with the MCP extra in your Python environment:

```bash
pip install "normet[mcp]"
```

Then install the plugin for your agent(s) using `normet-plugin`:

```bash
# Install for all five platforms (user scope)
normet-plugin install --target all

# Or target a single platform / project workspace
normet-plugin install --target antigravity --scope project --project-dir /path/to/repo
normet-plugin install --target claude-code --scope project --project-dir /path/to/repo
normet-plugin install --target dsh --scope user
normet-plugin install --target opencode --scope project --project-dir /path/to/repo
normet-plugin install --target codex --scope user

# Optional: use uvx for zero-local-install startup, restrict data roots, pass
# env vars, or expose only the 14-tool core workflow
normet-plugin install --target antigravity --uvx --roots /path/to/data --toolset core
normet-plugin install --target all --env OMP_NUM_THREADS=4

# Connect local agents to a remote GPU / HPC `normet-mcp --transport streamable-http` server
normet-plugin install --target all --url http://127.0.0.1:8765/mcp --auth-token "$NORMET_MCP_AUTH_TOKEN"
```

Validate, diagnose, inspect status, prune old jobs, or uninstall at any time:

```bash
normet-plugin validate
normet-plugin doctor
normet-plugin status
normet-plugin prune --days 30
normet-plugin uninstall --target all
```

## Platform Notes

- **Antigravity**: Installs to `~/.gemini/config/plugins/normet/` (`--scope user`) or
  `<project>/.agents/plugins/normet/` (`--scope project`). Manage/toggle it from the
  Customizations UI without manually editing `config.json`.
- **Claude Code**:
  - `--scope user` writes a ready-to-load plugin directory at
    `~/.normet/plugins/claude-code/normet/`; load it with
    `claude --plugin-dir ~/.normet/plugins/claude-code/normet`
    or `claude plugin add ~/.normet/plugins/claude-code/normet`.
  - `--scope project` merges `normet` into `<project>/.mcp.json` and installs skills into
    `<project>/.claude/skills/`.
- **DeepSeek Harness (`dsh`)**:
  - Installs skills into `~/.dsh/skills/` (or `<project>/.dsh/skills/`), merges `normet`
    into `~/.dsh/mcp.yml` (or `<project>/.dsh/mcp.yml`) for `dsh-mcp-project` / Armory,
    and for `--scope user` also patches existing `~/.dsh/profiles/*/cordis.patch.yml`
    files with `@deepseek-ai/dsh-mcp-client` so DeepSeek Harness Desktop picks it up
    automatically.
  - A ready-made DSH bundle with a title, description, icon and detail page is in
    [`dsh/`](dsh/README.md); add that directory through **Plugins > Add plugin**
    instead of running the installer when you want the bundle entry.
- **OpenCode**:
  - `--scope user` merges `mcp.normet` (`type: "local"`) into
    `~/.config/opencode/opencode.json` and installs skills into
    `~/.config/opencode/skills/`.
  - `--scope project` merges `mcp.normet` into `<project>/opencode.json` and installs
    skills into `<project>/.opencode/skills/`. Verify with `opencode mcp list`.
- **OpenAI Codex CLI (`codex`)**:
  - `--scope user` merges `[mcp_servers.normet]` into `~/.codex/config.toml` and installs
    skills into `~/.codex/skills/`.
  - `--scope project` merges `[mcp_servers.normet]` into `<project>/.codex/config.toml`
    and installs skills into `<project>/.codex/skills/`. Verify with `codex mcp list`.

## Security & Context Controls

- Set `NORMET_MCP_ROOTS` (via `--roots`) to restrict file access to your data directory.
- Jobs run as `normet` commands in their own folders under `$NORMET_MCP_HOME`
  (`~/.normet/mcp` by default). A job's command line runs again as it is; its folder
  keeps the inputs, outputs, log and exit status.
- Pass `--toolset core` to `normet-mcp` (or `normet-plugin install --toolset core`) to
  expose only the 14 tools of the standard de-weathering workflow instead of all 26.
- For remote GPU / HPC clusters, start `normet-mcp --transport streamable-http
  --host 127.0.0.1 --port 8765 --auth-token "$TOKEN"` and configure clients with
  `normet-plugin install --url ... --auth-token ...`.
- The studies quoted by the skills use normet's bundled examples (`my1`, `ulez`,
  `beijing`); no data leaves the machine unless a remote server was configured.
