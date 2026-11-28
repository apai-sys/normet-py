# AI assistants (MCP server)

`normet-mcp` serves normet to an AI assistant over the Model Context Protocol,
so Claude Code, Claude Desktop or any other MCP client can drive an analysis:
look at a table, train and normalise, decompose, run a synthetic control, read
the results and draw them.

```bash
pip install "normet[mcp]"
claude mcp add normet -- normet-mcp          # Claude Code
# Claude Desktop, claude_desktop_config.json:
#   {"mcpServers": {"normet": {"command": "normet-mcp"}}}
```

- **Jobs.** Each fit is a `normet` CLI command run in its own process and folder
  (`$NORMET_MCP_HOME`, `~/.normet/mcp` by default). Its command line is its
  record and runs again as it is, without the assistant.
- **Readings, not data.** The assistant is given normet's readings of a table or
  a result as text (a model's `modStats` on the held-out rows, the normalised
  series' means by year or month, a synthetic control's pre-period fit and
  effect), plus figures. The tables stay on your machine, and files are read only
  under `$NORMET_MCP_ROOTS` (your home folder by default).
- **You decide.** The server tells the assistant to propose the target, the
  predictors and the donor pool with its reasons, to let you confirm them, and to
  quote only the numbers the tools return.

Because each job is a CLI command, {doc}`cli` documents what the assistant can run.
The desktop app has its own assistant panel, described in {doc}`gui`.

## Fewer tools, or a remote server

`normet-mcp --toolset core` registers only the 14 tools of the standard
de-weathering workflow (inspect, start, status, read, trend, figure, report)
instead of all 27. Fewer tools means a shorter menu for the assistant to choose
from; the omitted ones are the specialised analyses (Chronos-2 and in-context
de-weathering, conserved-Ox and regime normalisation, Shapley/ICN
decomposition, benchmarks, met features, imputation and backtesting, event
detection, trend breaks, multi-receptor inversion), still reachable through
`normet` itself.

For a GPU or an HPC login node, `normet-mcp --transport streamable-http
--host 127.0.0.1 --port 8765 --path /mcp --auth-token "$TOKEN"` serves the
same tools over HTTP; clients elsewhere connect to the URL. Without a token the
server warns when bound to a non-loopback address. `--stateless` drops the
per-client session, for servers behind a load balancer.

## Installing into an agent

`normet-plugin` (`pip install "normet[mcp]"`) writes the server, a guardrails
rule and four skills (de-weathering workflow, model choice, SCM intervention,
transport/receptor analysis) into Antigravity, Claude Code, DeepSeek Harness,
OpenCode or OpenAI Codex:

```bash
normet-plugin install --target all --toolset core
normet-plugin status
normet-plugin doctor
normet-plugin uninstall --target all
```

`--scope project` targets a workspace instead of your home folder; `--uvx` starts
the server through `uvx --from "normet[mcp]"`; `--roots` and `--env` pass
`NORMET_MCP_ROOTS` and other environment variables; `--url`/`--auth-token`
point the client at a remote HTTP server. `normet-plugin prune --days 30`
retires old job folders under `NORMET_MCP_HOME`.

DeepSeek Harness users can instead add the ready-made bundle in `plugin/dsh/`
through **Plugins > Add plugin**; it carries a title, description, icon and
detail page, and starts the same server.
