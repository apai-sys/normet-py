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
