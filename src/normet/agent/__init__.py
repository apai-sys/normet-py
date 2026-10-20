"""normet for an AI agent: the tools an assistant drives normet with, and the
jobs they run.

One layer, two front ends -- the MCP server (:mod:`normet.mcp_server`,
``normet-mcp``) and the GUI's Assistant -- and nothing here depends on MCP.

- :mod:`~normet.agent.jobs` -- do-all, decompose, SCM and cross-validation runs,
  each a ``normet`` command in a child process of its own: the command run is
  the record of the job, and runs again as it is.
- :mod:`~normet.agent.readings` -- a table, a model, a normalised series, a
  decomposition, a synthetic control or a cross-validation, written out as text.
- :mod:`~normet.agent.tools` -- what an assistant asks of normet: the examples,
  a table's readings, a job started, watched or cancelled, a result read, drawn
  or reported.
- :mod:`~normet.agent.llm` -- a language model calling tools: Anthropic's and
  OpenAI-compatible APIs, and the loop between model and tools (the GUI's
  Assistant).
- :mod:`~normet.agent.mcp_http` -- tools served over MCP's HTTP transport on
  127.0.0.1, for a command-line agent the GUI's Assistant talks through.

What a tool gives back is normet's own readings: every number an assistant
quotes should be one it was given.
"""

from .jobs import JobStore
from .tools import EXAMPLES, FIGURES, Tools

__all__ = ["EXAMPLES", "FIGURES", "JobStore", "Tools"]
