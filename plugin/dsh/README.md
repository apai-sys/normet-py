# normet bundle for DeepSeek Harness

A DSH bundle that connects normet's MCP server as one row. It replaces the raw
`- insert: mcp-normet` that `normet-plugin install --target dsh` writes into a
profile patch, and gives the plugin a title, description, icon and detail page
in **Plugins**.

```
package.json          bundle manifest (dsh.bundle.patch, locale/icon exports)
cordis.patch.yml      inserts @deepseek-ai/dsh-mcp-client as row mcp-normet
locale/en.json        title, description, detail path
locale/zh.json        the same in Chinese
docs/DETAIL.md        one feature image (docs/preview.svg), per DSH's rules
docs/preview.svg      the de-weathering picture, text-free
icon.svg              normet's logo, from plugin/assets/logo.svg
```

## Install

**Plugins** > **Add plugin** > the `plugin/dsh` directory, or:

```
plugin_manager(action: "install_bundle", target: "/absolute/path/to/plugin/dsh")
```

A local install links this directory, so edits here reload at the next
recomposition (`set_plugin` on `mcp-normet` with `enabled: true` forces one).

## What the row starts

```yaml
command: normet-mcp
args:    []
```

The command must resolve in the environment where `normet[mcp]` is installed;
point it at that environment's `normet-mcp` when DSH does not inherit its
`PATH`. To reach a GPU or HPC host, replace the row's config with
`transport: streamable-http` and `url` from a running
`normet-mcp --transport streamable-http --auth-token ...`.

Jobs are kept under `$NORMET_MCP_HOME` (`~/.normet/mcp` by default), and the
server reads and writes files only under `$NORMET_MCP_ROOTS` (your home folder
when unset).

## Skills are separate

The four skills (`deweather-workflow`, `deweather-model-choice`,
`scm-intervention`, `transport-receptor`) are read from `~/.dsh/skills`, which
`normet-plugin install --target dsh` fills; this bundle does not ship them. Do
not run that installer again while the bundle is installed — it writes a second
`mcp-normet` row into `~/.dsh/profiles/<profile>/cordis.patch.yml`, and two
rows with one `serverName` compete for the same `mcp__normet__*` tool names.

## Uninstall

```
plugin_manager(action: "remove_bundle", target: "@local/normet")
```

It removes the row and leaves `~/.dsh/skills` and `~/.dsh/mcp.yml` alone.

## Artwork

`icon.svg` is normet's logo from `plugin/assets/logo.svg` (MIT, this
repository); `docs/preview.svg` is drawn for this bundle. Both are text-free.
