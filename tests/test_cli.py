"""Smoke tests for the click-based CLI (gated on click availability)."""

import importlib.util
import json

import pytest

needs_click = pytest.mark.skipif(
    importlib.util.find_spec("click") is None, reason="click not installed"
)


@needs_click
def test_cli_info_runs():
    from click.testing import CliRunner

    from normet.cli import _build_cli

    runner = CliRunner()
    result = runner.invoke(_build_cli(), ["info"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "normet" in data
    assert "python" in data


@needs_click
def test_cli_help():
    from click.testing import CliRunner

    from normet.cli import _build_cli

    runner = CliRunner()
    result = runner.invoke(_build_cli(), ["--help"])
    assert result.exit_code == 0
    for cmd in ("do-all", "decompose", "scm", "cv", "info"):
        assert cmd in result.output
