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


@needs_click
def test_cli_backend_choices_track_the_registry():
    """The --backend literal had gone stale, offering flaml long after lightgbm registered.

    Sourcing the choices from the registry is only worth anything if it stays
    sourced from the registry, so assert the two agree rather than pinning a list
    here that can drift the same way.
    """
    from click.testing import CliRunner

    from normet.backends import backend_registry
    from normet.cli import _build_cli

    runner = CliRunner()
    for cmd in ("do-all", "decompose", "cv"):
        out = runner.invoke(_build_cli(), [cmd, "--help"]).output
        line = next(ln for ln in out.splitlines() if "--backend" in ln)
        for name in backend_registry.available:
            assert name in line, f"{cmd}: {name} missing from {line!r}"


@needs_click
def test_only_do_all_offers_the_zero_shot_backend():
    """chronos-2 is not in the registry, and only do-all has a path for it.

    The registry's contract is train/save/load and Chronos-2 does none of the
    three, so it is added to one command's choice list rather than registered.
    decompose and cv have no zero-shot counterpart and must not advertise one.
    """
    from click.testing import CliRunner

    from normet.cli import _build_cli
    from normet.pipeline import CHRONOS_BACKEND

    runner = CliRunner()

    def backend_line(cmd: str) -> str:
        out = runner.invoke(_build_cli(), [cmd, "--help"]).output
        return next(ln for ln in out.splitlines() if "--backend" in ln)

    assert CHRONOS_BACKEND in backend_line("do-all")
    for cmd in ("decompose", "cv"):
        assert CHRONOS_BACKEND not in backend_line(cmd), cmd


@needs_click
def test_cli_deweather_is_advertised_and_documented():
    from click.testing import CliRunner

    from normet.cli import _build_cli

    runner = CliRunner()
    assert "deweather" in runner.invoke(_build_cli(), ["--help"]).output

    out = runner.invoke(_build_cli(), ["deweather", "--help"]).output
    for flag in ("--target", "--met-vars", "--n-samples", "--quantiles", "--device", "--out"):
        assert flag in out, flag


@needs_click
def test_cli_deweather_refuses_without_met_vars(tmp_path):
    """Without covariates the run is a plain forecast, not a de-weathering.

    Failing here costs a second; failing after the checkpoint download costs
    minutes and produces a column named `normalised` that normalises nothing.
    """
    import pandas as pd
    from click.testing import CliRunner

    from normet.cli import _build_cli

    src = tmp_path / "site.csv"
    pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=48, freq="h"), "PM2.5": range(48)}
    ).to_csv(src, index=False)

    result = CliRunner().invoke(
        _build_cli(),
        ["deweather", str(src), "--target", "PM2.5", "--out", str(tmp_path / "o.csv")],
    )
    assert result.exit_code != 0
    assert "--met-vars is required" in result.output


@needs_click
def test_cli_info_reports_the_foundation_dependencies():
    """`normet info` is where someone checks why the chronos-2 path is unavailable."""
    import json

    from click.testing import CliRunner

    from normet.backends import backend_registry
    from normet.cli import _build_cli

    data = json.loads(CliRunner().invoke(_build_cli(), ["info"]).output)
    for pkg in ("chronos-forecasting", "torch"):
        assert pkg in data["optional"], pkg
    assert data["backends"] == backend_registry.available


@needs_click
def test_cli_split_csv_accepts_a_yaml_list():
    """`covariates: [t2m, blh]` in a config file arrives as a list, not a string."""
    from normet.cli import _split_csv

    assert _split_csv("t2m, blh ,u10") == ["t2m", "blh", "u10"]
    assert _split_csv(["t2m", "blh"]) == ["t2m", "blh"]
    assert _split_csv([0.1, 0.9]) == ["0.1", "0.9"]
    assert _split_csv(None) is None
    assert _split_csv("") is None
