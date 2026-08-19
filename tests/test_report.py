"""HTML report generation — round-trip a synthetic NormetRun."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from normet.report import generate_html
from normet.utils.provenance import make_run

needs_click = pytest.mark.skipif(
    importlib.util.find_spec("click") is None, reason="click not installed"
)


def test_generate_html_minimal(tmp_path):
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    result = pd.DataFrame(
        {"observed": range(20), "normalised": [x * 1.1 for x in range(20)]},
        index=dates,
    )
    run = make_run(
        result=result,
        kind="normalise",
        seed=7,
        config={"n_samples": 100, "backend": "flaml"},
    )
    out = generate_html(run, tmp_path / "report.html")
    assert out.exists()
    content = out.read_text()
    assert "normet report" in content
    assert "normalise" in content
    assert "normalised" in content
    # Should contain a base64-encoded inline plot (data URI).
    assert "data:image/png;base64" in content
    # Provenance section
    assert "data_hash" in content or "kind" in content


def test_generate_html_for_scm(scm_panel, tmp_path):
    from normet.causal.scm import scm

    res = scm(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=["D1", "D2", "D3", "D4", "D5", "D6"],
    )
    run = make_run(
        result=res["synthetic"],
        kind="scm",
        seed=42,
        config={"cutoff_date": "2023-05-01", "scm_backend": "scm"},
    )
    out = generate_html(run, tmp_path / "scm_report.html", title="SCM toy report")
    assert out.exists()
    assert "SCM toy report" in out.read_text()


def test_report_to_markdown_minimal(tmp_path):
    from normet.report import report_to_markdown

    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    result = pd.DataFrame(
        {"observed": range(20), "normalised": [x * 1.1 for x in range(20)]},
        index=dates,
    )
    run = make_run(
        result=result,
        kind="normalise",
        seed=7,
        config={"n_samples": 100, "backend": "flaml"},
    )
    out = report_to_markdown(run, tmp_path / "report.md")
    assert out.exists()
    content = out.read_text()
    assert "# normet report" in content
    assert "## Result Preview" in content
    assert "observed" in content
    assert "normalised" in content
    assert "## Full Provenance Metadata" in content


def test_generate_html_for_bayesian_scm(tmp_path):
    from normet.report import generate_html

    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    result = pd.DataFrame(
        {
            "observed": np.sin(np.linspace(0, 10, 30)),
            "synthetic": np.sin(np.linspace(0, 10, 30)) + 0.1,
            "synthetic_low": np.sin(np.linspace(0, 10, 30)) - 0.2,
            "synthetic_high": np.sin(np.linspace(0, 10, 30)) + 0.3,
            "effect": -0.1 * np.ones(30),
            "effect_low": -0.3 * np.ones(30),
            "effect_high": 0.2 * np.ones(30),
        },
        index=dates,
    )
    run = make_run(
        result=result,
        kind="bayesian_scm",
        seed=42,
        config={"cutoff_date": "2023-01-15", "scm_backend": "bayesian_scm"},
    )
    out = generate_html(run, tmp_path / "bayesian_scm_report.html", title="Bayesian SCM report")
    assert out.exists()
    content = out.read_text()
    assert "Bayesian SCM report" in content
    assert "data:image/png;base64" in content


@needs_click
def test_cli_report_to_markdown(tmp_path):
    import joblib
    from click.testing import CliRunner

    from normet.cli import _build_cli

    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    result = pd.DataFrame(
        {"observed": range(20), "normalised": [x * 1.1 for x in range(20)]},
        index=dates,
    )
    run = make_run(
        result=result,
        kind="normalise",
        seed=7,
        config={"n_samples": 100, "backend": "flaml"},
    )
    # Save a run file
    run_file = tmp_path / "run.joblib"
    joblib.dump(run, run_file)

    runner = CliRunner()
    cli = _build_cli()

    # Test html generation via CLI
    html_out = tmp_path / "cli_report.html"
    res_html = runner.invoke(cli, ["report", str(run_file), "--out", str(html_out)])
    assert res_html.exit_code == 0
    assert html_out.exists()

    # Test md generation via CLI
    md_out = tmp_path / "cli_report.md"
    res_md = runner.invoke(cli, ["report", str(run_file), "--out", str(md_out)])
    assert res_md.exit_code == 0
    assert md_out.exists()
    assert "Result Preview" in md_out.read_text()
