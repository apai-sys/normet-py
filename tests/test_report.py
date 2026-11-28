"""HTML report generation — round-trip a synthetic NormetRun."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from normet.reporting.report import generate_html
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
    from normet.reporting.report import report_to_markdown

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
    from normet.reporting.report import generate_html

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

    from normet.app.cli import _build_cli

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


def test_generate_html_does_not_retain_the_figures_it_creates(tmp_path):
    """A report generated per site in a loop must not accumulate figures.

    generate_html builds its own plot, serialises it to an inline PNG and has
    no further use for it, but pyplot keeps every figure alive until closed.
    Figures handed in through extra_plots belong to the caller and must survive.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from normet import generate_html_report, make_run

    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    res = pd.DataFrame(
        {"observed": np.linspace(10, 20, 48), "normalised": np.linspace(11, 19, 48)}, index=idx
    )
    run = make_run(result=res, model=None, df_prep=None, df=None, kind="normalise", config={})

    plt.close("all")
    caller_fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    before = len(plt.get_fignums())

    for i in range(5):
        generate_html_report(run, tmp_path / f"r{i}.html", extra_plots=[caller_fig])

    assert len(plt.get_fignums()) == before, "generate_html retained a figure it created"
    assert caller_fig.number in plt.get_fignums(), "a caller-owned figure was closed"
    plt.close("all")


def _normalise_run():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    result = pd.DataFrame(
        {"observed": range(20), "normalised": [x * 1.1 for x in range(20)]}, index=dates
    )
    return make_run(result=result, kind="normalise", seed=7, config={"backend": "flaml"})


def test_a_report_in_chinese(tmp_path):
    """lang="zh": its headings, labels and footer in Chinese; the data's words as they are."""
    content = generate_html(_normalise_run(), tmp_path / "zh.html", lang="zh").read_text("utf-8")
    assert '<html lang="zh-CN">' in content
    for words in (
        "normet 报告——normalise",
        "<h2>图</h2>",
        "结果预览",
        "溯源信息",
        "随机种子",
        "由 normet 于",
    ):
        assert words in content, words
    assert "normalised" in content  # a column's name stays as it is
    en = generate_html(_normalise_run(), tmp_path / "en.html").read_text("utf-8")
    assert '<html lang="en">' in en and "Result preview" in en and "结果预览" not in en


def test_a_report_figure_in_chinese():
    """The figure's texts as the window writes them, where the machine has a Chinese face."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.text import Text

    from normet.gui._i18n import figure_font
    from normet.reporting.report import translate_figure
    from normet.viz.plotting import normalise_plot

    if figure_font("zh") is None:
        pytest.skip("no face with Chinese characters on this machine")
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({"observed": np.arange(60.0), "normalised": np.arange(60.0)}, index=dates)
    ax = normalise_plot(df, title="Meteorologically normalised NO2", ylabel="NO2")
    # A name the analyst wrote in Chinese is data: not translated, but it must get
    # the face all the same or the text draws boxes.
    ax.set_xlabel("交通源")
    translate_figure(ax.figure, "zh")
    texts = {t.get_text() for t in ax.figure.findobj(Text)}
    assert "NO2 的气象归一化序列" in texts
    assert {"观测", "归一化（去气象化）"} <= texts
    assert figure_font("zh") in ax.xaxis.label.get_fontfamily()


@needs_click
def test_cli_report_in_chinese(tmp_path):
    import joblib
    from click.testing import CliRunner

    from normet.app.cli import _build_cli

    run_file = tmp_path / "run.joblib"
    joblib.dump(_normalise_run(), run_file)
    out = tmp_path / "r.html"
    res = CliRunner().invoke(
        _build_cli(), ["report", str(run_file), "--out", str(out), "--lang", "zh"]
    )
    assert res.exit_code == 0, res.output
    assert "结果预览" in out.read_text("utf-8")


def test_log_scale_note_only_when_the_flag_is_on(tmp_path):
    """A run whose config carries log_transform=True gets a scale note with the
    low-value share; a linear run's report is exactly what it was before."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    observed = np.concatenate([np.linspace(0.1, 0.9, 10), np.linspace(2.0, 80.0, 10)])
    result = pd.DataFrame({"observed": observed, "normalised": observed * 1.1}, index=dates)
    log_run = make_run(result=result, kind="normalise", seed=7, config={"log_transform": True})
    content = generate_html(log_run, tmp_path / "log.html").read_text("utf-8")
    assert "Log-scale fit" in content
    assert "log1p(value)" in content
    assert "50%" in content  # 10 of 20 observations sit below 1

    lin_run = make_run(result=result, kind="normalise", seed=7, config={"backend": "flaml"})
    lin = generate_html(lin_run, tmp_path / "lin.html").read_text("utf-8")
    assert "Log-scale fit" not in lin
    assert "nm-note" not in lin

    zh = generate_html(log_run, tmp_path / "log_zh.html", lang="zh").read_text("utf-8")
    assert "对数尺度拟合" in zh


def test_auto_plot_handles_imputation_pdp_and_backtest(monkeypatch):
    """_auto_plot recognises Chronos-2 imputation, PDP, and backtest lead frames."""
    import matplotlib.pyplot as plt

    from normet.reporting.report import _auto_plot
    from normet.viz import plotting

    called: list[str] = []
    orig_plot_imp = plotting.plot_imputation
    orig_pdp_grid = plotting.pdp_grid

    def _spy_imp(*a, **kw):
        called.append("imputation")
        return orig_plot_imp(*a, **kw)

    def _spy_pdp(*a, **kw):
        called.append("pdp")
        return orig_pdp_grid(*a, **kw)

    monkeypatch.setattr(plotting, "plot_imputation", _spy_imp)
    monkeypatch.setattr(plotting, "pdp_grid", _spy_pdp)

    dates = pd.date_range("2024-01-01", periods=16, freq="h")
    imp_df = pd.DataFrame(
        {
            "observed": [1.0 if i not in (5, 6, 7) else np.nan for i in range(16)],
            "imputed": np.linspace(1.0, 5.0, 16),
            "is_imputed": [i in (5, 6, 7) for i in range(16)],
            "lower": np.linspace(0.5, 4.5, 16),
            "upper": np.linspace(1.5, 5.5, 16),
        },
        index=dates,
    )
    fig_imp = _auto_plot(make_run(result=imp_df, kind="impute"))
    assert fig_imp is not None
    assert "imputation" in called

    pdp_df = pd.DataFrame(
        {
            "variable": ["t2m"] * 4,
            "value": [0.0, 5.0, 10.0, 15.0],
            "pdp_mean": [20.0, 18.0, 16.0, 15.0],
            "pdp_lower": [17.0, 15.0, 13.0, 12.0],
            "pdp_upper": [23.0, 21.0, 19.0, 18.0],
        }
    )
    fig_pdp = _auto_plot(make_run(result=pdp_df, kind="pdp"))
    assert fig_pdp is not None
    assert "pdp" in called

    bt_df = pd.DataFrame(
        {
            "lead": [1, 2, 3, 4],
            "crps": [0.8, 0.9, 1.0, 1.1],
            "mae": [0.6, 0.7, 0.8, 0.9],
            "rmse": [0.9, 1.0, 1.1, 1.2],
            "coverage_80": [0.82, 0.81, 0.80, 0.78],
        }
    )
    fig_bt = _auto_plot(make_run(result=bt_df, kind="backtest"))
    assert fig_bt is not None and fig_bt.axes[0].get_xlabel() == "Lead step"
    plt.close("all")


def test_report_to_markdown_handles_non_dataframe_results(tmp_path):
    """report_to_markdown supports None, empty DataFrames, dicts, Series, and arbitrary objects."""
    from normet.reporting.report import report_to_markdown

    # 1. None and empty DataFrame
    for idx, res in enumerate((None, pd.DataFrame())):
        out = report_to_markdown(make_run(result=res, kind="empty"), tmp_path / f"empty_{idx}.md")
        assert "*(empty)*" in out.read_text("utf-8")

    # 2. Dict (e.g. SCM bundle with DataFrames, nested dict, scalars, empty DataFrame)
    scm_like = {
        "synthetic": pd.DataFrame(
            {"observed": [1.0, 2.0], "synthetic": [1.1, 1.9]},
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        ),
        "weights": {"D1": 0.6, "D2": 0.4},
        "pre_rmse": 0.123,
        "empty_df": pd.DataFrame(),
    }
    md_dict = report_to_markdown(
        make_run(result=scm_like, kind="scm"), tmp_path / "dict.md"
    ).read_text("utf-8")
    assert "### synthetic" in md_dict
    assert "### weights" in md_dict
    assert '"D1": 0.6' in md_dict
    assert "- **pre_rmse**: `0.123`" in md_dict
    assert "### empty_df" in md_dict
    assert "*(empty)*" in md_dict

    # 3. Series
    ser = pd.Series([1.5, 2.5], index=pd.date_range("2024-01-01", periods=2, freq="D"), name="pm25")
    md_ser = report_to_markdown(
        make_run(result=ser, kind="series"), tmp_path / "series.md"
    ).read_text("utf-8")
    assert "pm25" in md_ser and "1.5" in md_ser

    # 4. Arbitrary object
    class _CustomObj:
        def __repr__(self) -> str:
            return "CustomObj(alpha=42)"

    md_obj = report_to_markdown(
        make_run(result=_CustomObj(), kind="custom"), tmp_path / "obj.md"
    ).read_text("utf-8")
    assert "CustomObj(alpha=42)" in md_obj


def test_reports_normalise_missing_values_consistently(tmp_path):
    """_df_table, generate_html, and report_to_markdown render all missing values as '—'."""
    import re

    from normet.reporting.report import _df_table, report_to_markdown

    df = pd.DataFrame(
        {
            "flt": [1.23456, np.nan],
            "obj": pd.Series(["x", None], dtype=object),
            "str": pd.array(["a", None], dtype="string"),
            "dt": pd.to_datetime(["2021-01-01", None]),
            "intn": pd.array([7, pd.NA], dtype="Int64"),
            "booln": pd.array([True, pd.NA], dtype="boolean"),
        }
    )
    tbl_html = _df_table(df)
    assert "<td>1.235</td>" in tbl_html
    assert tbl_html.count("<td>—</td>") == 6
    for bad in ("<td>None</td>", "<td>&lt;NA&gt;</td>", "<td>NaT</td>", "<td>nan</td>"):
        assert bad not in tbl_html

    run = make_run(result=df, kind="normalise", seed=None)
    run.metadata["user"] = ""

    html_txt = generate_html(run, tmp_path / "missing.html").read_text("utf-8")
    assert re.search(r"<dt>seed</dt><dd>—</dd>", html_txt)
    assert re.search(r"<dt>user</dt><dd>—</dd>", html_txt)
    assert "<dd>None</dd>" not in html_txt

    md_txt = report_to_markdown(run, tmp_path / "missing.md").read_text("utf-8")
    assert "- **seed**: `—`" in md_txt
    assert "- **user**: `—`" in md_txt
    assert "`None`" not in md_txt
    for bad_md in ("| None |", "| <NA> |", "| NaT |", "| nan |"):
        assert bad_md not in md_txt
    assert "| — |" in md_txt
